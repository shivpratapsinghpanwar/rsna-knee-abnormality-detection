"""Out-of-fold predictions over the training corpus, and what they are worth.

Each fold's checkpoint scores only the studies it never trained on, so the
assembled matrix covers the corpus without leakage. Three things need this:

1. An honest ensemble number. Per-fold validation AUC measures one model on one
   fifth of the data; the OOF matrix measures the actual blend on all of it.
2. Per-target blend weights. Folds are not equally good at every target, and the
   weights can only be chosen on out-of-fold predictions.
3. Pseudo-labelling. Blending model predictions back into the weak labels
   denoises the labeller, and the predictions have to be out-of-fold or the
   model simply relearns its own training-set memorisation.

    python scripts/oof.py --ckpt work/models/fold*.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import CSV, MAX_SERIES, WORK  # noqa: E402
from rsna_knee.data import (  # noqa: E402
    index_corpus, make_folds, report_groups,
)
from rsna_knee.labels import ID_COL, TARGETS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import TTA, load_models, study_slices_from_prep, to_model_input  # noqa: E402


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, dict[str, float]]:
    per = {}
    for j, t in enumerate(TARGETS):
        col = y[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2 or len(np.unique(col[ok])) < 2:
            continue
        per[t] = float(roc_auc_score(col[ok], p[ok, j]))
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--tta", default="none")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    models = load_models([Path(p) for p in args.ckpt], device)
    folds_of = [int(torch.load(p, map_location="cpu", weights_only=False)["fold"])
                for p in args.ckpt]

    labels = pd.read_csv(WORK / "weak_labels.csv")
    corpus = index_corpus("train")
    if corpus.empty:
        raise SystemExit("no preprocessed mosaics found")
    labels = labels[labels.StudyInstanceUID.isin(set(corpus.StudyInstanceUID))]
    labels = labels.reset_index(drop=True)

    # Must match training exactly, or "out-of-fold" is a lie.
    fold = make_folds(labels, args.folds, args.seed,
                      groups=report_groups(labels, CSV))
    by_study: dict[str, dict[int, str]] = {}
    for r in corpus.itertuples():
        by_study.setdefault(r.StudyInstanceUID, {})[r.slot] = r.path

    tta = [t.strip() for t in args.tta.split(",") if t.strip()]
    oof = np.full((len(labels), len(TARGETS)), np.nan, dtype=np.float32)
    covered = np.zeros(len(labels), dtype=bool)

    for (model, size, _crop), f in zip(models, folds_of):
        idx = np.where(fold == f)[0]
        print(f"fold {f}: scoring {len(idx)} held-out studies at size {size}")
        for start in range(0, len(idx), args.batch):
            chunk = idx[start:start + args.batch]
            raw, masks = [], []
            for i in chunk:
                v, m = study_slices_from_prep(
                    labels.StudyInstanceUID.iloc[i], by_study[labels.StudyInstanceUID.iloc[i]], size)
                raw.append(v)
                masks.append(m)
            raw = np.stack(raw)
            msk = torch.from_numpy(np.stack(masks)).to(device)
            slots = torch.arange(MAX_SERIES, device=device).expand(len(chunk), MAX_SERIES)
            acc = None
            with torch.no_grad(), torch.autocast(device.type, torch.float16, enabled=amp):
                for name in tta:
                    x = torch.from_numpy(
                        np.stack([to_model_input(v, size) for v in TTA[name](raw)])
                    ).to(device)
                    out = torch.sigmoid(model(x, msk, slots).float())
                    acc = out if acc is None else acc + out
            oof[chunk] = (acc / len(tta)).cpu().numpy()
            covered[chunk] = True
            if (start // args.batch) % 25 == 0:
                print(f"  {start + len(chunk)}/{len(idx)}", flush=True)

    n = int(covered.sum())
    print(f"\ncovered {n}/{len(labels)} studies out of fold")
    if n == 0:
        raise SystemExit("no studies scored -- do the checkpoints match --folds?")

    weak = (labels[[f"weak_{t}" for t in TARGETS]].to_numpy() >= 0.5).astype(float)
    macro, per = macro_auc(weak[covered], oof[covered])
    print(f"\nOOF weak AUC (ensemble, {n} studies): {macro:.4f}")
    for t, v in sorted(per.items(), key=lambda kv: kv[1]):
        print(f"  {t:18s} {v:.3f}")

    gmask = covered & (labels.has_gold.to_numpy() == 1)
    if gmask.sum() >= 8:
        gold = labels.loc[gmask, [f"gold_{t}" for t in TARGETS]].to_numpy(float)
        gmacro, _ = macro_auc(gold, oof[gmask])
        print(f"\nOOF gold AUC ({int(gmask.sum())} studies): {gmacro:.4f}")

    out = Path(args.out) if args.out else WORK / "oof.csv"
    df = pd.DataFrame(oof, columns=TARGETS)
    df.insert(0, ID_COL, labels.StudyInstanceUID.values)
    df["fold"] = fold
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
