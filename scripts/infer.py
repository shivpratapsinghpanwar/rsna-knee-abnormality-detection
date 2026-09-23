"""Offline inference: DICOM test studies -> submission.csv.

This is what runs inside the Kaggle scoring notebook, where there is no internet
and the test set is hidden. It reuses the exact preprocessing the training corpus
was built with, so there is no train/serve skew: the same series selection, the
same 16-slice sampling, the same windowing.

    python scripts/infer.py --ckpt work/models/fold0_resnet34.pt --split test

Checkpoints are rank-averaged rather than probability-averaged. The metric is
ROC AUC, which only sees ordering, and averaging ranks keeps one badly calibrated
fold from dominating the blend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import CSV, MAX_SERIES, RAW, SIZE, SLICES  # noqa: E402
from rsna_knee.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from rsna_knee.dicom_io import read_series  # noqa: E402
from rsna_knee.labels import ID_COL, TARGETS  # noqa: E402
from rsna_knee.model import KneeStudyModel  # noqa: E402
from rsna_knee.preprocess import mosaic_to_volume, volume_to_mosaic  # noqa: E402
from rsna_knee.series import select_series  # noqa: E402


def study_tensor_from_prep(study: str, slots: dict[int, str]
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Build a study tensor from already-preprocessed mosaics.

    Used to score studies whose mosaics exist (the local dev corpus, or the
    training set for out-of-fold predictions) without re-reading DICOM.
    """
    from rsna_knee.preprocess import load_mosaic

    vol = np.zeros((MAX_SERIES, SLICES, SIZE, SIZE), dtype=np.float32)
    mask = np.zeros(MAX_SERIES, dtype=bool)
    for slot, path in slots.items():
        if slot >= MAX_SERIES:
            continue
        vol[slot] = mosaic_to_volume(load_mosaic(path)).astype(np.float32) / 255.0
        mask[slot] = True
    return (vol - IMAGENET_MEAN) / IMAGENET_STD, mask


def study_tensor(split: str, study: str, sel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build one study's (M, S, H, W) volume and its series mask from raw DICOM."""
    vol = np.zeros((MAX_SERIES, SLICES, SIZE, SIZE), dtype=np.float32)
    mask = np.zeros(MAX_SERIES, dtype=bool)
    for r in sel[sel.StudyInstanceUID == study].itertuples():
        d = RAW / f"{split}_series" / study / r.SeriesInstanceUID
        paths = sorted(d.glob("*.dcm"))
        if not paths:
            continue
        try:
            v = read_series(paths)
            if v.size == 0:
                continue
            # Round-tripping through the mosaic guarantees the tensor is identical
            # to what training saw, rather than merely similar.
            vol[r.slot] = mosaic_to_volume(volume_to_mosaic(v)).astype(np.float32) / 255.0
            mask[r.slot] = True
        except Exception:
            continue
    return (vol - IMAGENET_MEAN) / IMAGENET_STD, mask


def load_models(paths: list[Path], device) -> list[KneeStudyModel]:
    models = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = KneeStudyModel(ck.get("backbone", "resnet34"), pretrained=False)
        m.load_state_dict(ck["model"])
        m.to(device).eval()
        models.append(m)
        print(f"  loaded {p.name} (fold {ck.get('fold')}, "
              f"weak AUC {ck.get('weak_auc')})")
    return models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--from-prep", action="store_true",
                    help="read preprocessed mosaics instead of raw DICOM")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    models = load_models([Path(p) for p in args.ckpt], device)
    if not models:
        raise SystemExit("no checkpoints loaded")

    if args.from_prep:
        from rsna_knee.data import index_corpus

        corpus = index_corpus(args.split)
        if corpus.empty:
            raise SystemExit(f"no mosaics under PREP for split {args.split!r}")
        by_study: dict[str, dict[int, str]] = {}
        for r in corpus.itertuples():
            by_study.setdefault(r.StudyInstanceUID, {})[r.slot] = r.path
        ids = sorted(by_study)
        sel = None
        print(f"{len(ids)} studies from preprocessed mosaics")
    else:
        meta = pd.read_csv(CSV / f"{args.split}_series.csv")
        ids = pd.read_csv(CSV / f"{args.split}.csv")[ID_COL].tolist()
        sel = select_series(meta)
        by_study = {}
        print(f"{len(ids)} studies, {len(sel)} series selected")

    per_model = [[] for _ in models]
    batch: list[tuple[np.ndarray, np.ndarray]] = []

    def flush() -> None:
        if not batch:
            return
        vol = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        msk = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        slots = torch.arange(MAX_SERIES, device=device).expand(len(batch), MAX_SERIES)
        with torch.no_grad(), torch.autocast(device.type, torch.float16, enabled=amp):
            for k, m in enumerate(models):
                per_model[k].append(torch.sigmoid(m(vol, msk, slots).float()).cpu().numpy())
        batch.clear()

    for i, study in enumerate(ids, 1):
        if args.from_prep:
            batch.append(study_tensor_from_prep(study, by_study[study]))
        else:
            batch.append(study_tensor(args.split, study, sel))
        if len(batch) == args.batch:
            flush()
        if i % 25 == 0:
            print(f"  {i}/{len(ids)}", flush=True)
    flush()

    preds = [np.concatenate(p) for p in per_model]
    if len(preds) == 1:
        blend = preds[0]
    else:
        # Rank-average: AUC is invariant to monotone transforms, so ordering is
        # all that carries over, and ranks make the folds commensurate.
        ranked = []
        for p in preds:
            r = np.empty_like(p)
            for j in range(p.shape[1]):
                order = p[:, j].argsort().argsort().astype(np.float64)
                r[:, j] = order / max(len(order) - 1, 1)
            ranked.append(r)
        blend = np.mean(ranked, axis=0)

    sub = pd.DataFrame(blend, columns=TARGETS)
    sub.insert(0, ID_COL, ids)
    sub.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  {sub.shape}")
    print(sub.head().to_string())


if __name__ == "__main__":
    main()
