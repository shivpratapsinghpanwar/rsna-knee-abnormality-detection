"""Train the study-level model on weak labels, validate on weak + gold.

    python scripts/train.py --fold 0 --epochs 8 --backbone resnet34

Two validation numbers are reported each epoch and they answer different
questions:

  weak AUC  how well the image model reproduces the report-derived labels. It is
            computed over ~880 held-out studies, so it is stable enough to select
            checkpoints on, but it is capped by the labeler's own errors.
  gold AUC  the same metric against real human labels. It is the honest estimate
            of leaderboard performance, but only 58 studies carry gold labels, so
            roughly a dozen land in any one fold and the number is very noisy.

Model selection uses weak AUC; gold AUC is tracked to catch the case where the
model learns the labeler's mistakes rather than the anatomy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import SIZE, WORK  # noqa: E402
from rsna_knee.data import (  # noqa: E402
    KneeStudyDataset, collate, index_corpus, make_folds,
)
from rsna_knee.labels import TARGETS  # noqa: E402
from rsna_knee.model import (  # noqa: E402
    KneeStudyModel, masked_bce, prevalence_pos_weight,
)


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, dict[str, float]]:
    """Macro ROC AUC, skipping targets that are single-class in this split."""
    per: dict[str, float] = {}
    for j, t in enumerate(TARGETS):
        col = y[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2 or len(np.unique(col[ok])) < 2:
            continue
        per[t] = float(roc_auc_score(col[ok], p[ok, j]))
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps, ys = [], []
    for vol, mask, slots, target in loader:
        vol, mask, slots = vol.to(device), mask.to(device), slots.to(device)
        with torch.autocast(device.type, torch.float16, enabled=amp):
            out = model(vol, mask, slots)
        ps.append(torch.sigmoid(out.float()).cpu().numpy())
        ys.append(target.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--size", type=int, default=SIZE,
                    help="model input size; mosaics are rescaled to it if needed")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap study count")
    ap.add_argument("--out", default="")
    # Kaggle kills a GPU session at ~9 h. Stopping under our own budget means the
    # best checkpoint so far is already written and the run is still useful.
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop after the epoch that crosses this (0 = no limit)")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute activations; ~30%% slower, ~half the memory")
    ap.add_argument("--pos-weight-cap", type=float, default=0,
                    help="inverse-prevalence positive weighting, capped here "
                         "(0 disables). Aimed at the rare targets under macro AUC")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    labels = pd.read_csv(WORK / "weak_labels.csv")
    corpus = index_corpus("train")
    if corpus.empty:
        print("no preprocessed mosaics found -- run preprocess_series.py first")
        return

    # Only studies whose images actually exist can be trained on. Locally that is
    # the streamed dev subset; on Kaggle it is the whole corpus.
    have = set(corpus.StudyInstanceUID)
    labels = labels[labels.StudyInstanceUID.isin(have)].reset_index(drop=True)
    if args.limit:
        labels = labels.head(args.limit).reset_index(drop=True)
    print(f"{len(labels)} studies with images, {len(corpus)} mosaics, "
          f"{int(labels.has_gold.sum())} with gold")

    n_folds = min(args.folds, max(2, len(labels) // 2))
    fold = make_folds(labels, n_folds, args.seed)
    tr_df = labels[fold != args.fold].reset_index(drop=True)
    va_df = labels[fold == args.fold].reset_index(drop=True)
    print(f"fold {args.fold}/{n_folds}: train {len(tr_df)}  val {len(va_df)}")

    tr_ds = KneeStudyDataset(tr_df, corpus, train=True, size=args.size)
    va_ds = KneeStudyDataset(va_df, corpus, train=False, size=args.size)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, collate_fn=collate,
                       pin_memory=amp, drop_last=len(tr_ds) > args.batch)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, collate_fn=collate, pin_memory=amp)

    model = KneeStudyModel(args.backbone, pretrained=True,
                           grad_checkpointing=args.grad_checkpointing).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = max(1, len(tr_ld) // args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.25
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    pos_w = None
    if args.pos_weight_cap:
        rates = (tr_df[[f"weak_{t}" for t in TARGETS]].to_numpy() >= 0.5).mean(0)
        pos_w = prevalence_pos_weight(rates, args.pos_weight_cap, device)
        print("pos_weight:", {t: round(float(w), 2)
                              for t, w in zip(TARGETS, pos_w)})

    gold_cols = [f"gold_{t}" for t in TARGETS]
    out_dir = Path(args.out) if args.out else WORK / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"fold{args.fold}_{args.backbone}.pt"
    history, best = [], -1.0
    budget = args.max_minutes * 60 if args.max_minutes else float("inf")
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (vol, mask, slots, target) in enumerate(tr_ld):
            vol, mask = vol.to(device, non_blocking=True), mask.to(device)
            slots, target = slots.to(device), target.to(device)
            with torch.autocast(device.type, torch.float16, enabled=amp):
                loss = masked_bce(model(vol, mask, slots), target,
                                  pos_weight=pos_w) / args.accum
            scaler.scale(loss).backward()
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < steps - 1:
                    sched.step()
            total += loss.item() * args.accum
            nb += 1

        y, p = predict(model, va_ld, device, amp)
        weak_auc, per = macro_auc((y >= 0.5).astype(float), p)

        # Gold rows in this fold, scored against the human labels rather than the
        # labeler's. Tiny, but it is the only unbiased signal available.
        gold_auc = float("nan")
        gmask = va_df.has_gold.to_numpy() == 1
        if gmask.sum() >= 4:
            gold_auc, _ = macro_auc(va_df.loc[gmask, gold_cols].to_numpy(float),
                                    p[gmask])

        rec = dict(epoch=epoch, loss=round(total / max(nb, 1), 4),
                   weak_auc=round(weak_auc, 4),
                   gold_auc=None if gold_auc != gold_auc else round(gold_auc, 4),
                   secs=round(time.time() - t0, 1))
        history.append(rec)
        print(f"  epoch {epoch}  loss {rec['loss']:.4f}  weak AUC {weak_auc:.4f}"
              f"  gold AUC {rec['gold_auc']}  ({rec['secs']}s)", flush=True)

        if weak_auc > best:
            best = weak_auc
            torch.save(
                # `size` travels with the weights: inference must build its
                # tensor at the resolution the model was trained on, and that is
                # what lets checkpoints from different corpora be ensembled.
                dict(model=model.state_dict(), backbone=args.backbone,
                     fold=args.fold, weak_auc=weak_auc, epoch=epoch,
                     size=args.size, targets=TARGETS),
                ckpt,
            )
        # Written every epoch, not just at the end, so a session killed partway
        # still leaves a readable record of what happened.
        (out_dir / f"fold{args.fold}_history.json").write_text(
            json.dumps(history, indent=2)
        )

        if time.time() - started > budget:
            print(f"\nreached the {args.max_minutes:g} min budget after epoch "
                  f"{epoch}; stopping with the best checkpoint saved.")
            break

    print(f"\nbest weak AUC {best:.4f} -> {ckpt}")
    print("per-target (final epoch):")
    for t, v in sorted(per.items(), key=lambda kv: kv[1]):
        print(f"  {t:18s} {v:.3f}")


if __name__ == "__main__":
    main()
