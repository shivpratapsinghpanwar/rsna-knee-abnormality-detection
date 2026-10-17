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

from rsna_knee.config import (  # noqa: E402
    CROP_MM, CSV, MAX_SERIES, ROI_SLOTS, SIZE, WORK, prep_manifest,
)
from rsna_knee.data import (  # noqa: E402
    KneeStudyDataset, collate, index_corpus, make_folds, report_groups,
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


def param_groups(model, lr: float, backbone_mult: float, wd: float = 1e-2):
    """Split parameters into four groups: backbone/head x decay/no-decay.

    Two separate corrections live here.

    The pretrained backbone wants a far smaller step than the freshly
    initialised pooling and heads. A self-supervised encoder like DINOv3 arrives
    with features that are already better than anything eight epochs on 4.4k
    weakly-labelled studies can teach it, so a head-sized learning rate does not
    adapt those features, it overwrites them. The published consensus for ViT /
    ConvNeXt fine-tuning is a backbone rate of 1e-5..5e-5 against a head roughly
    two orders higher; our uniform 3e-4 sat ~40x above that band.

    Weight decay is also kept off norm parameters and biases. Decaying a
    LayerNorm gain pulls it toward zero, which rescales that layer's output for
    no regularisation benefit -- standard practice everywhere, and it matters
    more than usual here because the DINOv3 ConvNeXt is all LayerNorm.
    """
    seen, groups = set(), {
        ("backbone", True): [], ("backbone", False): [],
        ("head", True): [], ("head", False): [],
    }
    for name, prm in model.named_parameters():
        if not prm.requires_grad or id(prm) in seen:
            continue
        seen.add(id(prm))
        where = "backbone" if name.startswith("encoder.net.") else "head"
        # 1-D tensors are norm gains/betas and biases; 2-D+ are real weights.
        decay = prm.ndim > 1
        groups[(where, decay)].append(prm)
    out = []
    for (where, decay), prms in groups.items():
        if not prms:
            continue
        out.append(dict(
            params=prms,
            lr=lr * (backbone_mult if where == "backbone" else 1.0),
            weight_decay=wd if decay else 0.0,
            name=f"{where}_{'decay' if decay else 'nodecay'}",
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="head learning rate; the backbone gets this times "
                         "--backbone-lr-mult")
    ap.add_argument("--backbone-lr-mult", type=float, default=1.0,
                    help="backbone lr as a fraction of --lr. 1.0 reproduces "
                         "the old single-rate behaviour; ~0.05 puts a "
                         "pretrained encoder in the usual fine-tuning band")
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
    ap.add_argument("--max-vram-frac", type=float, default=0.0,
                    help="hard cap on this process's share of VRAM (0 = no cap). "
                         "Set it to leave room for another job on the same card: "
                         "the cap makes us OOM instead of starving the neighbour")
    ap.add_argument("--roi", action="store_true",
                    help="compartment ROI: split coronal and axial into two "
                         "overlapping half-width views (4 slots -> 6)")
    ap.add_argument("--encoder-chunk", type=int, default=0,
                    help="images per backbone call (0 = the whole study at "
                         "once). Caps activation peak so heavy backbones fit")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute activations; ~30%% slower, ~half the memory")
    ap.add_argument("--pos-weight-cap", type=float, default=0,
                    help="inverse-prevalence positive weighting, capped here "
                         "(0 disables). Aimed at the rare targets under macro AUC")
    # A spot VM can be reclaimed at any moment. The full training state is
    # written after every epoch, so a restarted run loses at most one epoch.
    ap.add_argument("--resume", action="store_true",
                    help="continue from fold<N>_last.pt in --out if present")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.max_vram_frac and device.type == "cuda":
        # A hard ceiling, not a hint. Another job shares this card, and the
        # caching allocator will otherwise grow to fill whatever is free and not
        # give it back -- starving the neighbour rather than failing ourselves.
        torch.cuda.set_per_process_memory_fraction(args.max_vram_frac)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"vram cap: {args.max_vram_frac:.0%} of {total:.1f} GB = "
              f"{args.max_vram_frac * total:.1f} GB", flush=True)
    amp = device.type == "cuda"

    # Which corpus this run trains on, so the submission can refuse to serve
    # these weights pixels preprocessed a different way.
    prep_info = prep_manifest()
    print("corpus:", prep_info)

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
    # Group by report hash: duplicates across folds are memorised, not
    # predicted, and inflate every offline number.
    fold = make_folds(labels, n_folds, args.seed,
                      groups=report_groups(labels, CSV))
    tr_df = labels[fold != args.fold].reset_index(drop=True)
    va_df = labels[fold == args.fold].reset_index(drop=True)
    print(f"fold {args.fold}/{n_folds}: train {len(tr_df)}  val {len(va_df)}")

    tr_ds = KneeStudyDataset(tr_df, corpus, train=True, size=args.size,
                             roi=args.roi)
    va_ds = KneeStudyDataset(va_df, corpus, train=False, size=args.size,
                             roi=args.roi)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, collate_fn=collate,
                       pin_memory=amp, drop_last=len(tr_ds) > args.batch)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, collate_fn=collate, pin_memory=amp)

    model = KneeStudyModel(args.backbone, pretrained=True,
                           grad_checkpointing=args.grad_checkpointing,
                           chunk=args.encoder_chunk,
                           n_slots=ROI_SLOTS if args.roi else MAX_SERIES
                           ).to(device)
    groups = param_groups(model, args.lr, args.backbone_lr_mult)
    opt = torch.optim.AdamW(groups)
    print("param groups:", {g["name"]: f"{len(g['params'])}t lr={g['lr']:.2g} "
                            f"wd={g['weight_decay']:g}" for g in groups})
    steps = max(1, len(tr_ld) // args.accum) * args.epochs
    # OneCycle needs one max_lr per group, or it flattens them back to one rate
    # and silently undoes the split above.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.25
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
    last = out_dir / f"fold{args.fold}_last.pt"
    history, best, start_epoch, per = [], -1.0, 0, {}
    if args.resume and last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        history, best = state["history"], state["best"]
        start_epoch = state["epoch"] + 1
        print(f"resumed from {last.name}: epoch {state['epoch']} done, "
              f"best weak AUC {best:.4f}", flush=True)
    budget = args.max_minutes * 60 if args.max_minutes else float("inf")
    started = time.time()

    for epoch in range(start_epoch, args.epochs):
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
                     size=args.size, crop_mm=CROP_MM,
                 lr=args.lr, backbone_lr_mult=args.backbone_lr_mult,
                 encoder_chunk=args.encoder_chunk, roi=args.roi,
                 prep=prep_info, targets=TARGETS),
                ckpt,
            )
        # Written every epoch, not just at the end, so a session killed partway
        # still leaves a readable record of what happened.
        (out_dir / f"fold{args.fold}_history.json").write_text(
            json.dumps(history, indent=2)
        )
        if args.resume:
            # Write-then-rename: a preemption mid-save must not leave a torn file
            # that the next boot would fail to load.
            tmp = last.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            sched=sched.state_dict(), scaler=scaler.state_dict(),
                            epoch=epoch, best=best, history=history), tmp)
            tmp.replace(last)

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
