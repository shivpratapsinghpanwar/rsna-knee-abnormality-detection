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


def study_slices_from_prep(study: str, slots: dict[int, str], size: int
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Build a uint8 study volume from already-preprocessed mosaics.

    Used to score studies whose mosaics exist (the local dev corpus, or the
    training set for out-of-fold predictions) without re-reading DICOM.
    """
    from rsna_knee.preprocess import load_mosaic, resize

    out = np.zeros((MAX_SERIES, SLICES, size, size), dtype=np.uint8)
    mask = np.zeros(MAX_SERIES, dtype=bool)
    for slot, path in slots.items():
        if slot >= MAX_SERIES:
            continue
        v = mosaic_to_volume(load_mosaic(path))
        if v.shape[-1] != size:
            v = np.stack([resize(sl, size) for sl in v])
        out[slot] = v
        mask[slot] = True
    return out, mask


def study_slices(split: str, study: str, sel: pd.DataFrame,
                 size: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode one study's DICOM to (M, SLICES, size, size) uint8 plus a mask.

    DICOM is read once at the largest resolution any model needs; smaller models
    get a resize of the same pixels rather than a second pass over the files.
    """
    out = np.zeros((MAX_SERIES, SLICES, size, size), dtype=np.uint8)
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
            out[r.slot] = mosaic_to_volume(volume_to_mosaic(v, size=size), size=size)
            mask[r.slot] = True
        except Exception:
            continue
    return out, mask


def to_model_input(vol_u8: np.ndarray, size: int) -> np.ndarray:
    """Rescale a uint8 study volume to one model's size and normalise it."""
    from rsna_knee.preprocess import resize

    if vol_u8.shape[-1] != size:
        vol_u8 = np.stack([
            np.stack([resize(sl, size) for sl in series]) for series in vol_u8
        ])
    return (vol_u8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


# Test-time augmentation. Horizontal flip is absent for the same reason it is
# absent from training: six of the twelve targets are side-specific, and on
# coronal and axial images image-left/right *is* medial/lateral, so a flip
# relabels the anatomy. These transforms leave laterality alone.
TTA = {
    "none": lambda v: v,
    # A small translation, matching the padded random crop used in training.
    "shift": lambda v: np.roll(v, (2, 2), axis=(-2, -1)),
    # Re-sample the stack one slice over; the 16 slices are a subsample of ~33,
    # so this genuinely shows the model different anatomy rather than the same
    # pixels twice.
    "slice": lambda v: np.roll(v, 1, axis=-3),
}


def load_models(paths: list[Path], device) -> list[tuple[KneeStudyModel, int]]:
    """Load checkpoints with the input size each was trained at.

    Checkpoints record their own `size`, so models trained on a 256 corpus and a
    384 corpus can be blended in one submission -- which is worth doing, since
    resolution diversity is one of the cheaper sources of ensemble gain.
    """
    models = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = KneeStudyModel(ck.get("backbone", "resnet34"), pretrained=False)
        m.load_state_dict(ck["model"])
        m.to(device).eval()
        size = int(ck.get("size", 256))  # corpora before `size` existed were 256
        models.append((m, size))
        print(f"  loaded {p.name} (fold {ck.get('fold')}, size {size}, "
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
    ap.add_argument("--tta", default="none",
                    help="comma-separated TTA transforms: " + ",".join(TTA))
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

    # Decode each study once at the largest size any model wants, then feed every
    # model a resize of those same pixels.
    read_size = max(size for _, size in models)
    tta = [t.strip() for t in args.tta.split(",") if t.strip()]
    unknown = [t for t in tta if t not in TTA]
    if unknown:
        raise SystemExit(f"unknown TTA {unknown}; choose from {sorted(TTA)}")
    print(f"TTA: {tta}")
    per_model = [[] for _ in models]
    batch: list[tuple[np.ndarray, np.ndarray]] = []

    def flush() -> None:
        if not batch:
            return
        raw = np.stack([b[0] for b in batch])
        msk = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        slots = torch.arange(MAX_SERIES, device=device).expand(len(batch), MAX_SERIES)
        with torch.no_grad(), torch.autocast(device.type, torch.float16, enabled=amp):
            cache: dict[tuple[int, str], torch.Tensor] = {}
            for k, (m, size) in enumerate(models):
                acc = None
                for name in tta:
                    key = (size, name)
                    if key not in cache:
                        aug = TTA[name](raw)
                        cache[key] = torch.from_numpy(
                            np.stack([to_model_input(v, size) for v in aug])
                        ).to(device)
                    out = torch.sigmoid(m(cache[key], msk, slots).float())
                    acc = out if acc is None else acc + out
                per_model[k].append((acc / len(tta)).cpu().numpy())
        batch.clear()

    for i, study in enumerate(ids, 1):
        if args.from_prep:
            batch.append(study_slices_from_prep(study, by_study[study], read_size))
        else:
            batch.append(study_slices(args.split, study, sel, read_size))
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
