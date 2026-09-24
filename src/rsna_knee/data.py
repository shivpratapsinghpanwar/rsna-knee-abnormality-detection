"""Dataset over the PNG mosaic corpus, plus the cross-validation split.

One item is a whole study: up to MAX_SERIES mosaics, decoded back into slice
volumes and stacked into a fixed (M, S, H, W) tensor with a validity mask, so
studies missing a plane still batch cleanly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import MAX_SERIES, PREP, SIZE, SLICES
from .labels import TARGETS
from .preprocess import mosaic_to_volume

IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226  # greyscale-collapsed ImageNet stats


def index_corpus(split: str = "train") -> pd.DataFrame:
    """Find every preprocessed mosaic on disk: one row per (study, slot)."""
    root = PREP / split
    rows = []
    if root.exists():
        for study_dir in root.iterdir():
            if not study_dir.is_dir():
                continue
            # Both extensions: the first corpus was PNG at 256, later ones are
            # JPEG at higher resolution, and a mixed corpus must still index.
            for mos in sorted(study_dir.iterdir()):
                if mos.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                slot = int(mos.name.split("_", 1)[0])
                rows.append(
                    dict(StudyInstanceUID=study_dir.name, slot=slot, path=str(mos))
                )
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "slot", "path"])


def make_folds(labels: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Multilabel-stratified fold assignment, one fold per study.

    Plain KFold would leave rare targets (MCL at 4.7%, Fracture at 6.7%) unevenly
    spread and make per-fold AUC for them meaningless. This greedily assigns each
    study to whichever fold is currently most short of its positives, which keeps
    every target's rate close to the corpus rate in every fold.
    """
    y = (labels[[f"weak_{t}" for t in TARGETS]].to_numpy() >= 0.5).astype(int)
    n, t = y.shape
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    # Rarest-first ordering matters: rare targets must be balanced before common
    # ones consume the remaining capacity.
    rarity = y.sum(0)
    order = sorted(order, key=lambda i: (y[i] * (1.0 / np.maximum(rarity, 1))).sum(),
                   reverse=True)

    want = y.sum(0) / n_folds
    have = np.zeros((n_folds, t))
    count = np.zeros(n_folds)
    fold = np.empty(n, dtype=int)
    for i in order:
        deficit = ((want - have) * y[i]).sum(1) if y[i].any() else -count
        best = int(np.argmax(deficit + 1e-6 * (count.max() - count)))
        fold[i] = best
        have[best] += y[i]
        count[best] += 1
    return fold


class KneeStudyDataset(Dataset):
    """Studies as fixed-shape tensors, with light augmentation for training.

    Horizontal flipping is deliberately absent. Six of the twelve targets are
    side-specific (medial vs lateral compartment and meniscus), and on coronal and
    axial images image-left/right *is* medial/lateral, so a flip would relabel the
    anatomy. Augmentation is therefore limited to transforms that leave laterality
    alone.
    """

    def __init__(self, labels: pd.DataFrame, corpus: pd.DataFrame,
                 train: bool = True, size: int = SIZE):
        self.labels = labels.reset_index(drop=True)
        self.train = train
        self.size = size
        self.target_cols = [f"weak_{t}" for t in TARGETS]
        by_study: dict[str, dict[int, str]] = {}
        for r in corpus.itertuples():
            by_study.setdefault(r.StudyInstanceUID, {})[r.slot] = r.path
        self.by_study = by_study

    def __len__(self) -> int:
        return len(self.labels)

    def _load_slot(self, path: str) -> np.ndarray:
        """Decode one mosaic to (SLICES, size, size), rescaling if it differs.

        The stored corpus and the model's input size are allowed to disagree: a
        256 corpus can train a 384 model or vice versa, which is what makes
        resolution an experiment rather than a rebuild.
        """
        from .preprocess import load_mosaic, resize

        vol = mosaic_to_volume(load_mosaic(path))
        if vol.shape[-1] != self.size:
            vol = np.stack([resize(s, self.size) for s in vol])
        return vol.astype(np.float32) / 255.0

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        rng = np.random
        # Intensity: MRI has no absolute scale, so gain and gamma vary between
        # scanners far more than they do in natural images.
        vol = np.clip(vol * rng.uniform(0.85, 1.15), 0, 1)
        vol = np.power(vol, rng.uniform(0.8, 1.25))
        # Geometry: a small translation via padded random crop.
        pad = max(1, self.size // 32)
        s = vol.shape[-1]
        padded = np.pad(vol, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
        i, j = rng.randint(0, 2 * pad + 1, size=2)
        vol = padded[:, i:i + s, j:j + s]
        # Slice jitter: drop or repeat an end slice so the stack is not always
        # sampled identically.
        if rng.rand() < 0.3:
            shift = rng.choice([-1, 1])
            vol = np.roll(vol, shift, axis=0)
        return vol

    def __getitem__(self, i: int):
        row = self.labels.iloc[i]
        study = row.StudyInstanceUID
        slots = self.by_study.get(study, {})

        vol = np.zeros((MAX_SERIES, SLICES, self.size, self.size), dtype=np.float32)
        mask = np.zeros(MAX_SERIES, dtype=bool)
        for slot in range(MAX_SERIES):
            path = slots.get(slot)
            if path is None:
                continue
            v = self._load_slot(path)
            if self.train:
                v = self._augment(v)
            vol[slot] = v
            mask[slot] = True

        vol = (vol - IMAGENET_MEAN) / IMAGENET_STD
        target = row[self.target_cols].to_numpy(dtype=np.float32)
        return (
            torch.from_numpy(vol),
            torch.from_numpy(mask),
            torch.arange(MAX_SERIES),
            torch.from_numpy(target),
        )


def collate(batch):
    vols, masks, slots, targets = zip(*batch)
    return (
        torch.stack(vols),
        torch.stack(masks),
        torch.stack(slots),
        torch.stack(targets),
    )
