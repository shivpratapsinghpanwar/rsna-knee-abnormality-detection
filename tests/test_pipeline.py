"""Tests for the image pipeline: preprocessing, folds, and the model's contracts.

These pin the properties that are expensive to notice later -- a silently
transposed mosaic, a fold split that starves a rare target, a mask that lets a
padded series leak into a prediction.

Run standalone: python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import MAX_SERIES, SIZE, SLICES  # noqa: E402
from rsna_knee.data import make_folds  # noqa: E402
from rsna_knee.labels import TARGETS  # noqa: E402
from rsna_knee.preprocess import (  # noqa: E402
    mosaic_to_volume, pick_slice_indices, volume_to_mosaic,
)
from rsna_knee.series import select_series  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def test_mosaic_roundtrip() -> None:
    """A volume that is already the target size must survive exactly.

    Any transposition or tiling-order bug shows up here: the mosaic is the only
    representation the model ever sees, so a scrambled one trains silently.
    """
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 256, (SLICES, SIZE, SIZE), dtype=np.uint8)
    back = mosaic_to_volume(volume_to_mosaic(vol))
    check(np.array_equal(back, vol), "mosaic round-trip is not exact")

    # Slices must stay distinguishable and in order, not collapse to one tile.
    ordered = np.stack([np.full((SIZE, SIZE), i, dtype=np.uint8) for i in range(SLICES)])
    back = mosaic_to_volume(volume_to_mosaic(ordered))
    check([int(s[0, 0]) for s in back] == list(range(SLICES)),
          "mosaic tiling does not preserve slice order")


def test_slice_sampling() -> None:
    """Sampling must always return exactly SLICES indices, in range and sorted."""
    for n in (1, 2, 7, 16, 17, 33, 160):
        idx = pick_slice_indices(n)
        check(len(idx) == SLICES, f"n={n}: got {len(idx)} indices, want {SLICES}")
        check(all(0 <= i < n for i in idx), f"n={n}: index out of range {idx}")
        check(idx == sorted(idx), f"n={n}: indices not ordered")
    check(pick_slice_indices(0) == [], "empty series should yield no indices")


def test_series_selection() -> None:
    """At most one series per slot, and no study silently dropped."""
    rows = []
    for study in ("a", "b"):
        for plane in ("Sagittal", "Sagittal", "Coronal", "Axial", "Sagittal"):
            rows.append(dict(
                StudyInstanceUID=study,
                SeriesInstanceUID=f"{study}-{plane}-{len(rows)}",
                Anatomical_Plane=plane, Fluid_Sensitive=len(rows) % 2,
                Fat_Suppression=len(rows) % 2,
            ))
    # A study whose series match no slot at all must still be represented.
    rows.append(dict(StudyInstanceUID="c", SeriesInstanceUID="c-1",
                     Anatomical_Plane="Oblique", Fluid_Sensitive=0,
                     Fat_Suppression=0))
    sel = select_series(pd.DataFrame(rows))

    check(set(sel.StudyInstanceUID) == {"a", "b", "c"},
          f"studies dropped by selection: {set(sel.StudyInstanceUID)}")
    for study, g in sel.groupby("StudyInstanceUID"):
        check(g.slot.is_unique, f"study {study} has duplicate slots")
        check(len(g) <= MAX_SERIES, f"study {study} kept {len(g)} > {MAX_SERIES}")
        check(g.SeriesInstanceUID.is_unique,
              f"study {study} reuses one series in two slots")


def test_folds_balanced() -> None:
    """Every target's rate must hold across folds, including the rare ones."""
    rng = np.random.default_rng(1)
    n, k = 2000, 5
    rates = np.linspace(0.04, 0.5, len(TARGETS))
    y = (rng.random((n, len(TARGETS))) < rates).astype(float)
    df = pd.DataFrame(y, columns=[f"weak_{t}" for t in TARGETS])
    df["StudyInstanceUID"] = [f"s{i}" for i in range(n)]

    fold = make_folds(df, k, seed=0)
    check(len(np.unique(fold)) == k, "not all folds were used")
    sizes = np.bincount(fold, minlength=k)
    check(sizes.max() - sizes.min() <= 1,
          f"fold sizes unbalanced: {sizes.tolist()}")
    for j, t in enumerate(TARGETS):
        overall = y[:, j].mean()
        worst = max(abs(y[fold == f, j].mean() - overall) for f in range(k))
        check(worst < 0.03,
              f"{t}: fold rate deviates {worst:.3f} from {overall:.3f}")


def test_model_contract() -> None:
    """Shapes, mask handling, and NaN-safety on a study with no usable series."""
    import torch

    from rsna_knee.model import KneeStudyModel, masked_bce

    torch.manual_seed(0)
    model = KneeStudyModel("resnet18", pretrained=False).eval()
    b, s = 2, SLICES
    vol = torch.randn(b, MAX_SERIES, s, 64, 64)
    slots = torch.arange(MAX_SERIES).expand(b, MAX_SERIES)

    full = torch.ones(b, MAX_SERIES, dtype=torch.bool)
    with torch.no_grad():
        out = model(vol, full, slots)
    check(out.shape == (b, len(TARGETS)), f"logits shape {tuple(out.shape)}")
    check(bool(torch.isfinite(out).all()), "logits contain non-finite values")

    # A study with no valid series must score finitely rather than NaN.
    with torch.no_grad():
        empty = model(vol, torch.zeros(b, MAX_SERIES, dtype=torch.bool), slots)
    check(bool(torch.isfinite(empty).all()), "empty-mask study produced NaN")

    # Masked-out series must not change the prediction: otherwise padding leaks.
    half = torch.tensor([[True, True, False, False]] * b)
    v2 = vol.clone()
    v2[:, 2:] = torch.randn_like(v2[:, 2:]) * 50
    with torch.no_grad():
        a = model(vol, half, slots)
        c = model(v2, half, slots)
    check(torch.allclose(a, c, atol=1e-4),
          f"masked series leaked into logits (max delta {(a - c).abs().max():.2e})")

    # NaN targets must be ignored rather than poisoning the loss.
    tgt = torch.rand(b, len(TARGETS))
    tgt[0, 0] = float("nan")
    loss = masked_bce(out, tgt)
    check(bool(torch.isfinite(loss)), "masked_bce returned non-finite loss")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as exc:  # a crashing test is a failing test
            failures.append(f"{t.__name__} raised {type(exc).__name__}: {exc}")

    for f in failures:
        print("FAIL ", f)
    print(f"{len(tests) - len({f.split(':')[0] for f in failures})}"
          f"/{len(tests)} test functions clean, {len(failures)} assertion failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
