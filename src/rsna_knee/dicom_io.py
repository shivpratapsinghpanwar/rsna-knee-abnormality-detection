"""Reading a DICOM series into a normalised uint8 volume.

MRI pixel values carry no absolute meaning -- unlike CT Hounsfield units, the
scale depends on scanner, sequence and gain. Every slice is therefore windowed
against its own robust percentile range rather than a fixed window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

from .config import HI_PCT, LO_PCT


def _slice_position(ds: pydicom.Dataset) -> float:
    """Best available through-plane ordering key for a slice.

    ImagePositionPatient projected on the slice normal is the correct ordering and
    survives gantry tilt; InstanceNumber is the fallback when geometry is absent.
    """
    try:
        iop = [float(v) for v in ds.ImageOrientationPatient]
        ipp = [float(v) for v in ds.ImagePositionPatient]
        normal = np.cross(iop[:3], iop[3:])
        return float(np.dot(ipp, normal))
    except Exception:
        try:
            return float(ds.InstanceNumber)
        except Exception:
            return 0.0


def _pixel_spacing(ds: pydicom.Dataset) -> float:
    """Millimetres per pixel in-plane, or 0.0 when the header does not say.

    This is the difference between a crop that always covers the same *anatomy*
    and one that covers whatever a particular scanner's field of view happened
    to be. Resizing a fixed pixel box to a fixed tensor makes a knee imaged at
    0.3 mm/px and one at 0.6 mm/px look like different-sized joints, and the
    model has to learn around that rather than from it.
    """
    for attr in ("PixelSpacing", "ImagerPixelSpacing"):
        try:
            val = getattr(ds, attr)
            if val is not None and len(val) >= 1:
                return float(val[0])
        except Exception:
            continue
    return 0.0


def read_slice(path: Path) -> tuple[np.ndarray, float, float]:
    """Return (float32 pixel array, ordering key, mm-per-pixel) for one DICOM."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        pass
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:  # rare multi-frame or RGB; collapse to a single plane
        arr = arr[arr.shape[0] // 2] if arr.shape[-1] != 3 else arr[..., 0]
    return arr, _slice_position(ds), _pixel_spacing(ds)


def window_uint8(arr: np.ndarray) -> np.ndarray:
    """Window one slice to uint8 using its own robust intensity range."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [LO_PCT, HI_PCT])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    return (out * 255.0).astype(np.uint8)


def read_series(paths: list[Path]) -> tuple[np.ndarray, float]:
    """Read a series into a (N, H, W) uint8 volume plus its mm-per-pixel.

    Slices whose in-plane size differs from the majority are dropped: a handful of
    series mix in a localiser or a differently sized repeat, and stacking those
    would either fail or silently misalign the volume.

    The returned spacing is the median over the slices that survived, so one
    mislabelled header cannot rescale a whole series.
    """
    read: list[tuple[float, np.ndarray, float]] = []
    for p in paths:
        try:
            arr, key, mm = read_slice(p)
        except Exception:
            continue
        read.append((key, arr, mm))
    if not read:
        return np.zeros((0, 0, 0), dtype=np.uint8), 0.0

    shapes: dict[tuple[int, int], int] = {}
    for _, a, _ in read:
        shapes[a.shape] = shapes.get(a.shape, 0) + 1
    keep_shape = max(shapes, key=lambda s: shapes[s])
    read = [(k, a, mm) for k, a, mm in read if a.shape == keep_shape]

    read.sort(key=lambda t: t[0])
    spacings = [mm for _, _, mm in read if mm > 0]
    spacing = float(np.median(spacings)) if spacings else 0.0
    return np.stack([window_uint8(a) for _, a, _ in read]), spacing
