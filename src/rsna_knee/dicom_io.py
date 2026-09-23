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


def read_slice(path: Path) -> tuple[np.ndarray, float]:
    """Return (float32 pixel array, ordering key) for one DICOM file."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        pass
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:  # rare multi-frame or RGB; collapse to a single plane
        arr = arr[arr.shape[0] // 2] if arr.shape[-1] != 3 else arr[..., 0]
    return arr, _slice_position(ds)


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


def read_series(paths: list[Path]) -> np.ndarray:
    """Read a whole series into a (N, H, W) uint8 volume ordered through-plane.

    Slices whose in-plane size differs from the majority are dropped: a handful of
    series mix in a localiser or a differently sized repeat, and stacking those
    would either fail or silently misalign the volume.
    """
    read: list[tuple[float, np.ndarray]] = []
    for p in paths:
        try:
            arr, key = read_slice(p)
        except Exception:
            continue
        read.append((key, arr))
    if not read:
        return np.zeros((0, 0, 0), dtype=np.uint8)

    shapes: dict[tuple[int, int], int] = {}
    for _, a in read:
        shapes[a.shape] = shapes.get(a.shape, 0) + 1
    keep_shape = max(shapes, key=lambda s: shapes[s])
    read = [(k, a) for k, a in read if a.shape == keep_shape]

    read.sort(key=lambda t: t[0])
    return np.stack([window_uint8(a) for _, a in read])
