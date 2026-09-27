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


def _laterality(ds) -> str:
    """Which knee this slice shows: 'L', 'R', or '' when it cannot be told.

    The `Laterality` tag is Type 2C -- conditionally required -- and whole
    vendors omit it, so roughly half the corpus has no tag at all. When it is
    missing the side is recovered from geometry instead: DICOM patient
    coordinates are LPS, so the sign of x at the *centre* of the image says
    which side of the midline the knee sits on.

    The centre matters. `ImagePositionPatient` is the corner of the image, and
    for a knee a 160 mm field puts the corner on the far side of the midline
    from the joint often enough that the corner's sign is close to a coin flip;
    a public analysis of this corpus measured tag agreement falling from 96.9%
    to 58.8% when the centre correction is skipped. Studies within 20 mm of the
    midline are left unresolved rather than guessed.
    """
    for attr in ("Laterality", "ImageLaterality"):
        v = getattr(ds, attr, None)
        if v:
            v = str(v).strip().upper()[:1]
            if v in ("L", "R"):
                return v
    try:
        ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        row_dir, col_dir = iop[:3], iop[3:]
        dr, dc = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])
        rows, cols = int(ds.Rows), int(ds.Columns)
        centre = (ipp
                  + row_dir * dc * (cols - 1) / 2.0
                  + col_dir * dr * (rows - 1) / 2.0)
        x = float(centre[0])
        return "" if abs(x) < 20.0 else ("L" if x > 0 else "R")
    except Exception:
        return ""


def window_series_uint8(vol: np.ndarray) -> np.ndarray:
    """Window a whole series to uint8 against one shared intensity range.

    Deliberately per-series rather than per-slice. On a fluid-sensitive sequence
    the *brightness* of fluid is the finding: effusion, synovitis, a Baker's cyst
    and bone-marrow oedema are all read off absolute signal. Rescaling each slice
    against its own percentiles destroys exactly that -- a slice through a large
    effusion and a slice through none are both stretched to full range, and the
    five intensity-based targets lose the signal that distinguishes them.
    """
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return np.zeros(vol.shape, dtype=np.uint8)
    # Percentiles over a stride rather than every voxel: this runs over the whole
    # stack now, and the difference is far below the JPEG quantisation we already
    # accept downstream.
    sample = finite[::7] if finite.size > 700_000 else finite
    lo, hi = np.percentile(sample, [LO_PCT, HI_PCT])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(vol.shape, dtype=np.uint8)
    out = (np.clip(vol, lo, hi) - lo) / (hi - lo)
    return (out * 255.0).astype(np.uint8)


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
    return arr, _slice_position(ds), _pixel_spacing(ds), _laterality(ds)


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


def read_series(paths: list[Path], per_series_window: bool = True
                ) -> tuple[np.ndarray, float, str]:
    """Read a series into a (N, H, W) uint8 volume, its mm-per-pixel, and side.

    Slices whose in-plane size differs from the majority are dropped: a handful of
    series mix in a localiser or a differently sized repeat, and stacking those
    would either fail or silently misalign the volume.

    The returned spacing is the median over the slices that survived, so one
    mislabelled header cannot rescale a whole series. The side is a majority vote
    for the same reason.
    """
    read: list[tuple[float, np.ndarray, float, str]] = []
    for p in paths:
        try:
            arr, key, mm, lat = read_slice(p)
        except Exception:
            continue
        read.append((key, arr, mm, lat))
    if not read:
        return np.zeros((0, 0, 0), dtype=np.uint8), 0.0, ""

    shapes: dict[tuple[int, int], int] = {}
    for _, a, _, _ in read:
        shapes[a.shape] = shapes.get(a.shape, 0) + 1
    keep_shape = max(shapes, key=lambda s: shapes[s])
    read = [t for t in read if t[1].shape == keep_shape]

    read.sort(key=lambda t: t[0])
    spacings = [mm for _, _, mm, _ in read if mm > 0]
    spacing = float(np.median(spacings)) if spacings else 0.0

    votes: dict[str, int] = {}
    for _, _, _, lat in read:
        if lat:
            votes[lat] = votes.get(lat, 0) + 1
    side = max(votes, key=lambda k: votes[k]) if votes else ""

    stack = np.stack([a for _, a, _, _ in read])
    # One window over the stacked series, not one per slice. The per-slice path
    # is kept reachable because checkpoints trained on a per-slice corpus must be
    # served the pixels they were trained on.
    vol = (window_series_uint8(stack) if per_series_window
           else np.stack([window_uint8(a) for a in stack]))
    return vol, spacing, side
