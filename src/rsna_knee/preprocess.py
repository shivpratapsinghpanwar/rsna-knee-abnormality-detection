"""Series volume -> fixed-size PNG mosaic, the storage format used for training.

One ordinary PNG per series keeps the corpus small enough to live inside a Kaggle
dataset and on local disk, needs no custom container, and is lossless on the
already 8-bit windowed data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import CROP_MM, GRID, MOSAIC_QUALITY, SIZE, SLICES


def pick_slice_indices(n: int, k: int = SLICES) -> list[int]:
    """Evenly spaced slice indices, biased away from the empty end slices.

    The first and last slices of a knee series are usually mostly background, so
    sampling spans the middle of the stack rather than the full extent.
    """
    if n <= 0:
        return []
    if n <= k:
        return list(range(n)) + [n - 1] * (k - n)
    lo, hi = 0.08 * n, 0.92 * n
    return [int(round(x)) for x in np.clip(np.linspace(lo, hi, k), 0, n - 1)]



def normalise_laterality(vol: np.ndarray, plane: str, side: str) -> np.ndarray:
    """Mirror a right knee onto the left-knee convention.

    Four of the twelve targets are medial/lateral pairs, and medial is not a
    fixed side of the image -- it is whichever side faces the other leg. On a
    coronal or axial slice the medial compartment of a left knee sits at image
    right and of a right knee at image left, so without this the same pixel
    column means opposite anatomy in different studies and the model is asked to
    separate medial from lateral with the evidence mirrored at random.

    The axis to flip depends on the plane. Coronal and axial carry medial-lateral
    *in plane*, so the width axis is reversed. Sagittal carries no medial-lateral
    within a slice at all -- it runs through the stack -- so the slice order is
    reversed instead.

    Left knees and studies whose side could not be determined are returned
    untouched; guessing is worse than leaving the minority unmirrored.
    """
    if side != "R" or vol.size == 0:
        return vol
    if plane.lower().startswith("sag"):
        return vol[::-1].copy()
    return vol[..., ::-1].copy()

def crop_mm(vol: np.ndarray, spacing: float, crop: float = CROP_MM) -> np.ndarray:
    """Centre-crop a volume to a fixed physical width, in millimetres.

    Without this the pipeline resizes a fixed *pixel* box to a fixed tensor, so a
    knee imaged at 0.3 mm/px and one at 0.6 mm/px arrive at the model as
    different-sized joints and it has to learn around the scanner rather than
    from the anatomy. Cropping to a fixed millimetre width makes every study
    cover the same field, which is what the published solutions do.

    A spacing of 0 means the header did not say; the volume is returned untouched
    rather than cropped on a guess. Likewise when the requested field is wider
    than the image, since padding would invent anatomy.
    """
    if spacing <= 0 or vol.size == 0:
        return vol
    want = int(round(crop / spacing))
    h, w = vol.shape[-2:]
    if want >= min(h, w) or want < 16:
        return vol
    top, left = (h - want) // 2, (w - want) // 2
    return vol[..., top:top + want, left:left + want]


def resize(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    if img.shape == (size, size):
        return img
    return np.asarray(
        Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8
    )


def volume_to_mosaic(vol: np.ndarray, size: int = SIZE, grid: int = GRID) -> np.ndarray:
    """Tile SLICES evenly-spaced slices into one (grid*size, grid*size) image."""
    idx = pick_slice_indices(len(vol), grid * grid)
    tiles = [resize(vol[i], size) for i in idx] if idx else []
    while len(tiles) < grid * grid:
        tiles.append(np.zeros((size, size), dtype=np.uint8))
    rows = [np.hstack(tiles[r * grid:(r + 1) * grid]) for r in range(grid)]
    return np.vstack(rows)


def mosaic_to_volume(mos: np.ndarray, size: int | None = None, grid: int = GRID
                     ) -> np.ndarray:
    """Inverse of volume_to_mosaic: (grid*size, grid*size) -> (grid*grid, size, size).

    The tile size is read from the mosaic rather than assumed from config, so a
    corpus built at one resolution still decodes correctly after SIZE changes --
    otherwise a 384 corpus read with SIZE=256 would silently return cropped tiles.
    """
    if size is None:
        size = mos.shape[0] // grid
    out = np.empty((grid * grid, size, size), dtype=mos.dtype)
    for r in range(grid):
        for c in range(grid):
            out[r * grid + c] = mos[r * size:(r + 1) * size, c * size:(c + 1) * size]
    return out


def save_mosaic(mos: np.ndarray, path: Path) -> None:
    """Write a mosaic, choosing the encoder from the file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(mos)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        # Quality 95 on already-8-bit windowed MRI is visually lossless and about
        # a quarter the size of PNG, which is what pays for the higher resolution.
        img.save(path, format="JPEG", quality=MOSAIC_QUALITY, subsampling=0,
                 optimize=True)
    else:
        img.save(path, format="PNG", compress_level=6)


def load_mosaic(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def mosaic_geometry(mos: np.ndarray, grid: int = GRID) -> int:
    """Tile size implied by a mosaic, so a corpus can be read at its own scale."""
    return mos.shape[0] // grid
