"""Paths and preprocessing constants, resolved for whichever host we run on.

The project runs in two places with the same code: a Kaggle notebook, where the
competition data is mounted read-only and training happens, and this workstation,
where only a small streamed subset exists for developing the pipeline. Nothing
below should need editing to move between them.
"""

from __future__ import annotations

import os
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"

ON_KAGGLE = Path("/kaggle/input").exists()


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


def find_input(*names: str, marker: str = "") -> Path | None:
    """Locate a mounted Kaggle input by name, whatever layout Kaggle uses.

    Mount points are not uniform and have changed: a competition lands at
    /kaggle/input/<slug>/, while datasets land at
    /kaggle/input/datasets/<user>/<slug>/ and kernel outputs under a similar
    nested path. Hard-coding one shape silently breaks the other -- which is
    exactly how the first preprocessing run failed -- so search instead.

    ``marker`` names a child that must exist, to reject a partial match.
    """
    roots = [Path("/kaggle/input")]
    for name in names:
        for root in roots:
            if not root.exists():
                continue
            # Direct child first, then one or two levels of nesting
            # (datasets/<user>/<slug>, kernels/<user>/<slug>).
            for pattern in (name, f"*/{name}", f"*/*/{name}"):
                for hit in sorted(root.glob(pattern)):
                    if not hit.is_dir():
                        continue
                    if marker and not (hit / marker).exists():
                        continue
                    return hit
    return None


if ON_KAGGLE:
    WORK = Path("/kaggle/working")
    RAW = find_input(COMPETITION, marker="train.csv") or Path("/kaggle/input") / COMPETITION

    # The mosaic corpus, when notebook 01's output is attached to a later stage.
    # Notebook 01 itself has no such input and falls through to
    # /kaggle/working/prep, which is what it writes and what gets saved.
    # RSNA_KNEE_PREP overrides the search entirely.
    _prep_env = os.environ.get("RSNA_KNEE_PREP")
    if _prep_env:
        PREP = Path(_prep_env)
    else:
        _hit = find_input("rsna-knee-preprocess", "rsna-knee-prep", marker="prep")
        PREP = (_hit / "prep") if _hit else _first_existing(
            *(p for p in [find_input("rsna-knee-preprocess", "rsna-knee-prep")] if p),
            WORK / "prep",
        )

    # Where this repo's own source was mounted, for scripts that shell out.
    CODE = find_input("rsna-knee-code", marker="src")
else:
    ROOT = Path(os.environ.get("RSNA_KNEE_ROOT", r"D:\kaggle_c"))
    RAW = ROOT / "data"
    WORK = ROOT / "work"
    PREP = ROOT / "prep"
    CODE = Path(__file__).resolve().parents[2]

CSV = RAW  # train.csv etc. sit alongside the series directories

# --- preprocessing geometry -------------------------------------------------
#
# Each series is reduced to a fixed grid of slices and stored as one mosaic image
# file. One ordinary image per series means no custom container to maintain.
#
# Resolution is the highest-value knob in this pipeline. Native slices run about
# 512x512 up to 960x960, so the first corpus at 256x256 discarded roughly 75% of
# the pixels -- and the findings that suffer most (meniscal tears, cartilage
# thinning, an MCL sprain) are exactly the subtle ones we score worst on.
#
# The first corpus stored lossless PNG at 256 and came to 8.86 GB, close enough
# to Kaggle's 20 GB notebook-output cap that raising resolution looked blocked.
# It was not: JPEG at high quality is 3-4x smaller than PNG on already-8-bit
# windowed MRI, and that compression artefact is negligible next to throwing
# away three quarters of the pixels. Switching format buys the resolution.
#
#   256 PNG   1024x1024 mosaic   ~520 KB   8.9 GB    (first corpus)
#   384 JPEG  1536x1536 mosaic   ~350 KB   ~6 GB     (current)
#
# RSNA_KNEE_SIZE overrides the resolution, so a second corpus can be built at a
# different scale and ensembled with the first.
SLICES = 16
SIZE = int(os.environ.get("RSNA_KNEE_SIZE", 384))
GRID = 4
assert GRID * GRID == SLICES

# Physical field of view kept per slice, in millimetres. The published 0.942
# solution uses 140 mm, sized so a knee fills the frame across scanners: with
# in-plane spacing running roughly 0.3-0.6 mm/px, a fixed pixel box covers very
# different anatomy per study, and the model ends up learning the scanner.
# Set RSNA_KNEE_CROP_MM=0 to disable and keep the old fixed-pixel behaviour.
CROP_MM = float(os.environ.get("RSNA_KNEE_CROP_MM", 140.0))

# Storage format for the mosaic. JPEG quality 95 on 8-bit MRI is visually
# lossless at a fraction of PNG's size; set RSNA_KNEE_EXT=.png to go back.
MOSAIC_EXT = os.environ.get("RSNA_KNEE_EXT", ".jpg")
MOSAIC_QUALITY = int(os.environ.get("RSNA_KNEE_QUALITY", 95))

# Intensity windowing percentiles. MRI has no standard scale like CT Hounsfield
# units, so each slice is windowed against its own robust range.
LO_PCT, HI_PCT = 0.5, 99.5

# --- series selection -------------------------------------------------------
#
# Studies carry 3-14 series (mean 5.5) and many are redundant repeats of a plane.
# We keep at most one per (plane, fluid-sensitivity) bucket, which covers the
# distinct diagnostic content:
#   sagittal fluid-sensitive   effusion, contusion, ACL, cartilage
#   sagittal non-fluid (PD/T1) meniscal tears -- the highest-value sequence
#   coronal  fluid-sensitive   MCL, meniscus, compartment OA
#   axial    fluid-sensitive   patellofemoral cartilage, synovitis, Baker's cyst
SERIES_SLOTS = [
    ("Sagittal", 1),
    ("Sagittal", 0),
    ("Coronal", 1),
    ("Axial", 1),
]
MAX_SERIES = len(SERIES_SLOTS)
