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


if ON_KAGGLE:
    RAW = Path("/kaggle/input") / COMPETITION
    WORK = Path("/kaggle/working")
    # The preprocessed corpus, when notebook 01's output is attached as an input
    # dataset. Notebook 01 itself has no such input and falls through to
    # /kaggle/working/prep, which is what it writes and what gets saved.
    # RSNA_KNEE_PREP overrides the search if the dataset is attached under a
    # different slug.
    # Notebook 01's output mounts at /kaggle/input/<kernel-slug>/, so its prep/
    # directory is found there without creating a separate dataset by hand.
    _prep_env = os.environ.get("RSNA_KNEE_PREP")
    PREP = Path(_prep_env) if _prep_env else _first_existing(
        *[
            Path("/kaggle/input") / slug / sub
            for slug in ("rsna-knee-preprocess", "rsna-knee-prep")
            for sub in ("prep", "")
        ],
        WORK / "prep",
    )
else:
    ROOT = Path(os.environ.get("RSNA_KNEE_ROOT", r"D:\kaggle_c"))
    RAW = ROOT / "data"
    WORK = ROOT / "work"
    PREP = ROOT / "prep"

CSV = RAW  # train.csv etc. sit alongside the series directories

# --- preprocessing geometry -------------------------------------------------
#
# Each series is reduced to a fixed grid of slices and stored as one lossless
# PNG mosaic. Keeping it to a single ordinary image file per series means no
# custom container, and PNG on 8-bit MRI is roughly half the size of raw uint8.
#
# 16 slices at 256x256 tiled 4x4 gives a 1024x1024 mosaic, about 300-500 KB per
# series. Across the ~17.6k series we actually keep that is well inside both the
# Kaggle 20 GB output cap and local disk.
SLICES = 16
SIZE = 256
GRID = 4
assert GRID * GRID == SLICES

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
