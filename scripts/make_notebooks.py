"""Generate the three Kaggle notebooks from a single source of truth.

Keeping the cell text here rather than in checked-in .ipynb JSON means the
notebooks cannot drift from the code they call, and a change to the pipeline is
one edit rather than three.

    python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"

# Kaggle does not mount every input under one shape: a competition appears at
# /kaggle/input/<slug>/ but a dataset at /kaggle/input/datasets/<user>/<slug>/.
# The bootstrap has to locate its own source before it can import anything, so
# it searches rather than assuming a path.
BOOT = '''import sys, subprocess
from pathlib import Path

CODE = next(
    (p for pat in ("rsna-knee-code", "*/rsna-knee-code", "*/*/rsna-knee-code")
     for p in sorted(Path("/kaggle/input").glob(pat)) if (p / "src").is_dir()),
    None,
)
assert CODE is not None, "rsna-knee-code dataset is not attached to this notebook"
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(CODE))
print("CODE", CODE)

# timm ships in the Kaggle image; the install is a fallback for images that
# lack it, and cannot run in the offline submission notebook.
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)
    import timm

import rsna_knee.config as cfg
print("RAW ", cfg.RAW, cfg.RAW.exists())
print("PREP", cfg.PREP, cfg.PREP.exists())'''

PREPROCESS = [
    "# 01 - Build the PNG mosaic corpus\n"
    "#\n"
    "# Converts the mounted DICOM series into one lossless PNG mosaic per selected\n"
    "# series (~8-9 GB for the full training set, inside the 20 GB output cap).\n"
    "# This kernel's output mounts straight into the training notebook as a kernel\n"
    "# source, so there is no dataset to build by hand.\n"
    "#\n"
    "# It is resumable: if the 12 h session limit hits, re-run and it skips what is\n"
    "# already written.",
    BOOT,
    '''!cd {CODE} && python scripts/build_weak_labels.py''',
    '''# Full corpus. Raise --workers to the number of available cores.
!cd {CODE} && python scripts/preprocess_series.py --split train --workers 4''',
    '''!cd {CODE} && python scripts/preprocess_series.py --split test --workers 4''',
    '''from pathlib import Path
prep = Path("/kaggle/working/prep")
pngs = list(prep.rglob("*.png"))
size = sum(p.stat().st_size for p in pngs)
print(f"{len(pngs)} mosaics, {size/1e9:.2f} GB")''',
]

TRAIN = [
    "# 02 - Train the study-level model\n"
    "#\n"
    "# Reads the mosaic corpus from notebook 01, mounted as a kernel source, and\n"
    "# trains one fold. Each fold runs as its own kernel (rsna-knee-train-f0, -f1,\n"
    "# ...), because pushing them all to one id would overwrite earlier weights.\n"
    "#\n"
    "# Validation reports two numbers: weak AUC over ~880 held-out studies (stable,\n"
    "# used for checkpoint selection) and gold AUC over the handful of human-labelled\n"
    "# studies in the fold (honest but very noisy).",
    BOOT,
    '''!cd {CODE} && python scripts/build_weak_labels.py''',
    '''FOLD = 0
!cd {CODE} && python scripts/train.py --fold {FOLD} --epochs 8 --backbone resnet34 \\
    --batch 2 --accum 8 --workers 2 --out /kaggle/working/models''',
    '''import json, glob
for h in sorted(glob.glob("/kaggle/working/models/*_history.json")):
    print(h)
    print(json.dumps(json.load(open(h)), indent=2))''',
]

SUBMIT = [
    "# 03 - Offline inference -> submission.csv\n"
    "#\n"
    "# Attach: the competition, the code dataset, and every train-fold kernel.\n"
    "# No internet is used, so this is valid as a code-competition submission.\n"
    "#\n"
    "# The hidden test set is preprocessed here from raw DICOM by the same code\n"
    "# path training used, so there is no train/serve skew.",
    BOOT,
    '''from pathlib import Path
# Each training fold ran as its own kernel, so its checkpoints mount under that
# kernel's own input directory.
CKPTS = sorted(
    str(p) for pat in ("*/models/*.pt", "*/*/models/*.pt", "*/*/*/models/*.pt")
    for p in Path("/kaggle/input").glob(pat)
    if "rsna-knee-train" in str(p)
)
assert CKPTS, "no checkpoints found - is every train fold attached as a source?"
for c in CKPTS:
    print(c)
# A shell magic interpolates a bare {name}; build the argument string here rather
# than relying on an expression inside the braces.
CKPT_ARGS = " ".join(CKPTS)''',
    '''!cd {CODE} && python scripts/infer.py --split test --out /kaggle/working/submission.csv \\
    --ckpt {CKPT_ARGS}''',
    '''import pandas as pd
sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.shape)
sub.head()''',
]


def notebook(cells: list[str]) -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown" if i == 0 else "code",
                "metadata": {},
                "source": (c if i else c).splitlines(keepends=True),
                **({} if i == 0 else {"outputs": [], "execution_count": None}),
            }
            for i, c in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in [
        ("01_preprocess.ipynb", PREPROCESS),
        ("02_train.ipynb", TRAIN),
        ("03_submit.ipynb", SUBMIT),
    ]:
        # The leading markdown cell doubles as the notebook's title block.
        cells = [cells[0].replace("# ", "", 1)] + cells[1:]
        (OUT / name).write_text(json.dumps(notebook(cells), indent=1))
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
