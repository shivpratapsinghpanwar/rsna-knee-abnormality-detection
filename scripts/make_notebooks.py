"""Generate the Kaggle notebooks from a single source of truth.

Keeping the cell text here rather than in checked-in .ipynb JSON means the
notebooks cannot drift from the code they call, and a change to the pipeline is
one edit rather than four.

    python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"

# Kaggle does not mount every input under one shape: a competition appears at
# /kaggle/input/<slug>/, a dataset at /kaggle/input/datasets/<user>/<slug>/ and a
# kernel output at /kaggle/input/notebooks/<user>/<slug>/. The bootstrap has to
# locate its own source before it can import anything, so it searches rather than
# assuming a path.
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

# timm ships in the Kaggle image; the install is a fallback for images that lack
# it, and cannot run in the offline submission notebook.
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)
    import timm

import rsna_knee.config as cfg
print("RAW ", cfg.RAW, cfg.RAW.exists())
print("PREP", cfg.PREP, cfg.PREP.exists())
print("SIZE", cfg.SIZE, cfg.MOSAIC_EXT)'''

PREPROCESS = [
    "# 01 - Build the mosaic corpus\n"
    "#\n"
    "# Converts the mounted DICOM series into one mosaic image per selected series:\n"
    "# 16 slices tiled 4x4, JPEG q95. At 384 that is ~8 GB for the full training\n"
    "# set, inside the 20 GB notebook-output cap.\n"
    "#\n"
    "# JPEG rather than PNG is what pays for the resolution: on already-8-bit\n"
    "# windowed MRI it is roughly half the size at 0.18% mean error, which is\n"
    "# nothing next to the 75% of pixels a 256 corpus throws away.\n"
    "#\n"
    "# This kernel's output mounts straight into the training notebook as a kernel\n"
    "# source, so there is no dataset to build by hand. It is resumable: if the\n"
    "# 12 h session limit hits, re-run and it skips what is already written.",
    BOOT,
    '''!cd {CODE} && python scripts/build_weak_labels.py''',
    '''# --max-minutes stops under Kaggle's session limit so the finished mosaics are
# saved; a killed session can save nothing at all.
!cd {CODE} && python scripts/preprocess_series.py --split train --workers 4 \\
    --max-minutes 660''',
    '''!cd {CODE} && python scripts/preprocess_series.py --split test --workers 4''',
    '''from pathlib import Path
prep = Path("/kaggle/working/prep")
files = [p for p in prep.rglob("*") if p.suffix.lower() in (".jpg", ".png")]
size = sum(p.stat().st_size for p in files)
print(f"{len(files)} mosaics, {size/1e9:.2f} GB")''',
]

TRAIN = [
    "# 02 - Train the study-level model\n"
    "#\n"
    "# Reads the mosaic corpus from notebook 01, mounted as a kernel source, and\n"
    "# trains one fold. Each fold runs as its own kernel (rsna-knee-train-fold-0,\n"
    "# -fold-1, ...), because pushing them all to one id would overwrite earlier\n"
    "# weights.\n"
    "#\n"
    "# Validation reports two numbers: weak AUC over ~880 held-out studies (stable,\n"
    "# used for checkpoint selection) and gold AUC over the ~12 human-labelled\n"
    "# studies in the fold (honest but very noisy - the noise floor is ~0.03).",
    BOOT,
    '''# Blended weak labels: 0.5 regex + 0.5 Qwen2.5-7B. The blend scores 0.846
# against the 58 gold studies where regex alone scores 0.738, and it is what the
# model actually learns from, so it is the single biggest lever in the pipeline.
!cd {CODE} && python scripts/build_weak_labels.py \
    --llm {CODE}/labels/llm_labels.csv --blend 0.5''',
    '''FOLD = 0
BACKBONE = "resnet34"
BATCH = 2
POSW = 0
# The MIL structure feeds MAX_SERIES * SLICES = 64 images per study, so batch 2
# is already 128 images per forward at 384. A heavier backbone needs batch 1 and
# more accumulation rather than a bigger batch -- the multiplier, not the model,
# is what fills the GPU here.
!cd {CODE} && python scripts/train.py --fold {FOLD} --epochs 8 --backbone {BACKBONE} \\
    --batch {BATCH} --accum {16 // BATCH} --workers 2 --max-minutes 480 \\
    --pos-weight-cap {POSW} --out /kaggle/working/models''',
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
    "# path training used, so there is no train/serve skew. Checkpoints carry the\n"
    "# resolution they were trained at, so models from different corpora blend.",
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

LABEL = [
    "# 04 - LLM report labeller\n"
    "#\n"
    "# The reports exist only at training time, so the labeller has no inference\n"
    "# cost and can be as expensive as we like. That asymmetry is the main one in\n"
    "# this competition, and the rule-based labeller does not exploit it: it tops\n"
    "# out at 0.738 gold AUC because hand-written regex cannot cover 12 languages\n"
    "# of hedged clinical prose.\n"
    "#\n"
    "# The 58 gold studies are scored first, through the same harness the\n"
    "# rule-based labeller is scored by, so the two are directly comparable before\n"
    "# any GPU time goes into the remaining 4349.",
    BOOT,
    '''import subprocess, sys
# bitsandbytes is not always in the Kaggle image and 4-bit is not optional here:
# a 7B model in 16-bit weights is ~14 GB against the T4's 14.56 GB, which loads
# and then OOMs as soon as attention allocates.
try:
    import bitsandbytes  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],
                   check=False)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
BATCH = 4
QUANT = "--load-4bit"''',
    '''# Validation pass: 58 reports, a couple of minutes.
!cd {CODE} && python scripts/llm_label.py --gold-only --model {MODEL} {QUANT} \\
    --batch {BATCH} --out /kaggle/working/llm_labels_gold.csv''',
    '''# Score the LLM labels against gold, side by side with the rule-based ones.
!cd {CODE} && python scripts/eval_labeler.py --llm /kaggle/working/llm_labels_gold.csv''',
]

# The full-corpus pass gets its own kernel. Running it after the gold pass in one
# session left the first process's allocator still holding the card, and the
# 4407-report run died at model load with 86 MB free.
LABEL_FULL = [
    "# 05 - LLM labels for the whole corpus\n"
    "#\n"
    "# Only worth the GPU time once notebook 04 shows the model beating the\n"
    "# rule-based labeller on the 58 gold studies. It did: 0.811 against 0.738,\n"
    "# and 0.844 blended.\n"
    "#\n"
    "# Writes incrementally, so the output is usable even if the session is cut\n"
    "# short -- build_weak_labels falls back to the regex score for any study the\n"
    "# LLM has no row for.",
    BOOT,
    '''import subprocess, sys
try:
    import bitsandbytes  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],
                   check=False)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
BATCH = 4
QUANT = "--load-4bit"''',
    '''!cd {CODE} && python scripts/llm_label.py --model {MODEL} {QUANT} --batch {BATCH} \\
    --out /kaggle/working/llm_labels.csv''',
    '''import pandas as pd
df = pd.read_csv("/kaggle/working/llm_labels.csv")
print(df.shape)
print(df.drop(columns=["StudyInstanceUID"]).mean().round(3).to_string())''',
]


def notebook(cells: list[str]) -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown" if i == 0 else "code",
                "metadata": {},
                "source": c.splitlines(keepends=True),
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
        ("04_label.ipynb", LABEL),
        ("05_label_full.ipynb", LABEL_FULL),
    ]:
        # The leading markdown cell doubles as the notebook's title block.
        cells = [cells[0].replace("# ", "", 1)] + cells[1:]
        (OUT / name).write_text(json.dumps(notebook(cells), indent=1))
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
