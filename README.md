# RSNA Knee Abnormality Detection

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

## Task
Predict 12 binary abnormality findings per knee MRI study, scored by
**macro-averaged ROC AUC**. Submission columns, in order:

```
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
```

## The two facts that determine the approach

**1. Labels come from reports, not from humans.** Only 58 of 4407 training
studies (1.3%) carry gold labels, but all 4407 carry a free-text radiology report
in roughly 12 languages. The image model therefore trains on report-derived weak
labels; the 58 gold studies are only good enough to validate the labeler, not to
train on. See [FINDINGS.md](FINDINGS.md).

**2. The images do not fit on a workstation.** The DICOM corpus is ~300-600 GB
(~800k files) and Kaggle serves it at a couple of MB/s. So **training runs on
Kaggle**, where the data is already mounted, and only a small streamed subset
lives locally for developing the code. Nothing in the pipeline assumes a full
local copy.

## Pipeline

| stage | code | runs where |
|---|---|---|
| report -> graded weak labels | `src/rsna_knee/reports.py`, `scripts/build_weak_labels.py` | anywhere (CPU) |
| series triage | `src/rsna_knee/series.py` | anywhere |
| DICOM -> PNG mosaic corpus | `src/rsna_knee/{dicom_io,preprocess}.py`, `scripts/preprocess_series.py` | Kaggle (full) / local (subset) |
| training | `src/rsna_knee/{data,model}.py`, `scripts/train.py` | Kaggle |
| submission | `scripts/infer.py` | Kaggle, offline |

### Storage format
Each selected series becomes **one lossless PNG**: 16 evenly-spaced slices at
256x256, tiled 4x4 into a 1024x1024 mosaic (~0.5 MB). That is ~8-9 GB for the
whole training set — inside Kaggle's 20 GB output cap and small enough to keep
locally — with no custom container to maintain. `preprocess.py` round-trips
mosaic <-> volume exactly, which `tests/test_pipeline.py` pins.

### Series selection
Studies carry 3-14 series (mean 5.5), mostly redundant repeats of a plane. We
keep at most one per slot:

| slot | sequence | what it carries |
|---|---|---|
| 0 | sagittal, fluid-sensitive | effusion, contusion, ACL, cartilage |
| 1 | sagittal, non-fluid (PD/T1) | meniscal tears — the highest-value sequence |
| 2 | coronal, fluid-sensitive | MCL, meniscus, compartment OA |
| 3 | axial, fluid-sensitive | patellofemoral cartilage, synovitis, Baker's cyst |

This covers all 4407 studies with 17,071 series (70% of the total) and gives the
study tensor a fixed shape. `train_series.csv` already supplies plane and
fluid-sensitivity, so no DICOM header parsing is needed.

### Model
The label is attached to the study but the evidence is in a few slices of one or
two series, so the model pools twice with **per-target attention**:

```
slices of a series --attention--> series embedding
series of a study  --attention--> study embedding --> 12 logits
```

Attention is per-target because the slices that matter for Baker's cyst are not
the ones that matter for PF OA. Slices enter a standard 2.5D backbone with their
two neighbours as RGB channels. Loss is soft-target BCE, which preserves the
labeler's grading instead of collapsing it to 0/1.

**No horizontal flip augmentation.** Six targets are side-specific, and on
coronal and axial images image-left/right *is* medial/lateral, so a flip
relabels the anatomy.

## Running it

```bash
# local: env lives off the project path (see note below)
uv venv --python 3.13 D:/kaggle_c/.venv
uv pip install --python D:/kaggle_c/.venv/Scripts/python.exe -r requirements.txt

python scripts/build_weak_labels.py          # reports -> weak_labels.csv
python tests/test_reports.py                 # 24 labeler regression cases
python tests/test_pipeline.py                # preprocessing / folds / model

python scripts/fetch_dev_subset.py --studies 48   # stream a local dev corpus
python scripts/train.py --fold 0 --epochs 8
python scripts/infer.py --ckpt <ckpt.pt> --split test
```

On Kaggle: `python scripts/kaggle_push_code.py` uploads this repo as a private
dataset, then run `notebooks/01_preprocess` -> `02_train` -> `03_submit`
(regenerate them with `scripts/make_notebooks.py`).

> **Windows note.** This checkout sits under a directory containing a `;`, which
> Windows treats as a PATH separator — a virtualenv created *inside* it cannot
> load C extensions (`numpy` fails with a DLL error). Keep the venv and data on a
> clean path such as `D:\kaggle_c`. Python 3.14 has no torch wheels; use 3.13.

## Status

- **Labeler** — done and validated: macro AUC **0.738** on the 58 gold studies,
  up from 0.663 at v1. 24 regression tests.
- **Image pipeline** — implemented and tested end to end (preprocess -> train ->
  submission) on a local dev subset. **Not yet trained at scale**: that needs a
  Kaggle run over the full corpus.
- **Next** — push the code dataset, run notebook 01 to build the corpus, then
  train 5 folds and submit.
