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
labels; the 58 gold studies are only good enough to validate the labeller, not
to train on.

Labelling therefore has no inference-time cost — the reports do not exist at
test time — which means it can be arbitrarily expensive. Blending a 7B LLM with
the regex lexicon takes the labels from 0.738 to **0.846** against gold. See
[FINDINGS.md](FINDINGS.md).

**2. The images do not fit on a workstation.** The DICOM corpus is ~300-600 GB
(~800k files) and Kaggle serves it at a couple of MB/s, so the corpus is built
**on Kaggle** where the data is already mounted. Nothing in the pipeline assumes
a full local copy of the DICOM.

The *mosaic* corpus is another matter: at 8.9 GB it fits anywhere, so training
runs wherever there is a GPU — Kaggle when quota allows, otherwise locally.

## Pipeline

| stage | code | runs where |
|---|---|---|
| report -> graded weak labels | `src/rsna_knee/reports.py` (regex) + `scripts/llm_label.py` (LLM), blended by `scripts/build_weak_labels.py` | Kaggle GPU once, then anywhere |
| series triage | `src/rsna_knee/series.py` | anywhere |
| DICOM -> JPEG mosaic corpus | `src/rsna_knee/{dicom_io,preprocess}.py`, `scripts/preprocess_series.py` | Kaggle CPU kernel |
| training | `src/rsna_knee/{data,model}.py`, `scripts/train.py` | Kaggle GPU or local 3080 |
| submission | `scripts/infer.py` | Kaggle, offline |

### Storage format
Each selected series becomes **one JPEG**: 16 evenly-spaced slices at 384x384,
tiled 4x4 into a 1536x1536 mosaic (~0.5 MB). That is ~8.9 GB for the whole
training set — inside Kaggle's 20 GB output cap — with no custom container to
maintain. `preprocess.py` round-trips mosaic <-> volume exactly, which
`tests/test_pipeline.py` pins.

**Format is what buys the resolution.** The first corpus was lossless PNG at
256, and 256 discards roughly 75% of the native pixels — worst for the subtle
findings we score lowest on. The 20 GB cap looked like the blocker but was not:
measured on real mosaics, JPEG q95 is **480 KB/series against PNG's 906 KB at
384**, so 2.25x the pixels fits in *less* space than the old corpus, at 0.18%
mean error. Set `RSNA_KNEE_SIZE` / `RSNA_KNEE_EXT` to build a corpus at another
scale; checkpoints record the size they trained at, so models from different
corpora can be blended.

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

### On Kaggle

```bash
python scripts/kaggle_push_code.py           # ship src/ as a private dataset
python scripts/kaggle_run.py preprocess      # build the mosaic corpus (CPU, hours)
python scripts/kaggle_run.py train --fold 0  # one GPU kernel per fold
python scripts/kaggle_run.py train --fold 1
python scripts/kaggle_run.py train --fold 2
python scripts/kaggle_run.py submit          # offline inference -> submission.csv

python scripts/kaggle_run.py preprocess --status
python scripts/kaggle_run.py train --status --all-folds
```

Re-run `kaggle_push_code.py` after any source change: kernels read the dataset,
not this checkout. Notebooks are generated by `scripts/make_notebooks.py`, so
edit that rather than the `.ipynb` files.

Each stage moves its output to the next as a **kernel source**, so nothing has to
be re-uploaded: notebook 01's corpus mounts into 02, and each fold's checkpoints
mount into 03. Folds get separate kernel ids (`rsna-knee-train-fold-0`, `-fold-1`,
...) because pushing them to one id would overwrite earlier weights.

> **Kaggle names a kernel from its title, not the `id` you supply.** An id of
> `rsna-knee-train-f0` under the title "RSNA Knee - train fold 0" silently
> creates `rsna-knee-train-fold-0`, and later lookups by id fail with a
> *permissions* error that reads like broken auth. `kaggle_run.py` derives the
> title from the slug so the two cannot disagree.

Both long stages stop under their own `--max-minutes` budget rather than being
killed at Kaggle's session limit, which can discard the output entirely.
Preprocessing is resumable, so a second run continues from what the first wrote.

> **Input paths are not uniform.** A competition mounts at `/kaggle/input/<slug>/`
> but a dataset at `/kaggle/input/datasets/<user>/<slug>/`. `config.find_input()`
> searches instead of assuming; hard-coding the dataset shape is what made the
> first preprocessing run die with `ModuleNotFoundError`.

> **Windows note.** This checkout sits under a directory containing a `;`, which
> Windows treats as a PATH separator — a virtualenv created *inside* it cannot
> load C extensions (`numpy` fails with a DLL error). Keep the venv and data on a
> clean path such as `D:\kaggle_c`. Python 3.14 has no torch wheels; use 3.13.

## Kaggle API rate limits

Two limits bite when pulling data locally, and both fail quietly:

- **Listing** — `competition_list_files` pages 200 at a time and returns **503**
  after a few thousand calls. Enumerating the ~800k files via the API is not
  viable; build the manifest inside a notebook with filesystem calls instead.
- **Downloading** — sustained parallel downloading returns **429**, and the
  observed `Retry-After` is ~13 hours. 16 workers trips it within about an hour.

`fetch_dev_subset.py` therefore defaults to 6 workers and shares one backoff gate
across threads, so a single 429 pauses every worker rather than having each retry
into the same wall. If the ban is longer than 15 minutes it stops cleanly; the
run is resumable, so re-running once the window clears picks up where it left off.

## Compute budget

Kaggle's GPU quota is **30 h per week, account-wide** — not per competition, so
other projects spend it too. When it runs out `kaggle_run.py` fails at push, and
the kernel keeps reporting its *previous* version's COMPLETE status, which looks
exactly like a successful run. Check the push output, not the status.

Two consequences shape the layout here:

- Anything that does not need a GPU runs as a **CPU kernel** (preprocessing,
  packing), which draws on a separate and much larger budget.
- Moving a corpus off Kaggle goes through `kaggle_run.py pack`, which tars it
  inside a CPU kernel first. The kernel-output API fetches loose files serially
  — the 17,071-file corpus came down at ~3 files/minute, versus ~2 MB/s for the
  single archive.

Training also runs locally: an RTX 3080 is faster than a Kaggle T4, and at 10 GB
it needs `--batch 1 --grad-checkpointing` at 384 rather than the T4's batch 2.

## Status

- **Labeller** — blended regex + Qwen2.5-7B, **0.846** macro AUC on the 58 gold
  studies, up from 0.663 (v1 regex) and 0.738 (final regex). Only 7.9% of
  studies now carry no signal at all, down from 19.4%. 24 regression tests.
- **Corpus** — 17,071 mosaics over all 4,407 studies. Rebuilt at **384 JPEG**
  (8.9 GB) after the first pass at 256 PNG.
- **Image model** — three folds at 256 on regex-only labels: weak **0.780**,
  gold **0.776**. One fold at 384 on the same labels: weak **0.787** (+0.008),
  gold 0.726.
- **Not yet trained on the blended labels.** That is the biggest pending gain
  and it is blocked only on GPU time, not on code.
- **Weakest target is MCL** — 0.611 in the image model, and the rarest at 4.7%
  positive. Its *labels* improved most from the LLM (0.813 → 0.959), so this is
  the first thing to re-check after a retrain.
- **Submission** — verified end to end offline, valid format; not yet entered on
  the leaderboard.

