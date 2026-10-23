# RSNA Knee Abnormality Detection

Team `_the_name _is_NERD` in the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection):
12 findings per knee MRI study, scored by macro-averaged ROC AUC.

**Result:** private leaderboard **0.942** (public 0.958), final rank **902 of 6,425 teams**. Every
0.958-public submission landed on 0.941-0.942 private, so none of the late blends moved the final score. Our own best model, a CoAtNet-2 trained on labels we extracted from the radiology reports,
scored **0.916** alone; the 0.958 came from blending into the public model stack. The full story, with every
architecture tried and every mistake we made, is in **[docs/REPORT.md](docs/REPORT.md)**.

**What is not in this repository or its releases:** every gold label column and all report text are left
out. Both are competition data; get them from the competition itself.

## Results at a glance

![Public leaderboard score per scored submission](docs/img/leaderboard_timeline.png)

| Model | Our validation (weak-label macro AUC) | Public leaderboard |
| --- | --- | --- |
| ResNet34, 3 folds, 256 px, regex labels | 0.780 | 0.746 |
| ResNet34, 5 folds, 384 px, regex + LLM labels | | 0.768 |
| DINOv3 ConvNeXt-tiny, one learning rate (bug) | fell to 0.540 | 0.789 |
| DINOv3 ConvNeXt-tiny, split learning rate | | 0.865 |
| **CoAtNet-2 (`coatnet_rmlp_2_rw_384`), fold 0** | 0.863 | **0.916** |
| CoAtNet-2, 5 folds | 0.858-0.868 per fold | blended only |
| DINOv2 ViT-L, 4 folds | 0.856-0.870 per fold | blended only |
| Public 0.958 stack + our CoAtNet arm at 0.03-0.15 | | 0.958 |
| Public 0.958 stack + DINOv2/CoAtNet arm at 0.05-0.10 | | 0.956-0.957 |
| Public 0.958 stack + heliosli R7 (final pick) | | 0.958 |

Report labellers, scored against the 58 gold-labelled studies:

| Labeller | Macro AUC |
| --- | --- |
| Regex lexicon, 12 languages (`src/rsna_knee/reports.py`) | 0.738 |
| Qwen2.5-7B-Instruct (`scripts/llm_label.py`) | 0.816 |
| 50/50 blend (`scripts/build_weak_labels.py --llm`) | **0.846** |

## Releases

Everything is on Hugging Face:
[shivpratapsingh/rsna-knee-coatnet-dinov2](https://huggingface.co/shivpratapsingh/rsna-knee-coatnet-dinov2).

| What | Folder there |
| --- | --- |
| CoAtNet-2 weights, 5 folds (fp32, 1.4 GB) | `coatnet_rmlp_2_rw_384/` |
| DINOv2 ViT-L weights, 4 folds (fp32, 4.6 GB) | `vit_large_patch14_dinov2/` |
| Report-derived labels: regex, Qwen2.5-7B, blend | `labels/` |
| Out-of-fold predictions + the cloud fold split | `oof/` |
| Checksums for every file | `SHA256SUMS` |

The final submission notebook is built by `scripts/submission/build_r7.py` from public notebooks.
Every Kaggle notebook we ran is public; the index is in [docs/KAGGLE_NOTEBOOKS.md](docs/KAGGLE_NOTEBOOKS.md).
Day-by-day working notes from the last week are in [docs/state_notes/](docs/state_notes/).

## How it works

![Pipeline from competition data to leaderboard](docs/img/pipeline.png)

**1. Labels come from reports, not from humans.** Only 58 of 4,407 training studies (1.3%) carry gold
labels, but all 4,407 carry a free-text radiology report in about 12 languages. The image model trains on
report-derived graded labels; the 58 gold studies only validate the labeller. Reports do not exist at test
time, so labelling can be as expensive as we like: blending a 7B LLM with the regex lexicon takes the labels
from 0.738 to 0.846 against gold. See [FINDINGS.md](FINDINGS.md).

**2. The images do not fit on a workstation.** The DICOM corpus is about 300-600 GB, so the training corpus
is built on Kaggle, where the data is mounted, and downloaded as one 11 GB archive.

| Stage | Code | Runs where |
| --- | --- | --- |
| report -> graded labels | `src/rsna_knee/reports.py` (regex) + `scripts/llm_label.py` (LLM), blended by `scripts/build_weak_labels.py` | Kaggle GPU once, then anywhere |
| series triage | `src/rsna_knee/series.py` | anywhere |
| DICOM -> 384 px corpus | `src/rsna_knee/{dicom_io,preprocess}.py`, `scripts/preprocess_series.py` | Kaggle CPU notebook |
| training | `src/rsna_knee/{data,model}.py`, `scripts/train.py` | local GPU, Kaggle GPU or GCP spot (`gcp/`) |
| out-of-fold scoring | `scripts/oof.py` | local GPU |
| submission | `scripts/infer.py` | Kaggle, offline |

### Corpus

Each study keeps at most one series per slot, so the study tensor has a fixed shape:

| Slot | Sequence | What it carries |
| --- | --- | --- |
| 0 | sagittal, fluid-sensitive | effusion, contusion, ACL, cartilage |
| 1 | sagittal, non-fluid (PD/T1) | meniscal tears |
| 2 | coronal, fluid-sensitive | MCL, meniscus, compartment OA |
| 3 | axial, fluid-sensitive | patellofemoral cartilage, synovitis, Baker's cyst |

Every series is cropped to a fixed **140 mm** physical field, windowed **per series** (not per slice), and
**mirrored for right knees** so that medial is always on the same side; without that, the four
medial/lateral targets are unlearnable. 16 slices per series at 384 px, stored as JPEG mosaics.

### Model

The label belongs to the study but the evidence sits in a few slices of one or two series, so the model
pools twice with per-target attention (multiple-instance learning):

```
slices of a series --attention--> series embedding
series of a study  --attention--> study embedding --> 12 logits
```

Slices enter a 2.5D backbone with their two neighbours as channels. Loss is soft-target BCE on the graded
labels. Pretrained backbones need split learning rates (about 3e-5 backbone, 1e-3 head); one rate drove
DINOv3 to chance. No horizontal flip augmentation: six targets are side-specific.

## Running it

```bash
# environment off the project path (a ';' in a Windows path breaks C extensions); Python 3.13
uv venv --python 3.13 D:/kaggle_c/.venv
uv pip install --python D:/kaggle_c/.venv/Scripts/python.exe -r requirements.txt

python scripts/build_weak_labels.py --llm labels/llm_labels.csv   # reports -> graded labels
python tests/test_reports.py                                      # labeller regression cases
python tests/test_pipeline.py                                     # preprocessing / folds / model

python scripts/train.py --fold 0 --epochs 8 --lr 1e-3 --backbone-lr-mult 0.03
python scripts/oof.py --ckpt <fold checkpoints> --fold-file <fold csv>   # leak-free OOF AUC
python scripts/infer.py --ckpt <ckpt.pt> --split test
```

### On Kaggle

```bash
python scripts/kaggle_push_code.py           # ship src/ as a dataset
python scripts/kaggle_run.py preprocess      # build the corpus (CPU notebook, hours)
python scripts/kaggle_run.py train --fold 0  # one GPU notebook per fold
python scripts/kaggle_run.py submit          # offline inference -> submission.csv
python scripts/push_weights.py               # upload local checkpoints as a dataset
```

Notes that cost us time:

- **Kaggle names a notebook from its title, not the `id` you supply**; `kaggle_run.py` derives the title
  from the slug so the two cannot disagree.
- **Input paths are not uniform.** A competition mounts at `/kaggle/input/<slug>/`, a dataset at
  `/kaggle/input/datasets/<user>/<slug>/`; `config.find_input()` searches instead of assuming.
- **The GPU quota is 30 h a week, account-wide.** A push that fails on quota leaves the notebook showing
  its previous COMPLETE status; check the push output. Run CPU work as CPU notebooks.
- **API limits.** File listing returns 503 after a few thousand calls; sustained downloading returns 429
  with a ~13 h `Retry-After`. Pack a corpus into one archive inside a notebook instead.
- **Fold splits must travel with checkpoints.** `make_folds` broke ties differently on AVX-512 cloud CPUs,
  so cloud and local splits agreed on only 22% of studies. Pass the saved split with `--fold-file`.

## Credits

The 0.958 submissions build on public work by goodpjw2008 (2.5D ConvNeXt reader and stack), nartaa
(Raptor CoAtNet weights), heliosli (R6/R7 compact models and per-finding weights) and rostigor (R6 copy),
credited in each notebook. Our v4 training labels mixed in public LLM labels by stevenleehans and pilkwang;
those are not redistributed here.

## Licence and data

Code in this repository: [MIT](LICENSE). Anything derived from the RSNA Knee competition data (labels, predictions,
weights) is subject to the competition's
[rules and data terms](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).
Research use only, not for clinical decisions.
