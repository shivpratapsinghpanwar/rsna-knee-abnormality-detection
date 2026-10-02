# Training a fold on a rented A100 (Lightning AI, Colab, Vast, anywhere)

Training may happen anywhere — the competition restricts external *data and
pretrained models*, not compute. Only the submission has to run in a Kaggle
notebook. We already train off-Kaggle on a local 3080; this is the same thing
with a bigger card.

## Why it is worth it

| | RTX 3080 10 GB | A100 40 GB |
|---|---|---|
| chunking | required (`--encoder-chunk 8`) | optional, far larger |
| batch | 1 study (64 images) | 4 studies (256 images) |
| one fold, 8 epochs | ~19 h | ~4-6 h |

## Do NOT upload the corpus from the workstation

It is 11 GB. The cloud box has far better bandwidth to Kaggle than a home
connection does, so have it pull the archive itself.

## Bootstrap

Paste into the Studio terminal. Needs your OWN kaggle.json (your account, your
machine — this is not account sharing).

```bash
set -e
git clone https://github.com/shivpratapsinghpanwar/rsna-knee-abnormality-detection.git
cd rsna-knee-abnormality-detection
pip install -q torch torchvision timm pandas numpy pillow pydicom scikit-learn kaggle

mkdir -p ~/.kaggle && cp /teamspace/uploads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

export KAGGLE_DIR=$PWD/kdata
mkdir -p $KAGGLE_DIR
# the preprocessed corpus, as packed by our own Kaggle `pack` kernel
python scripts/fetch_kernel_output.py rsna-knee-pack --dest $KAGGLE_DIR --only prep.tar
mkdir -p data && tar -xf $KAGGLE_DIR/prep.tar -C data
# sanity: must report laterality True and window series
python -c "import json;print(json.load(open('data/prep/manifest.json')))" || \
  echo "no manifest -- confirm this is the rebuilt corpus before training"

# labels: the public v4 blend, which beat our own labeller 0.8927 vs 0.8463 on gold
kaggle datasets download -d stevenleehans/rsna-knee-llm-report-labels -p lab --unzip
kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data
python - <<'PY'
import pandas as pd, sys
sys.path.insert(0, "src")
from rsna_knee.labels import ID_COL, TARGETS
v4 = pd.read_csv("lab/llm_labels_v4_blend.csv")
out = v4[[ID_COL]].copy()
for t in TARGETS:
    c = [x for x in v4.columns if x.strip().lower() == t.lower()][0]
    out[f"weak_{t}"] = pd.to_numeric(v4[c], errors="coerce")
    out[f"gold_{t}"] = 0.0
out["has_gold"] = 0
out.to_csv("work/weak_labels.csv", index=False)
print("labels written", out.shape)
PY
```

Point the code at these paths with the env vars `config.py` reads, then train.
Start with `--batch 4 --accum 4 --encoder-chunk 32`; if it OOMs, drop to
`--batch 2 --accum 8`.

```bash
python -u scripts/train.py --fold 2 --folds 5 --epochs 8 \
  --batch 4 --accum 4 --workers 8 \
  --backbone coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --size 384 \
  --lr 1e-3 --backbone-lr-mult 0.03 --grad-checkpointing \
  --encoder-chunk 32 --out work/coat
```

## Getting the result back

A checkpoint is 279 MB, so this is trivial — download `work/coat/fold*.pt` and
drop it into `D:\kaggle_c\work\coat\`. Inference reads the preprocessing and
chunk settings out of the checkpoint, so a fold trained on the A100 blends with
the local ones as long as the corpus manifest matched.

## Two things that will bite

- **gold labels are absent on the cloud box.** The 58 gold studies live in our
  local `weak_labels.csv`; the bootstrap writes zeros for them, so `gold AUC`
  prints as meaningless. Steer on **weak AUC** there, and re-measure gold
  locally once the checkpoint is back.
- **the corpus must be the rebuilt one.** If `manifest.json` is missing or says
  `window: slice`, the fold will not blend with ours — it would be served
  different pixels at inference. Check before spending the hours.
