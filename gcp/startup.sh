#!/bin/bash
# Runs as root on EVERY boot of a training VM -- the first one and every restart
# after a spot preemption. Each step is guarded so a reboot picks up where the
# last boot stopped:
#   setup     once per disk (marker .setup_done)
#   training  scripts/train.py --resume continues from fold<N>_last.pt
#   finish    DONE marker in the bucket, then power off so billing stops
#
# Settings arrive as instance metadata, written by gcp/knee_gcp.py launch.
set -u
exec >> /var/log/knee-startup.log 2>&1
echo "=== boot $(date -u +%FT%TZ)"

md() { curl -sf -H Metadata-Flavor:Google \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
FOLD=$(md fold); BUCKET=$(md bucket); EPOCHS=$(md epochs)
OUT=gs://$BUCKET/out/fold$FOLD
ROOT=/opt/knee
mkdir -p $ROOT/work/coat $ROOT/data $ROOT/hf
cd $ROOT

if gcloud storage ls "$OUT/DONE" >/dev/null 2>&1; then
  echo "fold $FOLD already done"; shutdown -h now; exit 0
fi

# The image installs the NVIDIA driver on first boot; wait for it.
for i in $(seq 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "no GPU driver"; exit 1; }

if [ ! -f .setup_done ]; then
  gcloud storage cp "gs://$BUCKET/stage/code.tar.gz" . && mkdir -p code &&
    tar -xzf code.tar.gz -C code
  gcloud storage cp "gs://$BUCKET/stage/train.csv" data/
  gcloud storage cp "gs://$BUCKET/stage/weak_labels.csv" work/
  # Streamed straight into tar: the archive never lands on disk twice.
  gcloud storage cat "gs://$BUCKET/stage/prep.tar" | tar -xf - -C $ROOT
  gcloud storage cp "gs://$BUCKET/stage/manifest.json" prep/
  ls prep/train | wc -l

  # uv brings its own Python, so this does not depend on what the image ships.
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
  uv venv /opt/kv --python 3.11
  uv pip install --python /opt/kv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python /opt/kv/bin/python timm==1.0.30 numpy pandas \
    scikit-learn pillow pydicom
  /opt/kv/bin/python -c "import torch; assert torch.cuda.is_available()" &&
    touch .setup_done
fi
[ -f .setup_done ] || { echo "setup failed"; exit 1; }

# A recreated disk starts empty: take back whatever the bucket already holds.
gcloud storage cp "$OUT/fold${FOLD}_*" work/coat/ 2>/dev/null || true

# Batch shape is chosen once and kept, so a resumed run continues the same
# schedule. 24 GB fits two studies with 32-image chunks; fall back if it OOMs.
[ -f cfg ] || echo "2 8 32" > cfg
read B A C < cfg

sync_out() {
  gcloud storage rsync work/coat "$OUT" --quiet
  gcloud storage cp train.log "$OUT/train.log" --quiet
}
( while true; do sleep 600; sync_out; done ) &
SYNC=$!

cd code
export RSNA_KNEE_ROOT=$ROOT HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1
run() {
  /opt/kv/bin/python -u scripts/train.py --fold "$FOLD" --folds 5 \
    --epochs "$EPOCHS" --batch "$B" --accum "$A" --workers 7 \
    --backbone coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --size 384 \
    --lr 1e-3 --backbone-lr-mult 0.03 --grad-checkpointing \
    --encoder-chunk "$C" --resume --out $ROOT/work/coat >> $ROOT/train.log 2>&1
}
run; rc=$?
if [ $rc -ne 0 ] && grep -q "OutOfMemoryError" $ROOT/train.log &&
   [ ! -f $ROOT/work/coat/fold${FOLD}_last.pt ]; then
  echo "OOM at $B/$A/$C, falling back to 1/16/16"
  echo "1 16 16" > $ROOT/cfg; read B A C < $ROOT/cfg
  run; rc=$?
fi
cd $ROOT
kill $SYNC 2>/dev/null
# A preemption kills training too. That is not a failure: leave no marker and
# let the next boot resume.
if [ "$(curl -sf -H Metadata-Flavor:Google \
  http://metadata.google.internal/computeMetadata/v1/instance/preempted)" = TRUE ]; then
  echo "preempted"; exit 0
fi
sync_out

if [ $rc -eq 0 ] && grep -q "best weak AUC" train.log; then
  echo done | gcloud storage cp - "$OUT/DONE"
else
  echo "rc=$rc" | gcloud storage cp - "$OUT/FAILED"
fi
shutdown -h now
