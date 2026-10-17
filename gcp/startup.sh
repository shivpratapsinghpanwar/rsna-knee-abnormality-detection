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
# tee rather than redirect: the metadata runner forwards stdout to the serial
# console, which the workstation can read with get-serial-port-output.
exec > >(tee -a /var/log/knee-startup.log) 2>&1
echo "=== boot $(date -u +%FT%TZ)"

md() { curl -sf -H Metadata-Flavor:Google \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
FOLD=$(md fold); BUCKET=$(md bucket); EPOCHS=$(md epochs)
OUT=gs://$BUCKET/out/fold$FOLD
ROOT=/opt/knee
mkdir -p $ROOT/work/coat $ROOT/data $ROOT/hf
cd $ROOT

# Never leave a broken VM running (an A100 bills ~Rs 110/h). Power off, and let
# the watcher restart it; three failed boots in a row mark the fold FAILED so a
# permanent fault cannot loop forever.
fail() {
  echo "setup failed: $1"
  n=$(( $(cat $ROOT/bootfails 2>/dev/null || echo 0) + 1 )); echo $n > $ROOT/bootfails
  gcloud storage cp /var/log/knee-startup.log "$OUT/startup.log" --quiet
  [ $n -ge 3 ] && echo "setup failed $n times: $1" | gcloud storage cp - "$OUT/FAILED"
  shutdown -h now; exit 1
}

if gcloud storage ls "$OUT/DONE" >/dev/null 2>&1; then
  echo "fold $FOLD already done"; shutdown -h now; exit 0
fi

# The image installs the NVIDIA driver on first boot; wait for it.
for i in $(seq 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || fail "no GPU driver"

if [ ! -f .setup_done ]; then
  gcloud storage cp "gs://$BUCKET/stage/code.tar.gz" . && mkdir -p code &&
    tar -xzf code.tar.gz -C code
  gcloud storage cp "gs://$BUCKET/stage/train.csv" data/
  gcloud storage cp "gs://$BUCKET/stage/weak_labels.csv" work/
  # Streamed straight into tar: the archive never lands on disk twice.
  gcloud storage cat "gs://$BUCKET/stage/prep.tar" | tar -xf - -C $ROOT
  gcloud storage cp "gs://$BUCKET/stage/manifest.json" prep/manifest.json
  # Without the data nothing below is worth doing. Stay up, unmarked, so the
  # serial log can be read and the next boot retries the download.
  n=$(ls prep/train 2>/dev/null | wc -l)
  echo "corpus studies: $n"
  if [ "$n" -lt 4400 ] || [ ! -s work/weak_labels.csv ] || [ ! -s data/train.csv ] ||
     [ ! -s prep/manifest.json ] || [ ! -d code/src ]; then
    fail "data incomplete"
  fi

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
[ -f .setup_done ] || fail "environment install"
rm -f $ROOT/bootfails

# A recreated disk starts empty: take back whatever the bucket already holds.
gcloud storage cp "$OUT/fold${FOLD}_*" work/coat/ 2>/dev/null || true

# Memory-for-speed ladder, fastest first: "batch accum chunk gradckpt".
# Encoder chunks are already recomputed in backward, so per-block gradient
# checkpointing on top of that recomputes twice; dropping it saves ~20% when the
# card has room (A100 40 GB). An OOM steps one rung down and resumes. The rung
# is kept on disk so a reboot does not climb back to one that failed.
LADDER=("2 8 64 0" "2 8 32 0" "2 8 32 1" "1 16 16 1")
# Leave two cores for the system; the rest decode JPEG mosaics.
WORKERS=$(( $(nproc) - 2 ))
[ -f rung ] || echo 0 > rung

sync_out() {
  gcloud storage rsync work/coat "$OUT" --quiet
  gcloud storage cp train.log "$OUT/train.log" --quiet
}
( while true; do sleep 600; sync_out; done ) &
SYNC=$!

cd code
export RSNA_KNEE_ROOT=$ROOT HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1
while true; do
  R=$(cat $ROOT/rung); read B A C GC <<< "${LADDER[$R]}"
  GCFLAG=""; [ "$GC" = 1 ] && GCFLAG="--grad-checkpointing"
  echo "run: batch $B accum $A chunk $C gradckpt $GC"
  /opt/kv/bin/python -u scripts/train.py --fold "$FOLD" --folds 5 \
    --epochs "$EPOCHS" --batch "$B" --accum "$A" --workers "$WORKERS" \
    --backbone coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --size 384 \
    --lr 1e-3 --backbone-lr-mult 0.03 $GCFLAG \
    --encoder-chunk "$C" --resume --out $ROOT/work/coat 2>&1 |
    tee $ROOT/run.log >> $ROOT/train.log
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ] && grep -q "OutOfMemoryError" $ROOT/run.log &&
     [ "$R" -lt $((${#LADDER[@]} - 1)) ]; then
    echo "OOM on rung $R, stepping down"; echo $((R + 1)) > $ROOT/rung
    continue
  fi
  break
done
cd $ROOT
kill $SYNC 2>/dev/null
# A preemption kills training too. That is not a failure: leave no marker and
# let the next boot resume.
if [ "$(curl -sf -H Metadata-Flavor:Google \
  http://metadata.google.internal/computeMetadata/v1/instance/preempted)" = TRUE ]; then
  echo "preempted"; exit 0
fi
# Same for any other stop from outside (the 12 h cap, a manual stop): the OS is
# already going down, which is not the run failing.
if [ "$(systemctl is-system-running)" = stopping ]; then
  echo "system stopping"; exit 0
fi
sync_out

if [ $rc -eq 0 ] && grep -q "best weak AUC" train.log; then
  echo done | gcloud storage cp - "$OUT/DONE"
else
  echo "rc=$rc" | gcloud storage cp - "$OUT/FAILED"
fi
shutdown -h now
