#!/bin/bash
# Boot script for the always-on watcher VM (knee-watch, e2-small, not spot).
#
# The workstation is not reliable enough to babysit spot VMs: a power cut there
# leaves a preempted trainer stopped until someone notices. This VM runs the
# same `knee_gcp.py watch` loop as a systemd service instead, so it survives its
# own reboots, and powers itself off once every fold is DONE.
#
# Metadata: bucket, folds ("2 3"), gpu ("a100,l4"), epochs.
set -u
exec > >(tee -a /var/log/knee-watcher-boot.log) 2>&1
md() { curl -sf -H Metadata-Flavor:Google \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
BUCKET=$(md bucket); FOLDS=$(md folds); GPU=$(md gpu); EPOCHS=$(md epochs)
ROOT=/opt/kw
mkdir -p $ROOT/code
# Fresh code every boot, so a fix staged from the workstation takes effect on
# the next restart of this VM.
gcloud storage cp "gs://$BUCKET/stage/code.tar.gz" $ROOT/ &&
  tar -xzf $ROOT/code.tar.gz -C $ROOT/code

cat > /etc/systemd/system/knee-watch.service <<EOF
[Unit]
Description=Supervise knee training VMs
After=network-online.target
Wants=network-online.target

[Service]
Environment=KNEE_GPU=$GPU KNEE_EPOCHS=$EPOCHS KNEE_LOCAL=$ROOT PYTHONUNBUFFERED=1
WorkingDirectory=$ROOT/code
ExecStart=/bin/bash -c 'python3 -u gcp/knee_gcp.py watch $FOLDS >> /var/log/knee-watch.log 2>&1 && gcloud storage cp /var/log/knee-watch.log gs://$BUCKET/watch/watch.log && shutdown -h now'
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF

# The log goes to the bucket every 5 min, so the workstation can read it.
cat > /etc/systemd/system/knee-watch-log.service <<EOF
[Unit]
Description=Copy the watcher log to the bucket

[Service]
ExecStart=/bin/bash -c 'while true; do gcloud storage cp /var/log/knee-watch.log gs://$BUCKET/watch/watch.log --quiet; sleep 300; done'
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now knee-watch.service knee-watch-log.service
echo "watcher up: folds $FOLDS on $GPU"
