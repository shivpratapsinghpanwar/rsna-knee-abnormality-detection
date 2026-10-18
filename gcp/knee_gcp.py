"""Train CoAtNet folds on Google Cloud spot L4s, driven from the workstation.

    python gcp/knee_gcp.py login            # activate D:\\kaggle_c\\gcloud\\sa.json
    python gcp/knee_gcp.py stage            # bucket + code, labels, corpus (once)
    python gcp/knee_gcp.py launch 2 3 4     # one spot VM per fold
    python gcp/knee_gcp.py watch 2 3 4      # restart after preemption, pull results
    python gcp/knee_gcp.py status
    python gcp/knee_gcp.py down             # delete every knee-*f<N> VM

KNEE_ARCH picks the model family (ARCHS below); CoAtNet is the default.

Each VM runs gcp/startup.sh on every boot, which trains with --resume and
syncs to gs://<bucket>/out/fold<N>/ every 10 min, so a preemption costs at most
one epoch. Finished checkpoints land in D:\\kaggle_c\\work\\coat_gcp\\, apart
from the local folds, because fold 2 also exists there as a partial run.

gcloud is called through its bundled Python rather than gcloud.cmd: cmd.exe
mangles `^`, `,` and `;`, and this repo's own path contains a `;`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SDK = Path(os.environ.get("GCLOUD_SDK", r"D:\kaggle_c\gcloud\google-cloud-sdk"))
# On the workstation gcloud runs through its bundled Python; on the always-on
# watcher VM (gcp/watcher.sh) it is the stock gcloud on PATH.
GCLOUD = ([str(SDK / "platform" / "bundledpython" / "python.exe"),
           str(SDK / "lib" / "gcloud.py")] if os.name == "nt" else ["gcloud"])
KAGGLE = Path(os.environ.get("KNEE_LOCAL", r"D:\kaggle_c"))

# Model families: tag -> (timm backbone, backbone lr multiplier). CoAtNet keeps
# its original names (knee-f<N>, out/fold<N>) so runs already in the bucket
# still resolve; every other family is namespaced by its tag.
ARCHS = {
    "coat": ("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "0.03"),
    # A pure CNN, unlike CoAtNet and the public stack's readers: a blend gains
    # from models whose mistakes differ, not from a second copy of the same one.
    "effv2m": ("tf_efficientnetv2_m.in21k_ft_in1k", "0.1"),
}
ARCH = os.environ.get("KNEE_ARCH", "coat")
BACKBONE, LR_MULT = ARCHS[ARCH]
PULL = KAGGLE / "work" / ("coat_gcp" if ARCH == "coat" else f"{ARCH}_gcp")
# The watcher VM has no use for the checkpoints, and a small disk.
DO_PULL = os.environ.get("KNEE_PULL", "1") == "1"

EPOCHS = os.environ.get("KNEE_EPOCHS", "8")
SPOT = os.environ.get("KNEE_SPOT", "1") == "1"
# Card -> (machine type, extra accelerator flag, US zones that stock it,
# roughly most-to-least available). Asia-southeast was always full when tried.
GPUS = {
    "l4": ("g2-standard-8", None,
           ["us-central1-a", "us-central1-b", "us-central1-c", "us-east4-a",
            "us-east4-c", "us-east1-b", "us-east1-c", "us-east1-d",
            "us-west1-a", "us-west1-b", "us-west4-a", "us-west4-c"]),
    # One H100 80 GB, 26 vCPUs: twice the decode workers of the A100 box too.
    "h100": ("a3-highgpu-1g", None,
             ["us-central1-a", "us-central1-b", "us-central1-c"]),
    "a100": ("a2-highgpu-1g", None,
             ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"]),
    "v100": ("n1-standard-8", "type=nvidia-tesla-v100,count=1",
             ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
              "us-east1-c", "us-west1-a", "us-west1-b"]),
}
# Cards to try, in order of preference, e.g. KNEE_GPU=a100,l4: the first one
# with capacity in any of its zones wins.
CARDS = os.environ.get("KNEE_GPU", "l4").split(",")
BUCKET_REGION = "us-central1"
# Service-account key: scoped to one project, nothing else on the Google
# account. The VMs run as the same account, so they need no other grants.
SA_KEY = Path(os.environ.get("KNEE_SA_KEY", r"D:\kaggle_c\gcloud\sa.json"))


def sa_email() -> str:
    """The account the VMs run as: the key's, else knee-runner if it exists.

    Many accounts forbid key creation by policy, so the CLI may be logged in as
    the user instead; the VMs then still run as the narrow account if it exists.
    """
    if SA_KEY.exists():
        return json.loads(SA_KEY.read_text())["client_email"]
    name = f"knee-runner@{project()}.iam.gserviceaccount.com"
    if gc("iam", "service-accounts", "describe", name, check=False, quiet=True):
        return name
    return ""


def login(_args) -> None:
    key = json.loads(SA_KEY.read_text())
    gc("auth", "activate-service-account", f"--key-file={SA_KEY}")
    gc("config", "set", "project", key["project_id"])
    print("active:", key["client_email"], "project:", key["project_id"])


def gc(*args, check=True, quiet=False) -> str:
    r = subprocess.run(GCLOUD + list(args), capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(r.stderr.strip()[-2000:])
    if not quiet and r.returncode:
        print(r.stderr.strip()[-600:])
    return r.stdout.strip() if r.returncode == 0 else ""


def project() -> str:
    p = gc("config", "get-value", "project", quiet=True)
    if not p or p == "(unset)":
        sys.exit("no project set: gcloud config set project <PROJECT_ID>")
    return p


def bucket() -> str:
    return os.environ.get("KNEE_BUCKET", f"{project()}-rsna-knee")


def vm(fold) -> str:
    return f"knee-f{fold}" if ARCH == "coat" else f"knee-{ARCH}-f{fold}"


def out(fold) -> str:
    sub = f"fold{fold}" if ARCH == "coat" else f"{ARCH}/fold{fold}"
    return f"gs://{bucket()}/out/{sub}"


def code_tarball() -> bytes:
    """src/, scripts/ and gcp/ exactly as they sit in the working tree."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ("src", "scripts", "gcp"):
            for f in sorted((REPO / sub).rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    tar.add(f, arcname=str(f.relative_to(REPO)).replace("\\", "/"))
    return buf.getvalue()


def stage(_args) -> None:
    b = bucket()
    if not gc("storage", "buckets", "describe", f"gs://{b}", check=False, quiet=True):
        gc("storage", "buckets", "create", f"gs://{b}",
           f"--location={BUCKET_REGION}", "--uniform-bucket-level-access")
        print("created bucket", b)
    tmp = KAGGLE / "gcloud" / "stage"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "code.tar.gz").write_bytes(code_tarball())
    small = [tmp / "code.tar.gz", KAGGLE / "data" / "train.csv",
             KAGGLE / "work" / "weak_labels.csv", KAGGLE / "prep" / "manifest.json"]
    for f in small:
        gc("storage", "cp", str(f), f"gs://{b}/stage/{f.name}")
        print("uploaded", f.name)
    # dl/prep.tar was checked against the local corpus mosaic-by-mosaic; it is
    # the rebuilt corpus. 11 GB, so skip it if an identical size is already up.
    tar = KAGGLE / "dl" / "prep.tar"
    have = gc("storage", "ls", "-l", f"gs://{b}/stage/prep.tar", check=False, quiet=True)
    if str(tar.stat().st_size) in have:
        print("prep.tar already staged")
    else:
        print(f"uploading prep.tar ({tar.stat().st_size / 1e9:.1f} GB)...", flush=True)
        gc("storage", "cp", str(tar), f"gs://{b}/stage/prep.tar")
        print("uploaded prep.tar")


def image() -> tuple[str, str]:
    """Newest Deep Learning VM image with a CUDA 12 driver baked in."""
    rows = json.loads(gc("compute", "images", "list",
                         "--project=deeplearning-platform-release",
                         "--no-standard-images", "--format=json(family,creationTimestamp)"))
    fams = [r for r in rows if r.get("family") and
            re.match(r"(common|pytorch).*cu12", r["family"]) and
            not re.search(r"arm|tpu|debug|nvidia-?4", r["family"])]
    fams.sort(key=lambda r: r["creationTimestamp"], reverse=True)
    if not fams:
        sys.exit("no Deep Learning VM image found")
    return "deeplearning-platform-release", fams[0]["family"]


def startup_file() -> Path:
    """startup.sh with LF endings, at a path cmd-safe for gcloud to read."""
    dst = KAGGLE / "gcloud" / "startup.sh"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes((REPO / "gcp" / "startup.sh").read_bytes().replace(b"\r\n", b"\n"))
    return dst


def create(fold, img=None) -> str | None:
    img_project, family = img or image()
    sa = sa_email()
    meta = (f"fold={fold},bucket={bucket()},epochs={EPOCHS},out={out(fold)},"
            f"backbone={BACKBONE},lrmult={LR_MULT},install-nvidia-driver=True")
    for card in CARDS:
        machine, accel, zones = GPUS[card]
        for zone in zones:
            args = ["compute", "instances", "create", vm(fold), f"--zone={zone}",
                    f"--machine-type={machine}", "--maintenance-policy=TERMINATE",
                    f"--image-project={img_project}", f"--image-family={family}",
                    "--boot-disk-size=150GB", "--boot-disk-type=pd-balanced",
                    "--scopes=cloud-platform", f"--metadata={meta}",
                    f"--metadata-from-file=startup-script={startup_file()}"]
            if sa:
                args.append(f"--service-account={sa}")
            if accel:
                args.append(f"--accelerator={accel}")
            if SPOT:
                args += ["--provisioning-model=SPOT", "--instance-termination-action=STOP"]
            if card == "h100":
                # a3 machines always carry a local SSD, and a stop needs to be
                # told what happens to it. Nothing is kept there.
                args.append("--discard-local-ssds-at-termination-timestamp=true")
            # Hard cap on one boot, so a hung run cannot bill forever: Google
            # stops it, the watcher starts it again and it resumes.
            args += ["--max-run-duration=12h"]
            try:
                gc(*args)
                print(f"{vm(fold)} created in {zone} on {card} ({family})", flush=True)
                return zone
            except RuntimeError as e:
                msg = str(e)
                short = msg.splitlines()[-1][:160] if msg else "?"
                print(f"  {card} {zone}: {short}", flush=True)
                if "Quota" in msg and "GPUS_ALL_REGIONS" in msg:
                    print("  global GPU quota reached -- not trying more zones")
                    return None
    return None


def launch(args) -> None:
    img = image()
    print("image:", img[1])
    for f in args.folds:
        create(f, img)


def instances() -> dict:
    # Trainers only: knee-f<N> and knee-<tag>-f<N>, never knee-watch.
    rows = json.loads(gc("compute", "instances", "list", "--filter=name~^knee-.*f[0-9]+$",
                         "--format=json(name,status,zone)") or "[]")
    return {r["name"]: (r["status"], r["zone"].rsplit("/", 1)[-1]) for r in rows}


def marker(fold, name) -> bool:
    return bool(gc("storage", "ls", f"{out(fold)}/{name}",
                   check=False, quiet=True))


def last_epoch(fold) -> str:
    log = gc("storage", "cat", f"{out(fold)}/train.log",
             check=False, quiet=True)
    lines = [l for l in log.splitlines() if "epoch" in l and "AUC" in l]
    return lines[-1].strip() if lines else "(no epoch yet)"


def status(_args=None) -> None:
    inst = instances()
    for name, (st, zone) in sorted(inst.items()):
        f = name.rsplit("-f", 1)[-1]
        print(f"{name:14s} {st:11s} {zone:15s} {last_epoch(f) if name == vm(f) else ''}")
    if not inst:
        print("no knee VMs")


def pull(fold) -> None:
    if not DO_PULL:
        return
    PULL.mkdir(parents=True, exist_ok=True)
    src = out(fold)
    for pat in (f"fold{fold}_{BACKBONE}.pt", "fold*_history.json", "train.log"):
        gc("storage", "cp", f"{src}/{pat}", str(PULL), check=False)
    log = PULL / "train.log"
    if log.exists():
        log.replace(PULL / f"fold{fold}_train.log")


def watch(args) -> None:
    pending, img, stuck = set(args.folds), None, {}
    while pending:
        before = len(pending)
        inst = instances()
        # With a one-GPU quota, folds go strictly in order: otherwise a stopped
        # VM frees the slot and a later fold grabs it before the restart can.
        active = sorted(pending) if args.parallel else sorted(pending)[:1]
        for f in active:
            name = vm(f)
            if marker(f, "DONE"):
                pull(f)
                if name in inst:
                    gc("compute", "instances", "delete", name,
                       f"--zone={inst[name][1]}", "--quiet", check=False)
                print(f"{ARCH} fold {f} DONE", flush=True)
                pending.discard(f)
                continue
            if marker(f, "FAILED"):
                print(f"{ARCH} fold {f} FAILED -- see {out(f)}/train.log")
                if name in inst:  # a stopped VM still bills for its disk
                    gc("compute", "instances", "delete", name,
                       f"--zone={inst[name][1]}", "--quiet", check=False)
                pending.discard(f)
                continue
            if name not in inst:
                img = img or image()
                create(f, img)
                continue
            st, zone = inst[name]
            if st in ("TERMINATED", "STOPPED"):
                # Spot reclaimed it. Restart in place keeps the disk; if the
                # zone stays out of L4s for 30 min, rebuild elsewhere -- the
                # bucket holds the progress.
                r = subprocess.run(GCLOUD + ["compute", "instances", "start", name,
                                             f"--zone={zone}"],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    stuck.pop(f, None)
                    print(f"fold {f}: restarted in {zone}", flush=True)
                else:
                    stuck.setdefault(f, time.time())
                    if time.time() - stuck[f] > 1800:
                        gc("compute", "instances", "delete", name, f"--zone={zone}",
                           "--quiet", check=False)
                        stuck.pop(f)
                        print(f"fold {f}: {zone} out of capacity, rebuilding", flush=True)
        print(time.strftime("[%H:%M]"), "; ".join(
            f"{ARCH} f{f}: {last_epoch(f)}" for f in sorted(pending)) or "all done", flush=True)
        # A fold that just finished frees the GPU: start the next one now
        # rather than five minutes from now.
        if pending and len(pending) == before:
            time.sleep(300)


def down(_args) -> None:
    for name, (_, zone) in instances().items():
        gc("compute", "instances", "delete", name, f"--zone={zone}", "--quiet",
           check=False)
        print("deleted", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login").set_defaults(fn=login)
    sub.add_parser("stage").set_defaults(fn=stage)
    for name, fn in (("launch", launch), ("watch", watch)):
        p = sub.add_parser(name)
        p.add_argument("folds", nargs="+", type=int)
        p.add_argument("--parallel", action="store_true",
                       help="run every fold at once (needs a GPU quota above 1)")
        p.set_defaults(fn=fn)
    sub.add_parser("status").set_defaults(fn=status)
    sub.add_parser("down").set_defaults(fn=down)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
