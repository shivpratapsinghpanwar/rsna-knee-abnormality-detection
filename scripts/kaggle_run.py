"""Push a notebook to Kaggle and run it, optionally waiting for the result.

Kaggle notebooks are where this project actually trains, since the DICOM corpus
does not fit locally. This wraps the kernels API so a run is one command:

    python scripts/kaggle_run.py preprocess          # build the mosaic corpus
    python scripts/kaggle_run.py train --fold 0      # train one fold
    python scripts/kaggle_run.py submit              # inference -> submission.csv

    python scripts/kaggle_run.py --status preprocess # check on a running kernel

Each stage declares its own resources: preprocessing is CPU-bound and wants the
12 h CPU session, training wants a GPU, and the submission kernel must have
internet disabled to be valid for a code competition.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))

REPO = Path(__file__).resolve().parents[1]
COMPETITION = "rsna-knee-abnormality-detection"
CODE_DS = "rsna-knee-code"
PREP_KERNEL = "rsna-knee-preprocess"

# stage -> (notebook, gpu, internet, dataset slugs, kernel slugs)
#
# The preprocessing notebook's output mounts straight into the later stages as a
# kernel source, so the mosaic corpus never has to be re-uploaded as a dataset.
STAGES = {
    # Cheap read-only check on the DICOM headers; run before a rebuild.
    "audit": ("00_audit.ipynb", False, True, [], []),
    # GPU, no data: measures what a CoAtNet costs on the card we would use.
    "bench": ("00b_bench.ipynb", True, True, [], []),
    "preprocess": ("01_preprocess.ipynb", False, True, [], []),
    "train": ("02_train.ipynb", True, True, [], [PREP_KERNEL]),
    # The submission notebook preprocesses the hidden test set itself from raw
    # DICOM, so it needs the weights but not the training corpus. Internet off is
    # required for the run to count as a code-competition entry.
    "submit": ("03_submit.ipynb", True, False, [], []),
    # Same notebook, but reading locally-trained weights uploaded as a dataset
    # instead of the Kaggle training kernels. Kept separate so a submission is
    # unambiguously one configuration rather than a blend of two eras.
    # GPU: a CPU submission at 384 exceeded the competition's runtime limit and
    # scored nothing. Inference cost is part of the design here, not an
    # afterthought -- five models at 384 is only affordable on a GPU.
    "submitlocal": ("03_submit.ipynb", True, False, ["rsna-knee-train-local"], []),
    # Internet on: the labeller downloads its weights. That is fine because the
    # labeller runs at training time only and never touches the submission.
    "label": ("04_label.ipynb", True, True, [], []),
    "labelfull": ("05_label_full.ipynb", True, True, [], []),
    # CPU only: packing costs no GPU quota, which matters when the weekly 30 h
    # is gone and training has to move to local hardware.
    "pack": ("06_pack.ipynb", False, False, [], [PREP_KERNEL]),
}

# Folds to train, and therefore which train kernels the submission blends. Each
# fold gets its own kernel: pushing every fold to one id would overwrite the
# previous fold's weights, leaving only the last one.
FOLDS = [0, 1, 2]


def kernel_slug(stage: str, fold: int | None = None, suffix: str = "") -> str:
    base = f"rsna-knee-train-fold-{fold}" if stage == "train" else f"rsna-knee-{stage}"
    return f"{base}-{suffix}" if suffix else base


def kernel_title(slug: str) -> str:
    """A title that slugifies back to `slug`.

    Kaggle derives the kernel's real slug from the *title*, not from the `id` in
    the metadata -- an id of rsna-knee-train-f0 under the title
    "RSNA Knee - train fold 0" silently created rsna-knee-train-fold-0 instead,
    and every later lookup by id failed. Deriving the title from the slug makes
    the two agree by construction.
    """
    return slug.replace("-", " ")


def metadata(user: str, slug: str, title: str, nb: str, gpu: bool, net: bool,
             datasets: list[str], kernels: list[str]) -> dict:
    return {
        "id": f"{user}/{slug}",
        "title": title,
        "code_file": nb,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": net,
        "competition_sources": [COMPETITION],
        "dataset_sources": [f"{user}/{CODE_DS}"] + [f"{user}/{s}" for s in datasets],
        "kernel_sources": [f"{user}/{s}" for s in kernels],
    }


def status(api, ref: str) -> tuple[str, str]:
    try:
        r = api.kernels_status(ref)
        # str() on the enum gives "KernelWorkerStatus.COMPLETE"; callers compare
        # against bare names, so keep only the last segment. Returning the
        # prefixed form made the poll loop below never recognise completion and
        # spin for 9.5 hours without ever entering the submission -- which cost
        # two submissions before it was spotted.
        raw = str(getattr(r, "status", "?"))
        return raw.rsplit(".", 1)[-1], str(getattr(r, "failure_message", "") or "")
    except Exception as exc:
        return f"error:{type(exc).__name__}", str(exc)[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--backbone", default="", help="train: timm backbone name")
    ap.add_argument("--batch", type=int, default=0, help="train: studies per step")
    ap.add_argument("--posw", type=float, default=-1,
                    help="train: inverse-prevalence positive weight cap")
    ap.add_argument("--cpu", action="store_true",
                    help="force a CPU kernel; the GPU quota is weekly and a "
                         "submission is cheap enough to run without one")
    ap.add_argument("--suffix", default="",
                    help="append to the kernel slug, to run variants side by side")
    ap.add_argument("--status", action="store_true", help="only report status")
    ap.add_argument("--all-folds", action="store_true",
                    help="with --status on train: report every fold")
    ap.add_argument("--wait", action="store_true", help="poll until the run ends")
    ap.add_argument("--tta", default="",
                    help="submit/submitlocal: comma-separated TTA transforms")
    ap.add_argument("--submit", metavar="MESSAGE", default="",
                    help="after a submit/submitlocal run completes, enter it in "
                         "the competition with this description and poll for the "
                         "public score. Implies --wait")
    ap.add_argument("--poll", type=int, default=120, help="seconds between polls")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.config_values["username"]
    fold = args.fold if args.stage == "train" else None
    slug = kernel_slug(args.stage, fold, args.suffix)
    ref = f"{user}/{slug}"

    if args.status:
        if args.stage == "train" and args.all_folds:
            for f in FOLDS:
                r = f"{user}/{kernel_slug('train', f)}"
                st, msg = status(api, r)
                print(f"{r}: {st}  {msg}")
            return
        st, msg = status(api, ref)
        print(f"{ref}: {st}  {msg}")
        return

    nb, gpu, net, datasets, kernels = STAGES[args.stage]
    if args.cpu:
        gpu = False
    # The submission blends every trained fold, each of which lives in its own
    # kernel output.
    if args.stage == "submit":
        kernels = [kernel_slug("train", f) for f in FOLDS]
    src = REPO / "notebooks" / nb
    if not src.exists():
        raise SystemExit(f"{src} missing -- run scripts/make_notebooks.py")

    stage_dir = REPO / ".kernel_stage"
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True)

    text = src.read_text(encoding="utf-8")
    # Notebook knobs to rewrite, if any. Empty for stages that take none, so the
    # substitution below is a no-op rather than a NameError.
    subs: dict[str, str] = {}
    if args.stage in ("submit", "submitlocal") and args.tta:
        subs = {"TTA": repr(args.tta)}
    elif args.stage == "train":
        # The notebook declares its knobs as plain assignments. Edit them through
        # the parsed notebook, not with a regex over the raw file: the .ipynb is
        # JSON, so a string like BACKBONE = "resnet34" is stored with escaped
        # quotes and a naive substitution corrupts the document.
        subs = {"FOLD": str(args.fold)}
        if args.backbone:
            subs["BACKBONE"] = repr(args.backbone)
        if args.batch:
            subs["BATCH"] = str(args.batch)
        if args.posw >= 0:
            subs["POSW"] = str(args.posw)

    doc = json.loads(text)
    applied = {k: False for k in subs}
    for cell in doc["cells"]:
        for i, line in enumerate(cell.get("source", [])):
            for key, val in subs.items():
                if applied[key] or not line.startswith(f"{key} = "):
                    continue
                end = "\n" if line.endswith("\n") else ""
                cell["source"][i] = f"{key} = {val}{end}"
                applied[key] = True
    missing = [k for k, ok in applied.items() if not ok]
    if missing:
        raise SystemExit(f"could not set {missing} in {nb}")
    if subs:
        text = json.dumps(doc, indent=1)
        print(f"  config: {subs}")
    (stage_dir / nb).write_text(text, encoding="utf-8")
    (stage_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(user, slug, kernel_title(slug), nb, gpu, net,
                            datasets, kernels), indent=2)
    )

    srcs = [f"{user}/{CODE_DS}"] + [f"{user}/{s}" for s in datasets + kernels]
    print(f"pushing {ref}  (gpu={gpu} internet={net})")
    print(f"  inputs: {COMPETITION} + {srcs}")
    resp = api.kernels_push(str(stage_dir))
    err = getattr(resp, "error", None)
    if err:
        raise SystemExit(f"push failed: {err}")
    # The version number comes back from the push and is the only reliable source:
    # kernels_list reports current_version_number as 0.
    version = next((v for v in (getattr(resp, "version_number", None),
                                getattr(resp, "versionNumber", None)) if v), None)
    print(f"  pushed -> https://www.kaggle.com/code/{ref}")
    shutil.rmtree(stage_dir, ignore_errors=True)

    if args.submit and args.stage not in ("submit", "submitlocal"):
        raise SystemExit("--submit only applies to the submit/submitlocal stages")
    if not args.wait and not args.submit:
        print()
        print("not waiting. check with:")
        print(f"  python scripts/kaggle_run.py {args.stage} --status")
        return

    print("polling ...")
    st = ""
    while True:
        time.sleep(args.poll)
        st, msg = status(api, ref)
        print(f"  {time.strftime('%H:%M:%S')}  {st}  {msg}", flush=True)
        if st.lower() in ("complete", "error", "cancelacknowledged"):
            break

    if not args.submit:
        return
    if st.lower() != "complete":
        raise SystemExit(f"not submitting: the kernel ended {st}")
    if not version:
        raise SystemExit("the push reported no version number to submit")

    # A code competition is entered by naming a kernel version. The grader re-runs
    # the notebook itself, so "submission.csv" is the name of the file that run is
    # expected to produce, not a file being uploaded from here.
    print()
    print(f"submitting {ref} v{version}: {args.submit!r}")
    api.competition_submit_code("submission.csv", args.submit, COMPETITION,
                                kernel=ref, kernel_version=version, quiet=True)
    print("submitted. The grader re-runs the notebook, so a score takes a while.")
    while True:
        time.sleep(args.poll)
        try:
            subs = api.competition_submissions(COMPETITION)
        except Exception as exc:
            print(f"  {time.strftime('%H:%M:%S')}  poll failed: "
                  f"{type(exc).__name__}", flush=True)
            continue
        me = next((x for x in subs
                   if str(getattr(x, "description", "")) == args.submit), None)
        if me is None:
            print(f"  {time.strftime('%H:%M:%S')}  not listed yet", flush=True)
            continue
        st2 = str(getattr(me, "status", "")).rsplit(".", 1)[-1].upper()
        score = str(getattr(me, "public_score", "") or "")
        oops = str(getattr(me, "error_description", "") or "")
        print(f"  {time.strftime('%H:%M:%S')}  {st2}  {score}  {oops}"[:130],
              flush=True)
        if st2 in ("COMPLETE", "ERROR"):
            print()
            print(f"public score: {score or 'none'}")
            if oops:
                print(f"error: {oops}")
            break


if __name__ == "__main__":
    main()
