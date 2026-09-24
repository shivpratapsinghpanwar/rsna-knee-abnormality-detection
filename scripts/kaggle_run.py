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
    "preprocess": ("01_preprocess.ipynb", False, True, [], []),
    "train": ("02_train.ipynb", True, True, [], [PREP_KERNEL]),
    # The submission notebook preprocesses the hidden test set itself from raw
    # DICOM, so it needs the weights but not the training corpus. Internet off is
    # required for the run to count as a code-competition entry.
    "submit": ("03_submit.ipynb", True, False, [], []),
    # Internet on: the labeller downloads its weights. That is fine because the
    # labeller runs at training time only and never touches the submission.
    "label": ("04_label.ipynb", True, True, [], []),
}

# Folds to train, and therefore which train kernels the submission blends. Each
# fold gets its own kernel: pushing every fold to one id would overwrite the
# previous fold's weights, leaving only the last one.
FOLDS = [0, 1, 2]


def kernel_slug(stage: str, fold: int | None = None) -> str:
    if stage == "train":
        return f"rsna-knee-train-fold-{fold}"
    return f"rsna-knee-{stage}"


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
        return str(getattr(r, "status", "?")), str(getattr(r, "failure_message", "") or "")
    except Exception as exc:
        return f"error:{type(exc).__name__}", str(exc)[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--status", action="store_true", help="only report status")
    ap.add_argument("--all-folds", action="store_true",
                    help="with --status on train: report every fold")
    ap.add_argument("--wait", action="store_true", help="poll until the run ends")
    ap.add_argument("--poll", type=int, default=120, help="seconds between polls")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.config_values["username"]
    fold = args.fold if args.stage == "train" else None
    slug = kernel_slug(args.stage, fold)
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
    if args.stage == "train":
        # The notebook declares FOLD = 0; rewrite it rather than keeping three
        # near-identical notebooks in the repo.
        text = text.replace("FOLD = 0", f"FOLD = {args.fold}")
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
    print(f"  pushed -> https://www.kaggle.com/code/{ref}")
    shutil.rmtree(stage_dir, ignore_errors=True)

    if not args.wait:
        print(f"\nnot waiting. check with:\n"
              f"  python scripts/kaggle_run.py {args.stage} --status")
        return

    print("polling ...")
    while True:
        time.sleep(args.poll)
        st, msg = status(api, ref)
        print(f"  {time.strftime('%H:%M:%S')}  {st}  {msg}", flush=True)
        if st.lower() in ("complete", "error", "cancelacknowledged"):
            break


if __name__ == "__main__":
    main()
