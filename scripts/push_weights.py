"""Upload locally-trained checkpoints to Kaggle so the submission can use them.

Training moved to the local GPU when Kaggle's weekly quota ran out, which left
the weights on the wrong machine: the submission notebook runs on Kaggle and
reads checkpoints from its mounted inputs.

The dataset is named `rsna-knee-train-local` on purpose. The submission notebook
globs `*/models/*.pt` and filters on "rsna-knee-train" in the path, so weights
shipped this way are picked up by exactly the same code that reads the Kaggle
training kernels -- no branch in the submission path, which is where a mistake
would be most expensive.

    python scripts/push_weights.py                  # everything in models384
    python scripts/push_weights.py --only fold0 fold1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))

from rsna_knee.config import WORK  # noqa: E402

SLUG = "rsna-knee-train-local"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(WORK / "models384"))
    ap.add_argument("--only", nargs="*", default=[],
                    help="checkpoint name prefixes to include, e.g. fold0 fold1")
    ap.add_argument("--message", default="local checkpoints")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.config_values["username"]

    src = Path(args.src)
    ckpts = sorted(p for p in src.glob("*.pt")
                   if not args.only or any(p.name.startswith(k) for k in args.only))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {src}")

    stage = Path(args.src).parent / ".weights_stage"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "models").mkdir(parents=True)
    total = 0
    for p in ckpts:
        shutil.copy(p, stage / "models" / p.name)
        # The history sits next to the weights so a submission can be traced back
        # to the run that produced it.
        hist = p.with_name(p.name.split("_")[0] + "_history.json")
        if hist.exists():
            shutil.copy(hist, stage / "models" / hist.name)
        total += p.stat().st_size
        print(f"  {p.name}  {p.stat().st_size/1e6:.0f} MB")

    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "RSNA Knee - local checkpoints",
        "id": f"{user}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))
    print(f"uploading {len(ckpts)} checkpoints, {total/1e6:.0f} MB ...")

    try:
        api.dataset_status(f"{user}/{SLUG}")
        exists = True
    except Exception:
        exists = False

    if exists:
        api.dataset_create_version(str(stage), version_notes=args.message,
                                   dir_mode="zip", quiet=False)
    else:
        api.dataset_create_new(str(stage), public=False, dir_mode="zip",
                               quiet=False)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"done -> /kaggle/input/{SLUG}")


if __name__ == "__main__":
    main()
