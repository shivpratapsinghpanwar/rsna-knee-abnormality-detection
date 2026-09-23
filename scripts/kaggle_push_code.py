"""Ship this repo's source to Kaggle as a dataset the notebooks can import.

Kaggle notebooks have no internet and cannot clone a private repo, so the code
travels as a small dataset instead. Notebooks then do:

    sys.path.insert(0, "/kaggle/input/rsna-knee-code/src")

    python scripts/kaggle_push_code.py            # create or update the dataset
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

REPO = Path(__file__).resolve().parents[1]
SLUG = "rsna-knee-code"
INCLUDE = ["src", "scripts", "tests"]


def build(stage: Path, username: str) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for rel in INCLUDE:
        src = REPO / rel
        if src.exists():
            shutil.copytree(
                src, stage / rel,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    for name in ("requirements.txt", "README.md", "FINDINGS.md"):
        if (REPO / name).exists():
            shutil.copy(REPO / name, stage / name)
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "RSNA Knee - project code",
        "id": f"{username}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", default="update code")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    username = api.config_values["username"]

    stage = REPO / ".kaggle_stage"
    build(stage, username)
    files = sum(1 for _ in stage.rglob("*") if _.is_file())
    print(f"staged {files} files in {stage}")

    try:
        api.dataset_status(f"{username}/{SLUG}")
        exists = True
    except Exception:
        exists = False

    if exists:
        print("updating existing dataset ...")
        api.dataset_create_version(str(stage), version_notes=args.message,
                                   dir_mode="zip", quiet=False)
    else:
        print("creating dataset ...")
        api.dataset_create_new(str(stage), public=False, dir_mode="zip", quiet=False)

    shutil.rmtree(stage, ignore_errors=True)
    print(f"done -> /kaggle/input/{SLUG}")


if __name__ == "__main__":
    main()
