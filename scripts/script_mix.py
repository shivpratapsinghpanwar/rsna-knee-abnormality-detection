"""Count reports by writing system, to target lexicon effort."""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsna_knee.labels import REPORT_COL, TARGETS  # noqa: E402

DATA = Path(r"D:\kaggle_c\data")

SCRIPTS = [
    ("greek", r"[\u0370-\u03FF\u1F00-\u1FFF]"),
    ("cyrillic", r"[\u0400-\u04FF]"),
    ("arabic", r"[\u0600-\u06FF]"),
    ("cjk", r"[\u4e00-\u9fff\u3040-\u30ff]"),
    ("hebrew", r"[\u0590-\u05FF]"),
    ("thai", r"[\u0E00-\u0E7F]"),
    ("devanagari", r"[\u0900-\u097F]"),
]


def script_of(t: str) -> str:
    for name, pat in SCRIPTS:
        if len(re.findall(pat, t)) > 20:
            return name
    return "latin"


def main() -> None:
    tr = pd.read_csv(DATA / "train.csv")
    tr["script"] = tr[REPORT_COL].map(script_of)
    counts = tr.script.value_counts()
    print("--- reports by writing system (n=%d) ---" % len(tr))
    for k, v in counts.items():
        print(f"{k:12s} {v:5d}  ({100*v/len(tr):5.1f}%)")

    gold = tr.dropna(subset=TARGETS)
    print("\n--- gold-labelled (n=%d) by script ---" % len(gold))
    print(gold.script.value_counts().to_string())


if __name__ == "__main__":
    main()
