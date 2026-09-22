"""Print report snippets for gold-positive studies the labeler missed.

Grounds lexicon expansion in real phrasing instead of guesswork.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.labels import REPORT_COL, TARGETS  # noqa: E402
from rsna_knee.reports import label_report, normalize  # noqa: E402

DATA = Path(r"D:\kaggle_c\data")
FOCUS = sys.argv[1] if len(sys.argv) > 1 else "Effusion"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def main() -> None:
    tr = pd.read_csv(DATA / "train.csv")
    gold = tr.dropna(subset=TARGETS).reset_index(drop=True)
    pred = pd.DataFrame([label_report(r) for r in gold[REPORT_COL]])[TARGETS]

    miss = gold[(gold[FOCUS] == 1) & (pred[FOCUS].values == 0)]
    print(f"=== {FOCUS}: {len(miss)} false negatives (showing {min(N, len(miss))}) ===\n")
    for i, (_, row) in enumerate(miss.head(N).iterrows()):
        txt = normalize(row[REPORT_COL])
        print(f"--- FN {i + 1} ---")
        print(txt[:520].encode("ascii", "backslashreplace").decode())
        print()

    fp = gold[(gold[FOCUS] == 0) & (pred[FOCUS].values == 1)]
    print(f"=== {FOCUS}: {len(fp)} false positives (showing {min(3, len(fp))}) ===\n")
    for i, (_, row) in enumerate(fp.head(3).iterrows()):
        txt = normalize(row[REPORT_COL])
        print(f"--- FP {i + 1} ---")
        print(txt[:400].encode("ascii", "backslashreplace").decode())
        print()


if __name__ == "__main__":
    main()
