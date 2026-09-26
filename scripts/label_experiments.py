"""Compare labeller configurations on the 58 gold studies, cheaply.

The labeller is the bottleneck: it scores 0.846 blended while the leaderboard
needs ~0.95, and the image model cannot outrun its teacher by much. So the
question worth spending GPU on is which labeller configuration is best, and the
58 gold studies answer it in minutes rather than the 2.5 h a full corpus pass
costs.

Each configuration is scored alone and blended with the regex lexicon, because
the blend has so far beaten either component:

    python scripts/label_experiments.py --model Qwen/Qwen2.5-7B-Instruct

Results land in WORK/label_experiments.csv. Read them with the n=58 noise floor
of roughly 0.03 in mind -- a configuration has to win by more than that to be
worth a full corpus pass.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import CSV, WORK  # noqa: E402
from rsna_knee.labels import ID_COL, REPORT_COL, TARGETS  # noqa: E402
from rsna_knee.reports import label_report  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# (name, scale, samples). Ordered cheapest first so an interrupted sweep still
# leaves the most informative comparisons done.
CONFIGS = [
    ("scale4_x1", 4, 1),    # the current production labeller, as the baseline
    ("scale11_x1", 11, 1),  # finer rungs: does resolution alone fix the ties?
    ("scale11_x3", 11, 3),  # + self-consistency, a second source of resolution
    ("scale4_x3", 4, 3),    # sampling without finer rungs, to separate the two
]


def macro(y: np.ndarray, p: np.ndarray) -> float:
    aucs = [roc_auc_score(y[:, j], p[:, j])
            for j in range(y.shape[1]) if len(set(y[:, j])) > 1]
    return float(np.mean(aucs))


def ties(p: np.ndarray) -> float:
    """Mean fraction of studies sharing a value, averaged over targets.

    This is the quantity the whole exercise is about: ROC AUC cannot rank tied
    studies, so a labeller that emits few distinct values is throwing away
    ordering it could have had.
    """
    return float(np.mean([1 - len(np.unique(p[:, j])) / len(p)
                          for j in range(p.shape[1])]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--only", default="", help="run just this config name")
    args = ap.parse_args()

    gold = pd.read_csv(CSV / "train.csv").dropna(subset=TARGETS).reset_index(drop=True)
    y = gold[TARGETS].to_numpy(int)
    rule = pd.DataFrame([label_report(r) for r in gold[REPORT_COL]])[TARGETS].to_numpy()
    print(f"gold studies: {len(gold)}")
    print(f"regex baseline: macro {macro(y, rule):.4f}  ties {ties(rule):.3f}\n")

    rows = []
    for name, scale, samples in CONFIGS:
        if args.only and name != args.only:
            continue
        out = WORK / f"lblexp_{name}.csv"
        if not out.exists():
            cmd = [str(Path("D:/kaggle_c/.venv/Scripts/python.exe")),
                   "-u", "scripts/llm_label.py", "--gold-only",
                   "--model", args.model, "--load-4bit",
                   "--batch", str(args.batch), "--scale", str(scale),
                   "--samples", str(samples), "--out", str(out)]
            print(f"--- {name}: scale={scale} samples={samples} ---", flush=True)
            r = subprocess.run(cmd, cwd=REPO)
            if r.returncode != 0 or not out.exists():
                print(f"  {name} failed, skipping")
                continue

        llm = gold[[ID_COL]].merge(pd.read_csv(out), on=ID_COL,
                                   how="left")[TARGETS].fillna(0).to_numpy()
        best_w, best_m = 0.0, -1.0
        for w in np.arange(0, 1.01, 0.1):
            m = macro(y, (1 - w) * rule + w * llm)
            if m > best_m:
                best_w, best_m = float(w), m
        rows.append(dict(config=name, scale=scale, samples=samples,
                         llm_macro=round(macro(y, llm), 4),
                         llm_ties=round(ties(llm), 3),
                         blend_macro=round(best_m, 4),
                         blend_w=round(best_w, 1),
                         blend50=round(macro(y, 0.5 * rule + 0.5 * llm), 4)))
        print(f"  {name}: llm {rows[-1]['llm_macro']:.4f} "
              f"ties {rows[-1]['llm_ties']:.3f}  "
              f"blend@0.5 {rows[-1]['blend50']:.4f}  "
              f"best blend {best_m:.4f} @ w={best_w:.1f}\n", flush=True)

    if rows:
        df = pd.DataFrame(rows).sort_values("blend50", ascending=False)
        df.to_csv(WORK / "label_experiments.csv", index=False)
        print(df.to_string(index=False))
        print("\nn=58, so the noise floor is ~0.03: a config has to beat the "
              "baseline by more than that to justify a full corpus pass.")


if __name__ == "__main__":
    main()
