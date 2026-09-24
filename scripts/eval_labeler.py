"""Score a report labeler against the 58 gold-labelled training studies.

This is the only ground truth available, so it is the yardstick for every change
to the labeler -- rule-based or LLM. Both go through the same function here, so
the comparison is like for like.

    python scripts/eval_labeler.py                       # rule-based only
    python scripts/eval_labeler.py --llm llm_gold.csv    # + LLM, + their blend

Caveat that governs how much any of this means: n=58, so a per-target AUC rests
on 10-25 positives and the noise floor on macro AUC is roughly 0.03. Differences
smaller than that are not real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import CSV  # noqa: E402
from rsna_knee.labels import ID_COL, REPORT_COL, TARGETS  # noqa: E402
from rsna_knee.reports import label_report  # noqa: E402

THRESH = 0.5


def score(name: str, gold: pd.DataFrame, pred: pd.DataFrame) -> float:
    """Print a per-target table for one labeler and return its macro AUC."""
    rows, aucs = [], []
    for t in TARGETS:
        y = gold[t].astype(int).to_numpy()
        s = pred[t].to_numpy(dtype=float)
        p = (s >= THRESH).astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        # AUC is undefined when a target is single-class in this tiny gold set.
        auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
        if auc == auc:
            aucs.append(auc)
        rows.append(dict(target=t, n_pos=int(y.sum()), pred_pos=int(p.sum()),
                         prec=round(prec, 3), rec=round(rec, 3), f1=round(f1, 3),
                         acc=round((tp + tn) / len(y), 3),
                         auc=round(auc, 3) if auc == auc else None))
    df = pd.DataFrame(rows)
    macro = float(np.mean(aucs))
    print(f"\n===== {name} =====")
    print(df.to_string(index=False))
    print(f"MACRO AUC {macro:.4f}   MEAN F1 {df.f1.mean():.4f}")
    return macro


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="", help="CSV of LLM labels to score too")
    args = ap.parse_args()

    tr = pd.read_csv(CSV / "train.csv")
    gold = tr.dropna(subset=TARGETS).reset_index(drop=True)
    print(f"gold-labelled studies: {len(gold)}")

    rule = pd.DataFrame([label_report(r) for r in gold[REPORT_COL]])[TARGETS]
    results = {"rule-based": score("rule-based (regex lexicon)", gold, rule)}

    if args.llm:
        llm_all = pd.read_csv(args.llm)
        # Align to the gold studies by id: the LLM file may hold the whole corpus.
        llm = (gold[[ID_COL]].merge(llm_all, on=ID_COL, how="left")[TARGETS]
               .fillna(0.0))
        missing = int(llm_all[ID_COL].isin(gold[ID_COL]).sum())
        print(f"\nLLM rows matching gold: {missing}/{len(gold)}")
        results["llm"] = score("LLM", gold, llm)

        # Their blend. Two labelers that fail differently are worth averaging,
        # and this is nearly free to check while we are here.
        blend = (rule.to_numpy() + llm.to_numpy()) / 2
        results["blend"] = score("blend (mean of both)",
                                 gold, pd.DataFrame(blend, columns=TARGETS))

    print("\n===== summary =====")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v:.4f}")
    if len(results) > 1:
        best, second = sorted(results.values(), reverse=True)[:2]
        print(f"\nbest beats runner-up by {best - second:.4f}; the n=58 noise floor "
              f"is ~0.03, so treat anything smaller as a tie.")


if __name__ == "__main__":
    main()
