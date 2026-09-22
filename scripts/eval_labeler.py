"""Validate the report labeler against the 58 gold-labelled training studies.

This is the only ground-truth signal available, so it is the yardstick for every
change to the lexicon in rsna_knee.reports.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.labels import ID_COL, REPORT_COL, TARGETS  # noqa: E402
from rsna_knee.reports import label_report  # noqa: E402

DATA = Path(r"D:\kaggle_c\data")


def main() -> None:
    tr = pd.read_csv(DATA / "train.csv")
    gold = tr.dropna(subset=TARGETS).reset_index(drop=True)
    print(f"gold-labelled studies: {len(gold)}")

    pred = pd.DataFrame([label_report(r) for r in gold[REPORT_COL]])[TARGETS]

    rows = []
    aucs = []
    for t in TARGETS:
        y = gold[t].astype(int).values
        p = pred[t].values
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / len(y)
        # AUC is undefined when a target is single-class in this tiny gold set.
        auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
        if auc == auc:
            aucs.append(auc)
        rows.append(
            dict(target=t, n_pos=int(y.sum()), pred_pos=int(p.sum()),
                 prec=round(prec, 3), rec=round(rec, 3), f1=round(f1, 3),
                 acc=round(acc, 3), auc=round(auc, 3) if auc == auc else None)
        )

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print(f"\nMACRO AUC (gold, binary preds): {sum(aucs) / len(aucs):.4f}")
    print(f"MEAN F1: {df.f1.mean():.4f}   MEAN ACC: {df.acc.mean():.4f}")

    # Coverage on the full unlabelled corpus: what the weak labels would look like.
    full = pd.DataFrame([label_report(r) for r in tr[REPORT_COL]])[TARGETS]
    print("\n--- weak-label positive rate across all 4407 studies (%) ---")
    print((full.mean() * 100).round(1).to_string())
    n_empty = int((full.sum(axis=1) == 0).sum())
    print(f"\nstudies with zero positive weak labels: {n_empty} "
          f"({100 * n_empty / len(full):.1f}%)")


if __name__ == "__main__":
    main()
