"""Leak-free OOF comparison of CoAtNet and DINOv2, and what a blend of the two buys.

Each oof2/<fam>_f<k>.csv holds one checkpoint's predictions on its own held-out
fold (NaN elsewhere). CoAtNet folds 0-1 used the local split, everything else
the cloud split, so the head-to-head uses only the cloud folds both families
have (2 and 4), where they scored the identical studies.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

O = Path(r"D:\kaggle_c\work\oof2")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
lab = pd.read_csv(r"D:\kaggle_c\work\weak_labels.csv").set_index("StudyInstanceUID")


def load(fam, folds):
    parts = []
    for k in folds:
        p = O / f"{fam}_f{k}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p).set_index("StudyInstanceUID")
        d = d[d[TARGETS].notna().all(axis=1)][TARGETS]
        d["k"] = k
        parts.append(d)
    return pd.concat(parts) if parts else None


def aucs(ids, pred):
    weak = (lab.loc[ids, [f"weak_{t}" for t in TARGETS]].to_numpy() >= 0.5).astype(int)
    per = {t: roc_auc_score(weak[:, j], pred[:, j]) for j, t in enumerate(TARGETS)
           if 0 < weak[:, j].sum() < len(ids)}
    g = lab.loc[ids, "has_gold"].to_numpy() == 1
    gold = None
    if g.sum() >= 8:
        gy = lab.loc[ids, [f"gold_{t}" for t in TARGETS]].to_numpy(float)[g]
        gp = pred[g]
        gs = [roc_auc_score(gy[:, j], gp[:, j]) for j in range(len(TARGETS))
              if 0 < np.nansum(gy[:, j]) < g.sum()]
        gold = (float(np.mean(gs)), int(g.sum()))
    return float(np.mean(list(per.values()))), per, gold


def ranks(a):
    return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])


coat = load("coat", range(5))
dino = load("dinol", [0, 1, 2, 4])

print("== each family, all its own folds ==")
for name, d in (("coat", coat), ("dinol", dino)):
    if d is None:
        continue
    for k in sorted(d.k.unique()):
        s = d[d.k == k]
        m, _, g = aucs(s.index, s[TARGETS].to_numpy())
        print(f"  {name} fold {k}: weak {m:.4f}  gold {g[0]:.4f} (n={g[1]})" if g else
              f"  {name} fold {k}: weak {m:.4f}")

common = [k for k in (2, 4) if dino is not None and k in set(dino.k) and k in set(coat.k)]
if not common:
    sys.exit("no shared cloud folds yet")
c = coat[coat.k.isin(common)]
d = dino[dino.k.isin(common)].reindex(c.index)
assert d[TARGETS].notna().all().all()
ids = c.index
C, Dn = c[TARGETS].to_numpy(), d[TARGETS].to_numpy()
print(f"\n== head-to-head on cloud folds {common}: {len(ids)} identical studies ==")
rows = {}
for name, w in (("coat", 0.0), ("dinol", 1.0), ("blend 70/30", 0.3), ("blend 50/50", 0.5)):
    # Ranks are computed within each fold, as the arm does on the test set.
    p = np.vstack([(1 - w) * ranks(C[(c.k == k).to_numpy()]) + w * ranks(Dn[(c.k == k).to_numpy()])
                   for k in common])
    order = np.concatenate([np.where((c.k == k).to_numpy())[0] for k in common])
    m, per, g = aucs(ids[order], p)
    rows[name] = per
    print(f"  {name:12s} weak {m:.4f}" + (f"   gold {g[0]:.4f} (n={g[1]})" if g else ""))

weak = (lab.loc[ids, [f"weak_{t}" for t in TARGETS]].to_numpy() >= 0.5).astype(int)
print("\n== per target (weak AUC) and within-class rank correlation coat vs dinol ==")
print(f"  {'target':18s} {'pos':>5s} {'coat':>6s} {'dinol':>6s} {'50/50':>6s}  rho+   rho-")
for j, t in enumerate(TARGETS):
    rc, rd = ranks(C)[:, j], ranks(Dn)[:, j]
    pos, neg = weak[:, j] == 1, weak[:, j] == 0
    rp = np.corrcoef(rc[pos], rd[pos])[0, 1] if pos.sum() > 2 else np.nan
    rn = np.corrcoef(rc[neg], rd[neg])[0, 1]
    print(f"  {t:18s} {int(pos.sum()):5d} {rows['coat'][t]:.3f}  {rows['dinol'][t]:.3f}  "
          f"{rows['blend 50/50'][t]:.3f}  {rp:.2f}  {rn:.2f}")
