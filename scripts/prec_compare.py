"""Does precision change fold-4 OOF predictions? Baseline = fp32 weights + fp16 autocast."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

O = Path(r"D:\kaggle_c\work\oof2")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
lab = pd.read_csv(r"D:\kaggle_c\work\weak_labels.csv").set_index("StudyInstanceUID")


def load(name):
    p = O / f"{name}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).set_index("StudyInstanceUID")
    return d[d[TARGETS].notna().all(axis=1)][TARGETS]


def score(ids, pred):
    weak = (lab.loc[ids, [f"weak_{t}" for t in TARGETS]].to_numpy() >= 0.5).astype(int)
    return np.mean([roc_auc_score(weak[:, j], pred[:, j]) for j in range(len(TARGETS))])


def ranks(a):
    return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])


runs = {
    "dinol fp32w+amp (Kaggle-equiv OOF)": "dinol_f4",
    "dinol fp16w+amp (what Kaggle ran)": "dinol_f4_h16w",
    "dinol full fp32": "dinol_f4_fp32",
    "coat  fp32w+amp (what Kaggle ran)": "coat_f4",
    "coat  full fp32": "coat_f4_fp32",
}
data = {k: load(v) for k, v in runs.items()}
ids = data["coat  fp32w+amp (what Kaggle ran)"].index
for k, d in data.items():
    if d is None:
        print(f"  {k:38s} (missing)")
        continue
    d = d.reindex(ids).to_numpy()
    base = data[("dinol fp32w+amp (Kaggle-equiv OOF)" if k.startswith("dinol")
                 else "coat  fp32w+amp (what Kaggle ran)")].reindex(ids).to_numpy()
    rho = np.mean([spearmanr(d[:, j], base[:, j])[0] for j in range(len(TARGETS))])
    print(f"  {k:38s} weak AUC {score(ids, d):.5f}   max|dp| {np.abs(d - base).max():.2e}"
          f"   rank rho vs baseline {rho:.6f}")

print("\n50/50 rank blend, fold 4:")
for label, dk, ck in (("as shipped (dino fp16w+amp, coat amp)", "dinol fp16w+amp (what Kaggle ran)",
                       "coat  fp32w+amp (what Kaggle ran)"),
                      ("all full fp32", "dinol full fp32", "coat  full fp32")):
    if data[dk] is None or data[ck] is None:
        continue
    p = 0.5 * ranks(data[dk].reindex(ids).to_numpy()) + 0.5 * ranks(data[ck].reindex(ids).to_numpy())
    print(f"  {label:40s} weak AUC {score(ids, p):.5f}")
