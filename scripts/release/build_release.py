"""Build the public labels + OOF release: our own derived labels only, no gold labels, no report text.

Writes D:\\kaggle_c\\release_labels with
  labels/regex_labels.csv, labels/qwen25_7b_labels.csv, labels/blend_regex_qwen.csv
  oof/<family>_f<k>.csv, oof/fold_cloud.csv
and prints each label set's macro AUC on the 58 gold studies (for the card; the gold labels stay out).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(r"D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection")
sys.path.insert(0, str(REPO / "src"))
from rsna_knee.config import CSV  # noqa: E402
from rsna_knee.labels import ID_COL, REPORT_COL, TARGETS  # noqa: E402
from rsna_knee.reports import label_report  # noqa: E402

OUT = Path(r"D:\kaggle_c\release_labels")
OOF = Path(r"D:\kaggle_c\work\oof2")
shutil.rmtree(OUT, ignore_errors=True)
(OUT / "labels").mkdir(parents=True)
(OUT / "oof").mkdir()

tr = pd.read_csv(CSV / "train.csv")
ids = tr[[ID_COL]]

regex = pd.concat([ids, pd.DataFrame([label_report(r) for r in tr[REPORT_COL]])[TARGETS]], axis=1)
qwen = ids.merge(pd.read_csv(REPO / "labels" / "llm_labels.csv"), on=ID_COL, how="left")
have = qwen[TARGETS].notna().all(axis=1)
blend = regex.copy()
blend.loc[have, TARGETS] = 0.5 * regex.loc[have, TARGETS].to_numpy() + 0.5 * qwen.loc[have, TARGETS].to_numpy()

# The shipped blend must equal what training used for our own labels.
ours = pd.read_csv(r"D:\kaggle_c\work\weak_labels_ours_backup.csv").set_index(ID_COL)
diff = np.abs(ours.loc[blend[ID_COL], [f"weak_{t}" for t in TARGETS]].to_numpy() - blend[TARGETS].to_numpy()).max()
print(f"blend vs training-time blend: max abs diff {diff:.2e}")

gold_mask = tr[TARGETS].notna().all(axis=1).to_numpy()
gold = tr.loc[gold_mask, TARGETS].to_numpy()


def gold_auc(df):
    p = df.loc[gold_mask, TARGETS].to_numpy(float)
    return np.mean([roc_auc_score(gold[:, j], np.nan_to_num(p[:, j])) for j in range(len(TARGETS))
                    if 0 < gold[:, j].sum() < len(gold)])


for name, df in (("regex_labels", regex), ("qwen25_7b_labels", qwen), ("blend_regex_qwen", blend)):
    assert list(df.columns) == [ID_COL] + TARGETS, df.columns
    df.to_csv(OUT / "labels" / f"{name}.csv", index=False, float_format="%.4f")
    print(f"{name:18s} rows {len(df)}  non-null {int(df[TARGETS].notna().all(axis=1).sum())}  "
          f"gold-58 macro AUC {gold_auc(df):.3f}")

for p in sorted(OOF.glob("*_f[0-4].csv")) + [OOF / "fold_cloud.csv"]:
    d = pd.read_csv(p)
    assert not any(c.startswith(("gold", "weak", "Report")) for c in d.columns), (p, d.columns)
    shutil.copy(p, OUT / "oof" / p.name)

# Nothing in the release may hold report text: every non-id cell must be numeric.
for p in OUT.rglob("*.csv"):
    d = pd.read_csv(p)
    for c in d.columns:
        if c != ID_COL:
            assert pd.api.types.is_numeric_dtype(d[c]), (p.name, c)
print("release ok:", sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*.csv")))
