"""Turn every training report into graded weak labels and write them to disk.

This is the training signal for the image model: 4407 studies' worth of labels
derived from reports, against the 58 gold rows that exist. Gold rows are kept in
a separate column block so training can weight or hold them out explicitly.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import CSV, WORK  # noqa: E402
from rsna_knee.labels import ID_COL, REPORT_COL, TARGETS  # noqa: E402
from rsna_knee.reports import label_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="",
                    help="CSV of LLM labels to use instead of / alongside regex")
    ap.add_argument("--blend", type=float, default=0.5,
                    help="with --llm: weight on the LLM labels (1.0 = LLM only)")
    args = ap.parse_args()

    tr = pd.read_csv(CSV / "train.csv")
    weak = pd.DataFrame([label_report(r) for r in tr[REPORT_COL]])[TARGETS]

    if args.llm:
        llm_raw = pd.read_csv(args.llm)
        # Align by study id and keep the regex score wherever the LLM has no row,
        # so a partial labelling run degrades instead of blanking those studies.
        llm = tr[[ID_COL]].merge(llm_raw, on=ID_COL, how="left")[TARGETS]
        have = llm.notna().all(axis=1)
        w = args.blend
        blended = weak.copy()
        blended[have] = (1 - w) * weak[have].to_numpy() + w * llm[have].to_numpy()
        print(f"LLM labels for {int(have.sum())}/{len(tr)} studies, "
              f"blend weight {w} (regex used for the remaining "
              f"{int((~have).sum())})")
        weak = blended

    weak.columns = [f"weak_{c}" for c in TARGETS]
    out = pd.concat([tr[[ID_COL]], weak], axis=1)
    out["has_gold"] = tr[TARGETS].notna().all(axis=1).astype(int)
    for t in TARGETS:
        out[f"gold_{t}"] = tr[t]

    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "weak_labels.csv"
    out.to_csv(path, index=False)

    print(f"wrote {path}  ({len(out)} studies, {out.has_gold.sum()} with gold)")
    print("\nweak label mean (graded, 0-1):")
    print(weak.mean().round(3).to_string())
    print(f"\nstudies with no weak signal at all: "
          f"{int((weak.sum(axis=1) == 0).sum())} "
          f"({100 * (weak.sum(axis=1) == 0).mean():.1f}%)")


if __name__ == "__main__":
    main()
