"""Choosing which series of a study to keep, and in which order.

Studies carry 3-14 series averaging 5.5, and the extras are largely repeats of a
plane rather than new diagnostic content. Keeping a fixed set of slots makes the
study tensor a fixed shape and cuts the corpus to roughly 4 series per study.

The organisers already supply Anatomical_Plane / Fluid_Sensitive / Fat_Suppression
per series, so no DICOM header parsing is needed to do this.
"""

from __future__ import annotations

import pandas as pd

from .config import SERIES_SLOTS

SLOT_NAMES = [f"{plane.lower()}_{'fs' if fluid else 'pd'}" for plane, fluid in SERIES_SLOTS]


def select_series(series_df: pd.DataFrame, slice_counts: dict[str, int] | None = None
                  ) -> pd.DataFrame:
    """Pick at most one series per slot per study.

    Returns the selected rows with two extra columns: ``slot`` (0-based index into
    SERIES_SLOTS) and ``slot_name``. When several series fill the same slot the
    one with the most slices wins, since a thicker stack covers more anatomy;
    without slice counts the first listed is taken.
    """
    df = series_df.copy()
    if slice_counts is not None:
        df["n_slices"] = df.SeriesInstanceUID.map(slice_counts).fillna(0).astype(int)
    else:
        df["n_slices"] = 0

    out = []
    for study, g in df.groupby("StudyInstanceUID", sort=False):
        used: set[str] = set()
        for slot, (plane, fluid) in enumerate(SERIES_SLOTS):
            cand = g[
                (g.Anatomical_Plane == plane)
                & (g.Fluid_Sensitive == fluid)
                & (~g.SeriesInstanceUID.isin(used))
            ]
            if cand.empty:
                continue
            row = cand.sort_values("n_slices", ascending=False).iloc[0]
            used.add(row.SeriesInstanceUID)
            r = row.to_dict()
            r["slot"] = slot
            r["slot_name"] = SLOT_NAMES[slot]
            out.append(r)
        if not used:
            # Degenerate study: no series matched any slot. Keep the largest one so
            # the study is still represented rather than dropped outright.
            row = g.sort_values("n_slices", ascending=False).iloc[0]
            r = row.to_dict()
            r["slot"] = 0
            r["slot_name"] = SLOT_NAMES[0]
            out.append(r)
    return pd.DataFrame(out).reset_index(drop=True)
