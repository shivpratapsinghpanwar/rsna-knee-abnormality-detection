"""Convert raw DICOM series into the PNG mosaic corpus used for training.

Runs unchanged on Kaggle, where the whole competition is mounted, and on this
workstation, where only a streamed dev subset exists -- it simply processes
whatever series are present on disk.

    python scripts/preprocess_series.py --split train --workers 8

Resumable: a series whose mosaic already exists is skipped, so an interrupted run
can be restarted and a Kaggle session that hits its 12 h limit can continue in a
second session.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import (  # noqa: E402
    CROP_MM, CSV, MOSAIC_EXT, PREP, RAW, SERIES_SLOTS, SIZE, SLICES,
)  # noqa: E402
from rsna_knee.dicom_io import read_series  # noqa: E402
from rsna_knee.preprocess import (  # noqa: E402
    crop_mm, normalise_laterality, save_mosaic, volume_to_mosaic,
)
from rsna_knee.series import select_series  # noqa: E402


def series_dir(split: str, study: str, series: str) -> Path:
    return RAW / f"{split}_series" / study / series


def count_slices(split: str, df: pd.DataFrame) -> dict[str, int]:
    """Slice count per series, from the filesystem. Absent series count 0."""
    counts: dict[str, int] = {}
    for study, series in zip(df.StudyInstanceUID, df.SeriesInstanceUID):
        d = series_dir(split, study, series)
        counts[series] = sum(1 for _ in d.glob("*.dcm")) if d.is_dir() else 0
    return counts


def _one(args: tuple[str, str, str, str, int]) -> tuple[str, str]:
    """Worker: read one series and write its mosaic. Returns (series, status)."""
    split, study, series, out_path, slot = args
    out = Path(out_path)
    if out.exists():
        return series, "skip"
    d = series_dir(split, study, series)
    paths = sorted(d.glob("*.dcm"))
    if not paths:
        return series, "missing"
    try:
        vol, spacing, side = read_series(paths)
        if vol.size == 0:
            return series, "empty"
        # Mirror right knees so medial is always the same side of the image.
        plane = SERIES_SLOTS[slot][0] if slot < len(SERIES_SLOTS) else "Sagittal"
        vol = normalise_laterality(vol, plane, side)
        # Crop to a fixed physical field before tiling, so every study shows the
        # model the same anatomy regardless of the scanner's field of view.
        vol = crop_mm(vol, spacing)
        save_mosaic(volume_to_mosaic(vol), out)
        return series, f"ok-{side.lower() or 'unknown'}"
    except Exception as exc:  # a single bad series must not stop the corpus
        return series, f"error:{type(exc).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="process only N series")
    # A Kaggle session is killed at the 12 h mark and a killed run may save no
    # output at all, losing hours of work. Stopping under our own budget keeps
    # whatever is finished; the next run resumes from it.
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop cleanly after this long (0 = no limit)")
    args = ap.parse_args()

    meta = pd.read_csv(CSV / f"{args.split}_series.csv")
    print(f"{args.split}: {len(meta)} series listed, "
          f"{meta.StudyInstanceUID.nunique()} studies")

    counts = count_slices(args.split, meta)
    present = sum(1 for v in counts.values() if v)
    print(f"present on disk: {present} series")
    if present == 0:
        print("nothing to do -- no raw DICOM found under", RAW)
        return

    sel = select_series(meta[meta.SeriesInstanceUID.map(counts).fillna(0) > 0], counts)
    if args.limit:
        sel = sel.head(args.limit)
    out_root = PREP / args.split
    jobs = [
        (args.split, r.StudyInstanceUID, r.SeriesInstanceUID,
         str(out_root / r.StudyInstanceUID
             / f"{r.slot}_{r.SeriesInstanceUID}{MOSAIC_EXT}"), int(r.slot))
        for r in sel.itertuples()
    ]
    print(f"selected {len(jobs)} series -> {out_root} "
          f"({SIZE}x{SIZE} tiles, {MOSAIC_EXT})")

    stats: dict[str, int] = {}
    t0 = time.time()
    budget = args.max_minutes * 60 if args.max_minutes else float("inf")
    stopped_early = False
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            _, status = f.result()
            key = status.split(":")[0]
            stats[key] = stats.get(key, 0) + 1
            el = time.time() - t0
            if i % 100 == 0 or i == len(futs):
                rate = i / el if el else 0
                eta = (len(futs) - i) / rate / 60 if rate else 0
                print(f"  {i}/{len(futs)}  {rate:.1f}/s  eta {eta:.0f}m  {stats}",
                      flush=True)
            if el > budget:
                stopped_early = True
                print(f"\nreached the {args.max_minutes:g} min budget after "
                      f"{i}/{len(futs)} series; cancelling the rest so the "
                      f"finished work is saved. Re-run to resume.", flush=True)
                for pending in futs:
                    pending.cancel()
                break

    print("\ndone:", stats, "(stopped early)" if stopped_early else "")
    size = sum(p.stat().st_size for p in out_root.rglob(f"*{MOSAIC_EXT}"))
    print(f"corpus: {size / 1e9:.2f} GB in {out_root}")


if __name__ == "__main__":
    main()
