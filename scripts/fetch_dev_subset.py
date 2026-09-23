"""Build a small local dev corpus by streaming: download, mosaic, delete raw.

The full competition is ~300-600 GB of DICOM against ~110 GB of free disk, so a
full local copy is impossible; real training runs on Kaggle. What this gives us
is enough genuine studies to develop and test the model code end to end without
waiting on Kaggle for every iteration.

Raw DICOM never accumulates: each series is downloaded to a scratch directory,
converted to its PNG mosaic, and deleted before the next one starts. Peak extra
disk is one series, a few tens of MB.

    python scripts/fetch_dev_subset.py --studies 60 --workers 16

Both the file listing and the finished mosaics are cached, so re-running resumes
rather than starting over.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import COMPETITION, CSV, PREP, WORK  # noqa: E402
from rsna_knee.dicom_io import read_series  # noqa: E402
from rsna_knee.preprocess import save_mosaic, volume_to_mosaic  # noqa: E402
from rsna_knee.series import select_series  # noqa: E402

os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))

LISTING = WORK / "train_listing.json"
_tl = threading.local()


def _client():
    """One Kaggle client and HTTP session per thread.

    Building a client per request triggers ConnectionAbortedError once more than a
    few threads are active, so they are cached per thread and reused.
    """
    if not hasattr(_tl, "kaggle"):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        _tl.api = api
        _tl.kaggle = api.build_kaggle_client().__enter__()
        _tl.session = requests.Session()
    return _tl.kaggle, _tl.session


def _reset_client() -> None:
    for attr in ("kaggle", "session", "api"):
        if hasattr(_tl, attr):
            delattr(_tl, attr)


def download(name: str, dest: Path, attempts: int = 4) -> bool:
    """Fetch one competition file. Two steps: signed URL, then a plain GET."""
    from kagglesdk.competitions.types.competition_api_service import (
        ApiDownloadDataFileRequest,
    )

    for attempt in range(attempts):
        try:
            kaggle, session = _client()
            req = ApiDownloadDataFileRequest()
            req.competition_name = COMPETITION
            req.file_name = name
            url = kaggle.competitions.competition_api_client.download_data_file(
                req
            ).request.url
            resp = session.get(url, timeout=180)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return True
        except Exception:
            _reset_client()
            if attempt == attempts - 1:
                return False
            time.sleep(1.5 * (attempt + 1))
    return False


def build_listing(want_studies: int) -> dict[str, list[str]]:
    """Map series UID -> DICOM file names, for the first N train studies.

    Kaggle's listing API pages 200 at a time and starts returning 503 after a few
    thousand calls, so this stops as soon as enough studies are covered and caches
    the result. The full-corpus equivalent is generated inside a Kaggle notebook
    with ordinary filesystem calls instead.
    """
    if LISTING.exists():
        cached = json.loads(LISTING.read_text())
        if cached.get("_studies", 0) >= want_studies:
            return {k: v for k, v in cached.items() if not k.startswith("_")}

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    by_series: dict[str, list[str]] = {}
    studies: set[str] = set()
    token = None
    pages = 0
    while len(studies) <= want_studies:
        try:
            resp = api.competition_list_files(
                COMPETITION, page_token=token, page_size=200
            )
        except Exception as exc:
            print(f"  listing stopped at page {pages}: {type(exc).__name__}")
            break
        if not resp.files:
            break
        for f in resp.files:
            parts = f.name.split("/")
            if len(parts) != 4 or parts[0] != "train_series":
                continue
            studies.add(parts[1])
            by_series.setdefault(parts[2], []).append(f.name)
        token = resp.next_page_token
        pages += 1
        if pages % 20 == 0:
            print(f"  listed {pages} pages, {len(studies)} studies", flush=True)
        if not token:
            break

    WORK.mkdir(parents=True, exist_ok=True)
    LISTING.write_text(json.dumps({**by_series, "_studies": len(studies)}))
    print(f"  listing cached: {len(studies)} studies, {len(by_series)} series")
    return by_series


def fetch_series(study: str, series: str, slot: int, names: list[str],
                 scratch: Path, workers: int) -> str:
    out = PREP / "train" / study / f"{slot}_{series}.png"
    if out.exists():
        return "skip"
    work = scratch / series
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(workers) as ex:
            ok = list(ex.map(lambda n: download(n, work / Path(n).name), names))
        if not any(ok):
            return "download-failed"
        vol = read_series(sorted(work.glob("*.dcm")))
        if vol.size == 0:
            return "empty"
        save_mosaic(volume_to_mosaic(vol), out)
        return "ok"
    finally:
        # Raw DICOM is discarded immediately -- this is the whole point of the
        # streaming approach; keeping it would exhaust the disk in a few hundred
        # studies.
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    print(f"listing train files for ~{args.studies} studies ...")
    by_series = build_listing(args.studies)
    if not by_series:
        print("no listing available; aborting")
        return

    meta = pd.read_csv(CSV / "train_series.csv")
    have = meta[meta.SeriesInstanceUID.isin(by_series)]
    counts = {s: len(v) for s, v in by_series.items()}

    # Only studies whose series are all present in the listing are usable; a
    # partly-listed study would silently lose a plane.
    sizes = meta.groupby("StudyInstanceUID").size()
    complete = [
        st for st, g in have.groupby("StudyInstanceUID") if len(g) == sizes[st]
    ][: args.studies]
    sel = select_series(have[have.StudyInstanceUID.isin(complete)], counts)
    print(f"{len(complete)} complete studies -> {len(sel)} series to fetch")

    scratch = WORK / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    t0 = time.time()
    for i, r in enumerate(sel.itertuples(), 1):
        status = fetch_series(r.StudyInstanceUID, r.SeriesInstanceUID, r.slot,
                              by_series[r.SeriesInstanceUID], scratch, args.workers)
        stats[status] = stats.get(status, 0) + 1
        el = time.time() - t0
        eta = (len(sel) - i) / (i / el) / 60 if el else 0
        print(f"  [{i}/{len(sel)}] {status:16s} eta {eta:5.0f}m  {stats}", flush=True)

    shutil.rmtree(scratch, ignore_errors=True)
    root = PREP / "train"
    pngs = list(root.rglob("*.png")) if root.exists() else []
    size = sum(p.stat().st_size for p in pngs)
    print(f"\ndone: {stats}")
    print(f"dev corpus: {size / 1e6:.0f} MB, {len(pngs)} mosaics")


if __name__ == "__main__":
    main()
