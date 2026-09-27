"""Check the geometric laterality recovery against the DICOM tag.

The `Laterality` tag is Type 2C and absent for whole vendors, so the side has to
be recovered from geometry for much of the corpus. This measures how good that
recovery is on the studies where the tag *is* present, and how much of the
accuracy comes from correcting the image corner to the image centre.

Cheap enough to run before committing to a full rebuild:

    python scripts/audit_laterality.py --limit 400
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import RAW  # noqa: E402


def x_at(ds, centre: bool) -> float:
    """Patient-space x of the image centre, or of its corner when centre=False."""
    ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    if not centre:
        return float(ipp[0])
    iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    dr, dc = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])
    c = (ipp
         + iop[:3] * dc * (int(ds.Columns) - 1) / 2.0
         + iop[3:] * dr * (int(ds.Rows) - 1) / 2.0)
    return float(c[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--dead-mm", type=float, default=20.0)
    args = ap.parse_args()

    root = RAW / f"{args.split}_series"
    if not root.is_dir():
        raise SystemExit(f"no DICOM under {root}")
    studies = sorted(p for p in root.iterdir() if p.is_dir())
    random.seed(0)
    random.shuffle(studies)

    tag = notag = dead = 0
    ok_centre = ok_corner = n = 0
    for st in studies:
        if tag + notag >= args.limit:
            break
        ser = next((s for s in st.iterdir() if s.is_dir()), None)
        if ser is None:
            continue
        f = next(iter(sorted(ser.glob("*.dcm"))), None)
        if f is None:
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        t = str(getattr(ds, "Laterality", None)
                or getattr(ds, "ImageLaterality", None) or "").strip().upper()[:1]
        if t not in ("L", "R"):
            notag += 1
            continue
        tag += 1
        try:
            xc, xk = x_at(ds, True), x_at(ds, False)
        except Exception:
            continue
        n += 1
        if abs(xc) < args.dead_mm:
            dead += 1
            continue
        ok_centre += ("L" if xc > 0 else "R") == t
        ok_corner += ("L" if xk > 0 else "R") == t

    usable = max(n - dead, 1)
    print(f"studies sampled        : {tag + notag}")
    print(f"  tag present          : {tag}")
    print(f"  tag absent           : {notag} "
          f"({notag / max(tag + notag, 1) * 100:.0f}% need geometry)")
    print(f"  within {args.dead_mm:g} mm midline : {dead} (left unresolved)")
    print()
    print(f"agreement with the tag, where the tag exists:")
    print(f"  centre-corrected     : {ok_centre}/{usable} "
          f"= {ok_centre / usable * 100:.1f}%")
    print(f"  corner (naive)       : {ok_corner}/{usable} "
          f"= {ok_corner / usable * 100:.1f}%")


if __name__ == "__main__":
    main()
