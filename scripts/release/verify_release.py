"""Integrity check of the staged release after the machine logged corrected RAM errors.

Writes SHA256SUMS for every file in D:\\kaggle_c\\hf_upload, then loads each checkpoint on CPU and checks
that every floating tensor is finite. A corrupted byte in a .pt usually fails the load or yields NaN/inf.
"""
import hashlib
import math
from pathlib import Path

import torch

ROOT = Path(r"D:\kaggle_c\hf_upload")
lines, bad = [], []
for p in sorted(q for q in ROOT.rglob("*") if q.is_file() and q.name != "SHA256SUMS"):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    lines.append(f"{h.hexdigest()}  {p.relative_to(ROOT).as_posix()}")
    if p.suffix == ".pt":
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            n = nonfinite = 0
            for v in ck["model"].values():
                if v.is_floating_point():
                    n += v.numel()
                    nonfinite += int((~torch.isfinite(v)).sum())
            auc = ck.get("weak_auc")
            status = "ok" if nonfinite == 0 else f"{nonfinite} NON-FINITE"
            print(f"{status:>14}  {p.name}  fold {ck.get('fold')}  {n / 1e6:.0f}M params  "
                  f"recorded weak AUC {auc if auc is None else round(float(auc), 4)}", flush=True)
            if nonfinite or (auc is not None and not math.isfinite(float(auc))):
                bad.append(p.name)
        except Exception as e:
            print(f"    LOAD FAILED  {p.name}: {type(e).__name__}: {e}", flush=True)
            bad.append(p.name)
(ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nwrote SHA256SUMS for {len(lines)} files; problems: {bad or 'none'}")
