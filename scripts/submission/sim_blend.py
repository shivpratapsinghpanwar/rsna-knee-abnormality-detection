"""Run the built notebook's blend cell on synthetic inputs: with R7, without R7, and R7 at low coverage."""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

nb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
blend = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
             and "".join(c["source"]).startswith("# Stage 3: per-finding rank blend"))
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
root = Path(sys.argv[2])
rng = np.random.default_rng(0)
n = 40
ids = [f"1.2.{i}" for i in range(n)]


def frame(v):
    d = pd.DataFrame(v, columns=L)
    d.insert(0, "StudyInstanceUID", ids)
    return d


def run(case, r7_ok, r7_cover):
    shutil.rmtree(root, ignore_errors=True)
    data, work, p6, p7 = root / "data", root / "work", root / "p6", root / "p7"
    for d in (data, work, p6 / "worker0", p7 / "worker0"):
        d.mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(data / "test.csv", index=False)
    pd.DataFrame(columns=["StudyInstanceUID"] + L).to_csv(data / "sample_submission.csv", index=False)
    base = rng.random((n, 12))
    raw = base + 0.3 * rng.random((n, 12))
    for nm, v in (("raptor_raw", raw), ("raptor_sub", raw), ("raptor224_sub", base + 0.3 * rng.random((n, 12))),
                  ("ours_sub", base + 0.3 * rng.random((n, 12))), ("r7_sub", base + 0.3 * rng.random((n, 12)))):
        frame(v).to_csv(work / f"{nm}.csv", index=False)
    frame(rng.random((n, 12))).to_csv(p6 / "worker0" / "predictions.csv", index=False)
    frame(rng.random((n, 12))).iloc[: int(r7_cover * n)].to_csv(p7 / "worker0" / "predictions.csv", index=False)
    g = dict(DATA=data, WORK=work, pred=p6, R7_OK=r7_ok, R7_PRED=p7 if r7_ok else None, R224_OK=True,
             RAPTOR_SECONDS=1.0, NB_T0=time.time(), display=lambda *_: None, time=time, json=json)
    exec(compile(blend, "blend", "exec"), g)
    rec = json.loads((work / "blend_receipt.json").read_text())
    sub = pd.read_csv(work / "submission.csv")
    print(f"{case:22s} r7_used={rec['r7_used']} r7_failed={rec['r7_failed']} "
          f"rows={len(sub)} range=[{sub[L].values.min():.3f},{sub[L].values.max():.3f}]")
    return sub[L].values


a = run("with R7", True, 1.0)
b = run("no R7", False, 1.0)
c = run("R7 coverage 50%", True, 0.5)
print("with vs without R7 differ:", not np.allclose(a, b), "| low-coverage == no-R7 path:",
      "yes" if c.shape == b.shape else "?")
