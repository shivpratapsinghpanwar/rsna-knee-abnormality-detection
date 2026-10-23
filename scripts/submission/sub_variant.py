"""Build, push and submit one arm variant whose weights are already uploaded.

    python sub_variant.py <slug> <W> <folds> <n_ckpts> <message> [--tta X]
"""
import subprocess, sys, time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

SP = Path(__file__).parent
PY = r"D:\kaggle_c\.venv\Scripts\python.exe"
slug, W, folds, n, msg = sys.argv[1:6]
extra = sys.argv[6:]
subprocess.run([PY, str(SP / "make_arm_variant.py"), slug, W, folds] + extra, check=True)
a = KaggleApi(); a.authenticate()
d = Path(rf"D:\kaggle_c\fork958\stage_{slug}")
while True:
    try:
        r = a.kernels_push(str(d)); err = getattr(r, "error", None)
    except Exception as e:
        err = e
    if not err:
        break
    print("push:", str(err)[:90], "-- retry in 5 min", flush=True)
    time.sleep(300)
v = getattr(r, "version_number", None) or getattr(r, "versionNumber", None)
Path(rf"D:\kaggle_c\work\{slug}.ver").write_text(str(v))
print("pushed", slug, "v", v, flush=True)
with open(rf"D:\kaggle_c\work\sub_{slug}.log", "w") as lf:
    subprocess.run([PY, str(SP / "sub_arm.py"), slug, n, msg], stdout=lf, stderr=subprocess.STDOUT)
print("done; see sub_" + slug + ".log", flush=True)
