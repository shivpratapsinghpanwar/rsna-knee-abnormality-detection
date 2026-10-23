"""Upload DINOv2 weights (fp16) as rsna-knee-train-dinol, then build, push, audit, submit.

    python fam_chain.py <slug> <W> <family spec> <n_ckpts> <message> [--skip-upload]
"""
import subprocess, sys, time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

SP = Path(__file__).parent
PY = r"D:\kaggle_c\.venv\Scripts\python.exe"
REPO = Path(r"D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection")
slug, W, spec, n, msg = sys.argv[1:6]
a = KaggleApi(); a.authenticate()
log = lambda *x: print(time.strftime("[%H:%M]"), *x, flush=True)

if "--skip-upload" not in sys.argv:
    r = subprocess.run([PY, "scripts/push_weights.py", "--src", r"D:\kaggle_c\work\dinol_gcp",
                        "--slug", "rsna-knee-train-dinol", "--half", "--message", f"DINOv2-L for {slug}"],
                       cwd=REPO, capture_output=True, text=True)
    log("push_weights rc", r.returncode, r.stdout[-400:], r.stderr[-300:])
    if r.returncode:
        sys.exit("weights upload failed")
    time.sleep(240)
    for _ in range(60):
        try:
            st = str(a.dataset_status("shivpratap0007/rsna-knee-train-dinol"))
        except Exception as e:
            st = repr(e)
        if "ready" in st.lower():
            break
        time.sleep(60)
    log("dataset status:", st)

subprocess.run([PY, str(SP / "make_arm_families.py"), slug, W, spec], check=True)
d = Path(rf"D:\kaggle_c\fork958\stage_{slug}")
while True:
    try:
        r = a.kernels_push(str(d)); err = getattr(r, "error", None)
    except Exception as e:
        err = e
    if not err:
        break
    log("push:", str(err)[:120], "-- retry in 5 min")
    time.sleep(300)
v = getattr(r, "version_number", None) or getattr(r, "versionNumber", None)
Path(rf"D:\kaggle_c\work\{slug}.ver").write_text(str(v))
log("pushed", slug, "v", v)
with open(rf"D:\kaggle_c\work\sub_{slug}.log", "w") as lf:
    subprocess.run([PY, str(SP / "sub_arm.py"), slug, n, msg], stdout=lf, stderr=subprocess.STDOUT)
log("chain finished; see sub_" + slug + ".log")
