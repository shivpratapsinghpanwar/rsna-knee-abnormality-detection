"""Wait for a blend kernel, audit its log (with retries), submit, poll score.

usage: sub_robust.py <slug> <expected_ckpts> <description>

The previous chain died on a single ReadTimeout while downloading kernel output,
so the submission was never entered. Here every network step retries, hangs are
turned into exceptions by a socket timeout, and only *.log files are fetched
when the client supports it.
"""
import glob, inspect, json, re, shutil, socket, sys, time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

socket.setdefaulttimeout(300)
slug, MSG = sys.argv[1], sys.argv[2]
REF, COMP = f"shivpratap0007/{slug}", "rsna-knee-abnormality-detection"
VER = int(Path(rf"D:\kaggle_c\work\{slug}.ver").read_text().strip())
OUT = rf"D:\kaggle_c\work\out_{slug}"
a = KaggleApi(); a.authenticate()

def retry(fn, tries=8, wait=60, what=""):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"  {what} attempt {i+1}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            time.sleep(wait)
    raise SystemExit(f"{what} failed after {tries} attempts")

last = ""
deadline = time.time() + 6 * 3600
while time.time() < deadline:
    st = retry(lambda: str(getattr(a.kernels_status(REF), "status", "?")).rsplit(".", 1)[-1],
               what="status")
    if st != last:
        print(f"[{time.strftime('%H:%M:%S')}] kernel {st}", flush=True); last = st
    if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
        break
    time.sleep(90)
if last != "COMPLETE":
    raise SystemExit(f"kernel ended {last}")

kw = {}
if "file_pattern" in inspect.signature(a.kernels_output).parameters:
    kw["file_pattern"] = r".*\.log$"
def fetch():
    shutil.rmtree(OUT, ignore_errors=True)
    a.kernels_output(REF, path=OUT, force=True, quiet=True, **kw)
    if not glob.glob(OUT + r"\*.log"):
        raise RuntimeError("no log in output")
retry(fetch, what="fetch log")

txt = ""
for f in glob.glob(OUT + r"\*.log"):
    raw = Path(f).read_text(encoding="utf-8", errors="replace")
    try:
        txt += "\n".join((e.get("data") or e.get("text") or "") for e in json.loads(raw) if isinstance(e, dict))
    except Exception:
        txt += raw
for m in re.findall(r"(?:R7_\w+|NOTEBOOK_COMPLETE|BLEND_FALLBACK|OURS_STAGE_COMPLETE).*", txt):
    print("  " + m[:220], flush=True)
r7_ok = bool(re.search(r"R7_STAGE ok=True", txt)) and "R7_STAGE_COMPLETE" in txt
r7_used = bool(re.search(r'"r7_used": true', txt))
bad = re.findall(r"R7_(?:FAILED|SKIPPED|DROPPED|UNUSABLE)|BLEND_FALLBACK", txt)
print(f"audit: r7_stage_ok={r7_ok} r7_used={r7_used} bad={bad}", flush=True)
if not (r7_ok and r7_used) or bad or "NOTEBOOK_COMPLETE" not in txt:
    raise SystemExit("audit failed -- not submitting")

print(f"submitting {REF} v{VER}: {MSG}", flush=True)
retry(lambda: a.competition_submit_code("submission.csv", MSG, COMP, kernel=REF,
                                        kernel_version=VER, quiet=True), what="submit")
while time.time() < deadline + 6 * 3600:
    time.sleep(150)
    try:
        subs = a.competition_submissions(COMP)
    except Exception:
        continue
    me = next((x for x in subs if str(getattr(x, "description", "")) == MSG), None)
    if not me:
        continue
    st = str(getattr(me, "status", "")).rsplit(".", 1)[-1].upper()
    sc = str(getattr(me, "public_score", "") or "")
    print(f"  [{time.strftime('%H:%M:%S')}] {st} {sc}", flush=True)
    if st in ("COMPLETE", "ERROR"):
        print(f"\nPUBLIC SCORE: {sc or 'none'}  ({MSG})", flush=True)
        break
