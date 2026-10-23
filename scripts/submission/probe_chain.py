"""Push each probe variant (waiting for a free GPU session, never cancelling one), then audit + submit it.

usage: probe_chain.py <slug>=<message>[=arm] ...
Each variant's push output goes to work/push_<slug>.log; its audit/submit/poll to work/sub_<slug>.log.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

W = Path(r"D:\kaggle_c\work")
PY = r"D:\kaggle_c\.venv\Scripts\python.exe"
env = {"PYTHONIOENCODING": "utf-8"}
import os
env = {**os.environ, **env}

subs = []
for spec in sys.argv[1:]:
    slug, msg, *flag = spec.split("=", 2)
    stage = Path(r"D:\kaggle_c\fork958") / f"stage_{slug}"
    for attempt in range(120):   # up to ~4 h of waiting for a GPU session
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(stage)], capture_output=True, text=True, env=env)
        out = (r.stdout + r.stderr).strip()
        (W / f"push_{slug}.log").write_text(out, encoding="utf-8")
        m = re.search(r"Kernel version (\d+) successfully pushed", out)
        if m:
            (W / f"{slug}.ver").write_text(m.group(1))
            print(f"[{time.strftime('%H:%M:%S')}] {slug} pushed v{m.group(1)}", flush=True)
            break
        if re.search(r"session count|Maximum batch", out, re.I):
            print(f"[{time.strftime('%H:%M:%S')}] {slug}: GPU sessions full, waiting", flush=True)
            time.sleep(120)
            continue
        if re.search(r"Connection|timed out|Max retries|Temporary failure|503|502", out):
            print(f"[{time.strftime('%H:%M:%S')}] {slug}: network error, retrying in 60 s", flush=True)
            time.sleep(60)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {slug}: push FAILED: {out[-300:]}", flush=True)
        break
    else:
        print(f"{slug}: gave up waiting for a GPU session", flush=True)
        continue
    if not (W / f"{slug}.ver").exists():
        continue
    log = open(W / f"sub_{slug}.log", "w", encoding="utf-8")
    subs.append(subprocess.Popen([PY, "-u", str(W / "tools" / "sub_r7.py"), slug, msg] + flag,
                                 stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(W)))
    time.sleep(30)

for p in subs:
    p.wait()
print("PROBE_CHAIN_DONE", flush=True)
