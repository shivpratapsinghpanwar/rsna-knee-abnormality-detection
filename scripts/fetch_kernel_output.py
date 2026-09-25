"""Download a Kaggle kernel's output files, streaming and resumable.

The stock `kernels_output` buffers each file entirely in RAM before writing it,
and has no resume. Pulling the 10.8 GB corpus archive that way broke at 6.46 GB
with an IncompleteRead and lost all of it.

This streams to disk in chunks and resumes with an HTTP Range request when the
connection drops, re-signing the URL each attempt because the signature expires.

    python scripts/fetch_kernel_output.py rsna-knee-pack --dest D:/kaggle_c/dl
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))

CHUNK = 8 << 20  # 8 MB


def list_output(api, owner: str, slug: str):
    """(file_name, url, size) for each output file of a kernel."""
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiListKernelSessionOutputRequest,
    )

    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        api._set_paging(req, 50, None)
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
    return [(f.file_name, f.url, getattr(f, "file_size", 0)) for f in (resp.files or [])]


def fetch(api, owner: str, slug: str, name: str, dest: Path,
          attempts: int = 40) -> bool:
    """Stream one output file to dest, resuming from whatever is already there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = None
    for attempt in range(attempts):
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have >= total:
            return True

        # Re-sign every attempt: these URLs are time-limited, so a resume an hour
        # into a large transfer cannot reuse the original one.
        url = next((u for n, u, _ in list_output(api, owner, slug) if n == name), None)
        if url is None:
            print(f"  {name}: not found in kernel output")
            return False

        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if have and r.status_code == 200:
                    # Server ignored the Range header; start over rather than
                    # append and silently corrupt the file.
                    have = 0
                    dest.unlink(missing_ok=True)
                elif have and r.status_code != 206:
                    r.raise_for_status()
                if total is None:
                    length = int(r.headers.get("Content-Length", 0))
                    total = have + length if length else None
                t0, start = time.time(), have
                with open(dest, "ab" if have else "wb") as fh:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            fh.write(chunk)
                            have += len(chunk)
                            el = time.time() - t0
                            if el > 20:
                                rate = (have - start) / el / 1e6
                                pct = f"{100 * have / total:.1f}%" if total else "?"
                                eta = ((total - have) / 1e6 / rate / 60
                                       if total and rate else 0)
                                print(f"  {have/1e9:.2f} GB  {pct}  "
                                      f"{rate:.1f} MB/s  eta {eta:.0f}m", flush=True)
                                t0, start = time.time(), have
            if total is None or have >= total:
                print(f"  {name}: done, {have/1e9:.2f} GB")
                return True
            print(f"  short read at {have/1e9:.2f}/{total/1e9:.2f} GB; resuming",
                  flush=True)
        except Exception as exc:
            print(f"  attempt {attempt + 1}: {type(exc).__name__} at "
                  f"{have/1e9:.2f} GB; resuming", flush=True)
            time.sleep(min(30, 3 * (attempt + 1)))
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", help="slug, or owner/slug")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--only", default="", help="only this file name")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    owner, slug = (args.kernel.split("/", 1) if "/" in args.kernel
                   else (api.config_values["username"], args.kernel))

    files = list_output(api, owner, slug)
    print(f"{owner}/{slug}: {len(files)} output file(s)")
    dest_dir = Path(args.dest)
    ok = True
    for name, _, size in files:
        if args.only and name != args.only:
            continue
        print(f"{name}  ({size/1e9:.2f} GB)" if size else name)
        ok &= fetch(api, owner, slug, name, dest_dir / name)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
