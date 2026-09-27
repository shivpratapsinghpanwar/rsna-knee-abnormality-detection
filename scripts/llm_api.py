"""Label the reports through a hosted API instead of a local model.

The labeller is the bottleneck -- blended labels score 0.846 against gold while
the leaderboard needs ~0.95, and an image model cannot outrun its teacher by
much. A 7B running in 4-bit on a 10 GB card is a blunt instrument for that, and
the constraint is VRAM, not money: the reports exist only at training time, so
labelling has no inference-time cost at all and any model is fair game.

Several providers have a genuinely free tier that comfortably covers this job
(4407 reports, ~1100 characters in, ~40 tokens out). Groq serves open models
through an OpenAI-compatible endpoint; Google AI Studio serves Gemini.

The key is read from the environment or from a local .env file, never passed on
the command line, so it stays out of shell history and out of the repo:

    D:\\kaggle_c\\.env
        GROQ_API_KEY=...
        GEMINI_API_KEY=...

    python scripts/llm_api.py --gold-only --provider groq
    python scripts/llm_api.py --provider groq --model llama-3.3-70b-versatile

Writes incrementally and resumes, because a free tier will rate-limit partway
through and the run has to survive that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import CSV, WORK  # noqa: E402
from rsna_knee.labels import ID_COL, REPORT_COL, TARGETS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_label import build_prompt, parse  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# KEY=value files.
ENV_FILES = [Path(r"D:\kaggle_c\.env"), REPO_ROOT / ".env", Path(".env")]

# Bare-token files: the whole file is the key, so the variable it maps to has to
# be inferred from the filename. Keys get dropped in with whatever capitalisation
# is convenient (Groq_key.txt, gemini_key.txt), so match case-insensitively on a
# substring rather than listing exact paths.
#
# Every directory searched here is gitignored. A key dropped in as
# groq/Groq_key.txt matched none of the original ignore patterns, and a routine
# `git add -A` would have committed it to a repo that is pushed to GitHub.
KEY_DIRS = [REPO_ROOT / "groq", REPO_ROOT / "gemini", REPO_ROOT / "secrets",
            Path(r"D:\kaggle_c")]
KEY_HINTS = {"GROQ_API_KEY": "groq", "GEMINI_API_KEY": "gemini"}

PROVIDERS = {
    # Groq: OpenAI-compatible, free tier, serves open models fast.
    "groq": dict(
        env="GROQ_API_KEY",
        url="https://api.groq.com/openai/v1/chat/completions",
        default_model="llama-3.3-70b-versatile",
        style="openai",
    ),
    # Anything else speaking the OpenAI schema (OpenRouter, Together, a local
    # vLLM server). Set RSNA_LLM_URL to point at it.
    "openai": dict(
        env="OPENAI_API_KEY",
        url=os.environ.get("RSNA_LLM_URL",
                           "https://api.openai.com/v1/chat/completions"),
        default_model="gpt-4o-mini",
        style="openai",
    ),
    "gemini": dict(
        env="GEMINI_API_KEY",
        url="https://generativelanguage.googleapis.com/v1beta/models",
        default_model="gemini-2.0-flash",
        style="gemini",
    ),
}


def load_env() -> None:
    """Load keys from .env files and from bare-token files, without overriding
    anything already set in the shell."""
    for path in ENV_FILES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

    for var, hint in KEY_HINTS.items():
        if os.environ.get(var):
            continue
        for d in KEY_DIRS:
            if not d.is_dir():
                continue
            for path in sorted(d.glob("*.txt")):
                name = path.name.lower()
                if hint not in name or "key" not in name:
                    continue
                # A bare-token file may still hold KEY=value; accept either, and
                # take the first non-comment line regardless.
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    os.environ[var] = (line.split("=", 1)[1] if "=" in line
                                       else line).strip().strip('"').strip("'")
                    break
                if os.environ.get(var):
                    break
            if os.environ.get(var):
                break


_tl = threading.local()


def session() -> requests.Session:
    if not hasattr(_tl, "s"):
        _tl.s = requests.Session()
    return _tl.s


def call(cfg: dict, key: str, model: str, messages: list[dict],
         temperature: float, attempts: int = 6) -> str | None:
    """One chat completion, retrying on rate limits and transient errors.

    Returns None when every attempt failed, which must stay distinguishable from
    a successful empty answer. Conflating them is not a cosmetic bug: parse("")
    yields all zeros, and writing that as a label turns an API failure into a
    confident "no findings" -- which is how a 40% failure rate on a hosted model
    came back looking like a model that was bad at the task.
    """
    for attempt in range(attempts):
        try:
            if cfg["style"] == "openai":
                # Reasoning models (the gpt-oss family) spend tokens thinking
                # before answering, and a tight budget is consumed entirely by
                # that -- the request succeeds with an empty `content`, which
                # looks exactly like a bad key. Budget for it.
                r = session().post(
                    cfg["url"],
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "messages": messages,
                          "temperature": temperature, "max_tokens": 1024},
                    timeout=180,
                )
            else:  # gemini
                sys_msg = next((m["content"] for m in messages
                                if m["role"] == "system"), "")
                user_msg = next(m["content"] for m in messages
                                if m["role"] == "user")
                # Same trap as the gpt-oss models: recent Gemini models think
                # before answering, and a 96-token budget was spent entirely on
                # that, returning a truncated "ACL=0" or nothing at all. Ask for
                # room, and turn thinking off where the model supports it -- this
                # is an extraction task, not one that benefits from deliberation.
                r = session().post(
                    f"{cfg['url']}/{model}:generateContent?key={key}",
                    json={
                        "system_instruction": {"parts": [{"text": sys_msg}]},
                        "contents": [{"parts": [{"text": user_msg}]}],
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": 1024,
                            "thinkingConfig": {"thinkingBudget": 0},
                        },
                    },
                    timeout=180,
                )
            if r.status_code == 429:
                # Free tiers rate-limit by design; honour Retry-After when given.
                wait = float(r.headers.get("Retry-After", 0)) or min(60, 5 * (attempt + 1))
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if cfg["style"] == "openai":
                msg = data["choices"][0]["message"]
                # A reasoning model that ran out of budget mid-thought leaves
                # content empty but the scores are often already in `reasoning`.
                # The parser only looks for KEY=digit, so reading it is safe.
                return msg.get("content") or msg.get("reasoning") or ""
            # A thinking model can return a candidate whose parts carry no
            # text, so join whatever text parts exist rather than indexing [0].
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(min(30, 3 * (attempt + 1)))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="groq", choices=sorted(PROVIDERS))
    ap.add_argument("--model", default="")
    ap.add_argument("--gold-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scale", type=int, default=11, choices=[4, 11])
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel requests; keep low, free tiers rate-limit")
    ap.add_argument("--out", default="")
    ap.add_argument("--list-models", action="store_true",
                    help="ask the provider what it serves, rather than trusting "
                         "a default model id that may have been renamed")
    ap.add_argument("--smoke", action="store_true",
                    help="one request end to end, to check the key and the "
                         "response shape before committing to a full pass")
    args = ap.parse_args()

    load_env()
    cfg = PROVIDERS[args.provider]
    key = os.environ.get(cfg["env"], "")
    if not key:
        raise SystemExit(
            f"{cfg['env']} is not set. Put it in D:\\kaggle_c\\.env as\n"
            f"    {cfg['env']}=your-key-here\n"
            "rather than passing it on the command line."
        )
    model = args.model or cfg["default_model"]

    if args.list_models:
        # Model ids get renamed and retired; ask rather than assume.
        try:
            if cfg["style"] == "openai":
                base = cfg["url"].rsplit("/chat/completions", 1)[0]
                r = requests.get(f"{base}/models",
                                 headers={"Authorization": f"Bearer {key}"},
                                 timeout=60)
                r.raise_for_status()
                names = sorted(m["id"] for m in r.json().get("data", []))
            else:
                r = requests.get(f"{cfg['url']}?key={key}", timeout=60)
                r.raise_for_status()
                names = sorted(m["name"].split("/")[-1]
                               for m in r.json().get("models", []))
            print(f"{args.provider} serves {len(names)} models:")
            for n in names:
                print("  ", n)
        except Exception as exc:
            print("could not list models:", type(exc).__name__, str(exc)[:200])
        return

    print(f"provider {args.provider}  model {model}  scale {args.scale}  "
          f"samples {args.samples}")

    if args.smoke:
        sample = pd.read_csv(CSV / "train.csv")[REPORT_COL].iloc[0]
        text = call(cfg, key, model, build_prompt(sample, args.scale), 0.0)
        print("\n--- raw response ---")
        print(text[:500] if text else "(empty -- key, model id or quota?)")
        print("\n--- parsed ---")
        print(json.dumps(parse(text, args.scale), indent=2) if text else "n/a")
        return

    tr = pd.read_csv(CSV / "train.csv")
    if args.gold_only:
        tr = tr.dropna(subset=TARGETS).reset_index(drop=True)
    if args.limit:
        tr = tr.head(args.limit).reset_index(drop=True)

    out_path = Path(args.out) if args.out else WORK / (
        f"api_labels_{args.provider}{'_gold' if args.gold_only else ''}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
    done = {r[ID_COL] for r in rows}
    todo = tr[~tr[ID_COL].isin(done)].reset_index(drop=True)
    print(f"{len(tr)} reports, {len(done)} already done, {len(todo)} to go")

    lock = threading.Lock()
    t0 = time.time()

    def one(rec) -> dict | None:
        """Label one report, or return None if every attempt failed.

        A None is dropped rather than written, so the study stays in `todo` and
        a later resume retries it. Writing zeros here would silently record an
        API failure as a confident negative.
        """
        sid, report = rec
        totals = {t: 0.0 for t in TARGETS}
        got = 0
        for _ in range(args.samples):
            text = call(cfg, key, model, build_prompt(report, args.scale),
                        args.temperature)
            if text is None:
                continue
            parsed = parse(text, args.scale)
            # An all-zero parse from a non-empty response is a real "nothing
            # found"; an all-zero parse from an empty one is a failure.
            if not text.strip():
                continue
            for t, v in parsed.items():
                totals[t] += v
            got += 1
        if not got:
            return None
        return {ID_COL: sid, **{t: v / got for t, v in totals.items()}}

    recs = list(zip(todo[ID_COL], todo[REPORT_COL]))
    failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, row in enumerate(ex.map(one, recs), 1):
            with lock:
                if row is None:
                    failed += 1
                else:
                    rows.append(row)
                if i % 25 == 0 or i == len(recs):
                    pd.DataFrame(rows).to_csv(out_path, index=False)
                    el = time.time() - t0
                    rate = i / el if el else 0
                    print(f"  {i}/{len(recs)}  {rate:.1f}/s  "
                          f"eta {(len(recs) - i) / rate / 60 if rate else 0:.0f}m"
                          f"  failed {failed}", flush=True)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(rows)} rows)")
    print("score it with:")
    print(f"  python scripts/eval_labeler.py --llm {out_path}")


if __name__ == "__main__":
    main()
