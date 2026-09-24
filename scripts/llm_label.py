"""Extract weak labels from the radiology reports with a multilingual LLM.

The reports exist only at training time -- the test set is images alone -- so the
labeler has no inference-time cost and can be arbitrarily expensive. That is the
main asymmetry in this competition, and the rule-based labeler in
`rsna_knee.reports` does not exploit it: it tops out at 0.738 macro AUC on the
58 gold studies because hand-written regex cannot cover 12 languages of hedged
clinical prose.

This scores the same 12 targets with an instruct model and is validated against
the same 58 gold studies by `scripts/eval_labeler.py`, so the two are directly
comparable and the better one wins on evidence.

    python scripts/llm_label.py --gold-only          # 58 reports, quick check
    python scripts/llm_label.py                      # all 4407

Writes `llm_labels.csv` to WORK incrementally, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import CSV, WORK  # noqa: E402
from rsna_knee.labels import ID_COL, REPORT_COL, TARGETS  # noqa: E402

# Short keys keep the generated output to ~40 tokens, which is what makes 4407
# reports affordable; the long names would triple it.
KEYS = ["ACL", "MCL", "MM", "LM", "MOA", "LOA", "PFOA",
        "EFF", "SYN", "BAK", "CON", "FRA"]
KEY_TO_TARGET = dict(zip(KEYS, TARGETS))

# The grade scale mirrors the rule-based labeler's: a graded score ranks better
# under ROC AUC than a binary one, and "degeneration only" must not train the
# same value as "radial tear".
GRADE = {0: 0.0, 1: 0.35, 2: 0.85, 3: 1.0}

SYSTEM = """You are a musculoskeletal radiologist reading knee MRI reports.
Reports may be in any language (English, Spanish, German, Turkish, Dutch, French,
Greek, Russian and others). Extract findings exactly as the report states them.
Do not infer findings that are not described."""

# The definitions are the same distinctions the gold labels make, in particular
# the tear-vs-degeneration split that the rule-based labeler had to learn the
# hard way: intrasubstance signal that does not reach the articular surface is
# degeneration, not a tear.
INSTRUCTION = """Report:
\"\"\"
{report}
\"\"\"

Rate each finding 0-3:
0 = not mentioned, or explicitly absent/normal/intact
1 = equivocal, trace/mild, or degeneration WITHOUT a tear
2 = clearly present
3 = severe, large, complete or extensive

Findings:
ACL  anterior cruciate ligament tear
MCL  medial collateral ligament tear
MM   medial meniscus TEAR (mucoid/intrasubstance degeneration alone = 1;
     signal reaching the articular surface, or grade 3, = a tear)
LM   lateral meniscus TEAR, same rule
MOA  medial compartment osteoarthritis / cartilage loss
LOA  lateral compartment osteoarthritis / cartilage loss
PFOA patellofemoral osteoarthritis / chondral damage of patella or trochlea
EFF  joint effusion (grade by amount: trace=1, moderate=2, large=3)
SYN  synovitis or synovial thickening
BAK  Baker's / popliteal cyst
CON  bone contusion or bone marrow oedema
FRA  fracture

Answer with exactly 12 items, nothing else:
ACL=? MCL=? MM=? LM=? MOA=? LOA=? PFOA=? EFF=? SYN=? BAK=? CON=? FRA=?"""


def build_prompt(report: str) -> list[dict]:
    # Very long reports are truncated rather than dropped; the findings section is
    # near the end, so keep the tail.
    text = report if len(report) <= 6000 else report[-6000:]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": INSTRUCTION.format(report=text)},
    ]


def parse(out: str) -> dict[str, float]:
    """Pull KEY=digit pairs out of a generation, defaulting anything absent to 0.

    Lenient by design: a model that adds prose around the answer, or emits the
    keys out of order, should still be usable rather than discarded.
    """
    found = {k: 0 for k in KEYS}
    for key, val in re.findall(r"\b(ACL|MCL|MM|LM|MOA|LOA|PFOA|EFF|SYN|BAK|CON|FRA)\s*[=:]\s*([0-3])",
                               out.upper()):
        found[key] = int(val)
    return {KEY_TO_TARGET[k]: GRADE[v] for k, v in found.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gold-only", action="store_true",
                    help="only the 58 gold-labelled studies, for validation")
    ap.add_argument("--out", default="")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise prompt building and parsing without a model")
    args = ap.parse_args()

    tr = pd.read_csv(CSV / "train.csv")
    if args.gold_only:
        tr = tr.dropna(subset=TARGETS).reset_index(drop=True)
    if args.limit:
        tr = tr.head(args.limit).reset_index(drop=True)
    print(f"{len(tr)} reports to label")

    out_path = Path(args.out) if args.out else WORK / (
        "llm_labels_gold.csv" if args.gold_only else "llm_labels.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = {r[ID_COL]: r for _, r in prev.iterrows()}
        print(f"resuming: {len(done)} already labelled")

    todo = tr[~tr[ID_COL].isin(done)].reset_index(drop=True)
    print(f"{len(todo)} remaining")

    if args.dry_run:
        msgs = build_prompt(todo[REPORT_COL].iloc[0])
        print("\n--- prompt ---\n" + msgs[1]["content"][:900])
        fake = "ACL=2 MCL=0 MM=1 LM=0 MOA=0 LOA=0 PFOA=1 EFF=3 SYN=0 BAK=0 CON=0 FRA=0"
        print("\n--- parse of", fake, "---")
        print(parse(fake))
        print("\n--- parse of messy output ---")
        print(parse("Sure! Here you go:\nACL: 3\nMM=2\nnothing else found"))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16, device_map="auto")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()

    rows = list(done.values())
    import time

    t0 = time.time()
    for start in range(0, len(todo), args.batch):
        chunk = todo.iloc[start:start + args.batch]
        prompts = [
            tok.apply_chat_template(build_prompt(r), tokenize=False,
                                    add_generation_prompt=True)
            for r in chunk[REPORT_COL]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        for sid, text in zip(chunk[ID_COL], outs):
            rows.append({ID_COL: sid, **parse(text)})

        n = len(rows) - len(done)
        el = time.time() - t0
        rate = n / el if el else 0
        eta = (len(todo) - n) / rate / 60 if rate else 0
        print(f"  {n}/{len(todo)}  {rate:.2f}/s  eta {eta:.0f}m", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(df)} rows)")
    print("\nmean score per target:")
    print(df[TARGETS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
