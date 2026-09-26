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
import os
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
#
# Four levels turned out to be the scale's weakness rather than its strength.
# On the gold set the LLM alone scores 0.811, but *any* blend with the regex
# scores lands at 0.842-0.852 -- flat across the whole weight range. That
# flatness is the tell: four values leave the corpus riddled with ties, ROC AUC
# cannot rank tied studies, and the regex score was acting as a tie-breaker
# rather than as a second opinion. A finer scale attacks that directly and costs
# nothing extra to generate.
GRADE4 = {0: 0.0, 1: 0.35, 2: 0.85, 3: 1.0}
GRADE = GRADE4

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


# A 0-10 rung asks the model for the same judgement on a scale fine enough to
# rank on. The wording keeps the clinically decisive boundaries -- tear versus
# degeneration, trace versus large effusion -- pinned to specific numbers, so the
# extra resolution lands inside the categories rather than blurring them.
INSTRUCTION_FINE = """Report:
\"\"\"
{report}
\"\"\"

Rate how strongly the report supports each finding, 0-10:
0  not mentioned, or explicitly absent / normal / intact
1-2 mentioned only to be doubted, or a trace amount
3-4 degeneration WITHOUT a tear, mild or equivocal change
5-6 probable, or a small but definite finding
7-8 clearly present as stated
9-10 severe, large, complete or extensive

Use the whole range. Two findings that are both present but differ in size or
certainty must not get the same number.

Findings:
ACL  anterior cruciate ligament tear
MCL  medial collateral ligament tear
MM   medial meniscus TEAR (mucoid/intrasubstance degeneration alone is 3-4;
     signal reaching the articular surface, or grade 3, is a tear)
LM   lateral meniscus TEAR, same rule
MOA  medial compartment osteoarthritis / cartilage loss
LOA  lateral compartment osteoarthritis / cartilage loss
PFOA patellofemoral osteoarthritis / chondral damage of patella or trochlea
EFF  joint effusion, graded by amount
SYN  synovitis or synovial thickening
BAK  Baker's / popliteal cyst
CON  bone contusion or bone marrow oedema
FRA  fracture

Answer with exactly 12 items, nothing else:
ACL=? MCL=? MM=? LM=? MOA=? LOA=? PFOA=? EFF=? SYN=? BAK=? CON=? FRA=?"""


def build_prompt(report: str, scale: int = 4) -> list[dict]:
    # Very long reports are truncated rather than dropped; the findings section is
    # near the end, so keep the tail.
    text = report if len(report) <= 6000 else report[-6000:]
    template = INSTRUCTION if scale == 4 else INSTRUCTION_FINE
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": template.format(report=text)},
    ]


def grade_map(scale: int) -> dict[int, float]:
    """Value for each rung of an N-point scale, normalised to [0, 1]."""
    if scale == 4:
        return GRADE4
    return {i: i / (scale - 1) for i in range(scale)}


def parse(out: str, scale: int = 4) -> dict[str, float]:
    """Pull KEY=digit pairs out of a generation, defaulting anything absent to 0.

    Lenient by design: a model that adds prose around the answer, or emits the
    keys out of order, should still be usable rather than discarded.
    """
    grade = grade_map(scale)
    found = {k: 0 for k in KEYS}
    pat = r"\b(ACL|MCL|MM|LM|MOA|LOA|PFOA|EFF|SYN|BAK|CON|FRA)\s*[=:]\s*(\d{1,2})"
    for key, val in re.findall(pat, out.upper()):
        v = int(val)
        if v in grade:  # ignore an out-of-range rung rather than crash on it
            found[key] = v
    return {KEY_TO_TARGET[k]: grade[v] for k, v in found.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gold-only", action="store_true",
                    help="only the 58 gold-labelled studies, for validation")
    ap.add_argument("--out", default="")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--scale", type=int, default=4, choices=[4, 11],
                    help="rungs in the output scale; 11 gives 0-10, which ranks "
                         "far better under AUC than 4 coarse levels")
    ap.add_argument("--samples", type=int, default=1,
                    help="self-consistency: sample this many times and average. "
                         "Averaging also breaks ties, independently of --scale")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="only used when --samples > 1")
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
        # Read from the full frame, not the remaining work: a dry run should
        # still show a prompt when a previous pass already labelled everything.
        msgs = build_prompt(tr[REPORT_COL].iloc[0], args.scale)
        print("\n--- prompt ---\n" + msgs[1]["content"][:900])
        fake = "ACL=2 MCL=0 MM=1 LM=0 MOA=0 LOA=0 PFOA=1 EFF=3 SYN=0 BAK=0 CON=0 FRA=0"
        print("\n--- parse of", fake, "---")
        print(parse(fake, args.scale))
        print("\n--- parse of messy output ---")
        print(parse("Sure! Here you go:\nACL: 3\nMM=2\nnothing else found",
                    args.scale))
        return

    # Fragmentation is the difference between fitting and not on a 14.5 GB card
    # when sequence lengths vary this much.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Keep model weights inside this project's own directory. The default cache
    # is under the user profile on C:, which is shared with unrelated projects,
    # and a 7B download is 15 GB in fp16 -- enough to matter to whatever else
    # lives there. WORK is on the data drive.
    os.environ.setdefault("HF_HOME", str(WORK.parent / "hf"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # A Kaggle T4 has 14.56 GB and no bfloat16 tensor cores. A 7B model in 16-bit
    # weights is ~14 GB, which loads and then OOMs the moment attention allocates
    # -- that is exactly how the first run died. 4-bit brings it to ~5 GB and
    # leaves room for a real batch, and fp16 is the right dtype on Turing.
    compute_dtype = torch.float16
    # Pin to one GPU rather than letting accelerate shard. In 4-bit the model is
    # ~5 GB and fits a single T4 with room for a batch; sharding across two
    # splits the weights but leaves each device holding activations for the whole
    # sequence, and the full-corpus run died that way with 86 MB free on GPU 1.
    device_map = {"": 0} if args.load_4bit and torch.cuda.is_available() else "auto"
    kw: dict = dict(dtype=compute_dtype, device_map=device_map)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"{free/1e9:.1f}/{total/1e9:.1f} GB free after load")

    rows = list(done.values())
    import time

    t0 = time.time()

    def run(frame) -> list[dict[str, float]]:
        """Label one batch, averaging over --samples generations."""
        prompts = [
            tok.apply_chat_template(build_prompt(r, args.scale), tokenize=False,
                                    add_generation_prompt=True)
            for r in frame[REPORT_COL]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to(model.device)
        totals = [{t: 0.0 for t in TARGETS} for _ in range(len(frame))]
        for _ in range(args.samples):
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=64,
                    do_sample=args.samples > 1,
                    temperature=args.temperature if args.samples > 1 else None,
                    pad_token_id=tok.pad_token_id,
                )
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            for i, text in enumerate(outs):
                for t, v in parse(text, args.scale).items():
                    totals[i][t] += v
        # Averaging several samples is a second, independent source of
        # resolution: it turns a discrete rung into a continuous score.
        return [{t: v / args.samples for t, v in row.items()} for row in totals]

    batch = args.batch
    start = 0
    while start < len(todo):
        chunk = todo.iloc[start:start + batch]
        try:
            outs = run(chunk)
        except torch.OutOfMemoryError:
            # Report length varies a lot (52 to 4743 chars), so one unlucky batch
            # can OOM where the rest are fine. Halve and retry rather than lose
            # the run; 4407 reports is too long to restart casually.
            torch.cuda.empty_cache()
            if batch > 1:
                batch = max(1, batch // 2)
                print(f"  OOM -- retrying with batch {batch}", flush=True)
                continue
            print("  OOM at batch 1; skipping this report", flush=True)
            outs = [{t: 0.0 for t in TARGETS} for _ in range(len(chunk))]
        for sid, scored in zip(chunk[ID_COL], outs):
            rows.append({ID_COL: sid, **scored})
        start += len(chunk)

        n = len(rows) - len(done)
        el = time.time() - t0
        rate = n / el if el else 0
        eta = (len(todo) - n) / rate / 60 if rate else 0
        print(f"  {n}/{len(todo)}  {rate:.2f}/s  eta {eta:.0f}m  batch={batch}",
              flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(df)} rows)")
    print("\nmean score per target:")
    print(df[TARGETS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
