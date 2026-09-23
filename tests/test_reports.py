"""Regression tests for the report labeler.

Each case below is a defect found by auditing the labeler against the full 4407
report corpus. They are pinned here so a future lexicon change cannot silently
reintroduce them.

Scores are graded, not binary, so CASES assert on which side of the 0.5 decision
boundary a score falls. That keeps every case's intent ("this is affirmed" /
"this is negated") while leaving the grading constants free to be tuned. Cases
that pin the grading itself live in ORDER_CASES and GRADE_CASES below.

Run standalone: python tests/test_reports.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.reports import label_report  # noqa: E402

CASES = [
    # (description, report text, target, expected)
    (
        "bug C: post-position negation must be detected",
        "medial meniscus tear is not identified on this study",
        "Medial Meniscus", 0.0,
    ),
    (
        "control: pre-position negation still works",
        "no tear of the medial meniscus is seen",
        "Medial Meniscus", 0.0,
    ),
    (
        "bug D: negation must not bleed across sentences",
        "acl tear is not seen. medial meniscus tear is present.",
        "Medial Meniscus", 1.0,
    ),
    (
        "bug D: the negated target in the same text stays negative",
        "acl tear is not seen. medial meniscus tear is present.",
        "ACL", 0.0,
    ),
    (
        "bug B: OA must not be credited across a clause boundary",
        "the lateral compartment demonstrates osteoarthritis "
        "while the medial compartment is normal",
        "Medial OA", 0.0,
    ),
    (
        "bug B: the correct compartment is still credited",
        "the lateral compartment demonstrates osteoarthritis "
        "while the medial compartment is normal",
        "Lateral OA", 1.0,
    ),
    (
        "bug A: 'roto' must not fire inside 'protocol' / 'proton'",
        "technique: proton density fat sat protocol. "
        "the medial meniscus is normal.",
        "Medial Meniscus", 0.0,
    ),
    (
        "bug F: 'osteoarthritic' must match",
        "osteoarthritic changes of the medial compartment",
        "Medial OA", 1.0,
    ),
    (
        "negation cues must be word-bounded (interno contains 'no ')",
        "rotura de menisco interno y lateral. derrame.",
        "Medial Meniscus", 1.0,
    ),
    (
        "same text: effusion must survive the interno false-negation",
        "rotura de menisco interno y lateral. derrame.",
        "Effusion", 1.0,
    ),
    (
        "turkish verb-final negation",
        "medial menisku yirtigi yoktur",
        "Medial Meniscus", 0.0,
    ),
    (
        "turkish affirmative",
        "medial menisku yirtigi mevcuttur",
        "Medial Meniscus", 1.0,
    ),
    (
        "negation terminator: 'but' releases the scope",
        "no effusion but there is a medial meniscus tear",
        "Medial Meniscus", 1.0,
    ),
    (
        "negation terminator: the negated finding stays negative",
        "no effusion but there is a medial meniscus tear",
        "Effusion", 0.0,
    ),
    (
        "PF OA written as a chondral defect of the trochlea",
        "full thickness cartilage defect at the femoral trochlea",
        "PF OA", 1.0,
    ),
    (
        "trochlear chondral defect is not lateral-compartment OA",
        "full thickness cartilage defect at the lateral trochlea",
        "Lateral OA", 0.0,
    ),
]


# Cases pinning the tear-vs-degeneration split. Intrasubstance degeneration and a
# stated tear are different entities and must not train the same target value.
GRADE_CASES = [
    (
        "mucoid degeneration alone is not a tear",
        "medial meniscus mucoid degeneration", "Medial Meniscus", "degen",
    ),
    (
        "signal not reaching the articular surface is degeneration",
        "increased signal in the posterior horn of the medial meniscus "
        "that does not contact the articular surface",
        "Medial Meniscus", "degen",
    ),
    (
        "grade 3 signal reaches the surface and is a tear",
        "grade iii signal in the posterior horn of the medial meniscus",
        "Medial Meniscus", "pos",
    ),
    (
        "grade 2 signal is not a tear",
        "grade ii signal in the posterior horn of the medial meniscus",
        "Medial Meniscus", "degen",
    ),
    (
        "a named tear morphology is a tear",
        "radial tear of the posterior root of the medial meniscus",
        "Medial Meniscus", "pos",
    ),
    (
        "meniscal extrusion counts as structural failure",
        "medial meniskus ekstrude gorunumde", "Medial Meniscus", "pos",
    ),
    (
        "hedged findings are discounted but stay positive",
        "possible tear of the lateral meniscus", "Lateral Meniscus", "hedged",
    ),
]

# Severity must be ordered, since ROC AUC scores ranking.
ORDER_CASES = [
    (
        "effusion severity is ordered",
        "Effusion",
        ["trace joint effusion", "small joint effusion",
         "moderate joint effusion", "large joint effusion"],
    ),
]

BANDS = {
    "pos": (0.99, 1.01),     # a stated, unhedged finding
    "hedged": (0.5, 0.99),   # affirmed but qualified
    "degen": (0.01, 0.5),    # degeneration without a tear
    "neg": (-0.01, 0.01),    # negated or absent
}


def main() -> int:
    failures = []
    for desc, text, target, expected in CASES:
        got = label_report(text)[target]
        # CASES predate grading: assert only the side of the decision boundary.
        if (got >= 0.5) != (expected >= 0.5):
            side = "pos" if expected >= 0.5 else "neg"
            failures.append((desc, target, f"{side} side", got, text))

    for desc, text, target, band in GRADE_CASES:
        lo, hi = BANDS[band]
        got = label_report(text)[target]
        if not (lo <= got <= hi):
            failures.append((desc, target, f"{band} band [{lo}, {hi}]", got, text))

    for desc, target, texts in ORDER_CASES:
        scores = [label_report(t)[target] for t in texts]
        if not all(a < b for a, b in zip(scores, scores[1:])):
            failures.append((desc, target, "strictly increasing", scores,
                             " | ".join(texts)))

    for desc, target, expected, got, text in failures:
        print("FAIL  " + desc)
        print(f"      target={target} expected={expected} got={got}")
        print(f"      text={text!r}")
        print()

    total = len(CASES) + len(GRADE_CASES) + len(ORDER_CASES)
    print(f"{total - len(failures)}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
