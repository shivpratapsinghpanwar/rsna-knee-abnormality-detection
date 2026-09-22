"""Regression tests for the report labeler.

Each case below is a defect found by auditing the labeler against the full 4407
report corpus. They are pinned here so a future lexicon change cannot silently
reintroduce them.

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


def main() -> int:
    failures = []
    for desc, text, target, expected in CASES:
        got = label_report(text)[target]
        if got != expected:
            failures.append((desc, target, expected, got, text))

    for desc, target, expected, got, text in failures:
        print(f"FAIL  {desc}\n      target={target} expected={expected} got={got}"
              f"\n      text={text!r}\n")

    print(f"{len(CASES) - len(failures)}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
