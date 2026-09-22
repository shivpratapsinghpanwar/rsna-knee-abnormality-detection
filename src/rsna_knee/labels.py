"""Canonical target definitions, confirmed against sample_submission.csv."""

# Exact column names and order as required by the submission file.
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

ID_COL = "StudyInstanceUID"
REPORT_COL = "Report"

# Series-level metadata supplied by the organisers in train_series.csv.
SERIES_META = ["Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
