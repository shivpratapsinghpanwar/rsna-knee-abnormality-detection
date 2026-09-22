# Confirmed dataset facts

All verified directly against the downloaded competition files, not inferred.

## Files

| file | shape | notes |
|---|---|---|
| `train.csv` | 4407 x 14 | `StudyInstanceUID`, `Report`, + 12 label columns |
| `train_series.csv` | 24371 x 5 | series-level metadata, ~5.5 series per study |
| `sample_submission.csv` | 3 x 13 | `StudyInstanceUID` + 12 targets, all filled with 0.5 |
| `test.csv` | 3 x 1 | `StudyInstanceUID` only; real test set is hidden |

## Targets

Exact column names and order required by the submission:

```
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
```

Values are binary 0/1. Metric is macro-averaged ROC AUC.

## The central constraint

**Only 58 of 4407 studies (1.32%) carry gold labels.** Every other label cell is
null. But `Report` is present for all 4407 studies with zero nulls.

So the image model cannot be trained on gold labels alone — the training signal
has to come from the reports. Report quality is the bottleneck, not model size.

The 58 gold studies are **enriched, not representative**: positive rates run
15–60% (Effusion 60.3%, MCL 15.5%). They cannot be used to estimate prevalence,
only to validate the labeler. At n=58 per-target AUC is extremely noisy, so
lexicon changes must not be tuned aggressively against it.

## Reports

- Encoding is **valid UTF-8** (verified at byte level; the mojibake seen in a
  Windows console is a codepage artifact, not a data defect).
- Length: mean 1098 chars, median 977, range 52–4743.
- Writing systems: **Latin 3866 (87.7%), Greek 321 (7.3%), Cyrillic 220 (5.0%)**.
  Within Latin: English, Spanish, German, Turkish, Dutch, French and others.
- Structure is typically `Technique / Findings / Impression`.

## Series metadata (a gift)

`train_series.csv` already supplies, per series:
`Fluid_Sensitive` (0/1), `Fat_Suppression` (0/1), `Anatomical_Plane`
(Sagittal / Axial / Coronal).

Sequence triage is therefore pre-solved — no DICOM header parsing is needed to
route series to the right heads.

## Labeler baseline

Rule-based multilingual lexicon (`src/rsna_knee/reports.py`), scored on the 58
gold studies:

- **Macro AUC 0.663**, mean F1 0.52.
- Reliable: MCL 0.79, Baker's 0.78, ACL 0.76, Lateral Meniscus 0.76, Fracture 0.75.
- Near chance: Lateral OA 0.51, PF OA 0.52, Effusion 0.53, Medial OA 0.58.

Known causes for the weak targets:
- **OA** — compartment attribution is too strict; reports often state
  osteoarthritis without naming a compartment, or describe it as chondropathy /
  cartilage loss / osteophytes instead.
- **Effusion** — a binary "is effusion mentioned" test is not discriminative,
  because trace effusion is mentioned in most reports. Gold appears to track
  *clinically significant* effusion, so this needs severity grading
  (trace/minimal vs moderate/large).
- **Greek** had no lexicon at all in v1.

A fixed bug worth remembering: unbounded negation cues matched inside words
(Spanish `menisco inter**no **y` contains `no `), silently negating findings
across Romance-language reports. All cues are now `\b`-anchored.
