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

## Labeler progress

Rule-based multilingual lexicon (`src/rsna_knee/reports.py`), scored on the 58
gold studies with `scripts/eval_labeler.py`:

| version | macro AUC | mean F1 | zero-label studies |
|---|---|---|---|
| v1 baseline | 0.663 | 0.519 | 29.6% |
| + word-bounded negation | 0.664 | 0.522 | — |
| + OA reworked around chondral findings | 0.701 | 0.607 | — |
| + effusion/synovitis vocabulary | 0.711 | 0.621 | — |
| + Greek lexicon | **0.720** | **0.633** | **19.0%** |

Current per-target AUC: MCL 0.85, ACL 0.78, Lateral Meniscus 0.78, Baker's 0.78,
Contusion 0.75, Fracture 0.73, PF OA 0.71, Medial OA 0.70, Synovitis 0.66,
Lateral OA 0.65, Medial Meniscus 0.63, Effusion 0.63.

### What moved the needle

1. **Negation bug.** Unbounded cues matched inside words — Spanish
   `menisco inter**no **y` contains `no `, silently negating the finding that
   followed. This corrupted Romance-language reports wholesale. All cues are now
   `\b`-anchored.
2. **OA is written as cartilage damage, not as "arthrosis".** Gold PF-OA cases
   read "chondral ulcer of the femoral trochlea" or "osteochondral defect at the
   lateral patellar facet". Adding chondral/cartilage vocabulary and
   patella/trochlea anchors lifted the three OA targets by 0.12–0.18 AUC each.
   `(?<!sub)` guards keep *subchondral* (e.g. fracture oedema) from firing.
3. **Compartment attribution.** "Cartilage defect at lateral trochlea" is
   patellofemoral, not lateral-compartment. PF terms now take precedence unless a
   femorotibial structure is named outright.
4. **Greek.** 321 studies (7.3%) had no coverage at all in v1.

### Caveat on overfitting

At n=58, per-target AUC is noise-dominated — the negation fix looked flat overall
while swinging Lateral Meniscus +0.09 and Medial Meniscus −0.06. Changes have
therefore been grounded in medical/linguistic vocabulary that generalises, not in
per-example tweaks. The corpus-level coverage figure (zero-label studies
29.6% → 19.0%) is independent evidence the gains are real, since the gold set
cannot observe it.

### Still weak

- **Effusion (0.63)** — recall is fixed but precision is capped, because trace
  effusion is mentioned in most reports while gold appears to track
  *clinically significant* effusion. Needs severity grading (trace/minimal vs
  moderate/large) rather than presence/absence.
- **Medial Meniscus (0.63)** — likely needs the same treatment for
  degenerative signal vs true tear.
