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
| + Greek lexicon | 0.720 | 0.633 | 19.0% |
| + sentence-scoped bidirectional negation (Phase 0) | 0.722 | 0.641 | 19.4% |
| + graded scores: severity, hedging, tear-vs-degeneration (Phase 1) | **0.738** | 0.639 | 19.4% |

### Phase 1 — graded labels instead of binary

The metric is ROC AUC, which scores *ranking*, so collapsing "massive effusion"
and "trace fluid" onto the same 1.0 throws away usable signal. `label_report`
now returns a score in [0, 1] built from three axes, and the image model trains
on those scores directly with soft-target BCE.

**Tear vs degeneration (Medial Meniscus 0.639 -> 0.712).** The labeler counted
any degenerative finding as a meniscus tear. Reading the gold cases shows these
are different entities and the labels treat them so:

- gold-negative: "mucoid degeneration", "increased signal ... that does **not**
  contact the articular surface", Stoller grade 1-2
- gold-positive: "radial tear of the posterior root", "bucket-handle tear",
  "yırtık", grade **3** signal

Grade 3 is by definition a tear because it reaches the articular surface, so the
rule is clinical rather than fitted: a named tear morphology, extrusion, grade 3,
or signal reaching the surface scores 1.0; plain intrasubstance degeneration
scores 0.30.

One scoping bug fell out of this. In "signal that does **not** contact the
articular surface", the `not` was negating the *meniscus* rather than the
*surface contact*, so the finding vanished entirely. Surface-contact phrasing is
now resolved before the anchor's negation is tested.

**Severity (Effusion 0.660 -> 0.780).** Reading the contexts first suggested
severity would *not* help: "small joint effusion" and "leve derrame" appear in
both gold-positive and gold-negative studies, and "moderate joint effusion" is
gold-negative. That is true for a binary decision — precision at threshold is
unchanged — but it is the wrong test for this metric. Grading trace/small/
moderate/large onto 0.35/0.55/0.80/1.00 lifts AUC by 0.12 because it ranks
correctly even where it cannot separate.

**Hedging.** "possible tear" scores 0.65 of a stated one.

#### These gains are structural, not tuned

Sweeping the three constants introduced here moves macro AUC by less than 0.01
in total:

| constant | range swept | macro AUC range |
|---|---|---|
| `DEGEN_ONLY` | 0.0 – 0.5 | 0.7381 – 0.7386 |
| `HEDGE_FACTOR` | 0.5 – 1.0 | 0.7381 – 0.7384 |
| `DEFAULT_SEVERITY` | 0.5 – 0.9 | 0.7301 – 0.7391 |

`DEGEN_ONLY` is flat between 0.1 and 0.5 for the reason AUC is the metric: only
the *ordering* of degeneration below a stated tear matters, not the value. So the
gain came from separating the categories, not from picking a number — which is
what the n=58 caveat below demands.

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

Both targets flagged here previously — Effusion and Medial Meniscus — were fixed
in Phase 1. What remains:

- **Lateral OA (0.61)** — the weakest target, and precision and recall are both
  0.36, so this is not a threshold problem. Only 11 gold positives exist, so the
  number itself is barely meaningful; the compartment-attribution logic is the
  place to look, not the vocabulary.
- **Synovitis (0.65)** — recall 0.41. Synovitis is often implied rather than
  named ("enhancing synovium", "pannus", effusion with thickened capsule) and the
  lexicon currently wants it stated.
- **The gold ceiling.** Six targets now sit between 0.71 and 0.89 and the labeler
  is approaching the point where the 58-study gold set cannot distinguish further
  improvements from noise. Additional labeler work should be judged on corpus-level
  coverage and on downstream image-model AUC, not on this set.

### Where the remaining headroom is

The labeler is no longer the bottleneck it was. At 0.738 macro AUC it is a usable
teacher, and the open question is how much of that signal an image model can
recover from pixels — which is what the rest of the pipeline exists to answer.
