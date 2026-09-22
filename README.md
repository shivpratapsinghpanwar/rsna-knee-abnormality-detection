# RSNA Knee Abnormality Detection

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

## Task
Predict 12 binary abnormality findings per knee MRI study. Scored by
**macro-averaged ROC AUC** across the 12 targets.

Targets (clinical names; exact CSV column spellings TBD from train.csv):
1. ACL tear
2. MCL tear
3. Medial meniscus injury
4. Lateral meniscus injury
5. Medial compartment osteoarthritis
6. Lateral compartment osteoarthritis
7. Patellofemoral osteoarthritis
8. Joint effusion
9. Synovitis
10. Baker's cyst
11. Bone contusion
12. Fracture

## Key facts driving the approach
- ~4.4k training studies; only a small fraction carry human 0/1 labels.
- **Every training study ships a free-text radiology report**, in ~12 languages.
- Data is DICOM MRI, multiple series per study (`train_series/<StudyUID>/<SeriesUID>/`).
- Code competition: inference runs in a Kaggle notebook, no internet at submit time.

The dominant signal is therefore **weak supervision from the reports**, not the
handful of gold labels. The report text exists only at training time; at test
time the model sees images alone.

## Planned pipeline
1. **Report -> labels.** Multilingual report parsing into the 12 targets,
   producing soft/weak labels for all training studies. Calibrate and validate
   this labeler against the small gold-labeled subset.
2. **Series triage.** Classify each series by plane/weighting from DICOM
   metadata; route the informative sequences to the right heads.
3. **Image model.** Per-series 2.5D encoder over slices with attention pooling,
   then cross-series aggregation to a study-level 12-logit output.
4. **Training.** Weak labels as the primary objective, gold labels for
   validation/calibration. Grouped CV by study, stratified per target.
5. **Inference.** Offline Kaggle notebook: weights shipped as a dataset,
   rank-averaged ensemble, submission.csv.

## Status
Scaffolding only. Blocked on Kaggle API credentials to download the data and
confirm the real schema. See PLAN.md for the step ordering.
