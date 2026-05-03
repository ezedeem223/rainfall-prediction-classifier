# Metric Provenance Matrix

This document traces every preserved metric referenced in this repository. Its purpose is to make the evidence chain auditable so that downstream researchers can assess what can and cannot be safely cited.

**Global constraints:**

- Do not call any metric here newly reproduced unless training and evaluation scripts were explicitly run against the local dataset in a documented pass.
- Do not present XGBoost numbers as notebook-verified — the tracked notebook does not contain XGBoost cells.
- Do not imply production readiness, operational forecast validity, climate decision validity, geographic generalisation beyond the dataset, or temporal robustness without temporal validation.

---

## Accuracy Metrics from `results/metrics.json` and `results/model_comparison.csv`

| Metric | Value | Source File | Source Context | Evidence Confidence | Allowed Wording | Forbidden Wording |
|---|---|---|---|---|---|---|
| Logistic Regression — Train Accuracy | 0.8403 | `results/metrics.json`, `results/model_comparison.csv` | `historical_repository_results` block; `source: original_repository_readme` | Preserved from pre-refactor README; not newly reproduced | "preserved historical training accuracy", "reported in earlier repository documentation" | "newly evaluated", "verified from fresh run", "notebook-verified" |
| Logistic Regression — Test Accuracy | 0.8424 | `results/metrics.json`, `results/model_comparison.csv` | `historical_repository_results` block; `source: original_repository_readme` | Preserved from pre-refactor README; not newly reproduced | "preserved historical test accuracy", "reported in earlier repository documentation" | "newly evaluated", "verified from fresh run", "notebook-verified" |
| Random Forest — Train Accuracy | 0.9999 | `results/metrics.json`, `results/model_comparison.csv` | `historical_repository_results` block; `source: original_repository_readme` | Preserved from pre-refactor README; not newly reproduced | "preserved historical training accuracy", "near-complete memorisation of training set" | "newly evaluated", "verified from fresh run" |
| Random Forest — Test Accuracy | 0.8428 | `results/metrics.json`, `results/model_comparison.csv` | `historical_repository_results` block; `source: original_repository_readme` | Preserved from pre-refactor README; not newly reproduced | "preserved historical test accuracy" | "newly evaluated", "verified from fresh run" |
| XGBoost — Train Accuracy | 0.8711 | `results/metrics.json`, `results/model_comparison.csv` | `historical_repository_results` block; `source: original_repository_readme` | Preserved from pre-refactor README; provenance gap — notebook does not contain XGBoost cells | "preserved from earlier repository reporting", "provenance gap applies" | "notebook-verified", "newly reproduced", "verified from fresh run" |
| XGBoost — Test Accuracy | 0.8519 | `results/metrics.json`, `results/model_comparison.csv` | `historical_repository_results` block; `source: original_repository_readme` | Preserved from pre-refactor README; provenance gap — notebook does not contain XGBoost cells | "preserved from earlier repository reporting", "provenance gap applies" | "notebook-verified", "newly reproduced", "verified from fresh run" |

---

## Confusion Matrix Metrics from `results/classification_report.txt`

These metrics were derived computationally from the confusion matrix PNG exports that were committed before the repository refactor.

| Metric | Value | Source File | Source Context | Evidence Confidence | Allowed Wording | Forbidden Wording |
|---|---|---|---|---|---|---|
| Random Forest — Confusion Matrix (TN, FP, FN, TP) | [[1098, 56], [174, 184]] | `results/metrics.json` (`preserved_notebook_exports`) | Derived from `results/confusion_matrix.png` | Verified from committed JSON; PNG originates from earlier notebook run | "confusion matrix derived from preserved PNG export", "archived evaluation artifact" | "newly generated", "fresh evaluation" |
| Random Forest — Derived Test Accuracy from CM | 0.8479 | `results/metrics.json` | `derived_test_accuracy` field under `preserved_notebook_exports.random_forest` | Computed from confusion matrix values; consistent with reported 0.8428 within rounding/subset differences | "derived from preserved confusion matrix", "consistent with historical reporting within rounding" | "authoritative fresh result" |
| Random Forest — Precision (No) | 0.8632 | `results/classification_report.txt` | Random Forest export section | Derived from preserved PNG confusion matrix; not a fresh evaluation | "preserved derived classification metric" | "newly evaluated" |
| Random Forest — Recall (No) | 0.9515 | `results/classification_report.txt` | Random Forest export section | Derived from preserved PNG confusion matrix | "preserved derived classification metric" | "newly evaluated" |
| Random Forest — Precision (Yes) | 0.7667 | `results/classification_report.txt` | Random Forest export section | Derived from preserved PNG confusion matrix | "preserved derived classification metric" | "newly evaluated" |
| Random Forest — Recall (Yes) | 0.5140 | `results/classification_report.txt` | Random Forest export section | Derived from preserved PNG confusion matrix | "preserved derived classification metric" | "newly evaluated" |
| Random Forest — F1 (No) | 0.9052 | `results/classification_report.txt` | Random Forest export section | Derived from preserved PNG confusion matrix | "preserved derived classification metric" | "newly evaluated" |
| Random Forest — F1 (Yes) | 0.6154 | `results/classification_report.txt` | Random Forest export section | Derived from preserved PNG confusion matrix | "preserved derived classification metric" | "newly evaluated" |
| Logistic Regression — Confusion Matrix (TN, FP, FN, TP) | [[1071, 83], [174, 184]] | `results/metrics.json` | Derived from `results/logistic_regression_confusion_matrix.png` | Verified from committed JSON; PNG from earlier notebook run | "confusion matrix derived from preserved PNG export" | "newly generated", "fresh evaluation" |
| Logistic Regression — Derived Test Accuracy from CM | 0.8300 | `results/metrics.json` | `derived_test_accuracy` field under `preserved_notebook_exports.logistic_regression` | Computed from confusion matrix values | "derived from preserved confusion matrix" | "authoritative fresh result" |
| Logistic Regression — Precision (No) | 0.8602 | `results/classification_report.txt` | Logistic Regression export section | Derived from preserved PNG confusion matrix | "preserved derived classification metric" | "newly evaluated" |
| Logistic Regression — Recall (No) | 0.9281 | `results/classification_report.txt` | Logistic Regression export section | Derived from preserved PNG confusion matrix | "preserved derived classification metric" | "newly evaluated" |
| Logistic Regression — Precision (Yes) | 0.6891 | `results/classification_report.txt` | Logistic Regression export section | Derived from preserved PNG confusion matrix | "preserved derived classification metric" | "newly evaluated" |
| Logistic Regression — Recall (Yes) | 0.5140 | `results/classification_report.txt` | Logistic Regression export section | Derived from preserved PNG confusion matrix | "preserved derived classification metric" | "newly evaluated" |

---

## XGBoost — Provenance Gap Summary

| Aspect | Status |
|---|---|
| XGBoost train/test accuracy in `metrics.json` | Present — preserved from pre-refactor README |
| XGBoost in `model_comparison.csv` | Present — annotated with provenance gap note |
| XGBoost cells in tracked notebook | **Absent** — confirmed from `results/model_comparison.csv` notes and `results/classification_report.txt` |
| XGBoost confusion matrix PNG | **Absent** — not committed |
| XGBoost classification report | **Absent** — `classification_report.txt` explicitly states none preserved |
| Safe citation posture | Present as "preserved from earlier repository reporting" only; do not present as notebook-verified or freshly reproduced |

---

## Metrics Not Present — Placeholder Status

| Metric | Status |
|---|---|
| Brier Score (any model) | **Not computed** — no calibration pass performed |
| ROC-AUC (any model) | **Not committed** — not present in preserved artifacts |
| Precision-Recall AUC | **Not committed** |
| Cross-validation scores | **Not committed** (config specifies 5-fold CV but no CV metrics preserved) |
| Temporal validation metrics | **Not computed** — no temporal split performed in this pass |
| SHAP values | **Not computed** — no SHAP pass performed in this pass |
| Permutation importance | **Not computed** — no permutation importance pass performed |
| Calibration curves | **Not computed** — no calibration pass performed |

These metrics are **placeholders only** and must not be used as performance evidence until generated from a documented run with local dataset and model artifact.

---

## Red Lines

1. Do not call any preserved metric in this repository newly reproduced unless the training and evaluation scripts were run with a local dataset and the output files were verified in a documented pass.
2. Do not present XGBoost accuracy numbers as notebook-verified under any circumstances.
3. Do not imply production readiness.
4. Do not imply operational forecast validity.
5. Do not imply climate or policy decision validity.
6. Do not imply geographic generalisation beyond the Australian weather station dataset.
7. Do not imply temporal robustness without documented temporal validation results.
8. Do not treat the Random Forest `test_accuracy=0.8428` as a reliable generalisation estimate without acknowledging the `train_accuracy=0.9999` overfitting pattern and the absence of temporal split validation.
