# Model Card — Rainfall Prediction Classifier

## Model Overview

| Field | Value |
|---|---|
| Task | Binary classification: predict `RainTomorrow` |
| Input type | Tabular daily weather observations |
| Output | Binary label (`Yes` / `No`) and optionally class probabilities |
| Language | Python 3.10 |
| Framework | scikit-learn pipeline with XGBoost classifier support |
| Artifact format | `joblib` bundle: fitted pipeline + label encoder + feature metadata |

## Model Families

Three classical model families are compared (verified from `results/model_comparison.csv`):

| Model | Identifier | Notes |
|---|---|---|
| Logistic Regression | `logistic_regression` | L1/L2, optional class weighting |
| Random Forest | `random_forest` | Bagging ensemble; overfitting risk verified |
| XGBoost | `xgboost` | Gradient boosting; provenance caveat applies |

## Target Variable

- **Column:** `RainTomorrow`
- **Type:** Binary string (`Yes` / `No`)
- **Encoding:** Label-encoded internally (`0` = No, `1` = Yes)

## Input Assumptions

The model expects tabular rows with the standard `weatherAUS` columns. Features used by the pipeline depend on the columns present after preprocessing:

- Numerical columns are scaled (StandardScaler by default).
- Categorical columns are one-hot encoded (unknown categories ignored at inference).
- A `Season` feature is derived from the `Date` column.
- The `Date` column is dropped before modelling.
- Rows with missing values are dropped by default (configurable in `configs/data.yaml`).

The exact feature set required at inference matches the columns seen during training. The saved artifact includes `feature_columns` metadata for validation.

## Dataset Status

- **Dataset:** Australian weather station observations (`weatherAUS.csv`)
- **Source:** Kaggle (`https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package`)
- **Bundled:** **No.** The dataset is not committed to this repository.
- **Expected path:** `data/raw/weatherAUS.csv`

## Artifact / Checkpoint Status

- **Default trained artifact:** `models/rainfall_prediction_pipeline.joblib`
- **Bundled:** **No.** No trained model artifact is committed by default.
- **To generate:** Run `python scripts/run_train.py --config configs/train.yaml` after placing the dataset.

## Intended Research Use

This model is intended for:

- Classical machine learning research on tabular weather data.
- Reproducibility and provenance studies.
- Methodology benchmarking (random vs. temporal splits, calibration, leakage auditing).
- Educational demonstration of scikit-learn pipeline patterns.

## Out-of-Scope Uses

This model is **not** suitable for:

- Operational weather forecasting or meteorological services.
- Climate-risk decision systems or any form of policy guidance.
- Safety-critical decision support.
- Real-time weather prediction.
- Geographic or temporal generalisation beyond the training data distribution without further validation.

## Preserved Historical Metrics

The following metrics are preserved from earlier repository reporting. They were **not** newly reproduced in this pass. They should be treated as historical reference values, not freshly verified benchmarks.

### From `results/metrics.json` and `results/model_comparison.csv`

| Model | Train Accuracy | Test Accuracy | Source | Evidence Confidence |
|---|---|---|---|---|
| Logistic Regression | 0.8403 | 0.8424 | Pre-refactor README | Preserved historical reporting |
| Random Forest | 0.9999 | 0.8428 | Pre-refactor README | Preserved historical reporting |
| XGBoost | 0.8711 | 0.8519 | Pre-refactor README | Preserved historical reporting — provenance gap (see below) |

### From `results/classification_report.txt` (derived from PNG confusion matrix exports)

| Model | Precision (No) | Recall (No) | Precision (Yes) | Recall (Yes) | Accuracy |
|---|---|---|---|---|---|
| Random Forest | 0.8632 | 0.9515 | 0.7667 | 0.5140 | 0.8479 |
| Logistic Regression | 0.8602 | 0.9281 | 0.6891 | 0.5140 | 0.8300 |

No XGBoost classification report is preserved in committed repository files.

### XGBoost Provenance Gap

The XGBoost accuracy values (`train_accuracy=0.8711`, `test_accuracy=0.8519`) were preserved from the pre-refactor README. The currently tracked notebook file does not include XGBoost training cells or a structured metric table re-deriving these values from source. These values must not be presented as notebook-verified results.

## Limitations

1. **No temporal validation performed.** The default random split may overestimate generalisation due to temporal autocorrelation in weather data.
2. **Random Forest exhibits near-complete training memorisation** (`train_accuracy=0.9999`). The test accuracy is not a reliable indicator of generalisation without examining depth constraints and temporal split behaviour.
3. **Class imbalance present.** Preserved confusion matrices suggest approximately 76% No-Rain and 24% Rain in the archived test partition. Accuracy alone is a misleading metric.
4. **Missing-value handling by row deletion** may induce selection bias.
5. **No probability calibration performed.** Model outputs should not be interpreted as calibrated rainfall probabilities.
6. **XGBoost metrics have a provenance gap.** See above.
7. **Notebook and package framings differ.** Notebook metrics are from a Melbourne-area same-day variant; package metrics are from Australia-wide `RainTomorrow` prediction.

## Ethical and Safety Limitations

- This model is a **research artifact**, not an operational weather forecasting service.
- Predictions must not be used for public safety, emergency management, or climate-risk decisions without further rigorous validation, calibration, and domain expert review.
- Geographic and temporal generalisation cannot be assumed from the preserved metrics.
- Any deployment beyond research experimentation requires independent re-evaluation on representative held-out data with appropriate temporal structure.

## Recommended Review

Before any downstream research use of these metrics, reviewers should:

1. Verify that the dataset used for evaluation matches the package framing (Australia-wide, `RainTomorrow`).
2. Check that the train/test split is reproducible and documented (random state 42, stratified, 20% test).
3. Confirm that temporal leakage has been assessed (see `TEMPORAL_VALIDATION_PROTOCOL.md`).
4. Review the XGBoost provenance gap (see `METRIC_PROVENANCE_MATRIX.md`).
5. Inspect the Random Forest overfitting pattern before interpreting its test accuracy as a reliable generalisation estimate.
