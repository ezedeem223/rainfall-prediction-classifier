# Academic Research Brief — Rainfall Prediction Classifier

## 1. Problem Definition

This repository addresses next-day rainfall classification from tabular weather observations collected across Australian weather stations. Given a row of daily meteorological measurements — including temperature, humidity, wind speed, atmospheric pressure, and a same-day rain indicator — the task is to predict the binary label `RainTomorrow`: whether rainfall will occur at the same location on the following day.

This is a supervised binary classification problem on structured tabular data.

## 2. Why Next-Day Rainfall Prediction Matters

Short-range rainfall prediction from tabular station observations is a well-studied benchmark that connects classical statistical methods to modern machine learning. It is relevant to:

- Agricultural decision support (irrigation scheduling, crop management).
- Hydrology research (runoff modelling, flood preparedness).
- Comparative evaluation of classical ML models on tabular weather data.
- Reproducibility studies examining how historical ML results hold up under rigorous re-evaluation.
- Methodology research on temporal leakage, label imbalance, and calibration in environmental classification tasks.

This repository does not claim operational forecasting capability. The preserved metrics and workflow are offered as a classical machine learning baseline for research purposes.

## 3. Task Scope

| Dimension | Value |
|---|---|
| Task type | Binary classification |
| Target column | `RainTomorrow` |
| Input | Daily tabular weather observations |
| Dataset | Australian weather station data (Kaggle, not bundled) |
| Geographic scope | Australia-wide (package framing) |
| Temporal scope | Multi-year station records (dataset-dependent) |
| Maintained runtime | `src/rainfall_prediction/` package + `scripts/` |

## 4. Maintained Runtime Surface

The supported workflow is defined by the installable Python package under `src/rainfall_prediction/` and the CLI scripts under `scripts/`. The archived exploratory notebooks under `notebooks/` are preserved for provenance but are not the canonical runtime entrypoint.

Key components:

- `src/rainfall_prediction/data.py` — dataset loading and validation.
- `src/rainfall_prediction/features.py` — preprocessing pipeline (numerical scaling, one-hot encoding).
- `src/rainfall_prediction/pipeline.py` — sklearn pipeline construction.
- `src/rainfall_prediction/train.py` — model training with grid search.
- `src/rainfall_prediction/evaluate.py` — evaluation metric computation.
- `src/rainfall_prediction/predict.py` — inference from saved artifact.
- `scripts/run_train.py`, `run_evaluate.py`, `run_predict.py` — CLI entry points.
- `configs/train.yaml`, `inference.yaml`, `data.yaml` — config-driven behaviour.

## 5. Classical ML Model Families

Three classical model families are compared (verified from `results/model_comparison.csv`):

| Model | Family | Notes |
|---|---|---|
| Logistic Regression | Linear | L1/L2 regularisation, class-weight options |
| Random Forest | Ensemble (bagging) | Susceptible to overfitting without depth constraints |
| XGBoost | Ensemble (gradient boosting) | Provenance caveat applies — see Section 7 |

## 6. Current Evidence Artifacts

The following artifacts are committed and verified:

| Artifact | Status |
|---|---|
| `results/metrics.json` | Preserved historical accuracy values for all three models |
| `results/model_comparison.csv` | Preserved accuracy table with source annotation |
| `results/classification_report.txt` | Preserved report derived from confusion matrix PNG exports (RF and LR only) |
| `results/confusion_matrix.png` | Random Forest confusion matrix PNG export from earlier development run |
| `results/logistic_regression_confusion_matrix.png` | Logistic Regression confusion matrix PNG from earlier development run |
| `results/feature_importance.png` | Random Forest feature importance PNG from earlier development run |
| `results/sample_predictions/example_input.json` | Example inference payload |

No XGBoost confusion matrix or classification report is committed. The dataset and trained model artifact are not committed.

## 7. Provenance Caveats

**XGBoost provenance gap (verified):** The historical accuracy for XGBoost (`train_accuracy=0.8711`, `test_accuracy=0.8519`) is preserved from the pre-refactor README. The currently tracked notebook file does not contain XGBoost training cells or structured metric tables that re-derive these numbers from source. These values must be treated as preserved historical reporting, not notebook-verified results.

**Notebook/package framing mismatch (verified):** The archived notebook (`notebooks/rainfall_prediction_classifier.ipynb`) narrows the analysis to Melbourne-area locations and temporarily renames the target to `RainToday` for a same-day rainfall variant. The public package is framed as Australia-wide `RainTomorrow` prediction. This mismatch means notebook-derived metrics (confusion matrices, accuracy) are not directly comparable to the package framing.

**Random Forest overfitting (verified):** The preserved Random Forest `train_accuracy=0.9999` against `test_accuracy=0.8428` indicates near-complete memorisation of the training set. This is a known risk of unconstrained ensemble methods and is noted as a methodological concern, not a validated result.

**Metrics not newly reproduced:** All metrics in `results/` were preserved during a repository refactor. They were not recomputed by running training and evaluation scripts in this pass.

## 8. Methodological Risks

1. **Random split temporal leakage:** The default split is random, not chronological. For weather data with temporal autocorrelation, this risks information leakage from future observations into the training set.
2. **Label imbalance:** The preserved confusion matrices suggest class imbalance (~76% No-Rain, ~24% Rain in the archived test partition). Accuracy is a misleading metric under imbalance.
3. **Missing-value strategy:** Row-dropping is the default strategy. This may induce selection bias if missingness is correlated with weather conditions.
4. **Geographic confounding:** A random split does not separate stations, so a model may partially memorise location-specific climatological priors rather than learning transferable patterns.
5. **Feature leakage candidates:** `RainToday` is derived from the same observation day's rainfall total and may carry indirect information about `RainTomorrow` through atmospheric persistence.
6. **Calibration unknown:** No probability calibration has been performed. Raw model probabilities should not be interpreted as calibrated rainfall probabilities.

## 9. Future Academic Research Directions

- **Temporal validation:** Replace random splits with chronological or rolling-origin splits.
- **Station-stratified evaluation:** Hold out entire stations to test geographic generalisation.
- **Probability calibration:** Apply Platt scaling or isotonic regression; report Brier score and reliability diagrams.
- **SHAP-based interpretability:** Produce SHAP value distributions for the XGBoost model.
- **Threshold optimisation:** Evaluate F1, precision, recall at multiple decision thresholds; consider asymmetric loss for rainfall early warning.
- **Missing-value imputation:** Compare row-dropping against mean/median/mode imputation and assess impact on class balance.
- **Temporal feature engineering:** Incorporate lag features (yesterday's rainfall, rolling humidity averages) while managing leakage risk carefully.
- **Cross-dataset validation:** Evaluate transferability of a model trained on one time period or region to another.
