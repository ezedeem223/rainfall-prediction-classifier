# Reproducibility Checklist

This checklist documents exactly what can and cannot be verified at each stage of reproducibility, and what local assets are required for each step.

---

## 1. Installation

### Steps

```bash
# Clone the repository
git clone https://github.com/ezedeem223/rainfall-prediction-classifier.git
cd rainfall-prediction-classifier

# Install runtime dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# (Optional) Install development dependencies
pip install -r requirements-dev.txt
```

### Verification (No Dataset or Model Required)

```bash
python -c "import rainfall_prediction; print(rainfall_prediction.__version__)"
```

**Expected output:** `1.0.0`

### Checklist

- [ ] Python 3.10 or higher available.
- [ ] `pip install -r requirements.txt` completes without errors.
- [ ] `pip install -e .` completes without errors.
- [ ] `import rainfall_prediction` succeeds.

---

## 2. Dataset Setup

### Steps

1. Download `weatherAUS.csv` from the Kaggle dataset: `https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package`
2. Place the file at `data/raw/weatherAUS.csv`.

### Verification

```bash
ls -lh data/raw/weatherAUS.csv
head -n 3 data/raw/weatherAUS.csv
```

**Expected:** File exists with headers including `Date`, `Location`, `RainTomorrow`, etc.

### Checklist

- [ ] Kaggle account created (required for download).
- [ ] `data/raw/weatherAUS.csv` present and readable.
- [ ] CSV contains `RainTomorrow` column (target column for the package framing).
- [ ] CSV contains `Date`, `Location`, `RainToday` columns required by the pipeline.

**Note:** The dataset is not bundled in this repository. Training and evaluation cannot proceed without this file.

---

## 3. Train Command

### Command

```bash
python scripts/run_train.py --config configs/train.yaml
```

### Prerequisites

- [ ] Dataset present at `data/raw/weatherAUS.csv`.
- [ ] All dependencies installed.

### Expected Outputs

- `models/rainfall_prediction_pipeline.joblib` — trained model artifact.
- `results/metrics.json` — fresh evaluation metrics (will overwrite preserved historical metrics).
- `results/model_comparison.csv` — fresh model comparison table (will overwrite preserved values).

### Checklist

- [ ] Command completes without errors.
- [ ] `models/rainfall_prediction_pipeline.joblib` is created.
- [ ] `results/metrics.json` is updated with fresh values.
- [ ] Fresh metrics are noted separately from the preserved historical metrics for provenance tracking.

**Warning:** Running this command will overwrite the preserved historical metrics in `results/metrics.json` and `results/model_comparison.csv`. Back up these files if you want to retain the preserved historical values alongside fresh results.

---

## 4. Evaluate Command

### Command

```bash
python scripts/run_evaluate.py --config configs/inference.yaml
```

### Prerequisites

- [ ] Dataset present at `data/raw/weatherAUS.csv`.
- [ ] Trained model artifact present at `models/rainfall_prediction_pipeline.joblib`.

### Expected Outputs

- Updated `results/metrics.json`.
- Updated `results/classification_report.txt`.
- Updated `results/confusion_matrix.png`.
- Updated `results/feature_importance.png` (if applicable to the selected model).

### Checklist

- [ ] Command completes without errors.
- [ ] `results/classification_report.txt` contains fresh metrics.
- [ ] `results/confusion_matrix.png` is updated.

---

## 5. Predict Command

### Command

```bash
python scripts/run_predict.py \
  --config configs/inference.yaml \
  --input results/sample_predictions/example_input.json
```

### Prerequisites

- [ ] Trained model artifact present at `models/rainfall_prediction_pipeline.joblib`.

**Note:** The predict script uses the saved model artifact and does not require the full dataset.

### Expected Outputs

- `results/sample_predictions/example_predictions.json` — predictions for the example input.

### Checklist

- [ ] Command completes without errors.
- [ ] Output file contains `RainTomorrow` predictions.
- [ ] If `include_probabilities: true` in `configs/inference.yaml`, output also contains `probability_No` and `probability_Yes` columns.

---

## 6. Test Command

### Command

```bash
pytest
```

### Prerequisites

- None. Tests do not require the dataset or trained model artifact.
- Development dependencies must be installed (`pip install -r requirements-dev.txt`).

### Expected Output

```
7 passed in X.XXs
```

(Plus any additional tests from the research pack.)

### Checklist

- [ ] All tests pass.
- [ ] No import errors.
- [ ] `tests/test_research_pack_exists.py` passes (verifies research pack files exist).
- [ ] `tests/test_metric_provenance.py` passes (verifies metric provenance correctness).

---

## 7. Evidence Pack Validation

### Command

```bash
python tools/evidence/validate_research_pack.py
```

### Prerequisites

- None. Validation tool does not require dataset, model artifact, or internet access.

### Expected Output

```
[PASS] All research pack files exist.
[PASS] results/metrics.json exists and is valid JSON.
[PASS] results/model_comparison.csv exists and is parseable.
[PASS] README.md references ACADEMIC_RESEARCH_BRIEF.md.
[PASS] No forbidden phrases found in docs/research_pack/.
[PASS] No institution-specific wording found in docs/research_pack/.
[PASS] Required limitation phrases present.
[PASS] Temporal protocol: no temporal results claimed.
[PASS] Interpretability protocol: no SHAP/permutation artifacts claimed.
[PASS] Calibration protocol: no calibration results claimed.
[PASS] Leakage audit: XGBoost provenance gap mentioned.
[PASS] Dataset/task card: notebook/package framing mismatch mentioned.

All checks passed.
```

---

## 8. Required Local Assets Summary

| Asset | Required For | Can Be Verified Without It? |
|---|---|---|
| `data/raw/weatherAUS.csv` | Training, evaluation | No |
| `models/rainfall_prediction_pipeline.joblib` | Evaluation, inference | No (generate with train command) |
| Python 3.10+ | All | Yes (check with `python --version`) |
| Installed packages | All | Verify with `pip list` |

---

## 9. What Can Be Verified Without Dataset or Model

| Verification | Command | Notes |
|---|---|---|
| Package import | `python -c "import rainfall_prediction"` | Must succeed |
| Config loading | `pytest tests/test_config.py` | Tests config loading logic |
| Pipeline construction | `pytest tests/test_pipeline.py` | Tests sklearn pipeline build |
| Evidence pack existence | `pytest tests/test_research_pack_exists.py` | Tests all research pack files exist |
| Metric provenance | `pytest tests/test_metric_provenance.py` | Tests preserved metric values and structure |
| Validation tool | `python tools/evidence/validate_research_pack.py` | Structural completeness check |
| Preserved metrics review | Manual: read `results/metrics.json` | Historical accuracy values available without dataset |
| Preserved confusion matrices | Manual: view `results/confusion_matrix.png` | PNG artifacts committed |

---

## 10. What Cannot Be Verified Without Dataset or Model

| Verification | Requires | Notes |
|---|---|---|
| Fresh training accuracy | Dataset | Must re-run `run_train.py` |
| Fresh classification report | Dataset + artifact | Must re-run `run_evaluate.py` |
| Temporal validation metrics | Dataset + artifact | Chronological split not yet implemented |
| SHAP values | Dataset + artifact | Not generated in this pass |
| Permutation importance | Dataset + artifact | Not generated in this pass |
| Calibration metrics | Dataset + artifact | Not generated in this pass |
| Per-location error rates | Dataset + artifact | Not generated in this pass |
| Rolling-origin CV results | Dataset | Not generated in this pass |

---

## 11. Known Limitations of Reproducibility

1. **Dataset not bundled:** Full reproduction of training and evaluation results requires independently obtaining `weatherAUS.csv` from Kaggle.
2. **Dataset version:** The Kaggle dataset may have been updated since the preserved historical metrics were computed. Fresh results may differ from historical results even with identical code.
3. **XGBoost provenance gap:** The XGBoost historical accuracy cannot be reproduced from the archived notebook (no XGBoost cells present). Only a fresh training run can independently verify or update these values.
4. **Notebook/package framing mismatch:** Preserved confusion matrix PNGs are from the Melbourne-area same-day framing. Fresh evaluation uses the Australia-wide `RainTomorrow` framing. Results are not directly comparable.
5. **Random seed:** The random split uses `random_state=42`. Results from a different random state will differ.
6. **Grid search selection:** The best model selected by grid search may vary depending on the dataset version and scikit-learn version. Hyperparameter selection should be documented alongside any fresh results.
