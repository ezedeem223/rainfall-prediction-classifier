# Rainfall Prediction Classifier

## Overview
A Python machine learning project that predicts whether it will rain the next day in Australia using classical ML on tabular weather data.

## Architecture
- **Language**: Python 3.10
- **Package layout**: Installable package under `src/rainfall_prediction/`
- **Build system**: setuptools with `pyproject.toml`
- **Config system**: YAML configs in `configs/`, with environment variable overrides

## Key Components
- `src/rainfall_prediction/` — installable Python package
  - `config.py` — YAML config loading with env var overrides
  - `data.py` — dataset loading
  - `features.py` — preprocessing (scaling, one-hot encoding)
  - `pipeline.py` — sklearn pipeline builder
  - `train.py` — model training (Logistic Regression, Random Forest, XGBoost)
  - `evaluate.py` — evaluation metrics and reporting
  - `predict.py` — inference from saved model artifact
  - `visualization.py` — confusion matrix, feature importance plots
  - `utils.py` — shared utilities and custom exceptions
- `scripts/` — CLI entry points (run_train.py, run_evaluate.py, run_predict.py)
- `configs/` — YAML configs (train.yaml, inference.yaml, data.yaml)
- `tests/` — pytest suite (7 tests)
- `results/` — evaluation artifacts (metrics.json, plots, model comparison)
- `models/` — saved model artifacts (.joblib)
- `data/raw/` — dataset location (weatherAUS.csv, not committed)
- `notebooks/` — archived exploratory notebooks

## Setup
```bash
pip install -r requirements.txt -r requirements-dev.txt -e .
```

## Usage
```bash
# Train models (requires data/raw/weatherAUS.csv)
python scripts/run_train.py --config configs/train.yaml

# Evaluate saved model
python scripts/run_evaluate.py --config configs/inference.yaml

# Run inference
python scripts/run_predict.py --config configs/inference.yaml --input results/sample_predictions/example_input.json
```

## Environment Variables
See `.env.example`:
- `RAINFALL_DATASET_PATH` — path to dataset CSV
- `RAINFALL_MODEL_PATH` — path to saved model artifact
- `RAINFALL_RESULTS_DIR` — directory for evaluation outputs

## Dependencies
- scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn, joblib, PyYAML
- Dev: pytest, black, isort, ruff, pre-commit

## Notes
- Dataset (`data/raw/weatherAUS.csv`) is NOT committed — must be obtained from Kaggle
- No trained model artifact is committed — run `make train` after obtaining the dataset
- This is a pure Python ML project with no frontend or backend server
