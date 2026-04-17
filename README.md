# Rainfall Prediction in Australia using Classical Machine Learning

This repository predicts whether it will rain the next day in Australia from tabular weather observations. The project keeps the original notebook-based work intact, but wraps it in a cleaner Python package, reproducible scripts, structured results, lightweight tests, and CI so it reads like a portfolio project instead of a one-off notebook submission.

## Quick Start

```bash
make setup
make train
make evaluate
```

## Why This Project Matters

This project demonstrates how to structure a production-ready machine learning pipeline beyond notebooks.

## Key Features

- Classical tabular classification for Australian weather data
- Comparison of Logistic Regression, Random Forest, and XGBoost
- Reusable preprocessing pipeline with categorical encoding and numerical scaling
- Config-driven train, evaluate, and predict scripts
- Saved model artifact workflow for later inference
- Preserved historical notebook and exported plots for reference

## Architecture Diagram

```mermaid
flowchart LR
    A[Weather Data] --> B[Preprocessing]
    B --> C[Model]
    C --> D[Evaluation]
    D --> E[Prediction]
```

## Project Structure

```text
rainfall-prediction-classifier/
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── configs/
│   ├── data.yaml
│   ├── inference.yaml
│   └── train.yaml
├── data/
│   ├── README.md
│   └── raw/
│       └── .gitkeep
├── models/
│   ├── .gitkeep
│   └── README.md
├── notebooks/
│   ├── exploration.ipynb
│   └── rainfall_prediction_classifier.ipynb
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
├── results/
│   ├── README.md
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── sample_predictions/
│       ├── README.md
│       └── example_input.json
├── scripts/
│   ├── run_evaluate.py
│   ├── run_predict.py
│   └── run_train.py
├── src/
│   └── rainfall_prediction/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── features.py
│       ├── pipeline.py
│       ├── predict.py
│       ├── train.py
│       ├── utils.py
│       └── visualization.py
└── tests/
    ├── test_config.py
    ├── test_pipeline.py
    └── test_predict.py
```

## Dataset

- Source: [Kaggle weather dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- Expected local CSV: `data/raw/weatherAUS.csv`
- Refactored package target column: `RainTomorrow`
- The raw dataset is not committed to the repository

Place the CSV at:

```text
data/raw/weatherAUS.csv
```

The historical notebook currently loads an IBM-hosted mirror named `weatherAUS-2.csv`. That notebook is preserved as-is for reference, while the refactored scripts expect a local Kaggle CSV for reproducibility.

## Methodology

The refactored package keeps the original modeling direction and preprocessing style:

1. Load tabular weather data.
2. Drop rows with missing values.
3. Derive a seasonal feature from `Date`.
4. One-hot encode categorical columns.
5. Scale numerical columns.
6. Split the data with a stratified train/test split.
7. Compare Logistic Regression, Random Forest, and XGBoost.

The preserved notebook also includes an exploratory variant that:

- narrows the data to Melbourne, Melbourne Airport, and Watsonia,
- renames `RainTomorrow` to `RainToday` for a same-day rainfall framing,
- exports Random Forest and Logistic Regression confusion matrices.

That notebook workflow is kept in `notebooks/rainfall_prediction_classifier.ipynb` as historical project context, but the package defaults back to the repository's public framing of predicting `RainTomorrow`.

## Results

The original repository README reported the following model comparison:

| Model | Train Accuracy | Test Accuracy |
| --- | ---: | ---: |
| Logistic Regression | 84.03% | 84.24% |
| Random Forest | 99.99% | 84.28% |
| XGBoost | 87.11% | **85.19%** |

These values are preserved in `results/metrics.json` and `results/model_comparison.csv` as historical repository results.

Important provenance note:

- The currently tracked notebook file and exported PNGs preserve Random Forest and Logistic Regression visual outputs.
- The current tracked notebook file does **not** contain XGBoost training cells or a structured metric table that re-verifies the XGBoost numbers from source.
- Because of that, the XGBoost row above is preserved from the original repository README rather than re-derived from the tracked notebook file.

## Why XGBoost Performed Best

On mixed tabular weather data, gradient boosting often captures non-linear thresholds and feature interactions more effectively than Logistic Regression, while generalizing better than an overfit Random Forest. That fits the historical pattern reported here: Random Forest nearly memorized the training data, while XGBoost achieved the strongest held-out accuracy.

## Sample Outputs

Preserved exported notebook artifacts:

Random Forest feature importance from the preserved notebook artifacts:

![Random Forest feature importance from the preserved notebook artifacts](results/feature_importance.png)

Additional preserved Logistic Regression confusion matrix:

`results/logistic_regression_confusion_matrix.png`

Additional preserved Random Forest confusion matrix:

`results/confusion_matrix.png`

Illustrative inference example after a trained model artifact is available:

Input:

```json
{
  "Date": "2017-06-01",
  "Location": "Melbourne",
  "MinTemp": 9.3,
  "MaxTemp": 15.2,
  "Humidity3pm": 71,
  "Pressure3pm": 1012.8,
  "RainToday": "Yes"
}
```

Output:

```text
RainTomorrow: Yes
```

The exact label depends on the trained model artifact and local dataset, but this is the intended prediction interface.

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Usage

Train all configured models and save the selected artifact:

```bash
python scripts/run_train.py --config configs/train.yaml
```

Evaluate the saved model and write structured outputs under `results/`:

```bash
python scripts/run_evaluate.py --config configs/inference.yaml
```

Run inference on the example payload:

```bash
python scripts/run_predict.py --config configs/inference.yaml --input results/sample_predictions/example_input.json
```

You can also use the `Makefile` shortcuts:

```bash
make setup
make install-dev
make test
make train
make evaluate
make predict
```

## Notebooks

- `notebooks/rainfall_prediction_classifier.ipynb`: preserved historical notebook workflow
- `notebooks/exploration.ipynb`: lightweight package-driven notebook stub that imports the refactored modules

## Limitations

- The dataset is not committed, so training and evaluation require a local CSV checkout.
- The preserved notebook and the original README do not line up perfectly: the notebook contains a Melbourne-area same-day rainfall variant, while the README frames the task as Australia-wide `RainTomorrow` prediction.
- No trained model artifact is committed by default.
- The committed results files are partly seeded from historical repository outputs until you rerun the scripts locally.

## Future Work

- Add stronger feature engineering beyond the season feature
- Use temporal validation instead of a purely random split
- Tune XGBoost and Random Forest more systematically
- Add probability calibration
- Package a lightweight API or dashboard around the saved model artifact

## License

This project is released under the [MIT License](LICENSE).
