# Dataset and Task Card

## Task Definition

| Field | Value |
|---|---|
| Task name | Next-day rainfall classification |
| Task type | Supervised binary classification |
| Target column | `RainTomorrow` |
| Positive class | `Yes` (rain will occur the next day) |
| Negative class | `No` (no rain the next day) |
| Input representation | Tabular daily weather observations (one row per station-day) |
| Evaluation metric (primary) | Accuracy (preserved historical metric) |
| Recommended additional metrics | F1, Precision, Recall (especially for the minority `Yes` class), Brier score, ROC-AUC |

## Dataset

| Field | Value |
|---|---|
| Name | Australian weather station observations (`weatherAUS.csv`) |
| Source reference | Kaggle: `https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package` (as documented in `data/README.md`) |
| Expected local path | `data/raw/weatherAUS.csv` |
| **Bundled in repository** | **No.** The dataset is not committed. The `.gitignore` excludes `data/raw/`. |
| **Trained model artifact bundled** | **No.** No trained artifact is committed by default. |
| Format | CSV |
| Approximate geographic scope | Weather stations across Australia |
| Approximate temporal scope | Multi-year (dataset-dependent; not verified in this pass without local file) |

## Key Features (from `results/sample_predictions/example_input.json` and package source)

The following columns are present in the example inference payload and are expected by the pipeline:

`Date`, `Location`, `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustDir`, `WindGustSpeed`, `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am`, `Temp3pm`, `RainToday`

The pipeline also derives a `Season` feature from `Date` before modelling, and drops `Date` itself.

## Public Package Framing vs. Archived Notebook Framing

**This mismatch is a documented provenance caveat, not something to minimise.**

| Dimension | Public Package (`src/` + `scripts/`) | Archived Notebook (`notebooks/rainfall_prediction_classifier.ipynb`) |
|---|---|---|
| Geographic scope | Australia-wide | Narrowed to Melbourne-area locations |
| Target column | `RainTomorrow` | Temporarily renamed to `RainToday` |
| Framing | Next-day rainfall prediction | Same-day rainfall variant |
| Dataset file expected | `data/raw/weatherAUS.csv` | References legacy mirror `weatherAUS-2.csv` |
| Runtime status | Maintained | Archived for provenance |

**Consequence for metric interpretation:** Confusion matrix PNG exports preserved in `results/` were produced by the archived notebook. They reflect the Melbourne-area same-day rainfall variant, not the Australia-wide `RainTomorrow` framing of the public package. Metrics derived from these PNGs carry this framing mismatch as a provenance caveat.

## Label and Target Caveats

- `RainTomorrow` is defined in the source dataset as whether ≥ 1 mm of rainfall was recorded the following day. The exact threshold definition should be confirmed from the Kaggle dataset documentation before downstream use.
- The class distribution in the preserved test partition (inferred from committed confusion matrices) is approximately **76% No-Rain, 24% Rain**. Accuracy is a misleading primary metric under this imbalance.
- `RainToday` (a binary yes/no indicator for same-day rainfall) is included as a model input feature. It may carry information relevant to `RainTomorrow` through atmospheric persistence, but it also represents a potential source of leakage — see `LEAKAGE_AND_SPLIT_RISK_AUDIT.md`.

## Missing Value Handling

- By default the pipeline **drops rows with missing values** (`drop_missing_rows: true` in `configs/data.yaml`).
- This strategy mirrors the earlier exploratory workflow in the archived notebook.
- **Caveat:** Dropping rows with missing values may induce selection bias if missingness is correlated with weather conditions (e.g., instrument failures during storms). The effective training population may not represent extreme weather events accurately.
- Alternative strategies (mean/median/mode imputation, indicator features for missingness) are not implemented in the current package but are recommended as future research directions.

## Geographic and Temporal Scope Caveats

- The Australia-wide framing covers diverse climatic regions (tropical, arid, temperate). A model trained on pooled station data may learn predominantly from the most-represented regions.
- No station-aware or region-aware evaluation has been performed in this pass.
- No temporal validation has been performed. The default split is random (`random_state=42`, stratified, `test_size=0.2`). Temporal autocorrelation in weather data means a random split likely inflates generalisation estimates relative to a chronological split.
- The archived notebook's Melbourne-area subset represents a substantially narrower climatic and geographic scope than the full dataset.

## Reproducibility Notes

| Step | Reproducible Without Dataset? | Notes |
|---|---|---|
| Package installation | Yes | `pip install -r requirements.txt -e .` |
| Unit tests | Yes | `pytest` — tests do not require dataset or model artifact |
| Evidence pack validation | Yes | `python tools/evidence/validate_research_pack.py` |
| Training | No | Requires `data/raw/weatherAUS.csv` |
| Evaluation | No | Requires dataset and trained artifact |
| Inference | No | Requires trained artifact |
| Reviewing preserved metrics | Yes | `results/metrics.json`, `results/model_comparison.csv` |

Full reproducibility of training and evaluation results requires obtaining the dataset from Kaggle and placing it at `data/raw/weatherAUS.csv`.
