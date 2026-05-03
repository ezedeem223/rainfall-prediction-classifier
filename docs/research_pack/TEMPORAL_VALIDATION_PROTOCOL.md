# Temporal Validation Protocol

## Important Notice

**No temporal validation results have been generated in this pass.** This document defines the protocol to be followed in a future pass when the local dataset is available. No temporal split metrics, no rolling-origin results, and no chronological evaluation outputs exist in this repository at this time.

---

## 1. Why Random Splits Are Insufficient for Weather Data

The default train/test split in this repository (`random_state=42`, stratified, `test_size=0.2`) is a random split. For tabular weather data with the structure of `weatherAUS.csv`, this introduces several methodological risks:

1. **Temporal autocorrelation:** Consecutive days at the same station are correlated. A random split places temporally adjacent rows in both training and test sets, allowing the model to exploit near-future information present in its training data.

2. **Label leakage through time:** `RainTomorrow` at time *t* becomes `RainToday` at time *t+1*. If rows *t* and *t+1* land in training and test respectively without chronological ordering, the model may implicitly learn the relationship through shuffled row ordering.

3. **Inflated generalisation estimates:** A model evaluated on a random 20% sample of the same time window as training will typically show better performance than one evaluated on a held-out future window, because the weather distribution is more similar.

4. **Distribution shift not captured:** A chronological split separates earlier data (training) from later data (test), exposing the model to potential distribution shift across time — which is the realistic deployment condition.

**A random split accuracy estimate is therefore an optimistic upper bound on true temporal generalisation, not a reliable measure of it.**

---

## 2. Recommended Chronological Split

### Protocol

1. Sort the full dataset by `Date` in ascending order.
2. Use all rows before a fixed cutoff date as training data, and all rows after as held-out test data.
3. The cutoff should be chosen so that the test period contains a representative seasonal cycle (at least 12 months recommended).
4. Apply the same preprocessing pipeline (fitted on training set only) to the test set.

### Configuration

Add to `configs/data.yaml` (future pass):

```yaml
split:
  method: chronological          # replace random with chronological
  cutoff_date: "2016-01-01"     # example; adjust based on dataset date range
  test_size: null                # not used when method is chronological
  random_state: null
  stratify: false
```

### Expected Output Files (future pass)

- `results/temporal/chronological_split_metrics.json`
- `results/temporal/chronological_split_classification_report.txt`
- `results/temporal/chronological_split_confusion_matrix.png`

---

## 3. Rolling-Origin Validation Design

Rolling-origin (walk-forward) validation is the most rigorous evaluation design for time-series-like tabular data. It simulates the real prediction scenario: train on all available history, predict the next window, then expand the training window and repeat.

### Protocol

1. Define an initial training window (e.g., first N years of data).
2. Define a forecast horizon (e.g., 30-day or 90-day windows).
3. For each fold:
   - Fit the full sklearn pipeline on the current training window.
   - Evaluate on the next forecast window.
   - Expand the training window by the forecast horizon.
   - Repeat until the dataset is exhausted.
4. Aggregate metrics (mean and standard deviation) across folds.

### Minimum Requirements Before Reporting

- At least 5 rolling folds.
- Each fold's test window must be entirely after all training data in that fold.
- Pipeline must be refit from scratch at each fold (no partial fitting leakage).
- Metrics must be reported per-fold, not only as an aggregate.

### Expected Output Files (future pass)

- `results/temporal/rolling_origin_fold_metrics.csv`
- `results/temporal/rolling_origin_summary.json`
- `results/temporal/rolling_origin_accuracy_over_time.png`

---

## 4. Station/Location-Aware Temporal Splits

Weather stations have distinct climatic profiles. A random or even chronological split that pools all stations may allow the model to learn location-specific priors. A more rigorous design holds out entire stations.

### Protocol Options

**A. Station hold-out:** Reserve a subset of stations entirely for evaluation. The model never sees those stations during training.

**B. Region-stratified chronological split:** Split chronologically within each climatic region, then evaluate across regions.

**C. Leave-one-region-out:** Train on all regions except one; evaluate on the held-out region.

### Recommended Minimum

For a credible academic evaluation, at minimum report performance broken down by `Location` or climatic region, even within a standard chronological split.

### Expected Output Files (future pass)

- `results/temporal/station_holdout_metrics.json`
- `results/temporal/per_location_metrics.csv`

---

## 5. Leakage Checks Involving Temporal Features

Before running any temporal validation, verify the following:

| Feature | Leakage Risk | Check |
|---|---|---|
| `Date` | Dropped before modelling — low direct risk | Confirm `drop_date_column: true` in config |
| `Season` | Derived from `Date` — encodes month/quarter seasonality only; low risk | Confirm derivation uses only the month component |
| `RainToday` | Derived from same-day rainfall; correlated with `RainTomorrow` through atmospheric persistence | Verify this is not derived from `RainTomorrow` itself |
| `RainTomorrow` | Target; must not appear as a feature | Confirm it is dropped from the feature matrix before preprocessing |
| Lag features (if added) | Any lag feature derived from future rows causes direct leakage | Ensure lags are computed before the chronological split and only use past values |
| Scaler/encoder fit | Must be fit on training data only | Confirm `Pipeline.fit` is called on training set; `transform` only on test |

---

## 6. How to Report Temporal Validation Results

When temporal validation is eventually performed, results must be reported with:

1. **Split method:** exact dates or fold boundaries.
2. **Training period:** start and end date.
3. **Test period:** start and end date.
4. **Metrics:** accuracy, F1 (Yes class), precision, recall, Brier score.
5. **Comparison to random split:** report both to quantify the inflation effect.
6. **Per-fold variance** (for rolling-origin): mean ± standard deviation.
7. **Dataset version:** Kaggle dataset version or download date.

---

## 7. Example Commands (Future Pass — Requires Local Dataset)

These commands will not execute correctly until `data/raw/weatherAUS.csv` is present and a trained artifact exists.

```bash
# Train with chronological split (once config is updated)
python scripts/run_train.py --config configs/train_temporal.yaml

# Evaluate on chronological held-out window
python scripts/run_evaluate.py --config configs/inference_temporal.yaml

# Rolling-origin cross-validation (requires additional script)
python scripts/run_rolling_validation.py --config configs/train_temporal.yaml --folds 5
```

---

## 8. Minimum Artifact Requirements Before Claiming Temporal Robustness

A claim of temporal robustness requires **all** of the following:

- [ ] Chronological split (not random) used for at least the primary reported result.
- [ ] Test window is entirely after all training data.
- [ ] Rolling-origin validation with at least 5 folds reported.
- [ ] Per-fold metric variance documented.
- [ ] Station/location breakdown reported.
- [ ] Leakage check completed and documented.
- [ ] Pipeline refit at each fold (no data leakage through pre-fitted transformers).
- [ ] Results files committed under `results/temporal/`.
