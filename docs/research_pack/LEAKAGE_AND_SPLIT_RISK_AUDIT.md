# Leakage and Split Risk Audit

This document is an academic reviewer-oriented audit of data leakage risks, split design limitations, and methodological vulnerabilities in this repository. It is intended to make risks transparent, not to overstate them or to dismiss the preserved results.

---

## 1. Target Leakage Risks

### 1.1 `RainToday` Feature

**Risk level:** Medium.

`RainToday` is a binary indicator (`Yes`/`No`) for whether rainfall occurred on the same day as the observation. The target is `RainTomorrow` — whether rainfall occurs the following day.

- `RainToday` and `RainTomorrow` are correlated through atmospheric persistence (consecutive rainy days are common in many Australian regions).
- This correlation is genuine and not necessarily leakage — knowing it rained today is legitimately useful for predicting tomorrow.
- However, if `RainToday` is derived from the same rainfall total that defines `RainTomorrow` at *t-1* (i.e., yesterday's `RainTomorrow` becomes today's `RainToday`), then it represents a direct structural relationship that may be inflated in a random split where both days' rows appear in training or test.
- **Mitigation:** In a chronological split with sufficient gap between training and test windows, this relationship is preserved but not exploited through future-data leakage.

### 1.2 Feature Columns Derived from Target-Adjacent Variables

**Risk level:** Low to medium (depends on dataset preprocessing).

Features such as `Rainfall` (the actual quantity in mm) may be closely related to `RainToday`, which in turn links to `RainTomorrow`. If `Rainfall` encodes same-day rain quantity and `RainToday` is derived from it, both carry overlapping information.

**Mitigation:** Inspect the dataset column definitions carefully before training. Features derived from or directly encoding the target observation window should be documented and potentially ablated.

### 1.3 Scaler and Encoder Fitting

**Risk level:** Low in the current implementation.

The preprocessing pipeline (scaler, one-hot encoder) is fitted inside a scikit-learn `Pipeline` object using `Pipeline.fit(X_train, y_train)`. This means the scaler and encoder parameters are estimated only from training data.

- **Verified:** The pipeline structure in `src/rainfall_prediction/pipeline.py` and `src/rainfall_prediction/features.py` uses a standard sklearn `Pipeline`, which correctly applies fit-only to training data and transform-only to test data.
- **Residual risk:** If any feature engineering step (e.g., season derivation, missing-value imputation in future extensions) is applied to the full dataset before splitting, it must be verified to use only training-observable information.

---

## 2. Temporal Leakage Risks

### 2.1 Random Split on Time-Series-Like Data

**Risk level:** High for temporal generalisation claims.

The default split is a stratified random split (`random_state=42`, `test_size=0.2`). For weather data collected over time at fixed stations:

- Temporally adjacent rows (consecutive days at the same station) may land in both training and test sets.
- The model effectively sees "near-future" patterns during training because shuffled rows from the test window are mixed into training.
- This causes the evaluated accuracy to be an optimistic upper bound on true temporal generalisation.
- **The preserved accuracy metrics (Logistic Regression 84.24%, Random Forest 84.28%, XGBoost 85.19%) must be interpreted as random-split results, not temporal-split results.**

**Mitigation:** Implement chronological split as described in `TEMPORAL_VALIDATION_PROTOCOL.md`. Report both random-split and chronological-split results side by side.

### 2.2 `Date` Column

**Risk level:** Low.

The `Date` column is dropped before modelling (`drop_date_column: true` in `configs/data.yaml`). Only the derived `Season` feature is retained. `Season` encodes seasonality (likely a categorical month-group mapping), which is non-leaking as it represents the observation's own time period.

**Residual risk:** If `Season` is encoded in a way that inadvertently correlates with temporal position within the dataset (e.g., assigns a monotonically increasing encoding), it could introduce weak temporal leakage. Verify that `Season` is purely categorical (Summer/Autumn/Winter/Spring) and not ordinal or continuous.

### 2.3 Future Lag Features

**Risk level:** Potentially high if added incorrectly.

No lag features are currently implemented. If lag features are added in future passes (e.g., yesterday's humidity, 3-day rolling average), they must be computed on the original time-ordered dataset before splitting, and they must only use past-observable values. Lag features derived after shuffling or derived from post-split data would introduce direct future leakage.

---

## 3. Location Leakage Risks

### 3.1 Location as a Feature

**Risk level:** Medium.

`Location` is one-hot encoded and used as a feature. Because the random split does not stratify by station, the model may partially memorise location-specific climatological priors (e.g., Darwin is rainy, Alice Springs is dry) rather than learning generalisable meteorological patterns.

- A model that learns "this row is from Darwin, therefore predict Yes" is exploiting location identity, not meteorological signal.
- This effect is not leakage in the strict sense, but it inflates apparent generalisation if the test set includes the same locations as training.

**Mitigation:** Run a station hold-out evaluation (reserve 2–3 stations entirely for testing) to separate location-memorisation from pattern generalisation.

### 3.2 Notebook/Package Framing Mismatch

**Risk level:** Medium for metric interpretation.

The archived notebook narrows the analysis to Melbourne-area stations and uses a same-day rainfall framing. The public package is Australia-wide with `RainTomorrow`. Confusion matrix PNG exports in `results/` reflect the notebook's narrower framing.

- Metrics derived from these PNGs are not directly comparable to Australia-wide `RainTomorrow` package results.
- Presenting the notebook PNG-derived metrics as package performance metrics would be misleading.

**Mitigation:** Clearly annotate all references to preserved metrics with their framing context. See `METRIC_PROVENANCE_MATRIX.md`.

---

## 4. Missing-Value Handling Risks

**Risk level:** Medium.

The default strategy drops all rows containing any missing value. This introduces several risks:

1. **Selection bias:** If missing values are more common during adverse weather conditions (e.g., instrument failures during storms), the effective training dataset skews toward mild/normal weather days. The model may perform poorly on extreme events.
2. **Class balance shift:** If the `Yes` class (rainy days) has proportionally more missing features than the `No` class, dropping missing rows changes the effective class balance.
3. **Reproducibility:** The number of dropped rows depends on the dataset version and any upstream preprocessing. Results may differ across dataset downloads.

**Mitigation:** Report the number of rows dropped and the before/after class distribution. Compare row-drop against imputation strategies in a future pass.

---

## 5. Random Split Limitations Summary

| Limitation | Severity | Mitigation |
|---|---|---|
| Temporal autocorrelation not respected | High | Chronological split |
| Consecutive-day rows may leak across splits | High | Chronological split with gap |
| Location distribution identical in train/test | Medium | Station hold-out |
| Class balance preserved but not temporal balance | Medium | Temporal stratification |
| Inflated accuracy relative to future-window generalisation | High | Report both split types |

---

## 6. Overfitting Signs — Random Forest (Verified)

**Verified from `results/metrics.json` and `results/model_comparison.csv`:**

- Random Forest `train_accuracy = 0.9999`
- Random Forest `test_accuracy = 0.8428`
- Generalisation gap: **0.1571**

This gap indicates near-complete memorisation of the training set. Possible causes:

- Unlimited tree depth (default `max_depth=None` allows trees to grow until pure leaves).
- Insufficient `min_samples_split` or `min_samples_leaf` constraints.
- The grid search in `configs/train.yaml` includes `max_depth: [null, 10, 20]` — if the best configuration found by cross-validation selects `null`, the final model will overfit.

**Implication for metric reporting:** The Random Forest `test_accuracy=0.8428` should not be presented as a reliable generalisation estimate without:
- Confirming the selected depth configuration.
- Running a chronological split evaluation.
- Noting the 0.1571 train/test gap explicitly.

---

## 7. XGBoost Provenance Risk (Verified)

**Verified from `results/model_comparison.csv`:** The XGBoost row carries the note: "current tracked notebook file does not include XGBoost cells."

**Risk:** Presenting XGBoost metrics (`train_accuracy=0.8711`, `test_accuracy=0.8519`) as notebook-verified or freshly reproduced would misrepresent their evidential status. These numbers originate from the pre-refactor README and cannot be traced to a committed training run or notebook cell.

**Safe posture:** Present XGBoost results as "preserved from earlier repository reporting" with an explicit provenance gap disclosure.

---

## 8. Feature Engineering Risks

| Feature | Risk | Notes |
|---|---|---|
| `Season` (derived from `Date`) | Low | Categorical seasonality; non-leaking if purely month-based |
| One-hot encoded `Location` | Medium | Location-memorisation risk in pooled model |
| One-hot encoded `WindGustDir`, `WindDir9am`, `WindDir3pm` | Low | Directional categories; no leakage concern |
| One-hot encoded `RainToday` | Medium | Structural relationship to target; see Section 1.1 |
| Numerical features (temperatures, pressure, humidity, etc.) | Low | Standard meteorological predictors; no leakage concern |

---

## 9. Mitigation Checklist

- [ ] Implement and report chronological split alongside random split.
- [ ] Implement rolling-origin validation (minimum 5 folds).
- [ ] Report per-location performance breakdown.
- [ ] Run station hold-out evaluation for at least 2–3 held-out stations.
- [ ] Document the number of rows dropped for missing values and the resulting class distribution.
- [ ] Verify that the `Season` feature is purely categorical and not ordinal/continuous.
- [ ] Confirm that `RainTomorrow` is not present in any feature column at training time.
- [ ] Document the selected model hyperparameters (especially Random Forest `max_depth`) from the grid search result.
- [ ] Report Random Forest train/test gap explicitly in all metric summaries.
- [ ] Add XGBoost provenance gap disclaimer to all metric references.
- [ ] Perform probability calibration before interpreting model outputs as probabilistic forecasts.

---

## 10. Safe vs. Unsafe Wording

| Context | Safe | Unsafe |
|---|---|---|
| Citing preserved accuracy | "preserved historical test accuracy of 84.24% under a random split" | "achieves 84.24% accuracy" |
| Describing Random Forest | "near-complete training memorisation (train accuracy 99.99%)" | "strong generalisation" |
| Describing XGBoost | "preserved from earlier reporting; provenance gap applies" | "best model with 85.19% accuracy" |
| Describing split design | "random stratified split; temporal validation not yet performed" | "rigorously evaluated" |
| Describing leakage | "leakage risk present; not yet audited with chronological split" | "no data leakage" |
| Describing calibration | "probabilities not calibrated; raw outputs only" | "calibrated probability estimates" |
