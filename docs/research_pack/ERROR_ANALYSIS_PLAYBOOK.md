# Error Analysis Playbook

## Important Notice

No new error analysis artifacts have been generated in this pass. This playbook defines the structured error analysis protocol to be followed in a future pass when the local dataset and trained model artifacts are available. All examples and breakdowns below are protocol definitions, not results from a completed analysis.

---

## 1. False Positives and False Negatives

For the binary classification task of predicting `RainTomorrow`:

| Error Type | Definition | Consequence |
|---|---|---|
| False Positive (FP) | Model predicts `Yes` (Rain); actual outcome is `No` (No Rain) | Unnecessary rain preparation; low operational cost in most research contexts |
| False Negative (FN) | Model predicts `No` (No Rain); actual outcome is `Yes` (Rain) | Missed rain event; higher cost in early-warning contexts |

### Preserved Confusion Matrix Values (Archived Test Partition)

From `results/metrics.json` (`preserved_notebook_exports`) — derived from Melbourne-area same-day rainfall framing. Not from the Australia-wide `RainTomorrow` package framing.

**Random Forest:**

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 1098 (TN) | 56 (FP) |
| **Actual Yes** | 174 (FN) | 184 (TP) |

- False Positive Rate: 56 / (1098 + 56) = **4.9%** — the model rarely predicts rain when there is none.
- False Negative Rate: 174 / (174 + 184) = **48.6%** — the model misses nearly half of actual rain events.

**Logistic Regression:**

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 1071 (TN) | 83 (FP) |
| **Actual Yes** | 174 (FN) | 184 (TP) |

- False Positive Rate: 83 / (1071 + 83) = **7.2%**
- False Negative Rate: 174 / (174 + 184) = **48.6%** — identical to Random Forest; both miss the same proportion of rain events.

### Interpretation

Both preserved models show a strong bias toward predicting No Rain. This is consistent with the class imbalance (~76% No-Rain in the archived test partition). The models conservatively avoid false positives at the expense of false negatives — a pattern that would be problematic in any rainfall early-warning application.

**Caveat:** These values reflect the Melbourne-area same-day framing from the archived notebook, not the Australia-wide `RainTomorrow` package framing. Error patterns in the full package may differ.

---

## 2. Season-Wise Breakdown Protocol

Weather patterns differ strongly by season. Error rates are likely season-dependent.

### Protocol (Future Pass — Requires Local Dataset and Trained Artifact)

1. Add season labels to the test partition using the same `Season` derivation as the pipeline.
2. Group test predictions by `Season`.
3. Compute confusion matrix, precision, recall, and F1 per season.

```python
import pandas as pd
from sklearn.metrics import classification_report

for season in ["Summer", "Autumn", "Winter", "Spring"]:
    mask = (X_test["Season"] == season)
    print(f"\n=== {season} ===")
    print(classification_report(y_test[mask], y_pred[mask], target_names=["No", "Yes"]))
```

### Expected Patterns (Hypothesis — Not Yet Verified)

- **Winter:** Higher base rate of rain in southern Australia; model may have higher recall for `Yes` class.
- **Summer:** More variable rainfall (thunderstorms, cyclones in north); potentially lower model precision.
- **Autumn/Spring:** Transition seasons; model may be less confident and produce more borderline probabilities.

### Required Output Files (Future Pass)

- `results/error_analysis/season_wise_metrics.csv`
- `results/error_analysis/season_wise_confusion_matrices.png`

---

## 3. Location-Wise Breakdown Protocol

Australia has highly diverse climatic regions. Error rates will vary substantially by location.

### Protocol (Future Pass)

```python
for location in X_test["Location"].unique():
    mask = (X_test["Location"] == location)
    if mask.sum() < 20:  # skip locations with too few test examples
        continue
    print(f"\n=== {location} ===")
    print(classification_report(y_test[mask], y_pred[mask], target_names=["No", "Yes"]))
```

### Expected Patterns (Hypothesis — Not Yet Verified)

- **Tropical north (Darwin, Cairns):** High base rate of wet-season rain; model may achieve high recall for `Yes` but at risk of memorising seasonal climatology.
- **Arid interior (Alice Springs):** Very low base rate; model likely predicts `No` almost always; recall for `Yes` expected to be poor.
- **Southern coast (Melbourne, Sydney, Adelaide):** Moderate rain frequency; most challenging discrimination task.

### Required Output Files (Future Pass)

- `results/error_analysis/location_wise_metrics.csv`
- `results/error_analysis/location_wise_false_negative_rate.png`

---

## 4. Rainfall Intensity Proxy Analysis

The dataset includes a `Rainfall` column (same-day rainfall in mm) and features such as `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, and `Cloud9am`, `Cloud3pm` that may serve as intensity proxies.

### Protocol (Future Pass)

Bin the test set by same-day rainfall amount (`Rainfall` column if available) and compute error rates per bin.

```python
bins = [0, 0.2, 1, 5, 20, float("inf")]
labels = ["Trace/dry", "Very light", "Light", "Moderate", "Heavy"]
X_test["rainfall_bin"] = pd.cut(X_test["Rainfall"], bins=bins, labels=labels)

for bin_label in labels:
    mask = (X_test["rainfall_bin"] == bin_label)
    if mask.sum() < 10:
        continue
    print(f"\n=== Rainfall bin: {bin_label} (n={mask.sum()}) ===")
    print(classification_report(y_test[mask], y_pred[mask], target_names=["No", "Yes"]))
```

### Expected Patterns (Hypothesis — Not Yet Verified)

- On heavy-rain days, `RainTomorrow=Yes` is likely more common; model recall for `Yes` may improve.
- On trace/dry days, the model is likely in the `No`-dominant regime; false negatives may concentrate here.

### Required Output Files (Future Pass)

- `results/error_analysis/rainfall_intensity_proxy_metrics.csv`

---

## 5. Edge Cases Around Key Features

The following feature combinations are expected to produce borderline or error-prone predictions:

| Edge Case | Expected Difficulty | Analysis Approach |
|---|---|---|
| High humidity (`Humidity3pm > 80`) + `RainToday=No` | Model may underpredict rain | Filter test set; compare FN rate |
| Low pressure (`Pressure3pm < 1005`) + no rain today | Approaching-storm scenario; model may miss | Filter; compare FN rate |
| Strong wind gust (`WindGustSpeed > 60`) | May signal frontal system; model sensitivity unknown | Filter; compare recall |
| `RainToday=Yes` + `Cloud3pm=8` (overcast) | Strong prior for rain tomorrow | Check if FN rate drops here |
| Summer + inland location + low humidity | Near-zero base rate; expect near-100% TN, low TP | Filter; check for trivial predictions |

### Protocol (Future Pass)

For each edge case, define a boolean filter on the test set and report the confusion matrix and class-level metrics for that subgroup.

### Required Output Files (Future Pass)

- `results/error_analysis/edge_case_subgroup_metrics.csv`

---

## 6. Subgroup Analysis Framework

Beyond the above breakdowns, a complete error analysis should include:

| Subgroup Axis | Grouping Variable | Metric Focus |
|---|---|---|
| Temporal | Year / Month | FN rate drift over time |
| Geographic | Location / State | FP/FN rate by region |
| Seasonal | Season | Recall for Yes class |
| Rain intensity | Rainfall bin | Model sensitivity by intensity |
| Atmospheric pressure | Pressure9am / Pressure3pm bins | Precision/recall sensitivity |
| Humidity level | Humidity3pm bins | False negative concentration |

### Required Output Files (Future Pass)

- `results/error_analysis/subgroup_analysis_summary.csv`

---

## 7. Temporal Drift Analysis

Weather patterns may shift over the multi-year period of the dataset. A model trained on earlier years may perform worse on later years if climatic conditions change.

### Protocol (Future Pass — Requires Chronological Split)

After implementing the chronological split (see `TEMPORAL_VALIDATION_PROTOCOL.md`):

1. Divide the test window into annual or semi-annual sub-periods.
2. Compute accuracy, recall (Yes), and F1 (Yes) for each sub-period using the fixed trained model (no retraining).
3. Plot metrics over time to identify drift.

### Required Output Files (Future Pass)

- `results/error_analysis/temporal_drift_metrics.csv`
- `results/error_analysis/temporal_drift_f1_yes.png`

---

## 8. Artifact Checklist for Future Error Analysis

- [ ] Test-set predictions saved with ground truth and predicted probabilities.
- [ ] Season labels added to test partition.
- [ ] Location labels available in test partition (not dropped before error analysis).
- [ ] Rainfall amount column retained in test partition for intensity proxy analysis.
- [ ] Season-wise confusion matrices computed and saved.
- [ ] Location-wise F1 (Yes class) computed and saved.
- [ ] Subgroup metrics for at least 3 feature-based edge cases computed and saved.
- [ ] Temporal drift analysis completed (requires chronological split).
- [ ] All error analysis outputs placed under `results/error_analysis/`.

---

## 9. How to Document Failure Cases Without Overclaiming

When documenting specific failure cases or error patterns, use the following guidelines:

| Practice | Correct Approach |
|---|---|
| Describing a high FN rate | "The model misses approximately 49% of actual rain events in the preserved archived test partition, under a random split and Melbourne-area framing." |
| Describing a subgroup finding | "In the [subgroup], the false negative rate is [X%] — higher than the overall rate of [Y%]. This pattern requires confirmation under a chronological split." |
| Describing edge cases | "Observations with [feature condition] show higher false negative rates in the subgroup analysis. This is consistent with [meteorological hypothesis], but further validation is required." |
| Generalising findings | "Error patterns from the archived Melbourne-area test partition may not generalise to the Australia-wide package framing without rerunning analysis on the full dataset." |

Never present error analysis results from the archived notebook's test partition as directly applicable to the Australia-wide package without explicit restatement of the framing caveat.
