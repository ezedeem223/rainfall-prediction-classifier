# Calibration and Thresholding Protocol

## Important Notice

**No calibration results have been generated in this pass.** This document defines the calibration and thresholding protocol to be followed in a future pass when the local dataset and trained model artifacts are available. Model probabilities in this repository should not be treated as calibrated unless a calibration procedure has been explicitly performed and documented.

---

## 1. Why Accuracy Alone Is Insufficient

The preserved metrics for this repository consist entirely of accuracy values. Accuracy has several well-known limitations for binary classification tasks, particularly for rainfall prediction:

1. **Class imbalance sensitivity:** The preserved confusion matrices suggest approximately 76% No-Rain and 24% Rain in the archived test partition. A trivial model predicting "No Rain" for every observation would achieve ~76% accuracy, outperforming the preserved Logistic Regression result (83%) by a smaller margin than it appears. Accuracy does not convey whether the model adds value beyond a majority-class baseline.

2. **Threshold dependence:** Accuracy is computed at the default 0.5 decision threshold. For rainfall prediction, the costs of false positives (unnecessarily predicted rain) and false negatives (missed rain) are asymmetric. The optimal threshold depends on the cost structure of the downstream use case, not on the midpoint of probability.

3. **No probabilistic information:** Accuracy does not capture whether model confidence (predicted probability) is well-calibrated. A model that predicts 0.9 probability of rain when it actually rains 50% of the time in those cases has poor calibration, regardless of its accuracy.

4. **No discrimination vs. calibration separation:** A model can be well-discriminating (good AUC) but poorly calibrated (systematically over- or under-confident), or vice versa. Accuracy conflates both.

---

## 2. Probability Calibration

### What Calibration Measures

A model is well-calibrated if, among all observations where it predicts probability *p* of rain, approximately *p* × 100% of those observations actually have rain. A reliability diagram (calibration curve) visualises this.

### Expected Calibration Behaviour by Model Family

| Model | Expected Calibration | Notes |
|---|---|---|
| Logistic Regression | Moderate to good | Linear models tend to produce better-calibrated probabilities by construction |
| Random Forest | Poor | Tends to predict probabilities near 0.5 due to averaging; Platt scaling or isotonic regression recommended |
| XGBoost | Moderate | Gradient boosting probabilities are generally better than RF but still benefit from calibration |

### Protocol (Future Pass — Requires Trained Artifact and Local Dataset)

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

# Option A: Post-hoc calibration with Platt scaling
calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
calibrated_model.fit(X_val, y_val)  # use a separate calibration set

# Option B: Isotonic regression
calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
calibrated_model.fit(X_val, y_val)

# Reliability diagram
prob_true, prob_pred = calibration_curve(y_test, y_prob_positive, n_bins=10)
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability Diagram")
plt.savefig("results/calibration/reliability_diagram_{model_name}.png")
```

### Required Output Files (Future Pass)

- `results/calibration/reliability_diagram_{model_name}.png`
- `results/calibration/calibration_metrics.json` — Expected Calibration Error (ECE), Maximum Calibration Error (MCE)

---

## 3. Brier Score

The Brier Score is the mean squared error between predicted probabilities and true binary outcomes. Lower is better; 0 is perfect; 0.25 is the score of an uninformative model predicting 0.5 for all observations.

### Formula

Brier Score = (1/N) × Σ (p_i − y_i)²

where p_i is the predicted probability of rain and y_i ∈ {0, 1} is the true outcome.

### Protocol (Future Pass)

```python
from sklearn.metrics import brier_score_loss

brier = brier_score_loss(y_test, y_prob_positive, pos_label=1)
print(f"Brier Score: {brier:.4f}")
```

### Required Output

- `results/calibration/brier_scores.json` — Brier score per model, before and after calibration.

### Reference Baselines

When reporting Brier Score, include:

- **Climatological baseline:** Brier score of a model that always predicts the training-set prevalence of `Yes`.
- **Majority-class baseline:** Brier score of always predicting 0 (No Rain).

---

## 4. Reliability Curves

A reliability diagram plots the fraction of positive outcomes (actual rain) against the mean predicted probability, binned into N intervals (typically 10). A perfectly calibrated model produces a diagonal line.

### Interpretation

- Points above the diagonal: model is under-confident (predicts lower probability than observed).
- Points below the diagonal: model is over-confident (predicts higher probability than observed).
- Clustering near 0 or 1: model has low uncertainty; assess whether this is appropriate or overconfident.

---

## 5. Threshold Selection

The default 0.5 threshold is arbitrary and may not be optimal for rainfall prediction. Threshold optimisation should be based on a defined cost function or target metric.

### Threshold Selection Protocol

1. Compute precision, recall, and F1 at multiple thresholds (0.1 to 0.9 in steps of 0.05).
2. Plot the precision-recall curve and identify the operating point that satisfies the research requirement.
3. For rainfall early warning applications, recall of the `Yes` class (sensitivity to rain events) is typically more important than precision.
4. For balanced general-purpose evaluation, the F1 score of the minority `Yes` class is a more informative threshold selection criterion than overall accuracy.

### Protocol (Future Pass)

```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_positive)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
optimal_threshold = thresholds[np.argmax(f1_scores[:-1])]
```

### Required Output Files (Future Pass)

- `results/calibration/precision_recall_curve_{model_name}.png`
- `results/calibration/threshold_metrics_{model_name}.csv` — precision, recall, F1 at each threshold.

---

## 6. Class Imbalance Considerations

**Verified from preserved confusion matrices (archived test partition):**

| Class | Count | Proportion |
|---|---|---|
| No Rain (0) | ~1154 | ~76.3% |
| Rain (1) | ~358 | ~23.7% |

This imbalance level (approximately 3:1) is sufficient to make accuracy misleading. Key implications:

- A majority-class baseline achieves ~76% accuracy with zero rainfall prediction capability.
- The preserved Logistic Regression recall for the `Yes` class is 0.5140 (from `classification_report.txt`) — the model correctly identifies only ~51% of actual rain events.
- Random Forest `Yes`-class recall is also 0.5140 — despite higher train accuracy, it does not improve minority-class recall.
- **F1 of the `Yes` class is a more informative metric than overall accuracy** for this task.

### Recommended Mitigation Strategies (Future Pass)

- `class_weight: balanced` in Logistic Regression (already in `configs/train.yaml` grid).
- `scale_pos_weight` in XGBoost (ratio of negative to positive class count).
- Threshold adjustment post-training to improve minority-class recall.
- Oversampling (SMOTE) or undersampling as a data-level intervention.
- Report all metrics stratified by class, not only overall accuracy.

---

## 7. Required Future Artifacts Before Claiming Calibrated Probability Estimates

- [ ] Calibration performed using a held-out calibration set (not the test set).
- [ ] Reliability diagram plotted for each model.
- [ ] Brier Score computed before and after calibration.
- [ ] Expected Calibration Error (ECE) computed.
- [ ] Calibrated models saved as separate artifacts from uncalibrated models.
- [ ] All calibrated probability reports clearly marked as post-calibration.

---

## 8. Safe Wording

| Context | Safe | Unsafe |
|---|---|---|
| Model output probabilities | "raw predicted probabilities; calibration not yet performed" | "calibrated probability of rainfall" |
| Accuracy metric | "accuracy on random split; class-imbalanced task" | "strong model performance" |
| F1 / Recall | "preserved F1 (Yes class) from archived test partition" | "reliable minority-class detection" |
| Threshold | "default 0.5 threshold; not optimised for this task" | "optimal decision threshold" |
