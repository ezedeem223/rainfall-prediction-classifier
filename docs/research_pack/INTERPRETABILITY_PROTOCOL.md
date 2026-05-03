# Interpretability Protocol

## Important Notice

**No new SHAP or permutation-importance artifacts are generated in this pass.** This document defines the interpretability protocol to be followed in a future pass when the local dataset and trained model artifacts are available.

---

## 1. Existing Interpretability Artifacts

The following interpretability artifact is committed and verified:

| Artifact | Status | Notes |
|---|---|---|
| `results/feature_importance.png` | Present — preserved from earlier development run | Random Forest feature importance (Gini impurity-based); derived from the archived notebook; reflects Melbourne-area same-day rainfall framing |

No other interpretability artifacts (SHAP values, permutation importance, coefficient tables) are committed.

### What the Committed Feature Importance Artifact Can and Cannot Say

**Can say:**
- Which features the Random Forest assigns the highest Gini impurity reduction to, under the notebook's Melbourne-area framing.
- Relative feature ranking within that specific model and dataset subset.

**Cannot say:**
- Whether the same feature ranking holds for the Australia-wide `RainTomorrow` package framing.
- Whether features ranked highly are causally related to rainfall (importance reflects correlation, not causation).
- Whether the feature importance would hold under a different split (chronological vs. random) or a different random seed.
- Whether importance rankings from Random Forest Gini impurity are consistent with SHAP-based or permutation-based rankings (Gini impurity importance is known to favour high-cardinality features).

---

## 2. Permutation Importance Protocol

Permutation importance measures the decrease in model performance when a single feature's values are randomly shuffled, breaking its relationship with the target.

### Why Prefer Permutation Importance Over Gini Impurity Importance

- Gini impurity importance is biased toward high-cardinality features (e.g., continuous temperature values over binary `RainToday`).
- Permutation importance is model-agnostic and reflects actual contribution to held-out performance.
- Permutation importance computed on the test set is more informative than training-set importance for assessing generalisation-relevant features.

### Protocol (Future Pass — Requires Local Dataset and Trained Artifact)

```python
from sklearn.inspection import permutation_importance
import pandas as pd

# Load fitted pipeline and test data
result = permutation_importance(
    fitted_pipeline,
    X_test,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring="accuracy",
)

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}).sort_values("importance_mean", ascending=False)

importance_df.to_csv("results/interpretability/permutation_importance.csv", index=False)
```

### Required Output Files (Future Pass)

- `results/interpretability/permutation_importance.csv` — mean and std of importance across repeats.
- `results/interpretability/permutation_importance.png` — bar chart with error bars.

### Reviewer Caveats

- Permutation importance can be misleading when features are highly correlated (shuffling one correlated feature may have little effect because the model compensates via correlated partners).
- Report importance on the test set (not training set) to reflect generalisation-relevant feature contributions.
- Report for all three model families separately, not only for the best model.

---

## 3. SHAP Protocol for Tree Models

SHAP (SHapley Additive exPlanations) provides theoretically grounded feature attribution that satisfies consistency, local accuracy, and missingness properties. For tree-based models (Random Forest, XGBoost), SHAP TreeExplainer is efficient and exact.

**No SHAP artifacts have been generated in this pass.**

### Protocol (Future Pass — Requires Trained XGBoost or Random Forest Artifact)

```python
import shap

# For XGBoost classifier extracted from the pipeline
classifier = fitted_pipeline.named_steps["classifier"]
X_test_transformed = fitted_pipeline.named_steps["preprocessor"].transform(X_test)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(shap_values, X_test_transformed, feature_names=transformed_feature_names, show=False)
# Save figure to results/interpretability/shap_summary.png
```

### Required Output Files (Future Pass)

- `results/interpretability/shap_summary.png` — beeswarm summary plot.
- `results/interpretability/shap_bar.png` — mean absolute SHAP value bar chart.
- `results/interpretability/shap_values.npy` — raw SHAP value array (optional, large).

### Reviewer Caveats

- SHAP values explain the model's predictions, not the underlying data-generating process. High SHAP importance does not imply causal importance.
- For a model with temporal leakage (random split), SHAP values may reflect test-set patterns that would not hold under a chronological split. Run SHAP analysis after temporal validation is completed.
- Do not present SHAP feature rankings as causal weather drivers.

---

## 4. Coefficient Interpretation for Logistic Regression

When a trained Logistic Regression artifact is available, the model's coefficients provide a direct linear interpretability signal.

### Protocol (Future Pass — Requires Trained Artifact)

```python
import pandas as pd

lr_model = fitted_pipeline.named_steps["classifier"]
feature_names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": lr_model.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)

coef_df.to_csv("results/interpretability/logistic_regression_coefficients.csv", index=False)
```

### Reviewer Caveats

- Coefficients are in the scaled feature space (after StandardScaler). They are comparable only after scaling; raw coefficient magnitudes should not be compared across features with different scales.
- Logistic Regression coefficients are sensitive to multicollinearity. If temperature features are highly correlated (e.g., `MinTemp`, `MaxTemp`, `Temp9am`, `Temp3pm`), individual coefficients may be unreliable even if the combined prediction is stable.
- Report coefficients with confidence intervals if possible (e.g., via bootstrapping), not only point estimates.

### Required Output Files (Future Pass)

- `results/interpretability/logistic_regression_coefficients.csv`
- `results/interpretability/logistic_regression_coefficients.png`

---

## 5. Local Explanation Protocol for Individual Predictions

For individual inference cases (such as the example in `results/sample_predictions/example_input.json`), local explanations help understand why a specific prediction was made.

### Protocol (Future Pass)

For tree models, SHAP's `force_plot` or `waterfall_plot` provides local explanations:

```python
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test_transformed[0],
        feature_names=transformed_feature_names,
    )
)
```

For Logistic Regression, the contribution of each feature to the log-odds can be computed directly from scaled features × coefficients.

### Required Output Files (Future Pass)

- `results/interpretability/local_explanation_example.png` — SHAP waterfall for the example input.

---

## 6. Required Artifact Naming Convention

All interpretability artifacts must be placed under `results/interpretability/` with the following naming scheme:

| Artifact Type | File Name Pattern |
|---|---|
| Permutation importance table | `permutation_importance_{model_name}.csv` |
| Permutation importance plot | `permutation_importance_{model_name}.png` |
| SHAP summary plot | `shap_summary_{model_name}.png` |
| SHAP bar chart | `shap_bar_{model_name}.png` |
| LR coefficients table | `logistic_regression_coefficients.csv` |
| LR coefficients plot | `logistic_regression_coefficients.png` |
| Local explanation | `local_explanation_{model_name}_{sample_id}.png` |

---

## 7. No Causal Claims

Interpretability methods in this protocol explain the model's learned associations, not causal relationships in the weather system. Statements such as "humidity causes rainfall" or "pressure drives the RainTomorrow outcome" must not be made based on feature importance or SHAP values alone. All interpretability results must be qualified as describing the model's behaviour under the given training data and split design.
