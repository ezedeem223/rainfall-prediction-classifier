"""Model pipeline helpers."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .utils import MissingDependencyError

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - exercised via graceful fallback
    XGBClassifier = None

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def humanize_model_name(model_name: str) -> str:
    """Return a friendly model name for reports and plots."""
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


def build_estimator(model_name: str, random_state: int = 42) -> Any:
    """Instantiate a classifier by name."""
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state)
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise MissingDependencyError(
                "xgboost is not installed. Install the runtime dependencies to train this model."
            )
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_model_pipeline(preprocessor: Any, model_name: str, random_state: int = 42) -> Pipeline:
    """Create an end-to-end sklearn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", build_estimator(model_name=model_name, random_state=random_state)),
        ]
    )


def get_feature_names(model_pipeline: Pipeline) -> list[str]:
    """Return transformed feature names from a fitted pipeline."""
    names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [name.split("__", 1)[-1] for name in names]
