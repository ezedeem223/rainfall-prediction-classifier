"""Feature engineering and preprocessing helpers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def date_to_season(value: pd.Timestamp | str) -> str:
    """Map a date to its Australian meteorological season."""
    date = pd.Timestamp(value)
    month = date.month
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


def infer_feature_columns(
    frame: pd.DataFrame,
    numerical_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve model feature groups, inferring them when omitted in config."""
    if numerical_features:
        numeric = [column for column in numerical_features if column in frame.columns]
    else:
        numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()

    if categorical_features:
        categorical = [column for column in categorical_features if column in frame.columns]
    else:
        categorical = [column for column in frame.columns if column not in numeric]

    return numeric, categorical


def build_preprocessor(
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    scaler: str = "standard",
    handle_unknown: str = "ignore",
) -> ColumnTransformer:
    """Create the shared preprocessing transformer."""
    if scaler != "standard":
        raise ValueError(f"Unsupported scaler: {scaler}")

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numerical_features:
        numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
        transformers.append(("num", numeric_pipeline, list(numerical_features)))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown=handle_unknown))]
        )
        transformers.append(("cat", categorical_pipeline, list(categorical_features)))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
