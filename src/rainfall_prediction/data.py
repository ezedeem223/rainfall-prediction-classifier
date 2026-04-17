"""Dataset loading and preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .features import date_to_season, infer_feature_columns
from .utils import DataNotFoundError


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    path = Path(dataset_path)
    if not path.exists():
        raise DataNotFoundError(
            "Dataset file not found at "
            f"{path}. Download weatherAUS.csv and place it under data/raw/ before training."
        )
    return pd.read_csv(path)


def validate_columns(frame: pd.DataFrame, expected_columns: list[str]) -> None:
    """Validate that required columns are present."""
    missing = [column for column in expected_columns if column and column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def apply_dataset_rules(
    frame: pd.DataFrame,
    dataset_config: dict[str, Any],
    training: bool = True,
) -> pd.DataFrame:
    """Apply dataset-level transformations shared across training and inference."""
    prepared = frame.copy()

    location_filter = dataset_config.get("location_filter") or []
    if location_filter and "Location" in prepared.columns:
        prepared = prepared[prepared["Location"].isin(location_filter)].copy()

    date_column = dataset_config.get("date_column")
    if (
        dataset_config.get("add_season_feature", False)
        and date_column
        and date_column in prepared.columns
    ):
        prepared[date_column] = pd.to_datetime(prepared[date_column], errors="coerce")
        if training and dataset_config.get("drop_missing_rows", False):
            prepared = prepared.dropna(subset=[date_column]).copy()
        prepared["Season"] = prepared[date_column].apply(date_to_season)
        if dataset_config.get("drop_date_column", True):
            prepared = prepared.drop(columns=[date_column])

    if training and dataset_config.get("drop_missing_rows", False):
        prepared = prepared.dropna().copy()

    return prepared


def prepare_training_frame(data_config: dict[str, Any]) -> pd.DataFrame:
    """Load and prepare the training dataframe according to config."""
    dataset_config = data_config["dataset"]
    frame = load_dataset(dataset_config["path"])
    frame = apply_dataset_rules(frame, dataset_config, training=True)

    target_column = dataset_config["target_column"]
    validate_columns(frame, [target_column])
    validate_columns(
        frame,
        list(dataset_config.get("numerical_features") or [])
        + list(dataset_config.get("categorical_features") or []),
    )
    return frame


def split_features_and_target(
    frame: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the dataframe into features and target."""
    validate_columns(frame, [target_column])
    return frame.drop(columns=[target_column]), frame[target_column]


def create_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    split_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/test split."""
    stratify = target if split_config.get("stratify", True) else None
    return train_test_split(
        features,
        target,
        test_size=split_config.get("test_size", 0.2),
        random_state=split_config.get("random_state", 42),
        stratify=stratify,
    )


def resolve_feature_groups(
    feature_frame: pd.DataFrame,
    data_config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Resolve numerical and categorical feature groups for preprocessing."""
    dataset_config = data_config["dataset"]
    return infer_feature_columns(
        feature_frame,
        numerical_features=dataset_config.get("numerical_features"),
        categorical_features=dataset_config.get("categorical_features"),
    )


def prepare_inference_frame(
    frame: pd.DataFrame,
    dataset_config: dict[str, Any],
    expected_features: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the training-time feature rules to raw prediction inputs."""
    prepared = frame.copy()
    target_column = dataset_config.get("target_column")
    if target_column and target_column in prepared.columns:
        prepared = prepared.drop(columns=[target_column])

    prepared = apply_dataset_rules(prepared, dataset_config, training=False)

    if expected_features:
        missing = [column for column in expected_features if column not in prepared.columns]
        if missing:
            raise ValueError(
                "Input data is missing required feature columns: " + ", ".join(missing)
            )
        prepared = prepared[expected_features]

    return prepared
