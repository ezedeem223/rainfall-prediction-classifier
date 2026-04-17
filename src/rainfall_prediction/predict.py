"""Prediction helpers for saved model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data import prepare_inference_frame
from .utils import ModelArtifactNotFoundError, ensure_directory


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    """Load a saved model bundle from disk."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ModelArtifactNotFoundError(
            "Model artifact not found at "
            f"{artifact_path}. Run the training script first to create it."
        )
    return joblib.load(artifact_path)


def load_prediction_input(path: str | Path) -> pd.DataFrame:
    """Load prediction input data from JSON or CSV."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction input not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)

    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        return pd.DataFrame(payload)

    raise ValueError("Prediction input must be a .json or .csv file.")


def parse_inline_json(row_json: str) -> pd.DataFrame:
    """Parse a single inline JSON object into a dataframe."""
    payload = json.loads(row_json)
    if not isinstance(payload, dict):
        raise ValueError("--row-json must contain a single JSON object.")
    return pd.DataFrame([payload])


def predict_from_frame(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    include_probabilities: bool = True,
) -> pd.DataFrame:
    """Generate predictions for a dataframe of raw input rows."""
    dataset_config = artifact.get("data_config", {}).get("dataset", {})
    expected_features = artifact.get("feature_columns", [])
    prepared = prepare_inference_frame(frame, dataset_config, expected_features=expected_features)

    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    predicted_encoded = pipeline.predict(prepared)
    predicted_labels = label_encoder.inverse_transform(predicted_encoded)

    result = frame.reset_index(drop=True).copy()
    target_column = dataset_config.get("target_column", "prediction")
    result[target_column] = predicted_labels

    if include_probabilities and hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(prepared)
        for index, label in enumerate(label_encoder.classes_):
            result[f"probability_{label}"] = probabilities[:, index]

    return result


def save_predictions(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist prediction outputs to JSON or CSV."""
    destination = Path(output_path)
    ensure_directory(destination.parent)

    if destination.suffix.lower() == ".csv":
        frame.to_csv(destination, index=False)
    else:
        frame.to_json(destination, orient="records", indent=2)

    return destination
