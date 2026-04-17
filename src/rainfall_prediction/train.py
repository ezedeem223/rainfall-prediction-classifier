"""Training helpers for the rainfall prediction workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .data import (
    create_train_test_split,
    prepare_training_frame,
    resolve_feature_groups,
    split_features_and_target,
)
from .features import build_preprocessor
from .pipeline import build_model_pipeline, humanize_model_name
from .utils import RainfallPredictionError, ensure_directory, write_json


def _select_model(comparison_frame: pd.DataFrame, training_config: dict[str, Any]) -> str:
    successful = comparison_frame[comparison_frame["status"] == "trained"].copy()
    if successful.empty:
        raise RainfallPredictionError("No models were trained successfully.")

    selected = training_config.get("selected_model", "best")
    if selected != "best":
        if selected not in successful["model_name"].values:
            raise RainfallPredictionError(f"Requested selected_model '{selected}' was not trained.")
        return selected

    metric = training_config.get("selection_metric", "test_accuracy")
    return successful.sort_values(metric, ascending=False).iloc[0]["model_name"]


def train_model_suite(config: dict[str, Any]) -> dict[str, Any]:
    """Train all configured models and persist the selected artifact."""
    data_config = config["data"]
    paths_config = config.get("paths", {})
    training_config = config.get("training", {})

    results_dir = ensure_directory(paths_config.get("results_dir", "results"))
    model_output_path = Path(
        paths_config.get("model_output_path", "models/rainfall_prediction_pipeline.joblib")
    )
    ensure_directory(model_output_path.parent)

    frame = prepare_training_frame(data_config)
    features, target = split_features_and_target(frame, data_config["dataset"]["target_column"])
    x_train, x_test, y_train, y_test = create_train_test_split(
        features,
        target,
        data_config.get("split", {}),
    )

    numerical_features, categorical_features = resolve_feature_groups(x_train, data_config)
    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
        scaler=data_config.get("preprocessing", {}).get("scaler", "standard"),
        handle_unknown=data_config.get("preprocessing", {}).get("one_hot_handle_unknown", "ignore"),
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(target)
    y_train_encoded = label_encoder.transform(y_train)

    cv = StratifiedKFold(
        n_splits=training_config.get("cross_validation_folds", 5),
        shuffle=True,
        random_state=training_config.get("random_state", 42),
    )

    comparison_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    for model_name in training_config.get("models", []):
        try:
            pipeline = build_model_pipeline(
                preprocessor=preprocessor,
                model_name=model_name,
                random_state=training_config.get("random_state", 42),
            )
            param_grid = training_config.get("param_grid", {}).get(model_name, {})
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="accuracy",
                verbose=training_config.get("verbose", 0),
            )
            search.fit(x_train, y_train_encoded)

            fitted_pipeline = search.best_estimator_
            train_predictions = label_encoder.inverse_transform(fitted_pipeline.predict(x_train))
            test_predictions = label_encoder.inverse_transform(fitted_pipeline.predict(x_test))

            row = {
                "model_name": model_name,
                "model_label": humanize_model_name(model_name),
                "status": "trained",
                "train_accuracy": float(accuracy_score(y_train, train_predictions)),
                "test_accuracy": float(accuracy_score(y_test, test_predictions)),
                "cv_best_score": float(search.best_score_),
                "best_params": search.best_params_,
            }
            comparison_rows.append(row)
            fitted_models[model_name] = {
                "pipeline": fitted_pipeline,
                "metrics": row,
                "best_params": search.best_params_,
            }
        except Exception as exc:
            comparison_rows.append(
                {
                    "model_name": model_name,
                    "model_label": humanize_model_name(model_name),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(results_dir / "model_comparison.csv", index=False)

    selected_model = _select_model(comparison_frame, training_config)
    selected_bundle = fitted_models[selected_model]

    artifact = {
        "pipeline": selected_bundle["pipeline"],
        "model_name": selected_model,
        "label_encoder": label_encoder,
        "feature_columns": features.columns.tolist(),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "data_config": data_config,
        "training_config": training_config,
        "split_config": data_config.get("split", {}),
        "best_params": selected_bundle["best_params"],
        "model_comparison": comparison_rows,
    }
    joblib.dump(artifact, model_output_path)

    summary = {
        "selected_model": selected_model,
        "selected_model_label": humanize_model_name(selected_model),
        "model_output_path": str(model_output_path),
        "model_comparison_path": str(results_dir / "model_comparison.csv"),
        "metrics": selected_bundle["metrics"],
    }
    write_json(summary, results_dir / "metrics.json")
    return summary
