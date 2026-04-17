"""Evaluation helpers for trained models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import create_train_test_split, prepare_training_frame, split_features_and_target
from .predict import load_model_artifact
from .utils import ensure_directory, write_json, write_text
from .visualization import save_confusion_matrix_plot, save_feature_importance_plot


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_true_encoded: Sequence[int] | None = None,
    probabilities: Any | None = None,
) -> dict[str, Any]:
    """Compute scalar evaluation metrics."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    has_binary_probabilities = (
        probabilities is not None
        and y_true_encoded is not None
        and getattr(probabilities, "shape", (0, 0))[1] == 2
    )
    if has_binary_probabilities:
        metrics["roc_auc"] = float(roc_auc_score(y_true_encoded, probabilities[:, 1]))

    return metrics


def evaluate_model(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a saved model artifact and write results under results/."""
    paths_config = config.get("paths", {})
    results_dir = ensure_directory(paths_config.get("results_dir", "results"))
    model_path = paths_config.get("model_path") or paths_config.get(
        "model_output_path",
        "models/rainfall_prediction_pipeline.joblib",
    )
    artifact = load_model_artifact(model_path)

    data_config = artifact.get("data_config", config.get("data", {}))
    frame = prepare_training_frame(data_config)
    features, target = split_features_and_target(frame, data_config["dataset"]["target_column"])
    _, x_test, _, y_test = create_train_test_split(
        features,
        target,
        artifact.get("split_config", data_config.get("split", {})),
    )

    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    predicted_encoded = pipeline.predict(x_test)
    predicted_labels = label_encoder.inverse_transform(predicted_encoded)

    probabilities = pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None
    y_test_encoded = label_encoder.transform(y_test)

    metrics = compute_classification_metrics(
        y_test,
        predicted_labels,
        y_test_encoded,
        probabilities,
    )
    metrics["model_name"] = artifact["model_name"]

    classification_text = classification_report(y_test, predicted_labels, digits=4)
    write_text(classification_text, results_dir / "classification_report.txt")
    write_json(metrics, results_dir / "metrics.json")

    if artifact.get("model_comparison"):
        import pandas as pd

        pd.DataFrame(artifact["model_comparison"]).to_csv(
            results_dir / "model_comparison.csv",
            index=False,
        )

    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=predicted_labels,
        output_path=results_dir / "confusion_matrix.png",
        model_name=artifact["model_name"],
        labels=list(label_encoder.classes_),
    )
    feature_plot = save_feature_importance_plot(
        model_pipeline=pipeline,
        model_name=artifact["model_name"],
        output_path=results_dir / "feature_importance.png",
    )

    return {
        "metrics_path": str(results_dir / "metrics.json"),
        "report_path": str(results_dir / "classification_report.txt"),
        "confusion_matrix_path": str(results_dir / "confusion_matrix.png"),
        "feature_importance_path": str(feature_plot) if feature_plot else None,
    }
