"""Plotting helpers for evaluation outputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline

from .pipeline import get_feature_names, humanize_model_name
from .utils import ensure_directory


def save_confusion_matrix_plot(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    output_path: str | Path,
    model_name: str,
    labels: Sequence[str],
) -> Path:
    """Save a confusion matrix figure."""
    destination = Path(output_path)
    ensure_directory(destination.parent)

    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(labels))
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(f"{humanize_model_name(model_name)} Confusion Matrix")
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def save_feature_importance_plot(
    model_pipeline: Pipeline,
    model_name: str,
    output_path: str | Path,
    top_n: int = 20,
) -> Path | None:
    """Save a feature-importance style chart when the estimator supports it."""
    estimator = model_pipeline.named_steps["classifier"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = abs(estimator.coef_).ravel()
    else:
        return None

    feature_names = get_feature_names(model_pipeline)
    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    destination = Path(output_path)
    ensure_directory(destination.parent)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(importance_frame["feature"], importance_frame["importance"], color="skyblue")
    axis.invert_yaxis()
    axis.set_title(f"Top {top_n} Features for {humanize_model_name(model_name)}")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination
