"""Tests that verify metric provenance integrity across preserved results artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "docs" / "research_pack"
RESULTS_DIR = ROOT / "results"

EXPECTED_MODEL_NAMES = {"logistic_regression", "random_forest", "xgboost"}

EXPECTED_METRICS = {
    "logistic_regression": {"train_accuracy": 0.8403, "test_accuracy": 0.8424},
    "random_forest": {"train_accuracy": 0.9999, "test_accuracy": 0.8428},
    "xgboost": {"train_accuracy": 0.8711, "test_accuracy": 0.8519},
}


def test_metrics_json_is_parseable() -> None:
    path = RESULTS_DIR / "metrics.json"
    assert path.exists(), "results/metrics.json must exist"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "metrics.json must be a JSON object"


def test_metrics_json_contains_historical_results() -> None:
    path = RESULTS_DIR / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "historical_repository_results" in data, (
        "metrics.json must contain 'historical_repository_results'"
    )
    models = data["historical_repository_results"]["models"]
    assert set(models.keys()) == EXPECTED_MODEL_NAMES, (
        f"Expected model keys {EXPECTED_MODEL_NAMES}, got {set(models.keys())}"
    )


def test_metrics_json_preserved_values_match_expected() -> None:
    path = RESULTS_DIR / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data["historical_repository_results"]["models"]
    for model_name, expected in EXPECTED_METRICS.items():
        assert model_name in models, f"Model '{model_name}' missing from metrics.json"
        for metric, value in expected.items():
            actual = models[model_name][metric]
            assert abs(actual - value) < 1e-6, (
                f"{model_name} {metric}: expected {value}, got {actual}"
            )


def test_model_comparison_csv_is_parseable() -> None:
    path = RESULTS_DIR / "model_comparison.csv"
    assert path.exists(), "results/model_comparison.csv must exist"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) > 0, "model_comparison.csv must contain at least one row"


def test_model_comparison_csv_contains_expected_models() -> None:
    path = RESULTS_DIR / "model_comparison.csv"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    model_names = {row["model_name"] for row in rows}
    assert EXPECTED_MODEL_NAMES == model_names, (
        f"Expected model names {EXPECTED_MODEL_NAMES} in model_comparison.csv, got {model_names}"
    )


def test_model_comparison_csv_accuracy_values_match_expected() -> None:
    path = RESULTS_DIR / "model_comparison.csv"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        name = row["model_name"]
        if name in EXPECTED_METRICS:
            train_acc = float(row["train_accuracy"])
            test_acc = float(row["test_accuracy"])
            assert abs(train_acc - EXPECTED_METRICS[name]["train_accuracy"]) < 1e-6, (
                f"{name} train_accuracy mismatch: expected "
                f"{EXPECTED_METRICS[name]['train_accuracy']}, got {train_acc}"
            )
            assert abs(test_acc - EXPECTED_METRICS[name]["test_accuracy"]) < 1e-6, (
                f"{name} test_accuracy mismatch: expected "
                f"{EXPECTED_METRICS[name]['test_accuracy']}, got {test_acc}"
            )


def test_metric_provenance_matrix_references_all_model_names() -> None:
    path = PACK_DIR / "METRIC_PROVENANCE_MATRIX.md"
    assert path.exists(), "METRIC_PROVENANCE_MATRIX.md must exist"
    text = path.read_text(encoding="utf-8").lower()
    for model_name in EXPECTED_MODEL_NAMES:
        assert model_name.replace("_", " ") in text or model_name in text, (
            f"METRIC_PROVENANCE_MATRIX.md must reference model '{model_name}'"
        )


def test_metric_provenance_matrix_mentions_xgboost_provenance_gap() -> None:
    path = PACK_DIR / "METRIC_PROVENANCE_MATRIX.md"
    text = path.read_text(encoding="utf-8").lower()
    assert "provenance gap" in text, (
        "METRIC_PROVENANCE_MATRIX.md must explicitly mention the XGBoost provenance gap"
    )


def test_random_forest_overfitting_pattern_visible_in_preserved_metrics() -> None:
    path = RESULTS_DIR / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rf = data["historical_repository_results"]["models"]["random_forest"]
    gap = rf["train_accuracy"] - rf["test_accuracy"]
    assert gap > 0.15, (
        f"Random Forest train/test gap expected > 0.15, got {gap:.4f}. "
        "Overfitting pattern must be visible in preserved metrics."
    )


def test_xgboost_best_test_accuracy_among_preserved_models() -> None:
    path = RESULTS_DIR / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data["historical_repository_results"]["models"]
    xgb_test = models["xgboost"]["test_accuracy"]
    lr_test = models["logistic_regression"]["test_accuracy"]
    rf_test = models["random_forest"]["test_accuracy"]
    assert xgb_test >= lr_test and xgb_test >= rf_test, (
        "XGBoost must have the highest preserved test accuracy among the three models"
    )


def test_model_comparison_csv_has_source_column() -> None:
    path = RESULTS_DIR / "model_comparison.csv"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert all("source" in row for row in rows), (
        "model_comparison.csv must contain a 'source' column for provenance tracking"
    )
