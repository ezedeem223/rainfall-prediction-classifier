import pytest

from rainfall_prediction.data import load_dataset
from rainfall_prediction.features import build_preprocessor
from rainfall_prediction.pipeline import build_model_pipeline
from rainfall_prediction.utils import DataNotFoundError


def test_build_model_pipeline_contains_expected_steps() -> None:
    preprocessor = build_preprocessor(["MinTemp"], ["Location"])
    pipeline = build_model_pipeline(preprocessor, model_name="logistic_regression", random_state=42)

    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    assert pipeline.named_steps["classifier"].__class__.__name__ == "LogisticRegression"


def test_load_dataset_raises_for_missing_path(tmp_path) -> None:
    with pytest.raises(DataNotFoundError):
        load_dataset(tmp_path / "missing.csv")
