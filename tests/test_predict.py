from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from rainfall_prediction.features import build_preprocessor
from rainfall_prediction.predict import load_model_artifact, predict_from_frame
from rainfall_prediction.utils import ModelArtifactNotFoundError


def _build_artifact(tmp_path: Path) -> Path:
    raw_training = pd.DataFrame(
        [
            {
                "Date": "2017-06-01",
                "Location": "Melbourne",
                "MinTemp": 9.0,
                "RainToday": "Yes",
                "Season": "Winter",
                "RainTomorrow": "Yes",
            },
            {
                "Date": "2017-01-10",
                "Location": "Sydney",
                "MinTemp": 27.0,
                "RainToday": "No",
                "Season": "Summer",
                "RainTomorrow": "No",
            },
            {
                "Date": "2017-07-05",
                "Location": "Melbourne",
                "MinTemp": 8.0,
                "RainToday": "Yes",
                "Season": "Winter",
                "RainTomorrow": "Yes",
            },
            {
                "Date": "2017-02-14",
                "Location": "Sydney",
                "MinTemp": 26.5,
                "RainToday": "No",
                "Season": "Summer",
                "RainTomorrow": "No",
            },
        ]
    )

    feature_frame = raw_training[["Location", "MinTemp", "RainToday", "Season"]]
    preprocessor = build_preprocessor(["MinTemp"], ["Location", "RainToday", "Season"])
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    encoder = LabelEncoder().fit(raw_training["RainTomorrow"])
    model.fit(feature_frame, encoder.transform(raw_training["RainTomorrow"]))

    artifact = {
        "pipeline": model,
        "label_encoder": encoder,
        "model_name": "logistic_regression",
        "feature_columns": ["Location", "MinTemp", "RainToday", "Season"],
        "data_config": {
            "dataset": {
                "target_column": "RainTomorrow",
                "date_column": "Date",
                "add_season_feature": True,
                "drop_date_column": True,
            }
        },
    }

    artifact_path = tmp_path / "model.joblib"
    joblib.dump(artifact, artifact_path)
    return artifact_path


def test_load_model_artifact_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactNotFoundError):
        load_model_artifact(tmp_path / "missing.joblib")


def test_predict_from_frame_returns_probabilities(tmp_path: Path) -> None:
    artifact = load_model_artifact(_build_artifact(tmp_path))
    inference_frame = pd.DataFrame(
        [
            {
                "Date": "2017-06-09",
                "Location": "Melbourne",
                "MinTemp": 10.2,
                "RainToday": "Yes",
            }
        ]
    )

    predictions = predict_from_frame(artifact, inference_frame, include_probabilities=True)

    assert "RainTomorrow" in predictions.columns
    assert "probability_No" in predictions.columns
    assert "probability_Yes" in predictions.columns
    assert predictions.loc[0, "RainTomorrow"] in {"No", "Yes"}
