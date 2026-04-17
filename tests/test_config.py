from pathlib import Path

from rainfall_prediction.config import load_config


def test_load_config_merges_data_defaults(tmp_path: Path) -> None:
    data_config = tmp_path / "data.yaml"
    train_config = tmp_path / "train.yaml"

    data_config.write_text(
        "dataset:\n  path: data/raw/weatherAUS.csv\n  target_column: RainTomorrow\n",
        encoding="utf-8",
    )
    train_config.write_text(
        "defaults:\n  data_config: data.yaml\ntraining:\n  models:\n    - logistic_regression\n",
        encoding="utf-8",
    )

    config = load_config(train_config)

    assert config["data"]["dataset"]["path"] == "data/raw/weatherAUS.csv"
    assert config["training"]["models"] == ["logistic_regression"]


def test_load_config_applies_environment_override(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "paths:\n  model_path: models/default.joblib\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAINFALL_MODEL_PATH", "models/override.joblib")

    config = load_config(config_path)

    assert config["paths"]["model_path"] == "models/override.joblib"
