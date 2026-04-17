"""CLI entrypoint for inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rainfall_prediction.config import load_config
from rainfall_prediction.predict import (
    load_model_artifact,
    load_prediction_input,
    parse_inline_json,
    predict_from_frame,
    save_predictions,
)
from rainfall_prediction.utils import RainfallPredictionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a trained rainfall model.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "inference.yaml"),
        help="Path to the inference config file.",
    )
    parser.add_argument(
        "--input",
        help="Path to a JSON or CSV input file. Defaults to the config path if omitted.",
    )
    parser.add_argument(
        "--row-json",
        help="Single inline JSON object for one prediction row.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for predictions. Defaults to the config path if omitted.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    paths = config.get("paths", {})

    try:
        artifact = load_model_artifact(
            paths.get("model_path", "models/rainfall_prediction_pipeline.joblib")
        )
        if args.row_json:
            frame = parse_inline_json(args.row_json)
        else:
            input_path = args.input or paths.get("input_path")
            if not input_path:
                raise RainfallPredictionError(
                    "No prediction input provided. Pass --input, --row-json, "
                    "or set paths.input_path."
                )
            frame = load_prediction_input(input_path)

        predictions = predict_from_frame(
            artifact,
            frame,
            include_probabilities=config.get("prediction", {}).get("include_probabilities", True),
        )

        output_path = args.output or paths.get("output_path")
        if output_path:
            save_predictions(predictions, output_path)
            print(f"Predictions written to: {output_path}")
        print(predictions.to_string(index=False))
        return 0
    except RainfallPredictionError as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - top-level safety net
        print(f"Unexpected prediction failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
