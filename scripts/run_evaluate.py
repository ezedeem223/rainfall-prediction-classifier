"""CLI entrypoint for model evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rainfall_prediction.config import load_config
from rainfall_prediction.evaluate import evaluate_model
from rainfall_prediction.utils import RainfallPredictionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained rainfall prediction model.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "inference.yaml"),
        help="Path to the evaluation config file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = evaluate_model(load_config(args.config))
    except RainfallPredictionError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - top-level safety net
        print(f"Unexpected evaluation failure: {exc}", file=sys.stderr)
        return 1

    print(f"Metrics written to: {outputs['metrics_path']}")
    print(f"Classification report: {outputs['report_path']}")
    print(f"Confusion matrix: {outputs['confusion_matrix_path']}")
    if outputs["feature_importance_path"]:
        print(f"Feature importance plot: {outputs['feature_importance_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
