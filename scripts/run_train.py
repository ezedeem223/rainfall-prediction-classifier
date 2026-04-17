"""CLI entrypoint for model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rainfall_prediction.config import load_config
from rainfall_prediction.train import train_model_suite
from rainfall_prediction.utils import RainfallPredictionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train rainfall prediction models.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "train.yaml"),
        help="Path to the training config file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = train_model_suite(load_config(args.config))
    except RainfallPredictionError as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - top-level safety net
        print(f"Unexpected training failure: {exc}", file=sys.stderr)
        return 1

    print(f"Selected model: {summary['selected_model_label']}")
    print(f"Saved artifact: {summary['model_output_path']}")
    print(f"Comparison table: {summary['model_comparison_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
