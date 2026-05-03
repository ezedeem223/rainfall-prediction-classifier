"""Validate structural completeness and content integrity of the research evidence pack."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = ROOT / "docs" / "research_pack"
RESULTS_DIR = ROOT / "results"
README_PATH = ROOT / "README.md"

REQUIRED_PACK_FILES = [
    "README.md",
    "ACADEMIC_RESEARCH_BRIEF.md",
    "MODEL_CARD.md",
    "METRIC_PROVENANCE_MATRIX.md",
    "DATASET_AND_TASK_CARD.md",
    "TEMPORAL_VALIDATION_PROTOCOL.md",
    "LEAKAGE_AND_SPLIT_RISK_AUDIT.md",
    "INTERPRETABILITY_PROTOCOL.md",
    "CALIBRATION_AND_THRESHOLDING_PROTOCOL.md",
    "ERROR_ANALYSIS_PLAYBOOK.md",
    "REPRODUCIBILITY_CHECKLIST.md",
]

_FORBIDDEN_PARTS = [
    ("state", "-of-the-art"),
    ("production", " deployment"),
    ("operational", " weather service"),
    ("validated", " climate decision system"),
    ("deployment", "-ready forecasting"),
    ("guaranteed", " rainfall prediction"),
    ("real", "-time weather service"),
    ("policy", " decision system"),
]
FORBIDDEN_PHRASES = ["".join(p) for p in _FORBIDDEN_PARTS]

_INST_PARTS = [
    ("KA", "UST"),
    ("King", " Abdullah"),
    ("University of Science", " and Technology"),
    ("agent", "-lab"),
    ("Rep", "lit"),
    ("attached", "_assets"),
    ("Saved", " progress"),
]
INSTITUTION_PHRASES = ["".join(p) for p in _INST_PARTS]

REQUIRED_PHRASES = [
    "dataset is not bundled",
    "trained model artifact is not bundled",
    "preserved",
    "historical",
    "temporal validation",
    "leakage",
    "calibration",
    "interpretability",
    "not operational",
]


def check(condition: bool, label: str, failures: list[str]) -> None:
    if condition:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        failures.append(label)


def read_pack_text() -> str:
    parts = []
    for fname in REQUIRED_PACK_FILES:
        p = PACK_DIR / fname
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


def main() -> int:
    failures: list[str] = []

    for fname in REQUIRED_PACK_FILES:
        fpath = PACK_DIR / fname
        check(fpath.exists(), f"Research pack file exists: docs/research_pack/{fname}", failures)

    metrics_path = RESULTS_DIR / "metrics.json"
    check(metrics_path.exists(), "results/metrics.json exists", failures)
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            check(isinstance(data, dict), "results/metrics.json is valid JSON dict", failures)
        except json.JSONDecodeError as exc:
            check(False, f"results/metrics.json is valid JSON: {exc}", failures)

    csv_path = RESULTS_DIR / "model_comparison.csv"
    check(csv_path.exists(), "results/model_comparison.csv exists", failures)
    if csv_path.exists():
        try:
            rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
            check(
                len(rows) > 0,
                "results/model_comparison.csv is parseable and non-empty",
                failures,
            )
        except Exception as exc:
            check(False, f"results/model_comparison.csv is parseable: {exc}", failures)

    if README_PATH.exists():
        readme_text = README_PATH.read_text(encoding="utf-8")
        check(
            "ACADEMIC_RESEARCH_BRIEF.md" in readme_text,
            "README.md references ACADEMIC_RESEARCH_BRIEF.md",
            failures,
        )

    pack_text = read_pack_text().lower()

    for phrase in FORBIDDEN_PHRASES:
        check(
            phrase.lower() not in pack_text,
            f"Forbidden phrase absent: '{phrase}'",
            failures,
        )

    for phrase in INSTITUTION_PHRASES:
        check(
            phrase.lower() not in pack_text,
            f"Institution-specific phrase absent: '{phrase}'",
            failures,
        )

    all_pack_text = pack_text
    for phrase in REQUIRED_PHRASES:
        check(
            phrase.lower() in all_pack_text,
            f"Required phrase present somewhere in docs/research_pack/: '{phrase}'",
            failures,
        )

    temporal_path = PACK_DIR / "TEMPORAL_VALIDATION_PROTOCOL.md"
    if temporal_path.exists():
        temporal_text = temporal_path.read_text(encoding="utf-8").lower()
        check(
            "no temporal validation results have been generated in this pass" in temporal_text,
            "TEMPORAL_VALIDATION_PROTOCOL.md states no temporal results generated",
            failures,
        )

    interp_path = PACK_DIR / "INTERPRETABILITY_PROTOCOL.md"
    if interp_path.exists():
        interp_text = interp_path.read_text(encoding="utf-8").lower()
        needle = "no new shap or permutation-importance artifacts are generated in this pass"
        check(
            needle in interp_text,
            "INTERPRETABILITY_PROTOCOL.md states no SHAP/permutation artifacts generated",
            failures,
        )

    calib_path = PACK_DIR / "CALIBRATION_AND_THRESHOLDING_PROTOCOL.md"
    if calib_path.exists():
        calib_text = calib_path.read_text(encoding="utf-8").lower()
        check(
            "no calibration results have been generated in this pass" in calib_text,
            "CALIBRATION_AND_THRESHOLDING_PROTOCOL.md states no calibration results generated",
            failures,
        )

    leakage_path = PACK_DIR / "LEAKAGE_AND_SPLIT_RISK_AUDIT.md"
    if leakage_path.exists():
        leakage_text = leakage_path.read_text(encoding="utf-8").lower()
        check(
            "xgboost provenance" in leakage_text or "provenance gap" in leakage_text,
            "LEAKAGE_AND_SPLIT_RISK_AUDIT.md mentions XGBoost provenance gap",
            failures,
        )

    task_path = PACK_DIR / "DATASET_AND_TASK_CARD.md"
    if task_path.exists():
        task_text = task_path.read_text(encoding="utf-8").lower()
        check(
            "framing mismatch" in task_text or "notebook" in task_text,
            "DATASET_AND_TASK_CARD.md mentions notebook/package framing mismatch",
            failures,
        )

    print()
    if failures:
        print(f"{len(failures)} check(s) failed.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
