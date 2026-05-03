"""Tests that verify the research evidence pack files exist and meet content requirements."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "docs" / "research_pack"

REQUIRED_FILES = [
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


def _pack_text() -> str:
    parts = []
    for fname in REQUIRED_FILES:
        p = PACK_DIR / fname
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_required_research_pack_files_exist() -> None:
    missing = [f for f in REQUIRED_FILES if not (PACK_DIR / f).exists()]
    assert not missing, f"Missing research pack files: {missing}"


def test_academic_research_brief_exists() -> None:
    assert (PACK_DIR / "ACADEMIC_RESEARCH_BRIEF.md").exists()


def test_validation_tool_exists() -> None:
    tool = ROOT / "tools" / "evidence" / "validate_research_pack.py"
    assert tool.exists(), "validate_research_pack.py not found"


def test_validation_tool_passes() -> None:
    tool = ROOT / "tools" / "evidence" / "validate_research_pack.py"
    result = subprocess.run(
        [sys.executable, str(tool)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"validate_research_pack.py failed:\n{result.stdout}\n{result.stderr}"
    )


def test_forbidden_phrases_absent_from_research_pack() -> None:
    text = _pack_text().lower()
    found = [p for p in FORBIDDEN_PHRASES if p.lower() in text]
    assert not found, f"Forbidden phrases found in docs/research_pack/: {found}"


def test_institution_specific_wording_absent() -> None:
    text = _pack_text().lower()
    found = [p for p in INSTITUTION_PHRASES if p.lower() in text]
    assert not found, f"Institution-specific phrases found in docs/research_pack/: {found}"


def test_temporal_protocol_states_no_results_generated() -> None:
    path = PACK_DIR / "TEMPORAL_VALIDATION_PROTOCOL.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8").lower()
    assert (
        "no temporal validation results have been generated in this pass" in text
    ), "TEMPORAL_VALIDATION_PROTOCOL.md must state no temporal results generated in this pass"


def test_interpretability_protocol_states_no_shap_generated() -> None:
    path = PACK_DIR / "INTERPRETABILITY_PROTOCOL.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8").lower()
    assert (
        "no new shap or permutation-importance artifacts are generated in this pass" in text
    ), "INTERPRETABILITY_PROTOCOL.md must state no SHAP/permutation artifacts generated"


def test_calibration_protocol_states_no_results_generated() -> None:
    path = PACK_DIR / "CALIBRATION_AND_THRESHOLDING_PROTOCOL.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8").lower()
    assert (
        "no calibration results have been generated in this pass" in text
    ), "CALIBRATION_AND_THRESHOLDING_PROTOCOL.md must state no calibration results generated"


def test_leakage_audit_mentions_xgboost_provenance_gap() -> None:
    path = PACK_DIR / "LEAKAGE_AND_SPLIT_RISK_AUDIT.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8").lower()
    assert "provenance" in text and ("xgboost" in text or "provenance gap" in text), (
        "LEAKAGE_AND_SPLIT_RISK_AUDIT.md must mention XGBoost provenance gap"
    )


def test_dataset_task_card_mentions_notebook_package_mismatch() -> None:
    path = PACK_DIR / "DATASET_AND_TASK_CARD.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8").lower()
    assert "framing mismatch" in text or ("notebook" in text and "package" in text), (
        "DATASET_AND_TASK_CARD.md must mention notebook/package framing mismatch"
    )


def test_readme_references_research_brief() -> None:
    readme = ROOT / "README.md"
    assert readme.exists()
    assert "ACADEMIC_RESEARCH_BRIEF.md" in readme.read_text(encoding="utf-8"), (
        "README.md must reference ACADEMIC_RESEARCH_BRIEF.md"
    )
