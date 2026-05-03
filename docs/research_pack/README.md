# Rainfall Prediction — Research Evidence Pack

This directory contains institution-neutral research documentation for the `rainfall-prediction-classifier` repository.

## Contents

| File | Purpose |
|---|---|
| [ACADEMIC_RESEARCH_BRIEF.md](ACADEMIC_RESEARCH_BRIEF.md) | Problem definition, scope, methodological risks, research directions |
| [MODEL_CARD.md](MODEL_CARD.md) | Model families, intended use, limitations, preserved metrics |
| [METRIC_PROVENANCE_MATRIX.md](METRIC_PROVENANCE_MATRIX.md) | Traceability table for every preserved metric |
| [DATASET_AND_TASK_CARD.md](DATASET_AND_TASK_CARD.md) | Task definition, dataset status, framing caveats |
| [TEMPORAL_VALIDATION_PROTOCOL.md](TEMPORAL_VALIDATION_PROTOCOL.md) | Chronological split and rolling-origin validation design |
| [LEAKAGE_AND_SPLIT_RISK_AUDIT.md](LEAKAGE_AND_SPLIT_RISK_AUDIT.md) | Target, temporal, location, and feature leakage risks |
| [INTERPRETABILITY_PROTOCOL.md](INTERPRETABILITY_PROTOCOL.md) | Feature importance, SHAP, and coefficient interpretation protocol |
| [CALIBRATION_AND_THRESHOLDING_PROTOCOL.md](CALIBRATION_AND_THRESHOLDING_PROTOCOL.md) | Probability calibration, Brier score, threshold selection |
| [ERROR_ANALYSIS_PLAYBOOK.md](ERROR_ANALYSIS_PLAYBOOK.md) | False-positive/negative analysis, subgroup breakdown, drift |
| [REPRODUCIBILITY_CHECKLIST.md](REPRODUCIBILITY_CHECKLIST.md) | Step-by-step checklist to reproduce training, evaluation, prediction |

## Validation Tool

```bash
python tools/evidence/validate_research_pack.py
```

This tool verifies structural completeness of the evidence pack without requiring a dataset, trained model artifact, or internet access.

## Important Caveats

- The dataset (`data/raw/weatherAUS.csv`) is **not bundled** in this repository.
- The trained model artifact is not bundled in this repository. No committed artifact exists by default.
- All preserved metrics are **historical** — they were not newly reproduced in this pass.
- This repository is a **research artifact** for classical tabular machine learning. It is not operational as a weather forecasting service.
- Temporal validation has **not** been performed in this pass. See `TEMPORAL_VALIDATION_PROTOCOL.md` for the planned protocol.
- No SHAP or permutation-importance artifacts have been generated in this pass. See `INTERPRETABILITY_PROTOCOL.md`.
- No probability calibration has been performed in this pass. See `CALIBRATION_AND_THRESHOLDING_PROTOCOL.md`.
