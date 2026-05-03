# Evidence Validation Tools

This directory contains tooling for validating the structural completeness and content integrity of the research evidence pack.

## validate_research_pack.py

Validates the `docs/research_pack/` directory without requiring a dataset, trained model artifact, or internet access.

### Usage

```bash
python tools/evidence/validate_research_pack.py
```

### What It Checks

- All required research pack files exist under `docs/research_pack/`.
- `results/metrics.json` exists and is valid JSON.
- `results/model_comparison.csv` exists and is parseable as CSV.
- `README.md` references `ACADEMIC_RESEARCH_BRIEF.md`.
- No forbidden phrases appear in `docs/research_pack/`.
- No institution-specific wording appears in `docs/research_pack/`.
- Required limitation phrases are present in `docs/research_pack/`.
- `TEMPORAL_VALIDATION_PROTOCOL.md` states no temporal results were generated in this pass.
- `INTERPRETABILITY_PROTOCOL.md` states no SHAP/permutation artifacts were generated in this pass.
- `CALIBRATION_AND_THRESHOLDING_PROTOCOL.md` states no calibration results were generated in this pass.
- `LEAKAGE_AND_SPLIT_RISK_AUDIT.md` mentions the XGBoost provenance gap.
- `DATASET_AND_TASK_CARD.md` mentions the notebook/package framing mismatch.

### Exit Codes

- `0` — All checks passed.
- `1` — One or more checks failed. Failed checks are printed to stdout.
