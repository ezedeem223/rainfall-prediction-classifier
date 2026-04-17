# Models Directory

Trained model artifacts are written here by `scripts/run_train.py`.

- Default artifact path: `models/rainfall_prediction_pipeline.joblib`
- Format: `joblib` bundle containing the fitted sklearn pipeline, label encoder, feature metadata, and training config
- Included in the repository: no committed trained model artifact

## Generate a model

```bash
python scripts/run_train.py --config configs/train.yaml
```

## Load a model for prediction

The CLI uses the artifact automatically:

```bash
python scripts/run_predict.py --config configs/inference.yaml --input results/sample_predictions/example_input.json
```

If the artifact is missing, the scripts fail with a clear message telling you to run training first.
