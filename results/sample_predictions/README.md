# Sample Prediction Inputs

Use the example payload in this folder to test the inference script once a trained model exists.

```bash
python scripts/run_predict.py --config configs/inference.yaml --input results/sample_predictions/example_input.json
```

The example file is illustrative and meant to document the expected schema. It is not a claim about model performance.

If you retrain the full package on the complete dataset, your saved artifact may expect more feature columns than the illustrative example payload included here. In that case, provide an input file that matches the trained artifact's feature set.
