PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: setup install install-dev lint format test train evaluate predict

setup: install-dev

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

lint:
	ruff check .

format:
	black .
	isort .

test:
	pytest

train:
	$(PYTHON) scripts/run_train.py --config configs/train.yaml

evaluate:
	$(PYTHON) scripts/run_evaluate.py --config configs/inference.yaml

predict:
	$(PYTHON) scripts/run_predict.py --config configs/inference.yaml --input results/sample_predictions/example_input.json
