.PHONY: help install smoke test lint fmt type bench train-a100 clean

PY ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PRE_COMMIT := $(VENV)/bin/pre-commit

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "NanoTSE targets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  %-13s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

$(VENV)/bin/activate:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate ## Create venv and install package + dev deps (editable)
	$(PIP) install -e ".[dev]"
	$(PRE_COMMIT) install || true

smoke: ## Run the 10-min M3 smoke train on the 200-clip subset
	$(PYTHON) scripts/train.py --config configs/smoke.yaml

test: ## Run pytest with coverage
	$(PYTEST)

lint: ## Ruff format check + lint
	$(RUFF) format --check .
	$(RUFF) check .

fmt: ## Apply ruff format + autofix
	$(RUFF) format .
	$(RUFF) check --fix .

type: ## mypy --strict on the nanotse package
	$(MYPY) nanotse

bench: ## Streaming latency benchmark (p50/p95/p99 chunk-latency)
	$(PYTHON) -m nanotse.eval.latency_bench

diagnose: ## Save mix/target/estimate wavs + SI-SDRi from a ckpt. Pass CKPT=path/to/model.pt
	@if [ -z "$$CKPT" ]; then echo "ERROR: pass CKPT=runs/<ts>/model.pt"; exit 1; fi
	$(PYTHON) scripts/diagnose.py --ckpt $$CKPT

train-a100: ## Submit full-data training to a vast.ai A100 (requires NANOTSE_A100_HOST)
	@if [ -z "$$NANOTSE_A100_HOST" ]; then echo "ERROR: set NANOTSE_A100_HOST"; exit 1; fi
	$(PYTHON) scripts/train.py --config configs/a100.yaml --remote $$NANOTSE_A100_HOST

clean: ## Remove venv + caches + build artifacts
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
