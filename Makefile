.PHONY: help install validate lint format test generate-data validate-data train \
        simulate-drift detect-drift dashboard pipeline clean

PYTHON := python

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependências com Poetry
	poetry install
	poetry run pre-commit install || true

validate: ## Valida ambiente local
	poetry run $(PYTHON) -m scripts.validate_env

lint: ## Lint com ruff
	poetry run ruff check src tests scripts
	poetry run ruff format --check src tests scripts

format: ## Formata código
	poetry run ruff format src tests scripts
	poetry run ruff check --fix src tests scripts

test: ## Executa testes
	poetry run pytest

generate-data: ## Gera dataset de referência + lote inválido
	poetry run $(PYTHON) -m scripts.generate_data

validate-data: ## Valida o dataset de referência (contrato Pandera)
	poetry run $(PYTHON) -m scripts.validate_data data/reference/credit_reference.csv

train: ## Treina modelo baseline e registra no MLflow
	poetry run $(PYTHON) -m scripts.train

simulate-drift: ## Gera dataset de produção com drift
	poetry run $(PYTHON) -m scripts.simulate_drift

detect-drift: ## Detecta drift (Evidently + PSI/KS)
	poetry run $(PYTHON) -m scripts.detect_drift

dashboard: ## Gera dashboard HTML consolidado
	poetry run $(PYTHON) -m scripts.build_dashboard

pipeline: ## Roda o pipeline completo ponta a ponta
	poetry run $(PYTHON) -m scripts.run_pipeline

clean: ## Remove caches locais
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
