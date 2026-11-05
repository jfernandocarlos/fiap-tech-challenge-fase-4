"""Configuração centralizada do projeto.

Duas fontes de verdade:

* ``Settings`` (Pydantic + ``.env``): infraestrutura e runtime.
* ``params.yaml``: hiperparâmetros de ML e limiares de drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DATA_PRODUCTION_DIR = PROJECT_ROOT / "data" / "production"
DATA_INVALID_DIR = PROJECT_ROOT / "data" / "invalid"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"

REFERENCE_CSV = DATA_REFERENCE_DIR / "credit_reference.csv"
PRODUCTION_CSV = DATA_PRODUCTION_DIR / "credit_production.csv"
INVALID_CSV = DATA_INVALID_DIR / "credit_invalid_batch.csv"
MODEL_PATH = MODELS_DIR / "credit_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
DRIFT_REPORT_HTML = REPORTS_DIR / "drift_report.html"
DRIFT_SUMMARY_JSON = REPORTS_DIR / "drift_summary.json"
MONITORING_DASHBOARD_HTML = REPORTS_DIR / "monitoring_dashboard.html"
PIPELINE_LOG = LOGS_DIR / "pipeline.jsonl"


class Settings(BaseSettings):
    """Configurações de infraestrutura lidas do ``.env``."""

    target_column: str = Field(default="default", description="Coluna alvo (inadimplência)")
    mlflow_tracking_uri: str = Field(default="mlruns", description="URI do tracking MLflow")
    drift_psi_threshold: float = Field(default=0.2, description="Limiar PSI para alerta")

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    """Carrega hiperparâmetros do ``params.yaml``."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


settings = Settings()
