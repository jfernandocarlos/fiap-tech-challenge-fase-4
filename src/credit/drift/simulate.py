"""Simula dataset de produção com Data Drift intencional.

Altera a distribuição de pelo menos duas variáveis importantes
(``income`` e ``late_payments``), espelhando choque econômico:
renda nominal sobe (inflação) e atrasos aumentam (stress financeiro).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit.config import PRODUCTION_CSV, load_params
from credit.logging_config import get_logger

logger = get_logger(__name__)


def apply_drift(reference: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Copia a referência e desloca features-chave."""
    params = load_params()["drift"]
    rng = np.random.default_rng(seed)
    prod = reference.copy()

    # Novos IDs para simular clientes "atuais"
    prod["customer_id"] = [f"PRD-{i:05d}" for i in range(len(prod))]

    income_mult = float(params["income_multiplier"])
    late_add = float(params["late_payments_add"])

    # Drift em income: inflação + ruído
    noise = rng.normal(1.0, 0.08, size=len(prod))
    prod["income"] = (prod["income"] * income_mult * noise).round(2)

    # Drift em late_payments: mais inadimplência parcial
    extra = rng.poisson(late_add, size=len(prod))
    prod["late_payments"] = (prod["late_payments"] + extra).astype(int)

    # Concept drift leve: debt_ratio sobe um pouco (endividamento)
    prod["debt_ratio"] = np.clip(prod["debt_ratio"] * 1.15 + 0.05, 0.01, 0.99).round(3)

    logger.info(
        "drift_applied",
        rows=len(prod),
        income_multiplier=income_mult,
        late_payments_add=late_add,
        shifted_features=params["shifted_features"],
    )
    return prod


def save_production(reference: pd.DataFrame, path: Path = PRODUCTION_CSV) -> Path:
    """Aplica drift e salva o dataset de produção."""
    prod = apply_drift(reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    prod.to_csv(path, index=False)
    logger.info("production_dataset_saved", path=str(path), rows=len(prod))
    return path
