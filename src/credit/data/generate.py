"""Gera dataset sintético de crédito (referência e lote inválido).

Inspirado em features típicas de risco de crédito (German Credit / Lending Club),
sem depender de download externo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit.config import INVALID_CSV, REFERENCE_CSV, load_params
from credit.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "customer_id",
    "age",
    "income",
    "employment_years",
    "loan_amount",
    "loan_duration_months",
    "debt_ratio",
    "num_credit_lines",
    "late_payments",
    "home_ownership",  # 0=rent, 1=mortgage, 2=owned
    "purpose",  # 0=personal, 1=car, 2=home, 3=education
]


def generate_reference_dataset(
    n_samples: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Gera o dataset de referência (limpo, usado no treino)."""
    params = load_params()
    n_samples = n_samples or int(params["data"]["n_samples"])
    seed = seed if seed is not None else int(params["data"]["random_seed"])
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 76, size=n_samples)
    income = rng.lognormal(mean=8.2, sigma=0.45, size=n_samples).round(2)
    income = np.clip(income, 800, 35000)
    employment_years = np.clip(rng.normal(age / 8, 3, size=n_samples), 0, 40).round(1)
    loan_amount = rng.uniform(2000, 80000, size=n_samples).round(2)
    loan_duration_months = rng.choice([12, 24, 36, 48, 60], size=n_samples)
    debt_ratio = np.clip(rng.beta(2, 5, size=n_samples), 0.01, 0.95).round(3)
    num_credit_lines = rng.integers(1, 12, size=n_samples)
    late_payments = rng.poisson(0.6, size=n_samples)
    home_ownership = rng.choice([0, 1, 2], size=n_samples, p=[0.35, 0.40, 0.25])
    purpose = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.40, 0.25, 0.20, 0.15])

    # Probabilidade de inadimplência (regra simples e interpretável)
    logit = (
        -2.5
        + 0.015 * (age - 40)
        - 0.00008 * income
        + 0.00003 * loan_amount
        + 2.5 * debt_ratio
        + 0.45 * late_payments
        - 0.04 * employment_years
        - 0.2 * home_ownership
    )
    prob = 1 / (1 + np.exp(-logit))
    default = (rng.random(n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"REF-{i:05d}" for i in range(n_samples)],
            "age": age,
            "income": income,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "loan_duration_months": loan_duration_months,
            "debt_ratio": debt_ratio,
            "num_credit_lines": num_credit_lines,
            "late_payments": late_payments,
            "home_ownership": home_ownership,
            "purpose": purpose,
            "default": default,
        }
    )
    return df


def generate_invalid_batch(seed: int = 99) -> pd.DataFrame:
    """Gera um lote propositalmente inválido para demonstrar o contrato falhando."""
    rng = np.random.default_rng(seed)
    n = 20
    df = pd.DataFrame(
        {
            "customer_id": [*["BAD-00001"] * 5, *[f"BAD-{i:05d}" for i in range(2, n - 3)]],
            "age": [16, 17, 90, 25, None, *rng.integers(20, 60, size=n - 5).tolist()],
            "income": [None, -100, 0, 3000, 4500, *rng.uniform(1000, 8000, size=n - 5).tolist()],
            "employment_years": list(rng.uniform(0, 10, size=n)),
            "loan_amount": [
                500,
                0,
                -50,
                10000,
                12000,
                *rng.uniform(2000, 20000, size=n - 5).tolist(),
            ],
            "loan_duration_months": list(rng.choice([12, 24, 36], size=n)),
            "debt_ratio": list(np.clip(rng.beta(2, 5, size=n), 0, 1)),
            "num_credit_lines": list(rng.integers(1, 8, size=n)),
            "late_payments": list(rng.integers(0, 3, size=n)),
            "home_ownership": list(rng.integers(0, 3, size=n)),
            "purpose": list(rng.integers(0, 4, size=n)),
            "default": list(rng.integers(0, 2, size=n)),
        }
    )
    return df


def save_reference(path: Path = REFERENCE_CSV) -> Path:
    """Gera e salva o dataset de referência."""
    params = load_params()
    df = generate_reference_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(
        "reference_dataset_saved",
        path=str(path),
        rows=len(df),
        default_rate=float(df["default"].mean()),
        seed=params["data"]["random_seed"],
    )
    return path


def save_invalid_batch(path: Path = INVALID_CSV) -> Path:
    """Gera e salva o lote inválido de demonstração."""
    df = generate_invalid_batch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("invalid_batch_saved", path=str(path), rows=len(df))
    return path
