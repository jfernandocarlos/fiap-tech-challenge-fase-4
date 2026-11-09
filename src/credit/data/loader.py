"""Carrega CSVs de referência, produção e lotes de validação."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit.config import INVALID_CSV, PRODUCTION_CSV, REFERENCE_CSV


def load_csv(path: Path) -> pd.DataFrame:
    """Carrega um CSV e garante tipos básicos."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


def load_reference() -> pd.DataFrame:
    """Carrega o dataset de referência (treino)."""
    return load_csv(REFERENCE_CSV)


def load_production() -> pd.DataFrame:
    """Carrega o dataset de produção (análise de drift)."""
    return load_csv(PRODUCTION_CSV)


def load_invalid_batch() -> pd.DataFrame:
    """Carrega o lote inválido de demonstração."""
    return load_csv(INVALID_CSV)
