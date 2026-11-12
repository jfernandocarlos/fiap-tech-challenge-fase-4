"""Testes de simulação e detecção de drift."""

from __future__ import annotations

import pandas as pd

from credit.drift.detect import feature_drift_stats, population_stability_index
from credit.drift.simulate import apply_drift


def test_apply_drift_changes_income_and_late_payments(reference_df: pd.DataFrame) -> None:
    prod = apply_drift(reference_df, seed=7)
    assert len(prod) == len(reference_df)
    assert prod["income"].mean() > reference_df["income"].mean()
    assert prod["late_payments"].mean() > reference_df["late_payments"].mean()
    assert list(prod["customer_id"]) != list(reference_df["customer_id"])


def test_psi_detects_shifted_distribution() -> None:
    import numpy as np

    ref = np.random.default_rng(0).normal(0, 1, 1000)
    cur = np.random.default_rng(1).normal(3, 1, 1000)
    psi = population_stability_index(ref, cur)
    assert psi > 0.2


def test_feature_drift_flags_shifted_features(reference_df: pd.DataFrame) -> None:
    prod = apply_drift(reference_df, seed=7)
    stats = feature_drift_stats(reference_df, prod)
    by_name = {row["feature"]: row for row in stats}
    assert by_name["income"]["drift_detected"] is True
    assert by_name["late_payments"]["drift_detected"] is True
