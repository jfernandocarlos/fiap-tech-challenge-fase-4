"""Testes do treino do modelo."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from credit.models.training import FEATURE_COLS, predict_dataframe, train_model


def test_train_model_returns_metrics(reference_df: pd.DataFrame, tmp_path: Path) -> None:
    model, metrics = train_model(reference_df)
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    assert metrics["n_train"] > 0
    assert set(metrics["feature_importance"]) == set(FEATURE_COLS)

    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "features": FEATURE_COLS}, model_path)
    scored = predict_dataframe(reference_df.head(10), model_path=model_path)
    assert "prediction" in scored.columns
    assert "probability" in scored.columns
