"""Treino do modelo baseline de credit scoring."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from credit.config import METRICS_PATH, MODEL_PATH, load_params, settings
from credit.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_COLS = [
    "age",
    "income",
    "employment_years",
    "loan_amount",
    "loan_duration_months",
    "debt_ratio",
    "num_credit_lines",
    "late_payments",
    "home_ownership",
    "purpose",
]


def train_model(df: pd.DataFrame) -> tuple[GradientBoostingClassifier, dict]:
    """Treina GradientBoosting e retorna (modelo, métricas)."""
    params = load_params()
    target = settings.target_column
    seed = int(params["data"]["random_seed"])

    x = df[FEATURE_COLS]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=params["data"]["test_size"],
        random_state=seed,
        stratify=y,
    )

    model = GradientBoostingClassifier(
        n_estimators=int(params["model"]["n_estimators"]),
        max_depth=int(params["model"]["max_depth"]),
        learning_rate=float(params["model"]["learning_rate"]),
        random_state=seed,
    )
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "default_rate_train": float(y_train.mean()),
        "feature_importance": {
            name: float(imp)
            for name, imp in zip(FEATURE_COLS, model.feature_importances_, strict=True)
        },
    }
    logger.info("model_trained", **{k: v for k, v in metrics.items() if k != "feature_importance"})
    return model, metrics


def save_artifacts(
    model: GradientBoostingClassifier,
    metrics: dict,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Persiste modelo joblib e métricas JSON."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLS}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("artifacts_saved", model=str(model_path), metrics=str(metrics_path))


def load_model(model_path: Path = MODEL_PATH) -> tuple[GradientBoostingClassifier, list[str]]:
    """Carrega modelo e lista de features."""
    payload = joblib.load(model_path)
    return payload["model"], payload["features"]


def predict_dataframe(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """Gera predições (classe + probabilidade) para um DataFrame."""
    model, features = load_model(model_path)
    out = df.copy()
    proba = model.predict_proba(out[features])[:, 1]
    out["prediction"] = (proba >= 0.5).astype(int)
    out["probability"] = np.round(proba, 4)
    return out
