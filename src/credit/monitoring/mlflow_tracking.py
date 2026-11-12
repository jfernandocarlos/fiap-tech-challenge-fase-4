"""Rastreamento de experimentos com MLflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.models import infer_signature

from credit.config import PROJECT_ROOT, settings
from credit.logging_config import get_logger

logger = get_logger(__name__)


def setup_mlflow(experiment_name: str = "credit-scoring") -> None:
    """Configura tracking URI e experimento."""
    uri = settings.mlflow_tracking_uri
    if not uri.startswith(("http://", "https://", "file:")):
        uri = (PROJECT_ROOT / uri).resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    logger.info("mlflow_configured", tracking_uri=uri, experiment=experiment_name)


def log_training_run(
    params: dict[str, Any],
    metrics: dict[str, Any],
    model,
    x_sample,
    run_name: str = "baseline-train",
) -> str:
    """Registra treino: params, métricas e modelo."""
    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        flat_params = {
            "n_estimators": params["model"]["n_estimators"],
            "max_depth": params["model"]["max_depth"],
            "learning_rate": params["model"]["learning_rate"],
            "test_size": params["data"]["test_size"],
            "random_seed": params["data"]["random_seed"],
        }
        mlflow.log_params(flat_params)
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
        signature = infer_signature(x_sample, model.predict(x_sample))
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
        run_id = run.info.run_id
        logger.info("mlflow_training_logged", run_id=run_id)
        return run_id


def log_drift_run(summary: dict[str, Any], report_html: Path, run_name: str = "drift-check") -> str:
    """Registra resultado da detecção de drift e anexa HTML."""
    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics(
            {
                "n_drifted": float(summary["n_drifted"]),
                "n_features": float(summary["n_features"]),
            }
        )
        mlflow.log_param("drifted_features", ",".join(summary["drifted_features"]) or "none")
        if report_html.exists():
            mlflow.log_artifact(str(report_html))
        run_id = run.info.run_id
        logger.info("mlflow_drift_logged", run_id=run_id, n_drifted=summary["n_drifted"])
        return run_id
