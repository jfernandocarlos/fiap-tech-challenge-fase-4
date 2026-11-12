"""Treina o modelo baseline e registra no MLflow."""

from __future__ import annotations

from credit.config import load_params
from credit.data.contracts import validate_dataframe
from credit.data.loader import load_reference
from credit.logging_config import setup_logging
from credit.models.training import FEATURE_COLS, save_artifacts, train_model
from credit.monitoring.mlflow_tracking import log_training_run
from credit.monitoring.pipeline_logs import append_event


def main() -> int:
    setup_logging()
    try:
        df = load_reference()
        validate_dataframe(df)
        model, metrics = train_model(df)
        save_artifacts(model, metrics)

        params = load_params()
        run_id = log_training_run(
            params=params,
            metrics=metrics,
            model=model,
            x_sample=df[FEATURE_COLS].head(50),
        )
        append_event(
            "train",
            "success",
            {"roc_auc": metrics["roc_auc"], "mlflow_run_id": run_id},
        )
        print(f"Modelo treinado — ROC-AUC={metrics['roc_auc']:.4f} | MLflow run={run_id}")
        return 0
    except Exception as exc:
        append_event("train", "failure", {"error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
