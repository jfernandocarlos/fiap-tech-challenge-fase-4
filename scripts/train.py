"""Treina o modelo baseline de credit scoring."""

from __future__ import annotations

from credit.data.contracts import validate_dataframe
from credit.data.loader import load_reference
from credit.logging_config import setup_logging
from credit.models.training import save_artifacts, train_model
from credit.monitoring.pipeline_logs import append_event


def main() -> int:
    setup_logging()
    try:
        df = load_reference()
        validate_dataframe(df)
        model, metrics = train_model(df)
        save_artifacts(model, metrics)
        append_event("train", "success", {"roc_auc": metrics["roc_auc"]})
        print(f"Modelo treinado — ROC-AUC={metrics['roc_auc']:.4f}")
        return 0
    except Exception as exc:
        append_event("train", "failure", {"error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
