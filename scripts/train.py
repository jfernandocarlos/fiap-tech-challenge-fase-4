"""Treina o modelo baseline de credit scoring."""

from __future__ import annotations

from credit.data.contracts import validate_dataframe
from credit.data.loader import load_reference
from credit.logging_config import setup_logging
from credit.models.training import save_artifacts, train_model


def main() -> int:
    setup_logging()
    df = load_reference()
    validate_dataframe(df)
    model, metrics = train_model(df)
    save_artifacts(model, metrics)
    print(f"Modelo treinado — ROC-AUC={metrics['roc_auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
