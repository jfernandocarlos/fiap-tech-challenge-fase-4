"""Gera dataset de produção com drift a partir da referência."""

from __future__ import annotations

from credit.data.loader import load_reference
from credit.drift.simulate import save_production
from credit.logging_config import setup_logging
from credit.monitoring.pipeline_logs import append_event


def main() -> int:
    setup_logging()
    try:
        reference = load_reference()
        path = save_production(reference)
        append_event("simulate_drift", "success", {"production": str(path), "rows": len(reference)})
        print(f"Produção com drift: {path}")
        return 0
    except Exception as exc:
        append_event("simulate_drift", "failure", {"error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
