"""Gera dataset de referência e lote inválido de demonstração."""

from __future__ import annotations

from credit.data.generate import save_invalid_batch, save_reference
from credit.logging_config import setup_logging
from credit.monitoring.pipeline_logs import append_event


def main() -> int:
    setup_logging()
    try:
        ref = save_reference()
        bad = save_invalid_batch()
        append_event(
            "generate_data",
            "success",
            {"reference": str(ref), "invalid_batch": str(bad)},
        )
        print(f"Referência: {ref}")
        print(f"Lote inválido: {bad}")
        return 0
    except Exception as exc:
        append_event("generate_data", "failure", {"error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
