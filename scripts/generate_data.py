"""Gera dataset de referência e lote inválido de demonstração."""

from __future__ import annotations

from credit.data.generate import save_invalid_batch, save_reference
from credit.logging_config import setup_logging


def main() -> int:
    setup_logging()
    ref = save_reference()
    bad = save_invalid_batch()
    print(f"Referência: {ref}")
    print(f"Lote inválido: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
