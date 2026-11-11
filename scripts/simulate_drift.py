"""Gera dataset de produção com drift a partir da referência."""

from __future__ import annotations

from credit.data.loader import load_reference
from credit.drift.simulate import save_production
from credit.logging_config import setup_logging


def main() -> int:
    setup_logging()
    reference = load_reference()
    path = save_production(reference)
    print(f"Produção com drift: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
