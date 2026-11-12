"""Valida um CSV contra o contrato Pandera.

Uso:
  poetry run python -m scripts.validate_data data/reference/credit_reference.csv
  poetry run python -m scripts.validate_data data/invalid/credit_invalid_batch.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from credit.data.contracts import DataContractError, validate_dataframe
from credit.data.loader import load_csv
from credit.logging_config import setup_logging
from credit.monitoring.pipeline_logs import append_event


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Valida CSV contra contrato Pandera")
    parser.add_argument("csv_path", type=Path, help="Caminho do CSV a validar")
    args = parser.parse_args()

    df = load_csv(args.csv_path)
    try:
        validate_dataframe(df, raise_on_error=True)
    except DataContractError as exc:
        append_event(
            "validate_data",
            "failure",
            {"path": str(args.csv_path), "errors": exc.failure_cases[:5]},
        )
        print(f"BLOQUEADO: {exc}")
        for case in exc.failure_cases[:10]:
            print(f"  - {case}")
        return 1

    append_event("validate_data", "success", {"path": str(args.csv_path), "rows": len(df)})
    print(f"OK: contrato válido ({len(df)} linhas) — {args.csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
