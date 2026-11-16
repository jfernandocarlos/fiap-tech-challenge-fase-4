"""Orquestra o pipeline completo: dados → validação → treino → drift → dashboard."""

from __future__ import annotations

import sys

from credit.config import INVALID_CSV, REFERENCE_CSV
from credit.logging_config import setup_logging
from scripts import (
    build_dashboard,
    detect_drift,
    generate_data,
    simulate_drift,
    train,
    validate_data,
)


def main() -> int:
    setup_logging()
    print("=== 1/6 generate_data ===")
    generate_data.main()

    print("=== 2/6 validate reference (deve passar) ===")
    sys.argv = ["validate_data", str(REFERENCE_CSV)]
    code = validate_data.main()
    if code != 0:
        return code

    print("=== 3/6 validate invalid batch (deve falhar/bloquear) ===")
    sys.argv = ["validate_data", str(INVALID_CSV)]
    code = validate_data.main()
    if code == 0:
        print("ERRO: lote inválido passou no contrato — isso não deveria acontecer.")
        return 1
    print("(esperado) lote inválido bloqueado pelo contrato.")

    print("=== 4/6 train ===")
    train.main()

    print("=== 5/6 simulate + detect drift ===")
    simulate_drift.main()
    detect_drift.main()

    print("=== 6/6 dashboard ===")
    build_dashboard.main()
    print("Pipeline completo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
