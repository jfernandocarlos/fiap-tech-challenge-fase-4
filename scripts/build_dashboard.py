"""Gera o dashboard HTML consolidado de monitoramento."""

from __future__ import annotations

from credit.logging_config import setup_logging
from credit.monitoring.pipeline_logs import append_event
from credit.reporting.dashboard import generate_dashboard


def main() -> int:
    setup_logging()
    try:
        path = generate_dashboard()
        append_event("build_dashboard", "success", {"path": str(path)})
        print(f"Dashboard: {path}")
        return 0
    except Exception as exc:
        append_event("build_dashboard", "failure", {"error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
