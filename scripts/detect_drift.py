"""Roda detecção de drift (Evidently + PSI/KS)."""

from __future__ import annotations

from credit.config import DRIFT_REPORT_HTML, PRODUCTION_CSV
from credit.data.loader import load_production, load_reference
from credit.drift.detect import run_drift_detection
from credit.logging_config import setup_logging
from credit.models.training import predict_dataframe
from credit.monitoring.pipeline_logs import append_event


def main() -> int:
    setup_logging()
    try:
        reference = load_reference()
        production = load_production()

        scored = predict_dataframe(production)
        scored_path = PRODUCTION_CSV.with_name("credit_production_scored.csv")
        scored.to_csv(scored_path, index=False)

        summary = run_drift_detection(reference, production)
        status = "alert" if summary["n_drifted"] > 0 else "success"
        append_event(
            "detect_drift",
            status,
            {
                "n_drifted": summary["n_drifted"],
                "drifted_features": summary["drifted_features"],
                "html": str(DRIFT_REPORT_HTML),
            },
        )
        print(
            f"Drift: {summary['n_drifted']} features — {summary['drifted_features']}\n"
            f"Relatório: {DRIFT_REPORT_HTML}"
        )
        return 0
    except Exception as exc:
        append_event("detect_drift", "failure", {"error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
