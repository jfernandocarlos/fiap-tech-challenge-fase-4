"""Gera dashboard HTML consolidado de monitoramento."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from credit.config import (
    DRIFT_SUMMARY_JSON,
    METRICS_PATH,
    MONITORING_DASHBOARD_HTML,
)
from credit.logging_config import get_logger
from credit.monitoring.pipeline_logs import summarize_health

logger = get_logger(__name__)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_dashboard_html(
    metrics: dict,
    drift: dict,
    health: dict,
) -> str:
    """Monta HTML simples e legível (sem framework)."""
    drifted = drift.get("drifted_features", [])
    feature_rows = ""
    for row in drift.get("features", []):
        flag = "SIM" if row["drift_detected"] else "não"
        css = "bad" if row["drift_detected"] else "ok"
        feature_rows += (
            f"<tr class='{css}'>"
            f"<td>{row['feature']}</td>"
            f"<td>{row['psi']}</td>"
            f"<td>{row['ks_statistic']}</td>"
            f"<td>{row['ks_pvalue']:.4g}</td>"
            f"<td>{flag}</td>"
            f"</tr>"
        )

    last = health.get("last_event") or {}
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Credit Scoring — Monitoring Dashboard</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; background: #f7f4ef; color: #1c1c1c; }}
    h1, h2 {{ font-family: Helvetica, Arial, sans-serif; }}
    .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
    .card {{ background: #fff; border: 1px solid #d9d2c5; padding: 1rem 1.25rem; min-width: 160px; }}
    .card strong {{ display: block; font-size: 1.4rem; margin-top: .35rem; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; }}
    th, td {{ border: 1px solid #d9d2c5; padding: .5rem .7rem; text-align: left; }}
    th {{ background: #ebe4d8; }}
    tr.bad {{ background: #f8e4e0; }}
    tr.ok {{ background: #e9f3ea; }}
    .alert {{ color: #8a1f11; font-weight: bold; }}
    .ok-text {{ color: #1f6b3a; }}
    footer {{ margin-top: 2rem; color: #666; font-size: .9rem; }}
  </style>
</head>
<body>
  <h1>Credit Scoring — Observabilidade</h1>
  <p>Saúde do pipeline, métricas do modelo e alertas de drift.</p>

  <div class="cards">
    <div class="card">ROC-AUC<strong>{metrics.get("roc_auc", "—")}</strong></div>
    <div class="card">Accuracy<strong>{metrics.get("accuracy", "—")}</strong></div>
    <div class="card">F1<strong>{metrics.get("f1", "—")}</strong></div>
    <div class="card">Features com drift<strong class="{'alert' if drifted else 'ok-text'}">{len(drifted)}</strong></div>
    <div class="card">Eventos OK<strong>{health.get("success", 0)}</strong></div>
    <div class="card">Falhas<strong>{health.get("failure", 0)}</strong></div>
    <div class="card">Alertas<strong>{health.get("alerts", 0)}</strong></div>
  </div>

  <h2>Drift detectado</h2>
  <p>
    Features degradadas:
    <strong class="{'alert' if drifted else 'ok-text'}">{", ".join(drifted) if drifted else "nenhuma"}</strong>
  </p>
  <table>
    <thead>
      <tr><th>Feature</th><th>PSI</th><th>KS</th><th>p-value KS</th><th>Drift?</th></tr>
    </thead>
    <tbody>
      {feature_rows or "<tr><td colspan='5'>Sem dados de drift. Rode detect-drift.</td></tr>"}
    </tbody>
  </table>

  <h2>Último evento do pipeline</h2>
  <pre>{json.dumps(last, indent=2, ensure_ascii=False) if last else "sem eventos"}</pre>

  <footer>Gerado em {generated}. Relatório detalhado Evidently: reports/drift_report.html</footer>
</body>
</html>
"""


def generate_dashboard(path: Path = MONITORING_DASHBOARD_HTML) -> Path:
    """Lê artefatos locais e grava o dashboard HTML."""
    metrics = _load_json(METRICS_PATH)
    drift = _load_json(DRIFT_SUMMARY_JSON)
    health = summarize_health()
    html = build_dashboard_html(metrics, drift, health)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    logger.info("dashboard_generated", path=str(path))
    return path
