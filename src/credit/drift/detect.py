"""Detecção de Data Drift com Evidently AI + resumo PSI/KS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from scipy import stats

from credit.config import (
    DRIFT_REPORT_HTML,
    DRIFT_SUMMARY_JSON,
    load_params,
    settings,
)
from credit.logging_config import get_logger
from credit.models.training import FEATURE_COLS

logger = get_logger(__name__)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """Calcula PSI entre duas distribuições univariadas."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(ref, quantiles))
    if len(breakpoints) < 3:
        return 0.0

    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)

    ref_perc = ref_counts / ref_counts.sum()
    cur_perc = cur_counts / cur_counts.sum()

    # Evita divisão/log por zero
    ref_perc = np.where(ref_perc == 0, 0.0001, ref_perc)
    cur_perc = np.where(cur_perc == 0, 0.0001, cur_perc)

    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))


def feature_drift_stats(reference: pd.DataFrame, current: pd.DataFrame) -> list[dict]:
    """PSI + Kolmogorov-Smirnov por feature numérica."""
    params = load_params()["drift"]
    psi_th = float(params.get("psi_threshold", settings.drift_psi_threshold))
    ks_th = float(params["ks_pvalue_threshold"])

    rows: list[dict] = []
    for col in FEATURE_COLS:
        psi = population_stability_index(
            reference[col].to_numpy(dtype=float),
            current[col].to_numpy(dtype=float),
        )
        ks_stat, ks_pvalue = stats.ks_2samp(
            reference[col].dropna(),
            current[col].dropna(),
        )
        drifted = bool(psi >= psi_th or ks_pvalue < ks_th)
        rows.append(
            {
                "feature": col,
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": float(ks_pvalue),
                "drift_detected": drifted,
            }
        )
    return rows


def run_drift_detection(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: Path = DRIFT_REPORT_HTML,
    summary_path: Path = DRIFT_SUMMARY_JSON,
) -> dict:
    """Gera relatório Evidently + JSON com PSI/KS."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Evidently compara apenas features (sem target/ids)
    ref_feat = reference[FEATURE_COLS].copy()
    cur_feat = current[FEATURE_COLS].copy()

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_feat, current_data=cur_feat)
    result.save_html(str(html_path))

    per_feature = feature_drift_stats(reference, current)
    drifted = [r for r in per_feature if r["drift_detected"]]
    summary = {
        "n_features": len(per_feature),
        "n_drifted": len(drifted),
        "drifted_features": [r["feature"] for r in drifted],
        "psi_threshold": float(load_params()["drift"]["psi_threshold"]),
        "features": per_feature,
        "evidently_html": str(html_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        "drift_detection_done",
        n_drifted=summary["n_drifted"],
        drifted_features=summary["drifted_features"],
        html=str(html_path),
    )
    return summary
