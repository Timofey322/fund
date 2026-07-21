"""
ML likelihood / calibration diagnostics for walk-forward OOS panels.

Uses BCE (log-loss) as the proper likelihood for binary direction labels —
aligned with HistGradientBoostingClassifier loss and HMM emission likelihood spirit.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.6931
    try:
        return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    except Exception:
        return 0.6931


def ml_likelihood_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    """
    Confusion-style matrix + global likelihood metrics.

    Rows = actual label, cols = predicted class (0=down, 1=up).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)

    cells: dict[str, float] = {}
    for actual in (0, 1):
        mask = y == actual
        n = int(mask.sum())
        if n == 0:
            cells[f"actual_{actual}_n"] = 0
            cells[f"actual_{actual}_p_pred_0"] = 0.0
            cells[f"actual_{actual}_p_pred_1"] = 0.0
            continue
        cells[f"actual_{actual}_n"] = n
        cells[f"actual_{actual}_p_pred_0"] = round(float((pred[mask] == 0).mean()), 4)
        cells[f"actual_{actual}_p_pred_1"] = round(float((pred[mask] == 1).mean()), 4)

    ll = _safe_log_loss(y, p)
    brier = float(brier_score_loss(y, p)) if len(y) > 0 else 0.25
    auc = 0.5
    if len(np.unique(y)) > 1:
        try:
            auc = float(roc_auc_score(y, p))
        except Exception:
            pass

    return {
        "threshold": threshold,
        "matrix": {
            "actual_down": {
                "n": cells.get("actual_0_n", 0),
                "p_pred_down": cells.get("actual_0_p_pred_0", 0.0),
                "p_pred_up": cells.get("actual_0_p_pred_1", 0.0),
            },
            "actual_up": {
                "n": cells.get("actual_1_n", 0),
                "p_pred_down": cells.get("actual_1_p_pred_0", 0.0),
                "p_pred_up": cells.get("actual_1_p_pred_1", 0.0),
            },
        },
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "auc": round(auc, 4),
        "random_log_loss": 0.6931,
        "likelihood_ratio_vs_random": round(0.6931 / max(ll, 1e-6), 4),
    }


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> list[dict]:
    """Reliability diagram bins: predicted proba vs observed frequency."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin_lo": round(float(lo), 2),
            "bin_hi": round(float(hi), 2),
            "n": n,
            "mean_pred": round(float(p[mask].mean()), 4),
            "observed_freq": round(float(y[mask].mean()), 4),
            "gap": round(float(y[mask].mean() - p[mask].mean()), 4),
        })
    return rows


def composite_ml_score(auc: float, logloss: float, ic: float = 0.0) -> float:
    """Purged-CV objective: ranking + likelihood penalty vs random (0.693)."""
    ll_pen = max(0.0, logloss - 0.693)
    return 0.40 * auc + 0.30 * max(ic, 0.0) - 0.30 * ll_pen
