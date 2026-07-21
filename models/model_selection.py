"""Multi-criteria purged-CV scoring for entry model selection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config as _cfg
from research.features.entry_ml import _spearman_ic

DEFAULT_CRITERIA_WEIGHTS: dict[str, float] = {
    "accuracy": 0.35,
    "top_decile_net_bps": 0.25,
    "edge_corr": 0.10,
    "ic_spearman": 0.08,
    "auc": 0.08,
    "pr_auc": 0.05,
    "log_loss": 0.04,
    "brier": 0.03,
    "calibration_mae": 0.02,
    "precision": 0.00,
    "recall": 0.00,
    "fold_stability": 0.00,
}

CRITERIA_KEYS = tuple(DEFAULT_CRITERIA_WEIGHTS.keys())


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return 0.5


def _safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float(np.mean(y)) if len(y) else 0.0
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float(np.mean(y))


def _safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.6931
    try:
        return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    except Exception:
        return 0.6931


def _calibration_mae(y: np.ndarray, p: np.ndarray, n_bins: int = 5) -> float:
    if len(y) == 0:
        return 0.25
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    gaps: list[float] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        gaps.append(abs(float(y[mask].mean()) - float(p[mask].mean())))
    return float(np.mean(gaps)) if gaps else 0.25


def _edge_corr(p: np.ndarray, fwd_ret: np.ndarray | None) -> float:
    if fwd_ret is None or len(p) < 6:
        return 0.0
    fr = np.asarray(fwd_ret, dtype=float)
    mask = np.isfinite(fr) & np.isfinite(p)
    if mask.sum() < 6 or np.std(fr[mask]) < 1e-12 or np.std(p[mask]) < 1e-12:
        return 0.0
    return float(np.corrcoef(p[mask], fr[mask])[0, 1])


def compute_fold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fwd_ret: np.ndarray | None = None,
    *,
    decision_threshold: float | None = None,
    commission_bps: float | None = None,
    tickers: np.ndarray | None = None,
) -> dict[str, float]:
    """Per-fold metrics for one validation slice."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    thr = decision_threshold
    if thr is None:
        thr = max(0.5, float(np.mean(y))) if len(y) else 0.5
    pred = (p >= thr).astype(int)

    prec = 0.0
    rec = 0.0
    if pred.sum() > 0:
        prec = float(precision_score(y, pred, zero_division=0))
    if y.sum() > 0:
        rec = float(recall_score(y, pred, zero_division=0))

    out: dict[str, float] = {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "auc": round(_safe_auc(y, p), 4),
        "pr_auc": round(_safe_pr_auc(y, p), 4),
        "log_loss": round(_safe_log_loss(y, p), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ic_spearman": round(_spearman_ic(p, fwd_ret if fwd_ret is not None else np.zeros_like(p)), 4),
        "edge_corr": round(_edge_corr(p, fwd_ret), 4),
        "calibration_mae": round(_calibration_mae(y, p), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "decision_threshold": round(float(thr), 4),
    }
    if fwd_ret is not None:
        from models.profit_metrics import top_decile_stats

        t_arr = np.asarray(tickers, dtype=object) if tickers is not None else None
        stats = top_decile_stats(
            p,
            fwd_ret,
            commission_bps=float(commission_bps or getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 1.1)),
            tickers=t_arr,
        )
        out["top_decile_net_bps"] = round(float(stats["top_decile_net_bps"]), 3)
        out["top_decile_gross_bps"] = round(float(stats["top_decile_gross_bps"]), 3)
        rt_v = stats.get("top_decile_rt_bps")
        if rt_v is not None and math.isfinite(float(rt_v)):
            out["top_decile_rt_bps"] = round(float(rt_v), 3)
        if t_arr is not None:
            vals: list[float] = []
            gross_vals: list[float] = []
            rt_vals: list[float] = []
            for sym in np.unique(t_arr):
                m = t_arr == sym
                if m.sum() < 50:
                    continue
                st = top_decile_stats(
                    p[m], fwd_ret[m], commission_bps=float(commission_bps or 1.1), tickers=t_arr[m],
                )
                net = st["top_decile_net_bps"]
                if np.isfinite(net):
                    vals.append(float(net))
                    gross_vals.append(float(st["top_decile_gross_bps"]))
                    if math.isfinite(float(st.get("top_decile_rt_bps", float("nan")))):
                        rt_vals.append(float(st["top_decile_rt_bps"]))
            if vals:
                out["mean_ticker_top_decile_net_bps"] = round(float(np.mean(vals)), 3)
                out["top_decile_net_bps"] = out["mean_ticker_top_decile_net_bps"]
                out["top_decile_gross_bps"] = round(float(np.mean(gross_vals)), 3)
                if rt_vals:
                    out["top_decile_rt_bps"] = round(float(np.mean(rt_vals)), 3)
    return out


def _criterion_score(name: str, value: float) -> float:
    """Map each metric to ~[0, 1] where higher is better."""
    if name == "auc":
        return max(0.0, min(1.0, (value - 0.5) * 2.0))
    if name == "pr_auc":
        return max(0.0, min(1.0, value))
    if name == "log_loss":
        return max(0.0, min(1.0, (0.693 - value) / 0.693))
    if name == "brier":
        return max(0.0, min(1.0, (0.25 - value) / 0.25))
    if name in ("ic_spearman", "edge_corr"):
        return max(0.0, min(1.0, (value + 1.0) / 2.0))
    if name == "calibration_mae":
        return max(0.0, min(1.0, (0.25 - value) / 0.25))
    if name in ("precision", "recall", "accuracy"):
        return max(0.0, min(1.0, value))
    if name == "fold_stability":
        return max(0.0, min(1.0, 1.0 - min(value / 0.15, 1.0)))
    if name == "top_decile_net_bps":
        from models.profit_metrics import profitability_score

        return profitability_score(value)
    return 0.0


def resolve_optuna_objective(metrics: dict[str, float]) -> float:
    """Map fold CV metrics to the scalar Optuna maximizes."""
    from models.profit_metrics import (
        optuna_reject_score,
        profitability_score,
        rejects_optuna_trial,
    )

    if rejects_optuna_trial(metrics):
        return optuna_reject_score()
    mode = str(getattr(_cfg, "FUSION_OPTUNA_OBJECTIVE", "composite")).lower()
    if mode == "accuracy":
        return float(metrics.get("accuracy", 0.0))
    if mode == "profit":
        base = profitability_score(metrics.get("top_decile_net_bps", float("-inf")))
        # Reward clearing RT with margin (bps → mild bump in [0, 1] space).
        gross = metrics.get("top_decile_gross_bps")
        rt = metrics.get("top_decile_rt_bps")
        if (
            gross is not None
            and rt is not None
            and math.isfinite(float(gross))
            and math.isfinite(float(rt))
        ):
            margin = float(gross) - float(rt)
            base = float(min(1.0, base + 0.15 * max(0.0, margin) / 25.0))
        # Penalize train≫val top-decile gap (same economics as composite path).
        if bool(getattr(_cfg, "FUSION_OPTUNA_APPLY_GAP_PENALTY", True)):
            gap = metrics.get("train_oos_gap_bps")
            if gap is None and metrics.get("gap_penalty") is not None:
                base = float(max(0.0, base - float(metrics["gap_penalty"])))
            elif gap is not None and math.isfinite(float(gap)):
                gap_w = float(getattr(_cfg, "FUSION_TRAIN_OOS_GAP_WEIGHT", 0.45))
                gap_scale = float(getattr(_cfg, "FUSION_TRAIN_OOS_GAP_SCALE_BPS", 8.0))
                penalty = gap_w * min(float(gap) / max(gap_scale, 1.0), 1.0)
                base = float(max(0.0, base - penalty))
        return base
    return float(metrics.get("composite", -1.0))


def entry_model_composite(
    metrics: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted multi-criteria score (higher = better entry model)."""
    w = weights or getattr(_cfg, "ENTRY_MODEL_CRITERIA_WEIGHTS", DEFAULT_CRITERIA_WEIGHTS)
    total_w = sum(w.get(k, 0.0) for k in CRITERIA_KEYS)
    if total_w <= 0:
        return 0.0
    score = 0.0
    for key in CRITERIA_KEYS:
        if key not in metrics:
            continue
        score += w.get(key, 0.0) * _criterion_score(key, float(metrics[key]))
    return round(score / total_w, 4)


def aggregate_cv_metrics(fold_rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean metrics across folds + fold stability penalty on composite."""
    if not fold_rows:
        return {"composite": -1.0}
    keys = [k for k in fold_rows[0] if k != "decision_threshold" and not isinstance(fold_rows[0].get(k), dict)]
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(r[key]) for r in fold_rows if key in r]
        if vals:
            out[key] = round(float(np.mean(vals)), 4)
            if len(vals) > 1:
                out[f"{key}_std"] = round(float(np.std(vals)), 4)

    fold_composites = [
        entry_model_composite(row, weights={k: v for k, v in DEFAULT_CRITERIA_WEIGHTS.items() if k != "fold_stability"})
        for row in fold_rows
    ]
    out["fold_stability"] = round(float(np.std(fold_composites)) if len(fold_composites) > 1 else 0.0, 4)
    out["composite"] = entry_model_composite(out)

    gaps = [float(r["train_oos_gap_bps"]) for r in fold_rows if r.get("train_oos_gap_bps") is not None]
    if gaps:
        mean_gap = float(np.mean(gaps))
        out["train_oos_gap_bps"] = round(mean_gap, 3)
        gap_w = float(getattr(_cfg, "FUSION_TRAIN_OOS_GAP_WEIGHT", 0.15))
        gap_scale = float(getattr(_cfg, "FUSION_TRAIN_OOS_GAP_SCALE_BPS", 10.0))
        penalty = gap_w * min(mean_gap / max(gap_scale, 1.0), 1.0)
        out["gap_penalty"] = round(penalty, 4)
        out["composite"] = round(max(-1.0, float(out["composite"]) - penalty), 4)
    return out


def criteria_breakdown(metrics: dict[str, float]) -> dict[str, Any]:
    """Human-readable contribution of each criterion to the composite."""
    w = getattr(_cfg, "ENTRY_MODEL_CRITERIA_WEIGHTS", DEFAULT_CRITERIA_WEIGHTS)
    rows: dict[str, Any] = {}
    for key in CRITERIA_KEYS:
        if key not in metrics:
            continue
        raw = float(metrics[key])
        norm = _criterion_score(key, raw)
        rows[key] = {
            "raw": raw,
            "normalized": round(norm, 4),
            "weight": w.get(key, 0.0),
            "contribution": round(w.get(key, 0.0) * norm, 4),
        }
    rows["composite"] = entry_model_composite(metrics)
    return rows
