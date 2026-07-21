"""Profitability metrics for model CV — aligned with decile gate economics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config as _cfg
from data_platform.universe import commission_bps_for_ticker
from simulation.execution_costs import round_trip_cost_bps, slippage_bps_per_side
from research.labels.trade import DEFAULT_SLIPPAGE_BPS


def top_decile_stats(
    proba: np.ndarray,
    fwd_ret: np.ndarray,
    *,
    commission_bps: float,
    slippage_bps: float | None = None,
    tickers: np.ndarray | pd.Series | None = None,
    n_deciles: int = 10,
    min_rows: int = 50,
) -> dict[str, float]:
    """Top-proba decile gross/net/RT (bps). Missing/small sample → -inf nets."""
    p = np.asarray(proba, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    mask = np.isfinite(p) & np.isfinite(r)
    need = max(n_deciles * 5, int(min_rows))
    if int(mask.sum()) < need:
        return {
            "top_decile_net_bps": float("-inf"),
            "top_decile_gross_bps": float("-inf"),
            "top_decile_rt_bps": float("nan"),
        }
    p = p[mask]
    r = r[mask]
    if tickers is not None:
        t = np.asarray(tickers, dtype=object)[mask]
        rt = np.array(
            [
                round_trip_cost_bps(commission_bps_for_ticker(str(sym)), slippage_bps_per_side(str(sym)))
                for sym in t
            ],
            dtype=float,
        )
    else:
        slip = DEFAULT_SLIPPAGE_BPS if slippage_bps is None else float(slippage_bps)
        rt = np.full(len(p), round_trip_cost_bps(float(commission_bps), slip), dtype=float)
    try:
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_deciles + 1))
        top = p >= edges[-2]
    except Exception:
        k = max(1, len(p) // n_deciles)
        order = np.argsort(p)
        top = np.zeros(len(p), dtype=bool)
        top[order[-k:]] = True
    if not top.any():
        return {
            "top_decile_net_bps": float("-inf"),
            "top_decile_gross_bps": float("-inf"),
            "top_decile_rt_bps": float("nan"),
        }
    gross_bps = float(np.mean(r[top])) * 10_000.0
    rt_mean = float(np.mean(rt[top]))
    net_bps = gross_bps - rt_mean
    return {
        "top_decile_net_bps": net_bps,
        "top_decile_gross_bps": gross_bps,
        "top_decile_rt_bps": rt_mean,
    }


def top_decile_net_bps(
    proba: np.ndarray,
    fwd_ret: np.ndarray,
    *,
    commission_bps: float,
    slippage_bps: float | None = None,
    tickers: np.ndarray | pd.Series | None = None,
    n_deciles: int = 10,
    min_rows: int = 50,
) -> float:
    """Mean net bps in the top probability decile (after round-trip costs)."""
    return float(
        top_decile_stats(
            proba,
            fwd_ret,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            tickers=tickers,
            n_deciles=n_deciles,
            min_rows=min_rows,
        )["top_decile_net_bps"]
    )


def profitability_score(net_bps: float, *, scale_bps: float = 25.0) -> float:
    """Map net bps to [0, 1] for composite weighting.

    ``-scale → 0``, ``0 → 0.5``, ``+scale → 1``. Negative edge is penalized
    (not floored at zero), so AUC alone cannot win Optuna.
    """
    if not np.isfinite(net_bps):
        return 0.0
    s = max(float(scale_bps), 1.0)
    return float(max(0.0, min(1.0, 0.5 + 0.5 * (float(net_bps) / s))))


def rejects_gross_below_rt(metrics: dict | None) -> bool:
    """True when top-decile gross cannot clear round-trip cost (+ optional margin)."""
    if not bool(getattr(_cfg, "FUSION_OPTUNA_REJECT_GROSS_BELOW_RT", True)):
        return False
    pol = metrics or {}
    gross = pol.get("top_decile_gross_bps")
    rt = pol.get("top_decile_rt_bps")
    if gross is None or rt is None:
        return False
    if not (math.isfinite(float(gross)) and math.isfinite(float(rt))):
        return False
    margin = float(getattr(_cfg, "FUSION_OPTUNA_MIN_GROSS_OVER_RT_BPS", 0.0))
    return float(gross) < float(rt) + margin


def rejects_optuna_trial(metrics: dict | None) -> bool:
    """Unified hard reject for Optuna (gross&lt;RT, net below gate, train/val gap)."""
    pol = metrics or {}
    if rejects_gross_below_rt(pol):
        return True
    min_net = getattr(_cfg, "FUSION_OPTUNA_MIN_NET_BPS", None)
    if min_net is not None:
        net = pol.get("top_decile_net_bps")
        if net is not None and math.isfinite(float(net)) and float(net) < float(min_net):
            return True
    max_gap = getattr(_cfg, "FUSION_OPTUNA_MAX_TRAIN_OOS_GAP_BPS", None)
    if max_gap is not None:
        gap = pol.get("train_oos_gap_bps")
        if gap is not None and math.isfinite(float(gap)) and float(gap) > float(max_gap):
            return True
    return False


def optuna_reject_score() -> float:
    """Scalar below any valid profitability/composite so rejected trials cannot win."""
    return -1.0


def optuna_reject_reason(metrics: dict | None) -> str | None:
    """Short reason string for rejected trials (logging / user_attr)."""
    pol = metrics or {}
    if rejects_gross_below_rt(pol):
        return "gross_below_rt"
    min_net = getattr(_cfg, "FUSION_OPTUNA_MIN_NET_BPS", None)
    net = pol.get("top_decile_net_bps")
    if (
        min_net is not None
        and net is not None
        and math.isfinite(float(net))
        and float(net) < float(min_net)
    ):
        return "net_below_min"
    max_gap = getattr(_cfg, "FUSION_OPTUNA_MAX_TRAIN_OOS_GAP_BPS", None)
    gap = pol.get("train_oos_gap_bps")
    if (
        max_gap is not None
        and gap is not None
        and math.isfinite(float(gap))
        and float(gap) > float(max_gap)
    ):
        return "train_oos_gap"
    return None
