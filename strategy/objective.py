"""Shared CV objective for impulse grid and threshold optimization."""

from __future__ import annotations

import numpy as np

import config as _cfg


def fusion_cv_objective(
    *,
    sharpe: float,
    mean_return_pct: float,
    active_rebalances: int,
    mean_trade_net_bps: float | None = None,
    constraint_penalty: float = 0.0,
    fold_days: float = 30.0,
    n_cv_folds: int = 1,
) -> float:
    """Composite desk objective: Sharpe + return − churn − loss penalty ± frequency/expectancy."""
    ret_scale = max(float(getattr(_cfg, "FUSION_RETURN_OBJECTIVE_SCALE", 10.0)), 1e-9)
    ret_component = float(mean_return_pct) / ret_scale
    turnover_penalty = float(getattr(_cfg, "FUSION_TURNOVER_PENALTY", 0.04)) * float(active_rebalances)
    negative_return_penalty = float(getattr(_cfg, "FUSION_NEGATIVE_RETURN_PENALTY", 2.0)) * max(0.0, -ret_component)

    target_tpy = float(getattr(_cfg, "FUSION_TARGET_TRADES_PER_YEAR", 30.0))
    freq_weight = float(getattr(_cfg, "FUSION_FREQUENCY_OBJECTIVE_WEIGHT", 0.0))
    est_tpy = float(active_rebalances) * (365.0 / max(float(fold_days), 7.0))
    freq_bonus = 0.0
    if freq_weight > 0.0 and ret_component > 0.0:
        freq_bonus = freq_weight * min(1.0, est_tpy / max(target_tpy, 1.0))

    expectancy_bonus = 0.0
    expectancy_penalty = 0.0
    exp_weight = float(getattr(_cfg, "FUSION_EXPECTANCY_OBJECTIVE_WEIGHT", 0.5))
    if mean_trade_net_bps is not None and np.isfinite(mean_trade_net_bps):
        if mean_trade_net_bps >= 0.0:
            expectancy_bonus = exp_weight * max(0.0, float(mean_trade_net_bps) / 20.0)
        else:
            expectancy_penalty = exp_weight * abs(float(mean_trade_net_bps)) / 10.0

    base = float(sharpe) + ret_component - turnover_penalty - negative_return_penalty - float(constraint_penalty)
    if base < 0.0 and freq_bonus > 0.0:
        base -= freq_bonus
    else:
        base += freq_bonus
    return base + expectancy_bonus - expectancy_penalty


def threshold_no_trade_objective() -> float:
    return float(getattr(_cfg, "FUSION_THRESHOLD_NO_TRADE_OBJECTIVE", -2.0))
