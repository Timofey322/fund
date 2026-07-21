"""
Stochastic survival simulations for intraday / bar-level equity curves.

Block bootstrap preserves short-term autocorrelation in bar returns.
Reports P(terminal loss), P(maxDD > X%), wealth percentiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import math

import numpy as np
import pandas as pd


def _bar_returns(equity: pd.Series) -> np.ndarray:
    rets = equity.pct_change().dropna().values
    return rets[np.isfinite(rets)]


def block_bootstrap_paths(
    returns: np.ndarray,
    n_paths: int,
    horizon: int,
    block: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Cumulative wealth paths W_t starting at 1.0."""
    if len(returns) < block + 2:
        return np.ones((min(n_paths, 1), max(horizon, 1)))
    rng = np.random.default_rng(seed)
    n_blocks = max(1, math.ceil(horizon / block))
    paths = np.ones((n_paths, horizon))
    for i in range(n_paths):
        chunks: list[float] = []
        while len(chunks) < horizon:
            start = int(rng.integers(0, len(returns) - block + 1))
            chunks.extend(returns[start : start + block].tolist())
        path_rets = np.array(chunks[:horizon])
        paths[i] = np.cumprod(1.0 + path_rets)
    return paths


def survival_simulation(
    equity: pd.Series,
    *,
    n_paths: int = 2000,
    horizon_bars: int | None = None,
    block_bars: int = 12,
    max_dd_threshold: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo survival stats from observed bar equity curve.

    horizon_bars: forward simulation length (default = len of observed returns).
    block_bars: bootstrap block (~1h on 5Min when block=12).
    """
    rets = _bar_returns(equity)
    if len(rets) < 10:
        return {"n_paths": 0, "n_bars": len(rets), "error": "insufficient_returns"}

    horizon = horizon_bars or len(rets)
    paths = block_bootstrap_paths(rets, n_paths, horizon, block_bars, seed=seed)
    terminal = paths[:, -1]
    running_max = np.maximum.accumulate(paths, axis=1)
    dd = paths / running_max - 1.0
    max_dd = dd.min(axis=1)

    return {
        "n_paths": n_paths,
        "horizon_bars": horizon,
        "block_bars": block_bars,
        "observed_bars": len(rets),
        "prob_terminal_loss": round(float((terminal < 1.0).mean()), 4),
        f"prob_maxdd_{int(max_dd_threshold * 100)}pct": round(
            float((max_dd < -max_dd_threshold).mean()), 4
        ),
        "terminal_wealth_p5": round(float(np.percentile(terminal, 5)), 4),
        "terminal_wealth_p50": round(float(np.percentile(terminal, 50)), 4),
        "terminal_wealth_p95": round(float(np.percentile(terminal, 95)), 4),
        "max_dd_p5": round(float(np.percentile(max_dd, 5)), 4),
        "max_dd_p50": round(float(np.percentile(max_dd, 50)), 4),
        "max_dd_p95": round(float(np.percentile(max_dd, 95)), 4),
        "survival_rate": round(float((terminal >= 1.0).mean()), 4),
    }
