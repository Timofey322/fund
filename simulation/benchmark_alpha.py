"""
Comparative alpha: strategy vs buy-and-hold benchmark.

Jensen's alpha + beta via OLS on excess returns (CAPM single-factor):

    r_strat - rf = alpha + beta * (r_bench - rf) + eps

Also: information ratio, tracking error, up/down capture, correlation.
All metrics annualized with PERIODS_PER_YEAR (bar-aware).
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

import config as _cfg


def _periods_per_year() -> float:
    return float(getattr(_cfg, "PERIODS_PER_YEAR", 252))


def _bar_returns(equity: pd.Series) -> pd.Series:
    r = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return r


def _align(strat_eq: pd.Series, bench_eq: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = _bar_returns(strat_eq)
    b = _bar_returns(bench_eq)
    aligned = pd.concat([s, b], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return aligned.iloc[:, 0], aligned.iloc[:, 1]


def comparative_alpha(
    strat_equity: pd.Series,
    bench_equity: pd.Series,
    *,
    risk_free_ann: float = 0.04,
    label: str = "strategy_vs_benchmark",
) -> dict:
    """
    Single-factor (CAPM) alpha/beta + active-management metrics.

    Returns annualized alpha, beta, information ratio, capture ratios.
    """
    s, b = _align(strat_equity, bench_equity)
    if len(s) < 10:
        return {"label": label, "error": "insufficient_overlap", "n_obs": int(len(s))}

    ppy = _periods_per_year()
    rf_per = risk_free_ann / ppy

    s_ex = s - rf_per
    b_ex = b - rf_per

    # OLS: s_ex = alpha + beta * b_ex
    var_b = float(np.var(b_ex, ddof=1))
    if var_b < 1e-18:
        beta = 0.0
        alpha_per = float(s_ex.mean())
    else:
        cov = float(np.cov(s_ex, b_ex, ddof=1)[0, 1])
        beta = cov / var_b
        alpha_per = float(s_ex.mean() - beta * b_ex.mean())

    resid = s_ex - (alpha_per + beta * b_ex)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((s_ex - s_ex.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0

    alpha_ann = (1.0 + alpha_per) ** ppy - 1.0 if alpha_per > -1 else -1.0

    # Active return / tracking error / information ratio
    active = s - b
    te_ann = float(active.std(ddof=1)) * math.sqrt(ppy)
    ir = float(active.mean() * ppy) / te_ann if te_ann > 1e-12 else 0.0

    corr = float(np.corrcoef(s, b)[0, 1]) if len(s) > 2 else 0.0

    # Up/down capture: how strategy behaves when benchmark up vs down
    up = b > 0
    dn = b < 0
    up_cap = (
        float(s[up].mean() / b[up].mean()) if up.sum() > 1 and abs(b[up].mean()) > 1e-12 else 0.0
    )
    dn_cap = (
        float(s[dn].mean() / b[dn].mean()) if dn.sum() > 1 and abs(b[dn].mean()) > 1e-12 else 0.0
    )

    tot_s = float(strat_equity.iloc[-1] / strat_equity.iloc[0] - 1)
    tot_b = float(bench_equity.iloc[-1] / bench_equity.iloc[0] - 1)

    return {
        "label": label,
        "n_obs": int(len(s)),
        "alpha_per_bar": round(alpha_per, 8),
        "alpha_annualized_pct": round(alpha_ann * 100, 3),
        "beta": round(beta, 4),
        "r_squared": round(r_squared, 4),
        "correlation": round(corr, 4),
        "information_ratio": round(ir, 3),
        "tracking_error_ann_pct": round(te_ann * 100, 3),
        "up_capture": round(up_cap, 3),
        "down_capture": round(dn_cap, 3),
        "total_return_pct": round(tot_s * 100, 2),
        "benchmark_return_pct": round(tot_b * 100, 2),
        "excess_return_pct": round((tot_s - tot_b) * 100, 2),
    }


def per_symbol_alpha(
    strat_equity: pd.Series,
    prices: pd.DataFrame,
    symbols: list[str],
    *,
    risk_free_ann: float = 0.04,
) -> dict[str, dict]:
    """Alpha of the strategy vs each symbol's buy-and-hold (rebased to 1.0 at overlap)."""
    out: dict[str, dict] = {}
    for sym in symbols:
        if sym not in prices.columns:
            continue
        bench = prices[sym].dropna()
        if bench.empty:
            continue
        common = strat_equity.index.intersection(bench.index)
        if len(common) < 10:
            continue
        s = strat_equity.reindex(common).ffill().dropna()
        b = bench.reindex(common).ffill().dropna()
        common = s.index.intersection(b.index)
        s = s.loc[common] / s.loc[common].iloc[0]
        b = b.loc[common] / b.loc[common].iloc[0]
        out[sym] = comparative_alpha(
            s, b, risk_free_ann=risk_free_ann, label=f"vs_{sym}",
        )
    return out


def equal_weight_benchmark(prices: pd.DataFrame, symbols: list[str]) -> pd.Series:
    """Equal-weight buy-and-hold basket of the given symbols (rebased to 1.0)."""
    cols = [s for s in symbols if s in prices.columns]
    if not cols:
        return pd.Series(dtype=float)
    sub = prices[cols].dropna(how="all").ffill()
    norm = sub / sub.iloc[0]
    basket = norm.mean(axis=1)
    return basket.rename("ew_benchmark")
