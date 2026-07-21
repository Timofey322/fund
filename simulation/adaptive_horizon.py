"""Regime-conditional adaptive holding horizon.

Horizons are chosen on **train or validation** rows only (never stitched OOS
test months). For each dominant HMM regime we maximise de-overlapped after-cost
expected value over ``HOLD_BUCKETS`` candidates.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from simulation.entry_signals import deoverlap_signals, round_trip_cost_bps
from common.naming import (
    COL_PROB_HMM_IMPULSE,
    COL_PROB_HMM_MEAN_REVERT,
    COL_PROB_HMM_STRESS,
    HMM_IMPULSE,
    HMM_MEAN_REVERT,
    HMM_STRESS,
)
from research.labels.trade import DEFAULT_SLIPPAGE_BPS, HOLD_BUCKETS

DEFAULT_HORIZON_CANDIDATES: tuple[int, ...] = HOLD_BUCKETS
DEFAULT_MIN_TRADES = 20

_REGIME_PROB_COL = {
    HMM_IMPULSE: COL_PROB_HMM_IMPULSE,
    HMM_MEAN_REVERT: COL_PROB_HMM_MEAN_REVERT,
    HMM_STRESS: COL_PROB_HMM_STRESS,
}


def dominant_regime_series(signals: pd.DataFrame) -> pd.Series:
    """Per-row dominant HMM regime from the three probability columns.

    Rows lacking the probability columns get ``mean_revert`` as a neutral
    default (never the blocked ``stress`` bucket).
    """
    cols = {name: col for name, col in _REGIME_PROB_COL.items() if col in signals.columns}
    if not cols or signals.empty:
        return pd.Series([HMM_MEAN_REVERT] * len(signals), index=signals.index, dtype=object)
    probs = signals[list(cols.values())].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    names = list(cols.keys())
    idx = probs.to_numpy().argmax(axis=1)
    return pd.Series([names[i] for i in idx], index=signals.index, dtype=object)


def _forward_returns_at_horizon(
    entries: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    """Net-of-nothing forward return held exactly ``horizon`` bars per entry.

    Returns are read from ``prices`` (close[p+h]/close[p] - 1) so the same entry
    set can be evaluated at different horizons. Entries without ``horizon`` future
    bars are dropped.
    """
    if entries.empty or prices is None or prices.empty:
        return np.array([], dtype=float)
    h = max(1, int(horizon))
    out: list[float] = []
    index = prices.index
    n = len(index)
    for ticker, grp in entries.groupby("ticker", sort=False):
        if ticker not in prices.columns:
            continue
        col = prices[ticker].to_numpy(dtype=float)
        dates = pd.to_datetime(grp["date"].to_numpy())
        pos = index.searchsorted(dates, side="left")
        for p in pos:
            p = int(p)
            q = p + h
            if p < 0 or q >= n:
                continue
            c0 = col[p]
            c1 = col[q]
            if not np.isfinite(c0) or not np.isfinite(c1) or c0 <= 0:
                continue
            out.append(c1 / c0 - 1.0)
    return np.asarray(out, dtype=float)


def horizon_ev_bps(
    entries: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
    *,
    commission_bps: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict:
    """De-overlapped after-cost EV (bps) of holding ``entries`` for ``horizon``."""
    empty = {"horizon": int(horizon), "n_trades": 0, "ev_bps": None, "ev_tstat": None}
    if entries.empty:
        return empty
    deov = deoverlap_signals(entries, prices, horizon)
    gross = _forward_returns_at_horizon(deov, prices, horizon)
    n = int(len(gross))
    if n == 0:
        return empty
    cost = round_trip_cost_bps(commission_bps, slippage_bps) / 10_000.0
    net = gross - cost
    mean = float(np.mean(net))
    std = float(np.std(net, ddof=1)) if n > 1 else 0.0
    tstat = float(mean / (std / np.sqrt(n))) if std > 1e-12 and n > 1 else 0.0
    return {
        "horizon": int(horizon),
        "n_trades": n,
        "ev_bps": round(mean * 10_000.0, 3),
        "ev_tstat": round(tstat, 4),
    }


def select_regime_horizons(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    commission_bps: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    candidates: tuple[int, ...] = DEFAULT_HORIZON_CANDIDATES,
    default_horizon: int,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> tuple[dict[str, int], dict]:
    """Pick the EV-maximising horizon per dominant HMM regime.

    A regime adopts a non-default horizon only with >= ``min_trades`` independent
    trades and a positive EV; otherwise it keeps ``default_horizon``. Returns
    ``(regime -> horizon, detail_table)``.
    """
    cand = tuple(sorted({int(c) for c in candidates if int(c) > 0}))
    chosen: dict[str, int] = {}
    detail: dict = {"default_horizon": int(default_horizon), "candidates": list(cand), "regimes": {}}
    if signals is None or signals.empty or not cand:
        return {r: int(default_horizon) for r in _REGIME_PROB_COL}, detail

    sig = signals.copy()
    sig["__regime__"] = dominant_regime_series(sig)
    for regime in _REGIME_PROB_COL:
        part = sig[sig["__regime__"] == regime]
        rows = [
            horizon_ev_bps(
                part, prices, h, commission_bps=commission_bps, slippage_bps=slippage_bps
            )
            for h in cand
        ]
        viable = [r for r in rows if r["n_trades"] >= min_trades and (r["ev_bps"] or -1.0) > 0.0]
        if viable:
            best = max(viable, key=lambda r: (r["ev_bps"], r["ev_tstat"], r["n_trades"]))
            chosen[regime] = int(best["horizon"])
            fallback = False
        else:
            chosen[regime] = int(default_horizon)
            best = None
            fallback = True
        detail["regimes"][regime] = {
            "chosen_horizon": chosen[regime],
            "fallback_to_default": fallback,
            "ev_by_horizon": rows,
        }
    return chosen, detail
