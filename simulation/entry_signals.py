"""Trade signal helpers: costs, de-overlap, entry filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from simulation.engine import _entry_eligible
from research.labels.trade import DEFAULT_SLIPPAGE_BPS
from simulation.execution_costs import (
    default_slippage_bps_per_side,
    round_trip_cost_bps as _rt_cost,
    round_trip_cost_bps_for_ticker,
    round_trip_cost_series,
    slippage_bps_per_side,
)


def round_trip_cost_bps(
    commission_bps: float,
    slippage_bps: float | None = None,
) -> float:
    """Total per-trade RT cost: 2×(commission + slippage) per side."""
    slip = default_slippage_bps_per_side() if slippage_bps is None else float(slippage_bps)
    return _rt_cost(float(commission_bps), slip)


def round_trip_commission_bps(commission_bps: float) -> float:
    """Entry + exit commission only (bps per side × 2)."""
    return 2.0 * float(commission_bps)


def default_stop_loss_bps() -> float:
    """Configured gross stop-loss distance (bps)."""
    import config as _cfg

    return float(getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0))


def min_tp_gross_bps(
    commission_bps: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    *,
    stop_loss_bps: float | None = None,
    buffer_bps: float | None = None,
) -> float:
    """Minimum gross take-profit (bps): ``SL + round-trip commission`` (+ optional buffer)."""
    import config as _cfg

    sl = float(stop_loss_bps if stop_loss_bps is not None else default_stop_loss_bps())
    comm = round_trip_commission_bps(commission_bps)
    slip = float(slippage_bps)
    buf = float(buffer_bps if buffer_bps is not None else getattr(_cfg, "FUSION_EDGE_BUFFER_BPS", 0.0))
    return sl + comm + slip + buf


def edge_floor_bps(
    commission_bps: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    *,
    stop_loss_bps: float | None = None,
    buffer_bps: float | None = None,
    mode: str | None = None,
) -> float:
    """Minimum predicted edge / gross TP (bps) for entry gating.

    ``sl_plus_commission`` (default): ``SL + 2×commission`` — each win must cover
    the stop distance plus fees.
    Legacy: ``commission_only``, ``full_cost``.
    """
    import config as _cfg

    buf = float(buffer_bps if buffer_bps is not None else getattr(_cfg, "FUSION_EDGE_BUFFER_BPS", 2.0))
    floor_mode = str(mode or getattr(_cfg, "FUSION_EDGE_FLOOR_MODE", "sl_plus_commission")).lower()
    if floor_mode in ("full_round_trip", "full_cost"):
        return round_trip_cost_bps(commission_bps, slippage_bps) + buf
    if floor_mode == "sl_plus_commission":
        return min_tp_gross_bps(
            commission_bps,
            slippage_bps,
            stop_loss_bps=stop_loss_bps,
            buffer_bps=buffer_bps,
        )
    return round_trip_commission_bps(commission_bps) + buf


def trade_returns_from_signals(
    signals: pd.DataFrame,
    commission_bps: float | None = None,
    *,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> np.ndarray:
    """Forward signal returns net of round-trip commission and slippage."""
    if signals.empty:
        return np.array([], dtype=float)
    ret_col = "fwd_ret_entry" if "fwd_ret_entry" in signals.columns else "fwd_ret"
    if ret_col not in signals.columns:
        return np.array([], dtype=float)
    gross = signals[ret_col].astype(float)
    if commission_bps is not None:
        cost = round_trip_cost_bps(commission_bps, slippage_bps) / 10_000.0
        return (gross - cost).to_numpy()
    if "ticker" in signals.columns:
        from data_platform.universe import commission_bps_for_ticker

        cost_bps = round_trip_cost_series(signals["ticker"])
        return (gross - cost_bps / 10_000.0).to_numpy()
    import config as _cfg

    comm = float(getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 1.1))
    cost = round_trip_cost_bps(comm, slippage_bps) / 10_000.0
    return (gross - cost).to_numpy()


def deoverlap_signals(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_bars: int | None = None,
) -> pd.DataFrame:
    """Keep only non-overlapping trades per ticker (one position per horizon)."""
    if signals.empty or "date" not in signals.columns or "ticker" not in signals.columns:
        return signals
    if "exit_horizon" in signals.columns:
        return deoverlap_signals_by_horizon(signals, prices)
    horizon = max(1, int(horizon_bars or 1))
    return _deoverlap_with_horizon(signals, prices, horizon)


def deoverlap_signals_by_horizon(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """De-overlap using per-row ``exit_horizon`` when present."""
    if signals.empty:
        return signals
    keep: list[Any] = []
    price_index = prices.index if prices is not None else None
    for ticker, grp in signals.groupby("ticker", sort=False):
        grp = grp.sort_values("date")
        dates = pd.to_datetime(grp["date"].to_numpy())
        horizons = grp["exit_horizon"].fillna(1).astype(int).to_numpy()
        if price_index is not None and ticker in getattr(prices, "columns", []):
            positions = price_index.searchsorted(dates)
        else:
            positions = np.arange(len(grp))
        last_exit = -1
        for pos, row_id, h in zip(positions, grp.index, horizons):
            if int(pos) >= last_exit:
                keep.append(row_id)
                last_exit = int(pos) + max(1, int(h))
    return signals.loc[keep].sort_values(["date", "ticker"]) if keep else signals.iloc[0:0]


def _deoverlap_with_horizon(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    keep: list[Any] = []
    price_index = prices.index if prices is not None else None
    for ticker, grp in signals.groupby("ticker", sort=False):
        grp = grp.sort_values("date")
        dates = pd.to_datetime(grp["date"].to_numpy())
        if price_index is not None and ticker in getattr(prices, "columns", []):
            positions = price_index.searchsorted(dates)
        else:
            positions = np.arange(len(grp))
        last_exit = -1
        for pos, row_id in zip(positions, grp.index):
            if int(pos) >= last_exit:
                keep.append(row_id)
                last_exit = int(pos) + horizon
    return signals.loc[keep].sort_values(["date", "ticker"]) if keep else signals.iloc[0:0]


def active_entry_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Rows that the backtest can enter — symmetric long and short eligibility."""
    if signals.empty:
        return signals
    import config as _cfg
    from simulation.engine import _entry_eligible, _short_entry_eligible

    allow_short = bool(getattr(_cfg, "FUSION_ALLOW_SHORT", False))
    keep = []
    for _, row in signals.iterrows():
        score = float(row.get("score", 50.0))
        risk_on = bool(row.get("risk_on", True))
        hold_th = row.get("hold_threshold")
        buy_th = row.get("buy_threshold")
        sell_th = row.get("sell_threshold")
        side = int(row.get("position_side", 0) or 0)
        if side == 1:
            keep.append(_entry_eligible(score, risk_on, hold_th, buy_th))
        elif allow_short and side == -1:
            keep.append(_short_entry_eligible(score, risk_on, sell_th, buy_th))
        elif score <= 0.0:
            keep.append(False)
        elif _entry_eligible(score, risk_on, hold_th, buy_th):
            keep.append(True)
        elif allow_short and _short_entry_eligible(score, risk_on, sell_th, buy_th):
            keep.append(True)
        else:
            keep.append(False)
    return signals[pd.Series(keep, index=signals.index)].copy()
