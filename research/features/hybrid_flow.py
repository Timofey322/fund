"""
Hybrid volume flow: candle imputation on full history + tick counts on recent window.

Where slim tick cache exists (buy_count/sell_count), override vol_imbalance with
real count_imbalance. Else keep CLV/wick candle decomposition.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd

from research.features.candle_flow import attach_candle_flow
from data_platform.binance import SLIM_COLS, load_slim_panel


def merge_hybrid_flow(
    ohlcv: pd.DataFrame,
    slim: pd.DataFrame | None = None,
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    """
    OHLCV -> candle flow, then overlay tick-based imbalance where available.

    Adds:
      flow_source: 1 = tick counts, 0 = candle imputation
      tick_imbalance: count_imbalance from ticks (0 where absent)
    """
    if ohlcv.empty:
        return ohlcv

    if slim is None and symbol:
        slim = load_slim_panel(symbol)

    out = attach_candle_flow(ohlcv)
    out["flow_source"] = 0.0
    out["tick_imbalance"] = 0.0

    if slim is None or slim.empty:
        return out

    slim = slim.reindex(out.index)
    bc = slim.get("buy_count", pd.Series(0.0, index=out.index)).fillna(0.0)
    sc = slim.get("sell_count", pd.Series(0.0, index=out.index)).fillna(0.0)
    denom = bc + sc
    tick_imb = np.where(denom > 0, (bc - sc) / denom, 0.0)
    tick_imb = pd.Series(tick_imb, index=out.index, dtype=float)

    has_tick = denom > 0
    if has_tick.any():
        out.loc[has_tick, "vol_imbalance"] = tick_imb[has_tick]
        out.loc[has_tick, "buy_share"] = (tick_imb[has_tick] + 1.0) / 2.0
        out.loc[has_tick, "buy_vol_est"] = out.loc[has_tick, "volume"] * out.loc[has_tick, "buy_share"]
        out.loc[has_tick, "sell_vol_est"] = out.loc[has_tick, "volume"] * (1.0 - out.loc[has_tick, "buy_share"])
        out.loc[has_tick, "flow_source"] = 1.0

    out["tick_imbalance"] = tick_imb
    return out


def hybrid_coverage(panel: pd.DataFrame) -> dict:
    """Share of bars with real tick flow vs candle-only."""
    if panel.empty or "flow_source" not in panel.columns:
        return {"tick_bars": 0, "candle_bars": len(panel), "tick_pct": 0.0}
    tick = int((panel["flow_source"] > 0).sum())
    total = len(panel)
    return {
        "tick_bars": tick,
        "candle_bars": total - tick,
        "tick_pct": round(100.0 * tick / total, 2) if total else 0.0,
    }


def filter_tick_only(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep bars with real tick buy/sell counts (no candle-imputed flow)."""
    if panel.empty:
        return panel
    out = panel.copy()
    if "flow_source" in out.columns:
        out = out[out["flow_source"] > 0]
    if "buy_count" in out.columns and "sell_count" in out.columns:
        out = out[(out["buy_count"].fillna(0) + out["sell_count"].fillna(0)) > 0]
    return out.reset_index(drop=True)
