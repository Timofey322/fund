"""Per-instrument execution costs — commission + vol-scaled slippage (aligned with backtest)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as _cfg
from data_platform.universe import commission_bps_for_ticker, is_crypto_symbol, is_tradfi_symbol


def slippage_bps_per_side(
    ticker: str,
    *,
    vol_ann: float | None = None,
) -> float:
    """Per-side slippage in bps.

    Backtest charges ``commission + slippage`` on each turnover leg (entry and exit).
    Defaults are conservative but realistic for liquid 5m bars (much lower than flat 15 bps).
    Optional vol scaling: ``base * clip(vol / vol_ref, 0.5, 2.0)``.
    """
    sym = str(ticker).upper()
    if is_tradfi_symbol(sym):
        base = float(getattr(_cfg, "SLIPPAGE_BPS_TRADFI", 2.0))
        vol_ref = float(getattr(_cfg, "SLIPPAGE_VOL_REF_TRADFI", 0.18))
    elif is_crypto_symbol(sym):
        base = float(getattr(_cfg, "SLIPPAGE_BPS_CRYPTO", 1.5))
        vol_ref = float(getattr(_cfg, "SLIPPAGE_VOL_REF_CRYPTO", 0.80))
    else:
        base = float(getattr(_cfg, "SLIPPAGE_BPS_DEFAULT", 2.0))
        vol_ref = float(getattr(_cfg, "SLIPPAGE_VOL_REF_DEFAULT", 0.50))

    if vol_ann is not None and np.isfinite(vol_ann) and vol_ann > 0:
        scale = float(np.clip(float(vol_ann) / max(vol_ref, 1e-6), 0.5, 2.0))
        if bool(getattr(_cfg, "SLIPPAGE_VOL_SCALE", True)):
            return base * scale
    return base


def round_trip_cost_bps(
    commission_bps_per_side: float,
    slippage_bps_per_side: float,
) -> float:
    """Round-trip cost matching ``engine._turnover_cost``: 2 legs × (comm + slip)."""
    return 2.0 * (float(commission_bps_per_side) + float(slippage_bps_per_side))


def round_trip_cost_bps_for_ticker(
    ticker: str,
    *,
    vol_ann: float | None = None,
) -> float:
    comm = commission_bps_for_ticker(ticker)
    slip = slippage_bps_per_side(ticker, vol_ann=vol_ann)
    return round_trip_cost_bps(comm, slip)


def round_trip_cost_series(
    tickers: pd.Series,
    *,
    vol_ann: pd.Series | None = None,
) -> pd.Series:
    """Per-row round-trip cost in bps."""
    t = tickers.astype(str)
    if vol_ann is not None:
        vol = pd.to_numeric(vol_ann, errors="coerce")
        return pd.Series(
            [
                round_trip_cost_bps_for_ticker(sym, vol_ann=float(v) if pd.notna(v) else None)
                for sym, v in zip(t, vol, strict=False)
            ],
            index=tickers.index,
            dtype=float,
        )
    return t.map(lambda sym: round_trip_cost_bps_for_ticker(sym))


def default_slippage_bps_per_side() -> float:
    """Legacy default for callers without a ticker (crypto baseline)."""
    return float(getattr(_cfg, "SLIPPAGE_BPS_CRYPTO", 1.5))
