"""Structural / path features complementary to spectral–VP–NW winners.

Designed from walk-forward importance (v3): volume profile, spectral cycles,
Nadaraya–Watson, and Hurst dominate; classic ROC/RSI stack is weak. New columns
extend those strong families without duplicating excluded flow microstructure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STRUCTURE_COLS: tuple[str, ...] = (
    "trend_efficiency_24",
    "ret_autocorr_24",
    "hl_range_z_24",
    "vol_of_vol_24",
    "nw_z",
    "vwap_dist",
)

# Extra CS ranks for the importance leaders (universal across universe size).
STRUCTURE_CS_COLS: tuple[str, ...] = (
    "hurst_rs_cs_rank",
    "spec_low_high_ratio_cs_rank",
    "vp_poc_dist_cs_rank",
    "garch_vol_ratio_cs_rank",
)


def _trend_efficiency(close: pd.Series, window: int = 24) -> pd.Series:
    """Kaufman efficiency ratio: net move / path length in [0, 1]."""
    c = close.astype(float)
    net = (c - c.shift(window)).abs()
    path = c.diff().abs().rolling(window, min_periods=max(4, window // 3)).sum()
    return (net / path.replace(0, np.nan)).clip(0.0, 1.0)


def _ret_autocorr(ret: pd.Series, window: int = 24, lag: int = 1) -> pd.Series:
    """Rolling lag-1 autocorrelation of returns (persistence vs mean-revert)."""
    r = pd.to_numeric(ret, errors="coerce")
    mu = r.rolling(window, min_periods=max(8, window // 2)).mean()
    x = r - mu
    x_l = x.shift(lag)
    num = (x * x_l).rolling(window, min_periods=max(8, window // 2)).mean()
    den = x.rolling(window, min_periods=max(8, window // 2)).var(ddof=0)
    return (num / den.replace(0, np.nan)).clip(-1.0, 1.0)


def _hl_range_z(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 24) -> pd.Series:
    """Z-score of bar range; falls back to |ret|*price when OHLC is flat/missing."""
    c = close.astype(float)
    hi = high.astype(float)
    lo = low.astype(float)
    rng = (hi - lo).clip(lower=0.0)
    # Synthetic range when high==low (common after sparse OHLC / adjusted closes).
    flat = rng <= 1e-12
    if bool(flat.mean() > 0.5):
        ret = c.pct_change(1, fill_method=None).abs().fillna(0.0)
        rng = (ret * c.abs()).clip(lower=0.0)
    mu = rng.rolling(window * 4, min_periods=max(8, window // 2)).mean()
    sd = rng.rolling(window * 4, min_periods=max(8, window // 2)).std()
    sd = sd.where(sd > 1e-12, np.nan)
    out = ((rng - mu) / sd).replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0)


def _vol_of_vol(ret: pd.Series, window: int = 24) -> pd.Series:
    rv = ret.rolling(12, min_periods=4).std()
    return rv.rolling(window, min_periods=max(6, window // 3)).std()


def attach_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach causal structure features for one ticker block or flat frame."""
    if df.empty or "close" not in df.columns:
        return df
    out = df.copy()
    close = out["close"].astype(float)
    ret = close.pct_change(1, fill_method=None)

    out["trend_efficiency_24"] = _trend_efficiency(close, 24)
    out["ret_autocorr_24"] = _ret_autocorr(ret, 24, 1)

    high = out["high"].astype(float) if "high" in out.columns else close
    low = out["low"].astype(float) if "low" in out.columns else close
    out["hl_range_z_24"] = _hl_range_z(high, low, close, 24)
    out["vol_of_vol_24"] = _vol_of_vol(ret, 24)

    if "nw_est" in out.columns and "nw_band_width" in out.columns:
        bw = out["nw_band_width"].astype(float).replace(0, np.nan)
        # nw_band_width is already relative to close in impulse.py
        out["nw_z"] = ((close - out["nw_est"].astype(float)) / close.replace(0, np.nan) / bw).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        out["nw_z"] = np.nan

    if "volume" in out.columns:
        typical = (high + low + close) / 3.0
        vol = out["volume"].astype(float).clip(lower=0.0)
        # Causal rolling VWAP (~1 session on 5m ≈ 78 RU / 78 US; use 78)
        win = 78
        pv = (typical * vol).rolling(win, min_periods=max(8, win // 4)).sum()
        vv = vol.rolling(win, min_periods=max(8, win // 4)).sum().replace(0, np.nan)
        vwap = pv / vv
        out["vwap_dist"] = ((close - vwap) / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    else:
        out["vwap_dist"] = np.nan

    return out


def attach_structure_features_by_ticker(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    if "ticker" not in panel.columns:
        return attach_structure_features(panel)
    parts: list[pd.DataFrame] = []
    for _, grp in panel.groupby("ticker", sort=False):
        sort_col = "bar_time" if "bar_time" in grp.columns else grp.index
        parts.append(attach_structure_features(grp.sort_values(sort_col)))
    if not parts:
        return panel
    return pd.concat(parts).sort_index()
