"""
Impulse / side-shift features for ML entry timing.

- Nadaraya-Watson envelope: smooth fair value + band position
- Momentum stack: ROC, RSI, MACD histogram
- Side-shift: short vs long momentum divergence, flow flip
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd

# ~4h context on 5Min bars
NW_LOOKBACK = 48
NW_BANDWIDTH = 6.0
NW_BAND_MULT = 2.0

IMPULSE_COLS = (
    "nw_est",
    "nw_slope",
    "nw_env_pos",
    "nw_band_width",
    "nw_breakout_up",
    "nw_breakout_dn",
    "roc_6",
    "roc_12",
    "roc_24",
    "rsi_14",
    "macd_hist",
    "mom_short",
    "mom_long",
    "side_shift",
    "impulse_raw",
    "power_shift",
)


def _gaussian_weights(dist: np.ndarray, bandwidth: float) -> np.ndarray:
    u = dist / max(bandwidth, 1e-6)
    w = np.exp(-0.5 * u * u)
    w /= w.sum() if w.sum() > 0 else 1.0
    return w


def nadaraya_watson(
    close: pd.Series,
    lookback: int = NW_LOOKBACK,
    bandwidth: float = NW_BANDWIDTH,
) -> pd.Series:
    """Rolling Nadaraya-Watson estimate (Gaussian kernel, causal)."""
    y = close.astype(float).values
    n = len(y)
    out = np.full(n, np.nan)
    x_idx = np.arange(n, dtype=float)
    for i in range(lookback - 1, n):
        sl = slice(i - lookback + 1, i + 1)
        xs = x_idx[sl]
        ys = y[sl]
        if np.any(np.isnan(ys)):
            continue
        dist = (xs[-1] - xs)
        w = _gaussian_weights(dist, bandwidth)
        out[i] = float(np.dot(w, ys))
    return pd.Series(out, index=close.index)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig).fillna(0.0)


def attach_impulse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add NW envelope, momentum, and side-shift columns to OHLCV(+flow) frame."""
    if df.empty or "close" not in df.columns:
        return df

    out = df.copy()
    close = out["close"].astype(float)
    vol_imb = out.get("vol_imbalance", pd.Series(0.0, index=out.index)).fillna(0.0)

    nw = nadaraya_watson(close)
    resid = close - nw
    band = resid.rolling(NW_LOOKBACK // 2, min_periods=8).std().fillna(resid.abs().rolling(8).mean())
    upper = nw + NW_BAND_MULT * band
    lower = nw - NW_BAND_MULT * band
    width = (upper - lower).replace(0, np.nan)

    out["nw_est"] = nw
    out["nw_slope"] = nw.diff(6) / nw.shift(6).replace(0, np.nan)
    out["nw_env_pos"] = ((close - lower) / width).clip(0, 1).fillna(0.5)
    out["nw_band_width"] = (width / close.replace(0, np.nan)).fillna(0.0)
    out["nw_breakout_up"] = (close > upper).astype(float)
    out["nw_breakout_dn"] = (close < lower).astype(float)

    out["roc_6"] = close.pct_change(6, fill_method=None)
    out["roc_12"] = close.pct_change(12, fill_method=None)
    out["roc_24"] = close.pct_change(24, fill_method=None)
    out["rsi_14"] = _rsi(close) / 100.0
    out["macd_hist"] = _macd_hist(close)
    out["mom_short"] = out["roc_6"].fillna(0.0)
    out["mom_long"] = out["roc_24"].fillna(0.0)

    # Side shift: short momentum crosses / diverges from long (regime of buyers vs sellers)
    sign_s = np.sign(out["mom_short"]).replace(0, np.nan)
    sign_l = np.sign(out["mom_long"]).replace(0, np.nan)
    out["side_shift"] = (sign_s != sign_l).astype(float).fillna(0.0)

    # Raw impulse: momentum + envelope extension + flow aligned with direction
    dir_mom = np.sign(out["mom_short"]).fillna(0.0)
    flow_align = vol_imb * dir_mom
    env_ext = (out["nw_env_pos"] - 0.5) * 2.0 * dir_mom
    out["impulse_raw"] = (
        0.4 * out["mom_short"].abs()
        + 0.3 * env_ext.abs()
        + 0.3 * flow_align.abs()
    ).fillna(0.0)

    # Power shift: absorption at lower band (long lower wick proxy via env) + positive flow
    lower_wick = out.get("lower_wick_ratio", pd.Series(0.0, index=out.index)).fillna(0.0)
    out["power_shift"] = (lower_wick * (vol_imb.clip(lower=0)) * (1.0 - out["nw_env_pos"])).fillna(0.0)

    return out
