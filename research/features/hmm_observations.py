"""
5Min bar-level HMM features — baseline = last 1 hour (12 bars).

ret_z on bar t:
  «Насколько сильно эта 5-минутка vs среднее за последний час?»

  log_ret_t = ln(close_t / close_{t-1})
  μ, σ    = mean/std(log_ret) over last 12 bars (1 hour)
  ret_z_t = (log_ret_t - μ) / σ

vol_ratio: 1h vol / typical 1-day vol.
vol_z:     vol_ratio vs last hour of vol_ratio.
risk_on:   close > 1-day SMA (288 bars).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import math

import numpy as np
import pandas as pd

from config import (
    CRYPTO_BARS_PER_DAY,
    CRYPTO_BARS_PER_HOUR,
    CRYPTO_TRADING_DAYS_PER_YEAR,
    HMM_BAR_RISK_SMA_BARS,
    HMM_BAR_ROLLING_MIN,
    HMM_BAR_VOL_MEDIAN_BARS,
    HMM_BAR_VOL_WINDOW_BARS,
    HMM_RET_Z_LOOKBACK_BARS,
)


def bars_per_year_crypto() -> float:
    return CRYPTO_TRADING_DAYS_PER_YEAR * float(CRYPTO_BARS_PER_DAY)


def _min_periods(window: int, floor: int = 6) -> int:
    return max(floor, window // 2)


def compute_bar_features(
    close: pd.Series,
    *,
    ret_z_lookback: int = HMM_RET_Z_LOOKBACK_BARS,
    vol_window: int = HMM_BAR_VOL_WINDOW_BARS,
    vol_median: int = HMM_BAR_VOL_MEDIAN_BARS,
    rolling_min: int = HMM_BAR_ROLLING_MIN,
    risk_sma_bars: int = HMM_BAR_RISK_SMA_BARS,
    ann_factor: float | None = None,
) -> pd.DataFrame:
    """
    Bar-level observations for Gaussian HMM on 5Min data (1-hour baseline).
    """
    if close.empty:
        return pd.DataFrame()

    c = close.dropna().astype(float)
    n = len(c)
    hour = ret_z_lookback
    vol_median_eff = min(vol_median, max(vol_window * 4, n // 2))
    lr = np.log(c / c.shift(1))

    # ret_z: current 5m bar vs last hour
    mu = lr.rolling(hour, min_periods=_min_periods(hour)).mean()
    sig = lr.rolling(hour, min_periods=_min_periods(hour)).std().replace(0, np.nan)
    ret_z = (lr - mu) / sig

    ann = ann_factor or math.sqrt(bars_per_year_crypto())
    vol = lr.rolling(vol_window, min_periods=_min_periods(vol_window)).std() * ann
    vol_med = vol.rolling(
        vol_median_eff, min_periods=max(vol_window, vol_median_eff // 4)
    ).median().replace(0, np.nan)
    vol_ratio = vol / vol_med

    # vol_z: vol_ratio vs last hour (not 7 days)
    pvol = vol_ratio
    vol_z = (pvol - pvol.rolling(hour, min_periods=rolling_min).mean()) / pvol.rolling(
        hour, min_periods=rolling_min
    ).std().replace(0, np.nan)

    sma_win = min(risk_sma_bars, max(n // 2, hour * 4))
    sma = c.rolling(sma_win, min_periods=hour).mean()
    risk_on = (c > sma).astype(float)

    # trend_z: directional drift of the last hour, standardized over ~1 day.
    #
    # ret_z is the single-bar return standardized over the *same* hour, so its
    # mean is ~0 inside any regime -> it carries volatility/outlier info but
    # almost no persistent direction. The HMM therefore cannot separate "up"
    # from "down" regimes from ret_z alone, and naming a state "impulse" by its
    # mean ret_z is essentially noise. trend_z measures the hourly drift relative
    # to its daily distribution, giving the HMM a genuine directional dimension.
    drift = lr.rolling(hour, min_periods=_min_periods(hour)).mean()
    trend_win = min(risk_sma_bars, max(n // 2, hour * 4))
    drift_mu = drift.rolling(trend_win, min_periods=hour).mean()
    drift_sig = drift.rolling(trend_win, min_periods=hour).std().replace(0, np.nan)
    trend_z = (drift - drift_mu) / drift_sig

    out = pd.DataFrame({
        "log_ret": lr,
        "vol_ratio": vol_ratio,
        "ret_z": ret_z,
        "vol_z": vol_z,
        "trend_z": trend_z,
        "risk_on": risk_on,
        "hmm_ret": ret_z,
        "hmm_vol": vol_z,
        "hmm_trend": trend_z,
    }, index=c.index)
    return out.dropna(subset=["hmm_ret", "hmm_vol", "hmm_trend"])


# HMM observation matrix column order. The directional dimension is first so the
# regime-naming logic (which orders states by ``means[:, 0]``) keys on persistent
# drift rather than single-bar return noise; ``means[:, 1]`` stays volatility.
HMM_BAR_OBS_COLS = ("hmm_trend", "hmm_vol", "hmm_ret")


def hmm_observation_matrix(feats: pd.DataFrame) -> np.ndarray:
    """Observation matrix for the Gaussian HMM in the canonical column order.

    Falls back to whatever observation columns are present (older callers built
    a 2-D ``[hmm_ret, hmm_vol]`` matrix) so the function is safe on partial
    feature frames.
    """
    cols = [c for c in HMM_BAR_OBS_COLS if c in feats.columns]
    if not cols:
        cols = [c for c in ("hmm_ret", "hmm_vol") if c in feats.columns]
    return feats[cols].to_numpy()


def ret_z_step_table(
    close: pd.Series,
    t_idx: int,
    lookback: int = CRYPTO_BARS_PER_HOUR,
) -> pd.DataFrame:
    """Small window table for ret_z explainer plots (default = 1 hour)."""
    c = close.dropna().astype(float)
    if t_idx >= len(c):
        t_idx = len(c) - 1
    sl = c.iloc[max(0, t_idx - lookback + 1): t_idx + 1]
    lr = np.log(sl / sl.shift(1))
    mu = lr.mean()
    sig = lr.std(ddof=1) if len(lr.dropna()) > 1 else np.nan
    cur = float(lr.iloc[-1]) if len(lr) else np.nan
    z = (cur - mu) / sig if sig and sig > 1e-12 else np.nan
    rows = []
    for i, (ts, r) in enumerate(lr.items()):
        rows.append({
            "bar": i,
            "time": ts,
            "close": float(sl.loc[ts]),
            "log_ret": float(r) if pd.notna(r) else np.nan,
            "is_current": i == len(lr) - 1,
        })
    df = pd.DataFrame(rows)
    df.attrs["mu"] = mu
    df.attrs["sigma"] = sig
    df.attrs["ret_z"] = z
    df.attrs["lookback_bars"] = lookback
    df.attrs["lookback_hours"] = lookback / CRYPTO_BARS_PER_HOUR
    return df
