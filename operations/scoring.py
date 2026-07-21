"""Math/physics-inspired signals: price trend, mean reversion (z), risk filter, vol."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import PRICE_TREND_LOOKBACK_DAYS, PRICE_TREND_SKIP_DAYS, TRADING_DAYS_PER_YEAR, VOL_WINDOW
from common.naming import (
    COL_FACTOR_MEAN_REV,
    COL_FACTOR_RISK,
    COL_FACTOR_TREND,
    COL_PRICE_TREND_12_1_PCT,
)


def ann_factor() -> float:
    return math.sqrt(TRADING_DAYS_PER_YEAR)


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def ann_vol(log_ret: pd.Series, window: int | None = None) -> pd.Series:
    w = window if window is not None else VOL_WINDOW
    return log_ret.rolling(w).std() * ann_factor()


def z_score_vs_sma(close: pd.Series, window: int = 200) -> pd.Series:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - sma) / std.replace(0, np.nan)


def price_trend_12_1(
    close: pd.Series,
    lookback: int = PRICE_TREND_LOOKBACK_DAYS,
    skip: int = PRICE_TREND_SKIP_DAYS,
) -> pd.Series:
    """12-1 price trend: return over lookback bars, skipping recent skip bars."""
    past = close.shift(skip)
    old = close.shift(lookback + skip)
    return (past / old) - 1


def regime_risk_on(close: pd.Series, sma_window: int = 200) -> pd.Series:
    sma = close.rolling(sma_window).mean()
    return (close > sma).astype(float)


def vol_regime_ratio(vol: pd.Series, median_window: int | None = None) -> pd.Series:
    if median_window is None:
        median_window = PRICE_TREND_LOOKBACK_DAYS * 5
    med = vol.rolling(median_window, min_periods=PRICE_TREND_LOOKBACK_DAYS).median()
    return vol / med.replace(0, np.nan)


def clip_score(x: float, lo: float = 0, hi: float = 100) -> float:
    if math.isnan(x) or math.isinf(x):
        return 50.0
    return max(lo, min(hi, x))


def score_price_trend(trend: float) -> float:
    """Map 12-1 price trend to 0-100 (sigmoid-like)."""
    if trend is None or (isinstance(trend, float) and math.isnan(trend)):
        return 50.0
    # +20% trend -> ~75, -20% -> ~25
    return clip_score(50 + trend * 125)


def score_mean_reversion(z: float) -> float:
    """Below SMA (z<0) = opportunity for long; deep discount scores higher."""
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return 50.0
    if z < -2.0:
        return clip_score(50 + min(abs(z), 3) * 18)
    if z < -1.0:
        return clip_score(50 + abs(z) * 12)
    if z > 1.5:
        return clip_score(50 - (z - 1.5) * 20)
    return 50.0


def score_risk_filter(risk_on: float, vol_ratio: float) -> float:
    s = 50.0
    if risk_on >= 0.5:
        s += 20
    else:
        s -= 25
    if vol_ratio is not None and not math.isnan(vol_ratio):
        if vol_ratio < 0.85:
            s += 10
        elif vol_ratio > 1.25:
            s -= 20
        elif vol_ratio > 1.0:
            s -= 8
    return clip_score(s)


def composite_score(
    price_trend: float,
    z: float,
    risk_on: float,
    vol_ratio: float,
    weight_trend: float = 0.30,
    weight_mean_rev: float = 0.35,
    weight_risk: float = 0.35,
) -> dict:
    trend_sub = score_price_trend(price_trend)
    mean_rev_sub = score_mean_reversion(z)
    risk_sub = score_risk_filter(risk_on, vol_ratio)
    total = weight_trend * trend_sub + weight_mean_rev * mean_rev_sub + weight_risk * risk_sub
    return {
        "score": round(total, 1),
        COL_FACTOR_TREND: round(trend_sub, 1),
        COL_FACTOR_MEAN_REV: round(mean_rev_sub, 1),
        COL_FACTOR_RISK: round(risk_sub, 1),
        COL_PRICE_TREND_12_1_PCT: round(price_trend * 100, 2) if price_trend == price_trend else None,
        "z_sma200": round(z, 2) if z == z else None,
        "risk_on": bool(risk_on >= 0.5),
        "vol_ratio": round(vol_ratio, 2) if vol_ratio == vol_ratio else None,
    }


def composite_score_frame(
    price_trend: pd.Series,
    z: pd.Series,
    risk_on: pd.Series,
    vol_ratio: pd.Series,
    weight_trend: float = 0.30,
    weight_mean_rev: float = 0.35,
    weight_risk: float = 0.35,
) -> pd.DataFrame:
    """Vectorized composite_score — same logic, for large intraday panels."""
    pt = pd.to_numeric(price_trend, errors="coerce")
    zz = pd.to_numeric(z, errors="coerce")
    ro = pd.to_numeric(risk_on, errors="coerce").fillna(0.0)
    vr = pd.to_numeric(vol_ratio, errors="coerce")

    # price trend sub-score
    trend_sub = np.clip(50 + pt * 125, 0, 100)
    trend_sub = trend_sub.where(pt.notna(), 50.0)

    # mean-reversion sub-score (piecewise on z)
    az = zz.abs()
    mr = np.select(
        [zz < -2.0, zz < -1.0, zz > 1.5],
        [
            np.clip(50 + np.minimum(az, 3) * 18, 0, 100),
            np.clip(50 + az * 12, 0, 100),
            np.clip(50 - (zz - 1.5) * 20, 0, 100),
        ],
        default=50.0,
    )
    mr = pd.Series(mr, index=zz.index).where(zz.notna(), 50.0)

    # risk sub-score
    risk = pd.Series(50.0, index=ro.index)
    risk = risk + np.where(ro >= 0.5, 20.0, -25.0)
    risk = risk + np.select(
        [vr < 0.85, vr > 1.25, vr > 1.0],
        [10.0, -20.0, -8.0],
        default=0.0,
    )
    risk = np.clip(risk, 0, 100)
    risk = pd.Series(risk, index=ro.index)

    total = weight_trend * trend_sub + weight_mean_rev * mr + weight_risk * risk
    out = pd.DataFrame({
        "score": total.round(1),
        COL_FACTOR_TREND: pd.Series(trend_sub, index=pt.index).round(1),
        COL_FACTOR_MEAN_REV: mr.round(1),
        COL_FACTOR_RISK: risk.round(1),
        COL_PRICE_TREND_12_1_PCT: (pt * 100).round(2),
        "z_sma200": zz.round(2),
        "risk_on": ro >= 0.5,
        "vol_ratio": vr.round(2),
    })
    return out


def decision_label(score: float, buy: float = 65, hold: float = 45) -> str:
    if score >= buy:
        return "BUY"
    if score >= hold:
        return "HOLD"
    return "REDUCE"


def ou_half_life(spread: pd.Series) -> float | None:
    """Ornstein-Uhlenbeck half-life from AR(1): delta = alpha + beta * x."""
    s = spread.dropna()
    if len(s) < 60:
        return None
    x = s.iloc[:-1].values
    dx = s.diff().iloc[1:].values
    if len(x) < 30:
        return None
    beta = np.polyfit(x, dx, 1)[0]
    if beta >= 0:
        return None
    return -math.log(2) / beta
