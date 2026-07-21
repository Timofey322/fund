"""
Market regime detection + dynamic score weights.

Scientific basis (literature priors):
- HMM_IMPULSE: Jegadeesh & Titman (1993) — cross-sectional trend factor works in bull phases
- HMM_MEAN_REVERT: Lo & MacKinlay (1988); contrarian at extremes in sideways markets
- HMM_STRESS: Ang et al. (2006) — volatility clusters; reduce risk, favour risk filter
- Regime switching: Ang & Bekaert (2002)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from dataclasses import dataclass

import pandas as pd

from config import REGIME_TICKER, SMA_LONG
from common.naming import (
    COL_HMM_DOMINANT,
    COL_WEIGHT_MEAN_REV,
    COL_WEIGHT_RISK,
    COL_WEIGHT_TREND,
    HMM_STRESS,
    HMM_MEAN_REVERT,
    HMM_IMPULSE,
)
from operations.scoring import (
    ann_vol,
    log_returns,
    price_trend_12_1,
    regime_risk_on,
    vol_regime_ratio,
    z_score_vs_sma,
)


@dataclass(frozen=True)
class RegimeProfile:
    name: str
    label_ru: str
    weight_factor_trend: float
    weight_factor_mean_rev: float
    weight_factor_risk: float
    buy_threshold: float
    hold_threshold: float
    exposure_cap: float
    literature: str


# Prior weights — not fitted on full sample (avoid look-ahead)
REGIMES: dict[str, RegimeProfile] = {
    HMM_IMPULSE: RegimeProfile(
        name=HMM_IMPULSE,
        label_ru="Тренд (HMM bull)",
        weight_factor_trend=0.55,
        weight_factor_mean_rev=0.10,
        weight_factor_risk=0.35,
        buy_threshold=60,
        hold_threshold=45,
        exposure_cap=1.0,
        literature="Jegadeesh & Titman (1993); Ang & Bekaert HMM bull state",
    ),
    HMM_MEAN_REVERT: RegimeProfile(
        name=HMM_MEAN_REVERT,
        label_ru="Флэт (HMM range)",
        weight_factor_trend=0.10,
        weight_factor_mean_rev=0.55,
        weight_factor_risk=0.35,
        buy_threshold=62,
        hold_threshold=45,
        exposure_cap=0.85,
        literature="Lo & MacKinlay (1988); HMM neutral state",
    ),
    HMM_STRESS: RegimeProfile(
        name=HMM_STRESS,
        label_ru="Стресс (HMM crisis)",
        weight_factor_trend=0.05,
        weight_factor_mean_rev=0.25,
        weight_factor_risk=0.70,
        buy_threshold=68,
        hold_threshold=50,
        exposure_cap=0.30,
        literature="Ang et al. (2006); HMM crisis state",
    ),
}


def detect_regime_at(
    risk_on: bool,
    vol_ratio: float,
    index_price_trend: float | None,
    z_index: float | None,
) -> str:
    """
    Classify market state from benchmark indices (QQQ + SPY) — no look-ahead.
    """
    vr = vol_ratio if vol_ratio == vol_ratio else 1.0
    trend = (
        index_price_trend
        if index_price_trend is not None and index_price_trend == index_price_trend
        else 0.0
    )
    z = z_index if z_index is not None and z_index == z_index else 0.0

    if vr > 1.25 or not risk_on:
        return HMM_STRESS
    if risk_on and trend > 0.05 and vr < 1.1:
        return HMM_IMPULSE
    if abs(z) < 1.0 and vr < 1.15:
        return HMM_MEAN_REVERT
    if trend > 0 and risk_on:
        return HMM_IMPULSE
    return HMM_MEAN_REVERT


def build_market_regime_frame(prices: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Daily regime via Gaussian HMM on REGIME_TICKERS (QQQ + SPY by default)."""
    from research.regime.hmm import build_hmm_regime_frame

    return build_hmm_regime_frame(prices, ticker)


def build_market_regime_frame_rules(prices: pd.DataFrame, ticker: str = REGIME_TICKER) -> pd.DataFrame:
    """Legacy rule-based regime (fallback)."""
    if ticker not in prices.columns:
        ticker = prices.columns[0]
    close = prices[ticker].dropna()
    lr = log_returns(close)
    vol = ann_vol(lr)
    vol_ratio = vol_regime_ratio(vol)
    risk_on = regime_risk_on(close, SMA_LONG)
    index_trend = price_trend_12_1(close)
    z = z_score_vs_sma(close, SMA_LONG)

    rows = []
    for dt in close.index:
        ro = bool(risk_on.loc[dt] >= 0.5) if pd.notna(risk_on.loc[dt]) else False
        vr = float(vol_ratio.loc[dt]) if pd.notna(vol_ratio.loc[dt]) else 1.0
        tr = float(index_trend.loc[dt]) if pd.notna(index_trend.loc[dt]) else None
        zz = float(z.loc[dt]) if pd.notna(z.loc[dt]) else None
        reg = detect_regime_at(ro, vr, tr, zz)
        prof = REGIMES[reg]
        rows.append(
            {
                "date": dt,
                COL_HMM_DOMINANT: reg,
                "regime_label": prof.label_ru,
                COL_WEIGHT_TREND: prof.weight_factor_trend,
                COL_WEIGHT_MEAN_REV: prof.weight_factor_mean_rev,
                COL_WEIGHT_RISK: prof.weight_factor_risk,
                "buy_threshold": prof.buy_threshold,
                "hold_threshold": prof.hold_threshold,
                "risk_on": ro,
                "vol_ratio": round(vr, 2),
                "index_price_trend_pct": round(tr * 100, 2) if tr is not None else None,
                "index_z_sma200": round(zz, 2) if zz is not None else None,
            }
        )
    return pd.DataFrame(rows)


def blend_weights(
    prior: tuple[float, float, float],
    optimized: tuple[float, float, float] | None,
    blend: float = 0.65,
) -> tuple[float, float, float]:
    """Shrink optimized weights toward literature prior (regularization)."""
    if optimized is None:
        return prior
    p = prior
    o = optimized
    w = (
        blend * p[0] + (1 - blend) * o[0],
        blend * p[1] + (1 - blend) * o[1],
        blend * p[2] + (1 - blend) * o[2],
    )
    s = sum(w)
    return (w[0] / s, w[1] / s, w[2] / s)
