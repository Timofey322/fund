"""
Buy-only rule entries: Nadaraya-Watson lower envelope touch + HMM growth filter.

We never sell. Buy when price touches the NW lower band at a low level and HMM
regime probabilities favor an upcoming bounce (mean-reversion / impulse fade).
"""

from __future__ import annotations

import pandas as pd

from common.naming import (
    COL_PROB_HMM_IMPULSE,
    COL_PROB_HMM_MEAN_REVERT,
    COL_PROB_HMM_STRESS,
)
from operations.scoring import ann_vol, log_returns
from research.features.impulse import nadaraya_watson
from research.regime.hmm import build_hmm_regime_frame

from rule.config import (
    RULE_HMM_GROWTH_MIN,
    RULE_NEUTRAL_SCORE,
    RULE_NW_BAND_MULT,
    RULE_NW_BAND_STD_WINDOW,
    RULE_NW_BANDWIDTH,
    RULE_NW_LOOKBACK,
    RULE_NW_TOUCH_MAX,
)
from rule.signals import impulse_stopping_probability


def nw_envelope_frame(close: pd.Series) -> pd.DataFrame:
    """Causal Nadaraya-Watson fair value + Gaussian-kernel envelope bands."""
    c = close.dropna().astype(float)
    if len(c) < RULE_NW_LOOKBACK + 10:
        return pd.DataFrame()

    nw = nadaraya_watson(c, lookback=RULE_NW_LOOKBACK, bandwidth=RULE_NW_BANDWIDTH)
    resid = c - nw
    win = max(int(RULE_NW_BAND_STD_WINDOW), 8)
    band = resid.rolling(win, min_periods=max(8, win // 4)).std()
    band = band.fillna(resid.abs().rolling(8, min_periods=4).mean())
    upper = nw + RULE_NW_BAND_MULT * band
    lower = nw - RULE_NW_BAND_MULT * band
    width = (upper - lower).replace(0, pd.NA)
    env_pos = ((c - lower) / width).clip(0.0, 1.0).fillna(0.5)
    touches_lower = env_pos <= float(RULE_NW_TOUCH_MAX)
    dist_to_lower = (c - lower) / width
    dev_below = (-dist_to_lower).clip(lower=0.0).fillna(0.0)
    slope = nw.pct_change(5, fill_method=None).fillna(0.0)

    return pd.DataFrame(
        {
            "nw_est": nw,
            "nw_upper": upper,
            "nw_lower": lower,
            "nw_width": width,
            "nw_env_pos": env_pos,
            "nw_touches_lower": touches_lower,
            "nw_dev_below": dev_below,
            "nw_slope": slope,
        },
        index=c.index,
    )


def hmm_growth_probability(
    prob_impulse: float,
    prob_mean_revert: float,
    prob_stress: float,
    prob_impulse_prev: float | None,
    *,
    risk_on: bool = True,
) -> float:
    """
    P(upside bounce after lower-band touch) from HMM regime probabilities.

    Combines fading downward impulse, rising mean-reversion, and risk-on tilt.
    """
    p_stop = impulse_stopping_probability(
        prob_impulse, prob_mean_revert, prob_stress, prob_impulse_prev,
    )
    bounce = float(prob_mean_revert) * 0.40 + float(p_stop) * 0.35
    if risk_on:
        bounce += 0.10
    if prob_impulse_prev is not None:
        fade = max(0.0, float(prob_impulse_prev) - float(prob_impulse))
        bounce += 0.20 * fade
    if float(prob_stress) > 0.50:
        bounce -= 0.18 * (float(prob_stress) - 0.50) / 0.50
    return float(min(1.0, max(0.0, bounce)))


def nw_hmm_buy_score(
    *,
    env_pos: float,
    touches_lower: bool,
    p_growth: float,
    neutral_score: float = RULE_NEUTRAL_SCORE,
    growth_min: float = RULE_HMM_GROWTH_MIN,
    touch_max: float = RULE_NW_TOUCH_MAX,
) -> tuple[float, str]:
    """
    Buy-only score: lower NW envelope touch + HMM expects growth.

    Never emits sell scores — neutral when conditions are not met.
    """
    if not touches_lower and env_pos > touch_max:
        return neutral_score, "neutral"
    if float(p_growth) < growth_min:
        return neutral_score, "neutral_hmm"

    # Closer to the lower band → stronger touch (env_pos=0 at lower wave)
    touch_strength = 1.0 - min(1.0, max(0.0, float(env_pos)) / max(touch_max, 1e-6))
    growth_strength = min(
        1.0,
        (float(p_growth) - growth_min) / max(1.0 - growth_min, 1e-6),
    )
    strength = min(1.0, touch_strength * (0.35 + 0.65 * growth_strength))
    score = neutral_score + (100.0 - neutral_score) * strength
    return float(min(100.0, score)), "buy_nw_hmm"


def _attach_ticker_nw_hmm_signals(close: pd.Series, ticker: str) -> pd.DataFrame:
    """Per-instrument NW envelope + HMM buy scores."""
    c = close.dropna().astype(float)
    if len(c) < max(RULE_NW_LOOKBACK + 30, 120):
        return pd.DataFrame()

    sym = str(ticker).upper()
    env = nw_envelope_frame(c)
    if env.empty:
        return pd.DataFrame()

    hmm = build_hmm_regime_frame(pd.DataFrame({sym: c}), ticker=sym)
    if hmm.empty:
        return pd.DataFrame()

    time_col = "bar_time" if "bar_time" in hmm.columns else "date"
    hmm = hmm.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
    hmm_idx = pd.to_datetime(hmm[time_col])
    prob_cols = [COL_PROB_HMM_IMPULSE, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_STRESS]
    hmm_feats = hmm.set_index(hmm_idx)[prob_cols + ["risk_on", "vol_ratio"]]

    merged = env.join(hmm_feats, how="inner")
    if merged.empty:
        return pd.DataFrame()

    vol = ann_vol(log_returns(c))
    p_imp = merged[COL_PROB_HMM_IMPULSE].astype(float)
    p_imp_prev = p_imp.shift(1)

    rows: list[dict] = []
    for ts, row in merged.iterrows():
        prev = p_imp_prev.loc[ts] if ts in p_imp_prev.index else None
        p_growth = hmm_growth_probability(
            float(row[COL_PROB_HMM_IMPULSE]),
            float(row[COL_PROB_HMM_MEAN_REVERT]),
            float(row[COL_PROB_HMM_STRESS]),
            float(prev) if prev is not None and pd.notna(prev) else None,
            risk_on=bool(row.get("risk_on", True)),
        )
        env_pos = float(row["nw_env_pos"]) if pd.notna(row["nw_env_pos"]) else 1.0
        touches = bool(row["nw_touches_lower"]) if pd.notna(row["nw_touches_lower"]) else False
        vol_ann = float(vol.loc[ts]) if ts in vol.index and pd.notna(vol.loc[ts]) else 0.25
        score, side = nw_hmm_buy_score(
            env_pos=env_pos,
            touches_lower=touches,
            p_growth=p_growth,
        )
        close_px = float(c.loc[ts]) if ts in c.index else float("nan")
        rows.append(
            {
                "date": pd.Timestamp(ts),
                "ticker": sym,
                "close": close_px,
                "vol_ann": vol_ann,
                "risk_on": bool(row.get("risk_on", True)),
                "vol_ratio": float(row.get("vol_ratio", 1.0) or 1.0),
                "score": score,
                "score_static": score,
                "p_growth": round(p_growth, 4),
                "nw_est": round(float(row["nw_est"]), 4),
                "nw_lower": round(float(row["nw_lower"]), 4),
                "nw_env_pos": round(env_pos, 4),
                "nw_touches_lower": touches,
                "nw_dev_below": round(float(row["nw_dev_below"]), 4),
                "signal_side": side,
                COL_PROB_HMM_IMPULSE: float(row[COL_PROB_HMM_IMPULSE]),
                COL_PROB_HMM_MEAN_REVERT: float(row[COL_PROB_HMM_MEAN_REVERT]),
                COL_PROB_HMM_STRESS: float(row[COL_PROB_HMM_STRESS]),
            }
        )
    return pd.DataFrame(rows)


def build_nw_hmm_buy_signal_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Multi-ticker NW + HMM buy-only signal frame."""
    if prices.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for col in prices.columns:
        part = _attach_ticker_nw_hmm_signals(prices[col], str(col))
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)
