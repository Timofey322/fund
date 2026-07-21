"""
HMM exhaustion rule strategy.

Buy:  downward price impulse (move > vol-scaled norm) + HMM says impulse stopping.
Sell: upward price impulse + HMM says impulse stopping.
"""

from __future__ import annotations

import math

import pandas as pd

from common.naming import (
    COL_PROB_HMM_IMPULSE,
    COL_PROB_HMM_MEAN_REVERT,
    COL_PROB_HMM_STRESS,
)
import config
from operations.scoring import ann_vol, log_returns
from research.regime.hmm import build_hmm_regime_frame

from rule.config import (
    RULE_BUY_MAINLY,
    RULE_DUMP_LOOKBACK_BARS,
    RULE_EXHAUSTION_MIN,
    RULE_IMPULSE_VOL_MULT,
    RULE_NEUTRAL_SCORE,
    RULE_PARTIAL_SELL_FRAC,
)


def expected_move_frac(vol_ann: float, lookback_bars: int) -> float:
    """Expected absolute return scale over lookback from annualized vol."""
    lb = max(int(lookback_bars), 1)
    vol = max(float(vol_ann), 1e-6)
    return vol * math.sqrt(lb / float(config.BARS_PER_YEAR))


def is_price_impulse(ret_frac: float, vol_ann: float, lookback_bars: int, mult: float) -> bool:
    """True when current |move| exceeds typical vol-scaled move (impulse)."""
    thr = expected_move_frac(vol_ann, lookback_bars) * max(float(mult), 1e-6)
    return abs(float(ret_frac)) > thr


def impulse_stopping_probability(
    prob_impulse: float,
    prob_mean_revert: float,
    prob_stress: float,
    prob_impulse_prev: float | None = None,
) -> float:
    """P(directional impulse is ending) from HMM regime probabilities."""
    p_stop = float(prob_mean_revert) + 0.12 * float(prob_stress)
    if prob_impulse < 0.38:
        p_stop += 0.38 - float(prob_impulse)
    if prob_impulse_prev is not None:
        p_stop += 0.45 * max(0.0, float(prob_impulse_prev) - float(prob_impulse))
    return float(min(1.0, max(0.0, p_stop)))


def exhaustion_score(
    ret_frac: float,
    vol_ann: float,
    p_stop: float,
    *,
    lookback_bars: int = RULE_DUMP_LOOKBACK_BARS,
    impulse_vol_mult: float = RULE_IMPULSE_VOL_MULT,
    exhaustion_min: float = RULE_EXHAUSTION_MIN,
    neutral_score: float = RULE_NEUTRAL_SCORE,
) -> tuple[float, str]:
    """
    Map vol-relative impulse + HMM exhaustion -> score for the backtest engine.

  Impulse: |ret| > mult * vol-scaled expected move for this instrument.
    """
    thr = expected_move_frac(vol_ann, lookback_bars) * max(float(impulse_vol_mult), 1e-6)
    is_dump = float(ret_frac) <= -thr
    is_rally = float(ret_frac) >= thr
    if not (is_dump or is_rally) or p_stop < exhaustion_min:
        return neutral_score, "neutral"

    exc = min(1.0, (p_stop - exhaustion_min) / max(1.0 - exhaustion_min, 1e-6))
    mag = min(1.0, abs(float(ret_frac)) / max(thr, 1e-9))
    strength = exc * mag
    if is_dump:
        score = neutral_score + (100.0 - neutral_score) * strength
        return float(min(100.0, score)), "buy_dump_exhaustion"
    # Buy-mainly: ignore rally sells unless partial trim is enabled.
    if RULE_BUY_MAINLY and RULE_PARTIAL_SELL_FRAC <= 0.0:
        return neutral_score, "sell_rally_exhaustion"
    score = neutral_score - neutral_score * strength
    return float(max(0.0, score)), "sell_rally_exhaustion"


def _attach_ticker_hmm_signals(close: pd.Series, ticker: str) -> pd.DataFrame:
    """Per-instrument bar-level HMM regime + exhaustion scores."""
    c = close.dropna().astype(float)
    if len(c) < 100:
        return pd.DataFrame()

    sym = str(ticker).upper()
    hmm = build_hmm_regime_frame(pd.DataFrame({sym: c}), ticker=sym)
    if hmm.empty:
        return pd.DataFrame()

    time_col = "bar_time" if "bar_time" in hmm.columns else "date"
    hmm = hmm.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
    hmm_idx = pd.to_datetime(hmm[time_col])
    prob_cols = [COL_PROB_HMM_IMPULSE, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_STRESS]
    hmm_feats = hmm.set_index(hmm_idx)[prob_cols + ["risk_on", "vol_ratio"]]

    bar_idx = pd.to_datetime(c.index)
    frame = pd.DataFrame({"close": c.values}, index=bar_idx)
    merged = frame.join(hmm_feats, how="inner")
    if merged.empty:
        return pd.DataFrame()

    lookback = int(RULE_DUMP_LOOKBACK_BARS)
    merged["ret_frac"] = merged["close"].pct_change(lookback, fill_method=None)
    vol = ann_vol(log_returns(merged["close"]))

    p_imp = merged[COL_PROB_HMM_IMPULSE].astype(float)
    p_imp_prev = p_imp.shift(1)

    rows: list[dict] = []
    for ts, row in merged.iterrows():
        prev = p_imp_prev.loc[ts] if ts in p_imp_prev.index else None
        p_stop = impulse_stopping_probability(
            float(row[COL_PROB_HMM_IMPULSE]),
            float(row[COL_PROB_HMM_MEAN_REVERT]),
            float(row[COL_PROB_HMM_STRESS]),
            float(prev) if prev is not None and pd.notna(prev) else None,
        )
        ret_frac = float(row["ret_frac"]) if pd.notna(row["ret_frac"]) else 0.0
        vol_ann = float(vol.loc[ts]) if ts in vol.index and pd.notna(vol.loc[ts]) else 0.25
        exp_pct = expected_move_frac(vol_ann, lookback) * 100.0
        score, side = exhaustion_score(ret_frac, vol_ann, p_stop)
        rows.append(
            {
                "date": pd.Timestamp(ts),
                "ticker": sym,
                "close": float(row["close"]),
                "vol_ann": vol_ann,
                "risk_on": bool(row.get("risk_on", True)),
                "vol_ratio": float(row.get("vol_ratio", 1.0) or 1.0),
                "score": score,
                "score_static": score,
                "p_stop": round(p_stop, 4),
                "ret_pct": round(ret_frac * 100.0, 3),
                "impulse_thr_pct": round(exp_pct * RULE_IMPULSE_VOL_MULT, 3),
                "is_impulse": is_price_impulse(ret_frac, vol_ann, lookback, RULE_IMPULSE_VOL_MULT),
                "signal_side": side,
                COL_PROB_HMM_IMPULSE: float(row[COL_PROB_HMM_IMPULSE]),
                COL_PROB_HMM_MEAN_REVERT: float(row[COL_PROB_HMM_MEAN_REVERT]),
                COL_PROB_HMM_STRESS: float(row[COL_PROB_HMM_STRESS]),
            }
        )
    return pd.DataFrame(rows)


def build_hmm_exhaustion_signal_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Build multi-ticker HMM exhaustion signals (bar-level when HMM_FREQUENCY=bar)."""
    if prices.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for col in prices.columns:
        part = _attach_ticker_hmm_signals(prices[col], str(col))
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)
