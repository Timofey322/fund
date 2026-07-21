"""
Volume Profile (POC / Value Area) — causal, session-expanding.

Computed only from bars seen so far in the session (no lookahead).
HMM regime modulates how VP levels are interpreted (trend vs range vs crisis).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd

from common.naming import COL_PROB_HMM_STRESS, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_IMPULSE

VP_BINS = 24
VALUE_AREA_PCT = 0.70
MIN_BARS_VP = 6

VP_COLS = (
    "vp_poc",
    "vp_vah",
    "vp_val",
    "vp_poc_dist",
    "vp_va_width",
    "vp_in_value",
    "vp_above_poc",
    "vp_below_val",
    "vp_near_poc",
)

VP_HMM_COLS = (
    "vp_hmm_impulse_sig",
    "vp_hmm_mean_revert_sig",
    "vp_hmm_composite",
)


def _typical_price(df: pd.DataFrame) -> np.ndarray:
    return ((df["high"] + df["low"] + df["close"]) / 3.0).values


def _compute_vp_levels(
    sl: pd.DataFrame,
    n_bins: int = VP_BINS,
    va_pct: float = VALUE_AREA_PCT,
) -> tuple[float, float, float]:
    """POC, VAH, VAL from a bar slice."""
    if len(sl) < MIN_BARS_VP:
        c = float(sl["close"].iloc[-1])
        return c, c, c

    lo = float(sl["low"].min())
    hi = float(sl["high"].max())
    if hi <= lo:
        c = float(sl["close"].iloc[-1])
        return c, c, c

    tp = _typical_price(sl)
    vol = sl["volume"].fillna(0.0).values
    if vol.sum() <= 0:
        c = float(sl["close"].iloc[-1])
        return c, c, c

    edges = np.linspace(lo, hi, n_bins + 1)
    hist, _ = np.histogram(tp, bins=edges, weights=vol)
    if hist.sum() <= 0:
        c = float(sl["close"].iloc[-1])
        return c, c, c

    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_idx = int(np.argmax(hist))
    poc = float(centers[poc_idx])

    # Value area: expand from POC until va_pct of volume captured
    order = np.argsort(-hist)
    cum = 0.0
    target = hist.sum() * va_pct
    included = set()
    for idx in order:
        cum += hist[idx]
        included.add(idx)
        if cum >= target:
            break
    if not included:
        included = {poc_idx}
    va_lo = float(centers[min(included)])
    va_hi = float(centers[max(included)])
    return poc, va_hi, va_lo


def attach_volume_profile(
    df: pd.DataFrame,
    session: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Session-causal volume profile: at each bar, VP uses only prior bars in session.
    """
    if df.empty or "close" not in df.columns:
        return df

    out = df.copy()
    if session is None:
        session = pd.Series(out.index.normalize(), index=out.index)

    poc_arr = np.full(len(out), np.nan)
    vah_arr = np.full(len(out), np.nan)
    val_arr = np.full(len(out), np.nan)

    sess_vals = session.values
    for sess in pd.unique(sess_vals):
        mask = sess_vals == sess
        idx = np.where(mask)[0]
        for j, i in enumerate(idx):
            sl = out.iloc[idx[: j + 1]]
            poc, vah, val = _compute_vp_levels(sl)
            poc_arr[i] = poc
            vah_arr[i] = vah
            val_arr[i] = val

    close = out["close"].astype(float).values
    poc_arr = np.where(np.isnan(poc_arr), close, poc_arr)
    vah_arr = np.where(np.isnan(vah_arr), close, vah_arr)
    val_arr = np.where(np.isnan(val_arr), close, val_arr)

    width = np.maximum(vah_arr - val_arr, 1e-9)
    out["vp_poc"] = poc_arr
    out["vp_vah"] = vah_arr
    out["vp_val"] = val_arr
    out["vp_poc_dist"] = (close - poc_arr) / poc_arr
    out["vp_va_width"] = width / np.maximum(poc_arr, 1e-9)
    out["vp_in_value"] = ((close >= val_arr) & (close <= vah_arr)).astype(float)
    out["vp_above_poc"] = (close > poc_arr).astype(float)
    out["vp_below_val"] = (close < val_arr).astype(float)
    out["vp_near_poc"] = (np.abs(close - poc_arr) / np.maximum(poc_arr, 1e-9) < 0.002).astype(float)
    return out


def vp_hmm_interpretation(row: pd.Series) -> dict[str, float]:
    """
    Regime-conditioned VP signal (Markov context):

    - HMM_IMPULSE: price above POC / breaking VAH → continuation long bias
    - HMM_MEAN_REVERT: price near VAL → bounce; near VAH → fade
    - HMM_STRESS: dampen all VP signals
    """
    p_trend = float(row.get(COL_PROB_HMM_IMPULSE, 0.33))
    p_range = float(row.get(COL_PROB_HMM_MEAN_REVERT, 0.33))
    p_crisis = float(row.get(COL_PROB_HMM_STRESS, 0.34))

    poc_dist = float(row.get("vp_poc_dist", 0.0) or 0.0)
    above_poc = float(row.get("vp_above_poc", 0.5) or 0.5)
    below_val = float(row.get("vp_below_val", 0.0) or 0.0)
    in_val = float(row.get("vp_in_value", 0.5) or 0.5)
    width = max(float(row.get("vp_va_width", 0.01) or 0.01), 1e-6)

    # Trend regime: momentum from POC breakout
    trend_sig = above_poc * (1.0 + poc_dist) + (1.0 - in_val) * 0.5
    trend_sig = float(np.clip(trend_sig, -1.0, 2.0))

    # Range regime: mean-revert toward POC from extremes
    range_sig = below_val * 1.0 - float(row.get("vp_above_poc", 0)) * (poc_dist / width) * 0.5
    if poc_dist < -0.001 * width:
        range_sig += abs(poc_dist) * 0.5
    range_sig = float(np.clip(range_sig, -1.0, 1.5))

    crisis_damp = max(0.0, 1.0 - p_crisis * 1.5)
    composite = crisis_damp * (p_trend * trend_sig + p_range * range_sig)

    return {
        "vp_hmm_impulse_sig": round(p_trend * trend_sig, 4),
        "vp_hmm_mean_revert_sig": round(p_range * range_sig, 4),
        "vp_hmm_composite": round(composite, 4),
    }


def attach_vp_hmm_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add HMM × Volume Profile interaction columns (requires VP + HMM cols)."""
    if panel.empty:
        return panel
    out = panel.copy()
    rows = [vp_hmm_interpretation(row) for _, row in out.iterrows()]
    vp_df = pd.DataFrame(rows, index=out.index)
    for c in VP_HMM_COLS:
        out[c] = vp_df[c]
    return out
