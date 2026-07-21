"""Balanced binary entry labels + TP/SL regression targets per instrument."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as _cfg
from research.labels.trade import (
    FWD_RET_ENTRY,
    TARGET_ENTRY,
    TARGET_ENTRY_SHORT,
    forward_return,
    future_mae,
    future_mfe,
)

TARGET_TP_BPS = "target_tp_bps"
TARGET_SL_BPS = "target_sl_bps"
BALANCED_HORIZON = "balanced_horizon"
BALANCED_POSITIVE_RATE = "balanced_positive_rate"


def _positive_rate(fwd: pd.Series, thr: float) -> float:
    valid = fwd.dropna()
    if valid.empty:
        return 0.5
    return float((valid >= thr).mean())


def pick_balanced_horizon(
    close: pd.Series,
    *,
    horizons: tuple[int, ...] | None = None,
    target_rate: float = 0.5,
) -> tuple[int, float, float]:
    """
    Choose horizon with class balance closest to ``target_rate``.

    Threshold = rolling median of forward return at each horizon (per instrument).
    Returns (horizon_bars, threshold_return_frac, achieved_positive_rate).
    """
    target = float(getattr(_cfg, "FUSION_BALANCED_TARGET_RATE", target_rate))
    if horizons is None:
        if getattr(_cfg, "IS_INTRADAY", False):
            horizons = (12, 24, 48, 96, 144, 192)
        else:
            horizons = (5, 10, 15, 20, 21, 42, 63)
    c = close.astype(float)
    best_h = int(horizons[0])
    best_thr = 0.0
    best_rate = 0.5
    best_score = -1.0
    for h in horizons:
        h = int(h)
        if h < 1 or len(c) < h + 50:
            continue
        fwd = forward_return(c, h)
        valid = fwd.dropna()
        if len(valid) < 100:
            continue
        thr = float(valid.median())
        rate = _positive_rate(fwd, thr)
        score = 1.0 - abs(rate - target)
        if score > best_score:
            best_score = score
            best_h = h
            best_thr = thr
            best_rate = rate
    return best_h, best_thr, best_rate


def build_balanced_entry_label(
    close: pd.Series,
    *,
    horizons: tuple[int, ...] | None = None,
    target_rate: float = 0.5,
) -> tuple[pd.Series, pd.Series, dict]:
    """Balanced binary label using median forward-return threshold per instrument."""
    h, thr, rate = pick_balanced_horizon(close, horizons=horizons, target_rate=target_rate)
    fwd = forward_return(close.astype(float), h)
    label = (fwd >= thr).astype(float)
    label[fwd.isna()] = np.nan
    meta = {
        "horizon": h,
        "threshold_return": thr,
        "positive_rate": round(rate, 4),
        "label_type": "balanced",
    }
    return label.rename(TARGET_ENTRY), fwd.rename(FWD_RET_ENTRY), meta


def build_balanced_short_label(
    close: pd.Series,
    *,
    horizons: tuple[int, ...] | None = None,
    target_rate: float = 0.5,
) -> tuple[pd.Series, dict]:
    """Balanced short label: forward return <= -median threshold."""
    h, thr, rate = pick_balanced_horizon(close, horizons=horizons, target_rate=target_rate)
    fwd = forward_return(close.astype(float), h)
    label = (fwd <= -thr).astype(float)
    label[fwd.isna()] = np.nan
    meta = {
        "horizon": h,
        "threshold_return": -thr,
        "positive_rate": round(rate, 4),
        "label_type": "balanced_short",
    }
    return label.rename(TARGET_ENTRY_SHORT), meta


def attach_tp_sl_regression_targets(
    df: pd.DataFrame,
    *,
    horizon_bars: int,
    close_col: str = "close",
) -> pd.DataFrame:
    """MFE/MAE over horizon as TP/SL regression targets (bps)."""
    if df.empty or close_col not in df.columns:
        return df
    out = df.copy()
    close = out[close_col].astype(float)
    h = max(int(horizon_bars), 1)
    mfe = future_mfe(close, h)
    mae = future_mae(close, h)
    out[TARGET_TP_BPS] = (mfe * 10_000.0).clip(5.0, 500.0)
    out[TARGET_SL_BPS] = (mae * 10_000.0).clip(5.0, 500.0)
    return out


def attach_balanced_entry_label(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    symbol: str | None = None,
) -> pd.DataFrame:
    """Attach balanced ``label_entry`` + TP/SL targets for one instrument panel."""
    if df.empty or close_col not in df.columns:
        return df
    out = df.copy()
    label, fwd, meta = build_balanced_entry_label(out[close_col])
    out[TARGET_ENTRY] = label.values
    out[FWD_RET_ENTRY] = fwd.values
    short_label, _ = build_balanced_short_label(out[close_col])
    out[TARGET_ENTRY_SHORT] = short_label.values
    out[BALANCED_HORIZON] = int(meta["horizon"])
    out[BALANCED_POSITIVE_RATE] = float(meta["positive_rate"])
    out = attach_tp_sl_regression_targets(out, horizon_bars=int(meta["horizon"]), close_col=close_col)
    if symbol:
        out.attrs["balanced_spec"] = {**meta, "symbol": str(symbol).upper()}
    return out
