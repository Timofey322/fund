"""Entry ML feature engineering and LightGBM hyperparameter helpers for fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from research.features.hybrid_flow import merge_hybrid_flow

FWD_HORIZON_BARS = 12
TRAIN_SESSIONS_MIN = 8
WARMUP_SESSIONS = 3
REOPT_EVERY_SESSIONS = 3

FLOW_FEATURE_COLS = [
    "ret_1",
    "ret_6",
    "ret_12",
    "vol_imbalance",
    "imb_ma_6",
    "imb_ma_12",
    "buy_share",
    "clv",
    "lower_wick_ratio",
    "upper_wick_ratio",
    "body_ratio",
    "hammer_score",
    "flow_source",
    "tick_imbalance",
    "vol_z",
    "body_pct",
    "session_progress",
]

PARAM_GRID = [
    {"max_depth": 2, "learning_rate": 0.10, "max_iter": 80, "min_samples_leaf": 40},
    {"max_depth": 3, "learning_rate": 0.08, "max_iter": 100, "min_samples_leaf": 30},
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 150, "min_samples_leaf": 25},
    {"max_depth": 4, "learning_rate": 0.06, "max_iter": 150, "min_samples_leaf": 20},
    {"max_depth": 5, "learning_rate": 0.04, "max_iter": 200, "min_samples_leaf": 15},
    {"max_depth": 4, "learning_rate": 0.08, "max_iter": 120, "min_samples_leaf": 30, "l2_regularization": 0.1},
]


def _session_key(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.normalize(), index=index, name="session")


def _engineer_features(df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    """OHLCV + hybrid volume (ticks recent window, candles elsewhere)."""
    out = merge_hybrid_flow(df, symbol=symbol)
    close = out["close"]
    out["ret_1"] = close.pct_change(1, fill_method=None)
    out["ret_6"] = close.pct_change(6, fill_method=None)
    out["ret_12"] = close.pct_change(12, fill_method=None)

    imb = out["vol_imbalance"].fillna(0.0)
    out["imb_ma_6"] = imb.rolling(6, min_periods=1).mean()
    out["imb_ma_12"] = imb.rolling(12, min_periods=1).mean()

    vol = out["volume"].fillna(0.0)
    vol_ma = vol.rolling(20, min_periods=5).mean().replace(0, np.nan)
    out["vol_z"] = (vol / vol_ma).replace([np.inf, -np.inf], np.nan)
    out["body_pct"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)

    sess = _session_key(out.index)
    if symbol:
        from common.timeframe import session_id
        sess = session_id(out.index, symbol=symbol)
    out["session"] = sess
    pos = out.groupby(sess.values).cumcount()
    sess_len = out.groupby(sess.values)["close"].transform("count").clip(lower=1)
    out["session_progress"] = pos / (sess_len - 1).clip(lower=1)

    from research.features.registry import attach_price_flow_derived

    return attach_price_flow_derived(out)


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 6:
        return 0.0
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) < 1e-9 or np.std(ry) < 1e-9:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return 0.5


def effective_purge_sessions(
    max_label_horizon_bars: int | None = None,
    *,
    purge: int | None = None,
) -> int:
    """Purge gap (sessions) wide enough to separate overlapping forward labels."""
    if purge is not None:
        return max(1, int(purge))
    bars = max_label_horizon_bars
    if bars is None:
        try:
            from strategy.target_opt import applied_hold_default

            bars = applied_hold_default(FWD_HORIZON_BARS)
        except Exception:
            bars = FWD_HORIZON_BARS
    try:
        import config as _cfg

        bars_per_session = int(getattr(_cfg, "CRYPTO_BARS_PER_DAY", 288))
        floor = int(getattr(_cfg, "PURGE_PERIODS", 1) or 1)
    except Exception:
        bars_per_session = 288
        floor = 1
    computed = max(1, int(np.ceil(int(bars) / max(bars_per_session, 1))))
    return max(floor, computed)


def _purged_session_folds(
    sessions: list,
    n_splits: int = 4,
    purge: int | None = None,
    *,
    max_label_horizon_bars: int | None = None,
) -> list[tuple[set, set]]:
    purge = effective_purge_sessions(max_label_horizon_bars, purge=purge)
    n = len(sessions)
    if n < n_splits * 2:
        mid = n // 2
        pl, ph = max(0, mid - purge), min(n, mid + purge)
        train_s = set(sessions[:pl]) if pl > 0 else set(sessions[:max(1, mid)])
        test_s = set(sessions[ph:]) if ph < n else set(sessions[mid:])
        if train_s and test_s:
            return [(train_s, test_s)]
        return [(set(sessions[:mid]), set(sessions[mid:]))]
    fold = max(1, n // n_splits)
    folds = []
    for k in range(n_splits):
        ts = k * fold
        te = n if k == n_splits - 1 else min((k + 1) * fold, n)
        test_s = set(sessions[ts:te])
        pl, ph = max(0, ts - purge), min(n, te + purge)
        train_s = set(sessions) - set(sessions[pl:ph])
        if train_s and test_s:
            folds.append((train_s, test_s))
    return folds
