"""Trade target engineering for entry, stop-loss, take-profit, and hold time."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_12_POSITIVE = "label_12_positive"
TARGET_20_1PCT = "label_20_1pct"
TARGET_12_AFTER_COSTS = "label_12_after_costs"
TARGET_20_AFTER_COSTS = "label_20_after_costs"
TARGET_TRIPLE_BARRIER_20 = "label_triple_barrier_20"

# Unified per-instrument entry target column (built from a per-symbol spec).
TARGET_ENTRY = "label_entry"
TARGET_ENTRY_SHORT = "label_entry_short"
TARGET_DIRECTION = "label_direction"
DIRECTION_FLAT = 0
DIRECTION_LONG = 1
DIRECTION_SHORT = 2
DIRECTION_NAMES = ("flat", "long", "short")
FWD_RET_ENTRY = "fwd_ret_entry"
ENTRY_LABEL_HORIZON = "entry_label_horizon"
ENTRY_LONG_POSITIVE_RATE = "entry_long_positive_rate"
ENTRY_SHORT_POSITIVE_RATE = "entry_short_positive_rate"
LABEL_TYPES = ("after_costs", "positive", "triple_barrier")

# Holding-horizon buckets span 1h..24h (5Min bars) so the regime-adaptive exit
# selector can match long per-instrument target horizons, not just intraday holds.
HOLD_BUCKETS = (12, 24, 48, 96, 144, 192, 288)
DEFAULT_COMMISSION_THRESHOLD_BPS = 20.0

import config as _cfg

DEFAULT_SLIPPAGE_BPS = float(getattr(_cfg, "SLIPPAGE_BPS_CRYPTO", 1.5))
DEFAULT_AFTER_COST_THRESHOLD_BPS = DEFAULT_COMMISSION_THRESHOLD_BPS + DEFAULT_SLIPPAGE_BPS


def forward_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    """Forward percent return over `horizon_bars`."""
    return close.astype(float).pct_change(horizon_bars, fill_method=None).shift(-horizon_bars)


def future_path_returns(close: pd.Series, horizon_bars: int) -> pd.DataFrame:
    """Forward returns for every bar from +1 to +horizon."""
    c = close.astype(float)
    return pd.concat(
        {i: c.shift(-i) / c - 1.0 for i in range(1, horizon_bars + 1)},
        axis=1,
    )


def future_mae(close: pd.Series, horizon_bars: int) -> pd.Series:
    """Long-side maximum adverse excursion as a positive loss magnitude."""
    path = future_path_returns(close, horizon_bars)
    return (-path.min(axis=1)).clip(lower=0.0)


def future_mfe(close: pd.Series, horizon_bars: int) -> pd.Series:
    """Long-side maximum favorable excursion."""
    path = future_path_returns(close, horizon_bars)
    return path.max(axis=1).clip(lower=0.0)


def triple_barrier_label(
    close: pd.Series,
    *,
    horizon_bars: int,
    take_profit_bps: float,
    stop_loss_bps: float,
) -> pd.Series:
    """1 when TP is touched before SL inside the horizon, else 0."""
    path = future_path_returns(close, horizon_bars)
    tp = float(take_profit_bps) / 10_000.0
    sl = -float(stop_loss_bps) / 10_000.0
    arr = path.to_numpy(dtype=float)
    valid = np.isfinite(arr)
    any_valid = valid.any(axis=1)
    full_window = valid.sum(axis=1) >= int(horizon_bars)
    tp_hit = arr >= tp
    sl_hit = arr <= sl
    has_tp = tp_hit.any(axis=1)
    has_sl = sl_hit.any(axis=1)
    # argmax returns 0 when no hit, so use has_* masks before comparing.
    first_tp = np.argmax(tp_hit, axis=1)
    first_sl = np.argmax(sl_hit, axis=1)

    labels = np.full(len(path), np.nan, dtype=float)
    labels[has_tp & ~has_sl] = 1.0
    labels[has_sl & ~has_tp] = 0.0
    both = has_tp & has_sl
    labels[both] = (first_tp[both] < first_sl[both]).astype(float)
    # Timeout (full window, neither barrier): 0 — do not dilute cost floor with
    # "any positive path max" (that labeled weak moves as wins).
    no_hit = any_valid & ~has_tp & ~has_sl & full_window
    labels[no_hit] = 0.0
    return pd.Series(labels, index=close.index, name=TARGET_TRIPLE_BARRIER_20)


def triple_barrier_label_short(
    close: pd.Series,
    *,
    horizon_bars: int,
    take_profit_bps: float,
    stop_loss_bps: float,
) -> pd.Series:
    """1 when short TP (down move) is touched before short SL (up move) within the horizon."""
    path = future_path_returns(close, horizon_bars)
    tp = -float(take_profit_bps) / 10_000.0
    sl = float(stop_loss_bps) / 10_000.0
    arr = path.to_numpy(dtype=float)
    valid = np.isfinite(arr)
    any_valid = valid.any(axis=1)
    full_window = valid.sum(axis=1) >= int(horizon_bars)
    tp_hit = arr <= tp
    sl_hit = arr >= sl
    has_tp = tp_hit.any(axis=1)
    has_sl = sl_hit.any(axis=1)
    first_tp = np.argmax(tp_hit, axis=1)
    first_sl = np.argmax(sl_hit, axis=1)

    labels = np.full(len(path), np.nan, dtype=float)
    labels[has_tp & ~has_sl] = 1.0
    labels[has_sl & ~has_tp] = 0.0
    both = has_tp & has_sl
    labels[both] = (first_tp[both] < first_sl[both]).astype(float)
    no_hit = any_valid & ~has_tp & ~has_sl & full_window
    labels[no_hit] = 0.0
    return pd.Series(labels, index=close.index, name=TARGET_ENTRY_SHORT)


def _first_barrier_win_bar(
    arr: np.ndarray,
    valid: np.ndarray,
    *,
    tp_hit: np.ndarray,
    sl_hit: np.ndarray,
) -> np.ndarray:
    """Bar index of TP touch when TP is before SL; else -1."""
    has_tp = tp_hit.any(axis=1)
    has_sl = sl_hit.any(axis=1)
    first_tp = np.argmax(tp_hit, axis=1)
    first_sl = np.argmax(sl_hit, axis=1)
    wins = np.full(len(arr), -1, dtype=int)
    tp_only = has_tp & ~has_sl
    wins[tp_only] = first_tp[tp_only]
    both = has_tp & has_sl
    wins[both & (first_tp < first_sl)] = first_tp[both & (first_tp < first_sl)]
    return wins


def triple_barrier_direction_label(
    close: pd.Series,
    *,
    horizon_bars: int,
    take_profit_bps: float,
    stop_loss_bps: float,
) -> pd.Series:
    """
    Mutually exclusive direction: 0=flat, 1=long, 2=short.

  First winning triple-barrier touch decides direction; timeout/no-win => flat.
    """
    path = future_path_returns(close, horizon_bars)
    tp = float(take_profit_bps) / 10_000.0
    sl = float(stop_loss_bps) / 10_000.0
    arr = path.to_numpy(dtype=float)
    valid = np.isfinite(arr)
    any_valid = valid.any(axis=1)
    full_window = valid.sum(axis=1) >= int(horizon_bars)

    long_tp_hit = arr >= tp
    long_sl_hit = arr <= -sl
    short_tp_hit = arr <= -tp
    short_sl_hit = arr >= sl

    long_win = _first_barrier_win_bar(arr, valid, tp_hit=long_tp_hit, sl_hit=long_sl_hit)
    short_win = _first_barrier_win_bar(arr, valid, tp_hit=short_tp_hit, sl_hit=short_sl_hit)

    labels = np.full(len(path), np.nan, dtype=float)
    only_long = (long_win >= 0) & (short_win < 0)
    only_short = (short_win >= 0) & (long_win < 0)
    both = (long_win >= 0) & (short_win >= 0)
    labels[only_long] = DIRECTION_LONG
    labels[only_short] = DIRECTION_SHORT
    labels[both & (long_win < short_win)] = DIRECTION_LONG
    labels[both & (short_win < long_win)] = DIRECTION_SHORT
    labels[both & (long_win == short_win)] = DIRECTION_FLAT
    no_win = any_valid & (long_win < 0) & (short_win < 0) & full_window
    labels[no_win] = DIRECTION_FLAT
    return pd.Series(labels, index=close.index, name=TARGET_DIRECTION)


def build_direction_label(
    close: pd.Series,
    *,
    horizon_bars: int,
    label_type: str = "triple_barrier",
    threshold_bps: float = DEFAULT_AFTER_COST_THRESHOLD_BPS,
    stop_loss_bps: float | None = None,
    commission_bps: float | None = None,
) -> pd.Series:
    """3-class direction label aligned with economic entry specs."""
    if label_type != "triple_barrier":
        long_l, _ = build_entry_label(
            close,
            horizon_bars=horizon_bars,
            label_type=label_type,
            threshold_bps=threshold_bps,
            stop_loss_bps=stop_loss_bps,
            commission_bps=commission_bps,
        )
        short_l = build_short_entry_label(
            close,
            horizon_bars=horizon_bars,
            label_type=label_type,
            threshold_bps=threshold_bps,
            stop_loss_bps=stop_loss_bps,
            commission_bps=commission_bps,
        )
        out = np.full(len(close), DIRECTION_FLAT, dtype=float)
        lv = long_l.fillna(-1).astype(int).to_numpy()
        sv = short_l.fillna(-1).astype(int).to_numpy()
        out[(lv == 1) & (sv != 1)] = DIRECTION_LONG
        out[(sv == 1) & (lv != 1)] = DIRECTION_SHORT
        out[(lv == 1) & (sv == 1)] = DIRECTION_FLAT
        out[(long_l.isna()) | (short_l.isna())] = np.nan
        return pd.Series(out, index=close.index, name=TARGET_DIRECTION)

    tp_bps, sl_bps, _ = _resolve_label_costs(
        threshold_bps=threshold_bps,
        stop_loss_bps=stop_loss_bps,
        commission_bps=commission_bps,
    )
    return triple_barrier_direction_label(
        close.astype(float),
        horizon_bars=int(horizon_bars),
        take_profit_bps=tp_bps,
        stop_loss_bps=sl_bps,
    )


def direction_class_rates(label: pd.Series) -> dict[str, float]:
    """Share of flat/long/short in a direction label series."""
    valid = label.dropna().astype(int)
    if valid.empty:
        return {name: 0.0 for name in DIRECTION_NAMES}
    n = len(valid)
    return {
        name: float((valid == code).sum() / n)
        for code, name in zip((0, 1, 2), DIRECTION_NAMES)
    }


def _resolve_label_costs(
    *,
    threshold_bps: float,
    stop_loss_bps: float | None = None,
    commission_bps: float | None = None,
) -> tuple[float, float, float]:
    """Return (tp_bps, sl_bps, threshold_frac) for one instrument spec."""
    from simulation.entry_signals import min_tp_gross_bps

    comm = float(
        commission_bps
        if commission_bps is not None
        else getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 10.0)
    )
    sl_bps = float(
        stop_loss_bps
        if stop_loss_bps is not None
        else getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0)
    )
    min_tp = min_tp_gross_bps(comm, buffer_bps=0.0, stop_loss_bps=sl_bps)
    tp_bps = max(float(threshold_bps), min_tp)
    return tp_bps, sl_bps, tp_bps / 10_000.0


def default_entry_spec(symbol: str) -> dict:
    """Fixed economic entry spec (no full-sample statistics — no label lookahead)."""
    from data_platform.universe import commission_bps_for_ticker
    from research.features.entry_ml import FWD_HORIZON_BARS

    sym = str(symbol).upper()
    comm = commission_bps_for_ticker(sym)
    tp_bps, _, _ = _resolve_label_costs(
        threshold_bps=float(getattr(_cfg, "FUSION_DEFAULT_ENTRY_THRESHOLD_BPS", 0.0) or 0.0),
        commission_bps=comm,
    )
    horizon = int(getattr(_cfg, "FUSION_DEFAULT_ENTRY_HORIZON", FWD_HORIZON_BARS))
    return {
        "horizon": max(horizon, 1),
        "label_type": str(getattr(_cfg, "FUSION_DEFAULT_ENTRY_LABEL_TYPE", "triple_barrier")),
        "threshold_bps": float(tp_bps),
    }


def resolve_entry_spec(symbol: str) -> dict:
    """Applied target-opt cache > inline config > fixed economic default per symbol."""
    sym = str(symbol).upper()
    try:
        from strategy.target_opt import per_instrument_specs

        specs = per_instrument_specs(tradeable_only=False)
        if sym in specs:
            return dict(specs[sym])
    except Exception:
        pass
    return default_entry_spec(sym)


def build_short_entry_label(
    close: pd.Series,
    *,
    horizon_bars: int,
    label_type: str = "after_costs",
    threshold_bps: float = DEFAULT_AFTER_COST_THRESHOLD_BPS,
    stop_loss_bps: float | None = None,
    commission_bps: float | None = None,
) -> pd.Series:
    """Short-side binary label mirroring the long ``build_entry_label`` economics."""
    if label_type not in LABEL_TYPES:
        raise ValueError(f"Unknown label_type={label_type!r}; expected one of {LABEL_TYPES}")
    h = int(horizon_bars)
    if h < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    c = close.astype(float)
    fwd = forward_return(c, h)
    tp_bps, sl_bps, thr = _resolve_label_costs(
        threshold_bps=threshold_bps,
        stop_loss_bps=stop_loss_bps,
        commission_bps=commission_bps,
    )

    if label_type == "triple_barrier":
        label = triple_barrier_label_short(
            c, horizon_bars=h, take_profit_bps=tp_bps, stop_loss_bps=sl_bps
        )
    elif label_type == "positive":
        label = (fwd < 0).astype(float)
        label[fwd.isna()] = np.nan
    else:  # after_costs
        label = (fwd <= -thr).astype(float)
        label[fwd.isna()] = np.nan

    return label.rename(TARGET_ENTRY_SHORT)


def attach_economic_entry_labels(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    symbol: str | None = None,
    spec: dict | None = None,
) -> pd.DataFrame:
    """Attach long + short entry labels from fixed economic specs (SL + commission aligned)."""
    if df.empty or close_col not in df.columns:
        return df
    from research.labels.balanced import attach_tp_sl_regression_targets

    sym = str(symbol or "").upper()
    resolved = dict(spec or (resolve_entry_spec(sym) if sym else default_entry_spec("SPY")))
    comm = None
    if sym:
        from data_platform.universe import commission_bps_for_ticker

        comm = commission_bps_for_ticker(sym)

    close = df[close_col].astype(float)
    h = int(resolved.get("horizon", 12))
    label_type = str(resolved.get("label_type", "triple_barrier"))
    threshold_bps = float(resolved.get("threshold_bps", DEFAULT_AFTER_COST_THRESHOLD_BPS))

    label, fwd = build_entry_label(
        close,
        horizon_bars=h,
        label_type=label_type,
        threshold_bps=threshold_bps,
        commission_bps=comm,
    )
    short_label = build_short_entry_label(
        close,
        horizon_bars=h,
        label_type=label_type,
        threshold_bps=threshold_bps,
        commission_bps=comm,
    )

    out = df.copy()
    out[TARGET_ENTRY] = label.values
    out[FWD_RET_ENTRY] = fwd.values
    out[TARGET_ENTRY_SHORT] = short_label.values
    direction = build_direction_label(
        close,
        horizon_bars=h,
        label_type=label_type,
        threshold_bps=threshold_bps,
        commission_bps=comm,
    )
    out[TARGET_DIRECTION] = direction.values
    rates = direction_class_rates(direction)
    out["direction_flat_rate"] = rates["flat"]
    out["direction_long_rate"] = rates["long"]
    out["direction_short_rate"] = rates["short"]
    out[ENTRY_LABEL_HORIZON] = h
    valid_long = label.dropna()
    valid_short = short_label.dropna()
    out[ENTRY_LONG_POSITIVE_RATE] = float(valid_long.mean()) if not valid_long.empty else np.nan
    out[ENTRY_SHORT_POSITIVE_RATE] = float(valid_short.mean()) if not valid_short.empty else np.nan
    out = attach_tp_sl_regression_targets(out, horizon_bars=h, close_col=close_col)
    if sym:
        out.attrs["entry_spec"] = {**resolved, "symbol": sym}
    return out


def best_hold_bucket(close: pd.Series, buckets: tuple[int, ...] = HOLD_BUCKETS) -> pd.Series:
    """Bucket with the best forward return among candidate hold horizons."""
    c = close.astype(float)
    fwd = pd.concat({b: forward_return(c, b) for b in buckets}, axis=1)
    labels = fwd.dropna(how="all").idxmax(axis=1)
    labels = labels.reindex(close.index)
    labels[fwd.isna().all(axis=1)] = np.nan
    return labels


def build_entry_label(
    close: pd.Series,
    *,
    horizon_bars: int,
    label_type: str = "after_costs",
    threshold_bps: float = DEFAULT_AFTER_COST_THRESHOLD_BPS,
    stop_loss_bps: float | None = None,
    commission_bps: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Binary entry label + its forward return for one (horizon, type, threshold) spec.

    - ``after_costs``: 1 when fwd_ret(H) >= min TP (``max(threshold, SL + commission)``).
    - ``positive``: 1 when fwd_ret(H) > 0 (directional baseline).
    - ``triple_barrier``: 1 when TP is touched before SL within H (asymmetric: TP >= SL + commission).

    Returns ``(label, fwd_ret)`` aligned to ``close.index``. NaN labels where the
    forward window runs past the available data are preserved (excluded downstream).
    """
    if label_type not in LABEL_TYPES:
        raise ValueError(f"Unknown label_type={label_type!r}; expected one of {LABEL_TYPES}")
    h = int(horizon_bars)
    if h < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    c = close.astype(float)
    fwd = forward_return(c, h)

    tp_bps, sl_bps, thr = _resolve_label_costs(
        threshold_bps=threshold_bps,
        stop_loss_bps=stop_loss_bps,
        commission_bps=commission_bps,
    )

    if label_type == "triple_barrier":
        label = triple_barrier_label(
            c, horizon_bars=h, take_profit_bps=tp_bps, stop_loss_bps=sl_bps
        )
    elif label_type == "positive":
        label = (fwd > 0).astype(float)
        label[fwd.isna()] = np.nan
    else:  # after_costs
        label = (fwd >= thr).astype(float)
        label[fwd.isna()] = np.nan

    return label.rename(TARGET_ENTRY), fwd.rename(FWD_RET_ENTRY)


def attach_entry_label(
    df: pd.DataFrame,
    spec: dict,
    *,
    close_col: str = "close",
    symbol: str | None = None,
) -> pd.DataFrame:
    """Attach long/short ``label_entry`` columns from a per-symbol spec."""
    sym = str(symbol or spec.get("symbol") or "").upper() or None
    return attach_economic_entry_labels(df, close_col=close_col, symbol=sym, spec=spec)


def attach_trade_targets(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    baseline_horizon: int = 12,
    old_horizon: int = 20,
    old_threshold: float = 0.01,
    after_cost_threshold_bps: float = DEFAULT_AFTER_COST_THRESHOLD_BPS,
    slippage_bps: float = 0.0,
    triple_barrier_horizon: int = 20,
    triple_barrier_tp_bps: float | None = None,
    triple_barrier_sl_bps: float | None = None,
    commission_bps: float | None = None,
    risk_horizons: tuple[int, ...] = (20, 48, 96),
) -> pd.DataFrame:
    """Attach all trade ML targets while keeping legacy `fwd_ret`/`label` columns."""
    if df.empty or close_col not in df.columns:
        return df

    import config as _cfg
    from simulation.entry_signals import edge_floor_bps, min_tp_gross_bps

    out = df.copy()
    close = out[close_col].astype(float)
    comm = float(
        commission_bps
        if commission_bps is not None
        else getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 10.0)
    )
    sl_bps = float(
        triple_barrier_sl_bps
        if triple_barrier_sl_bps is not None
        else getattr(_cfg, "FUSION_STOP_LOSS_BPS", float(after_cost_threshold_bps))
    )
    label_mode = str(getattr(_cfg, "FUSION_LABEL_TP_MODE", "sl_plus_commission"))
    if label_mode in ("commission_only", "full_round_trip", "full_cost"):
        min_tp = edge_floor_bps(comm, slippage_bps, mode=label_mode, buffer_bps=0.0)
    else:
        min_tp = min_tp_gross_bps(comm, slippage_bps, stop_loss_bps=sl_bps, buffer_bps=0.0)
    tp_bps = float(triple_barrier_tp_bps if triple_barrier_tp_bps is not None else min_tp)
    tp_bps = max(tp_bps, min_tp)
    after_cost_threshold = max(float(after_cost_threshold_bps), min_tp) / 10_000.0
    after_cost_threshold += float(slippage_bps) / 10_000.0
    out["fwd_ret_12"] = forward_return(close, baseline_horizon)
    out[TARGET_12_POSITIVE] = (out["fwd_ret_12"] > 0).astype(int)
    out[TARGET_12_AFTER_COSTS] = (out["fwd_ret_12"] >= after_cost_threshold).astype(int)
    out["fwd_ret_20"] = forward_return(close, old_horizon)
    out[TARGET_20_1PCT] = (out["fwd_ret_20"] >= old_threshold).astype(int)
    out[TARGET_20_AFTER_COSTS] = (out["fwd_ret_20"] >= after_cost_threshold).astype(int)
    out[TARGET_TRIPLE_BARRIER_20] = triple_barrier_label(
        close,
        horizon_bars=triple_barrier_horizon,
        take_profit_bps=tp_bps,
        stop_loss_bps=sl_bps,
    )

    # Backward-compatible columns used by existing strategy/backtest code.
    out["fwd_ret"] = out["fwd_ret_12"]
    out["label"] = out[TARGET_12_POSITIVE]

    for h in risk_horizons:
        out[f"future_mae_{h}"] = future_mae(close, h)
        out[f"future_mfe_{h}"] = future_mfe(close, h)

    out["best_hold_bucket"] = best_hold_bucket(close)
    return out

