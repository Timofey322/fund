"""Monthly rebalance backtest, vol targeting (Moreira & Muir), per-ticker stats."""



from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))




import math



import numpy as np

import pandas as pd



import config as _cfg
from config import (
    BARS_PER_YEAR,
    BENCHMARK,
    COMMISSION_BPS_PER_SIDE,
    FORECAST_HORIZON_BARS,
    IS_INTRADAY,
    MAX_WEIGHT,
    PERIODS_PER_YEAR,
    REBALANCE_FREQ,
    REGIME_TICKER,
    REGIME_TICKERS,
    SCORE_BUY,
    SCORE_ENTER,
    SCORE_EXIT,
    SCORE_HOLD,
    SMA_LONG,
    TAIL_VOL_RATIO_FREEZE,
    TARGET_VOL_ANN,
)

from operations.scoring import (

    ann_vol,

    composite_score,

    composite_score_frame,

    log_returns,

    price_trend_12_1,

    regime_risk_on,

    vol_regime_ratio,

    z_score_vs_sma,

)





def _blended_regime_series(prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """risk_on + vol_ratio from REGIME_TICKERS (QQQ + SPY): AND for risk, max for vol."""
    tickers = [t for t in REGIME_TICKERS if t in prices.columns]
    if not tickers:
        tickers = [REGIME_TICKER] if REGIME_TICKER in prices.columns else [prices.columns[0]]

    risk_frames: list[pd.Series] = []
    vol_frames: list[pd.Series] = []
    for t in tickers:
        close = prices[t].dropna()
        risk_frames.append(regime_risk_on(close, SMA_LONG))
        vol_frames.append(vol_regime_ratio(ann_vol(log_returns(close))))

    risk_on = risk_frames[0]
    for ro in risk_frames[1:]:
        aligned = ro.reindex(risk_on.index).fillna(0)
        risk_on = risk_on.astype(bool) & aligned.astype(bool)

    vol_ratio = vol_frames[0]
    for vr in vol_frames[1:]:
        vol_ratio = pd.concat([vol_ratio, vr], axis=1).max(axis=1)

    return risk_on, vol_ratio


# Daily-factor windows used by the intraday builder (trading-day units)
_D_SMA_LONG = 200
_D_PT_LOOKBACK = 252
_D_PT_SKIP = 21
_D_VOL_WIN = 20
_D_VOL_MED = 1260


def _daily_closes_from_intraday(prices: pd.DataFrame) -> pd.DataFrame:
    """One close per session per ticker, indexed by each session's last-bar timestamp."""
    from common.timeframe import session_last_close

    cols = {}
    for t in prices.columns:
        s = session_last_close(prices[t].dropna())
        if not s.empty:
            cols[t] = s
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index().ffill()


def _intraday_regime_series(daily: pd.DataFrame, regime_ticker: str) -> tuple[pd.Series, pd.Series]:
    """risk_on (AND) + vol_ratio (max) over REGIME_TICKERS on the daily series."""
    tickers = [t for t in REGIME_TICKERS if t in daily.columns]
    if not tickers:
        tickers = [regime_ticker] if regime_ticker in daily.columns else [daily.columns[0]]
    risk_frames, vol_frames = [], []
    for t in tickers:
        c = daily[t].dropna()
        lr = np.log(c / c.shift(1))
        risk_frames.append(regime_risk_on(c, _D_SMA_LONG))
        vol = ann_vol(lr, _D_VOL_WIN)
        med = vol.rolling(_D_VOL_MED, min_periods=_D_VOL_WIN).median().replace(0, np.nan)
        vol_frames.append(vol / med)
    risk_on = risk_frames[0]
    for ro in risk_frames[1:]:
        risk_on = risk_on.astype(bool) & ro.reindex(risk_on.index).fillna(0).astype(bool)
    vol_ratio = vol_frames[0]
    for vr in vol_frames[1:]:
        vol_ratio = pd.concat([vol_ratio, vr], axis=1).max(axis=1)
    return risk_on, vol_ratio


def _build_signal_frame_intraday(prices: pd.DataFrame, regime_ticker: str) -> pd.DataFrame:
    """
    Intraday signals: a daily-factor strategy updates once per session, so we
    compute scores on per-session closes (fast) and date them at the session
    close. The per-bar backtest broadcasts each signal to subsequent bars and
    handles intraday entries/exits via the forecast horizon + signal exit.
    """
    daily = _daily_closes_from_intraday(prices)
    if daily.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "vol_ann", "score", "score_static"])

    risk_on, vol_ratio = _intraday_regime_series(daily, regime_ticker)
    parts = []
    for col in daily.columns:
        c = daily[col].dropna()
        if len(c) < _D_SMA_LONG + 30:
            continue
        lr = np.log(c / c.shift(1))
        df = pd.DataFrame(
            {
                "ticker": col,
                "close": c,
                "vol_ann": ann_vol(lr, _D_VOL_WIN),
                "price_trend_raw": price_trend_12_1(c, _D_PT_LOOKBACK, _D_PT_SKIP),
                "z_raw": z_score_vs_sma(c, _D_SMA_LONG),
                "risk_on": risk_on.reindex(c.index).ffill(),
                "vol_ratio": vol_ratio.reindex(c.index).ffill(),
            },
            index=c.index,
        ).dropna(subset=["close", "z_raw"])
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["date", "ticker", "close", "vol_ann", "score", "score_static"])

    merged = pd.concat(parts).reset_index(names="date")
    scores = composite_score_frame(
        merged["price_trend_raw"], merged["z_raw"], merged["risk_on"], merged["vol_ratio"]
    ).reset_index(drop=True)
    out = pd.concat([merged[["date", "ticker", "close", "vol_ann"]].reset_index(drop=True), scores], axis=1)
    out["score_static"] = out["score"]
    return out


def build_signal_frame(prices: pd.DataFrame, regime_ticker: str = REGIME_TICKER) -> pd.DataFrame:

    if _cfg.IS_INTRADAY:
        return _build_signal_frame_intraday(prices, regime_ticker)

    if [t for t in REGIME_TICKERS if t in prices.columns]:
        regime_on, regime_vol_ratio = _blended_regime_series(prices)
    elif regime_ticker in prices.columns:
        regime_close = prices[regime_ticker]
        regime_on = regime_risk_on(regime_close, SMA_LONG)
        regime_vol_ratio = vol_regime_ratio(ann_vol(log_returns(regime_close)))
    else:
        for alt in (REGIME_TICKER, "SPY", "AAPL", prices.columns[0]):
            if alt in prices.columns:
                regime_ticker = alt
                break
        regime_close = prices[regime_ticker]
        regime_on = regime_risk_on(regime_close, SMA_LONG)
        regime_vol_ratio = vol_regime_ratio(ann_vol(log_returns(regime_close)))



    parts = []

    for col in prices.columns:

        close = prices[col].dropna()

        if len(close) < SMA_LONG + 30:

            continue

        lr = log_returns(close)

        mom = price_trend_12_1(close)

        z = z_score_vs_sma(close, SMA_LONG)

        vol = ann_vol(lr)

        regime_aligned = regime_on.reindex(close.index).ffill()

        vol_ratio_aligned = regime_vol_ratio.reindex(close.index).ffill()



        df = pd.DataFrame(

            {

                "ticker": col,

                "close": close,

                "vol_ann": vol,

                "price_trend_raw": mom,

                "z_raw": z,

                "risk_on": regime_aligned,

                "vol_ratio": vol_ratio_aligned,

            },

            index=close.index,

        )

        df = df.dropna(subset=["close", "z_raw"])

        parts.append(df)



    merged = pd.concat(parts)

    merged = merged.reset_index(names="date")



    if _cfg.IS_INTRADAY:

        # Vectorized: row-wise apply is intractable over intraday panels

        scores = composite_score_frame(

            merged["price_trend_raw"],

            merged["z_raw"],

            merged["risk_on"],

            merged["vol_ratio"],

        ).reset_index(drop=True)

    else:

        scores = merged.apply(

            lambda r: pd.Series(

                composite_score(

                    float(r["price_trend_raw"]) if pd.notna(r["price_trend_raw"]) else float("nan"),

                    float(r["z_raw"]),

                    float(r["risk_on"]) if pd.notna(r["risk_on"]) else 0.0,

                    float(r["vol_ratio"]) if pd.notna(r["vol_ratio"]) else 1.0,

                )

            ),

            axis=1,

        )

    out = pd.concat([merged[["date", "ticker", "close", "vol_ann"]].reset_index(drop=True), scores], axis=1)

    out["score_static"] = out["score"]

    return out





def _eligible(
    score: float,
    risk_on: bool,
    hold_threshold: float | None = None,
    buy_threshold: float | None = None,
) -> bool:
    """Legacy: single threshold for monthly backtest without hysteresis."""
    return _entry_eligible(score, risk_on, hold_threshold, buy_threshold)


def _short_entry_eligible(
    score: float,
    risk_on: bool,
    sell_threshold: float | None = None,
    buy_threshold: float | None = None,
) -> bool:
    from strategy.fusion_direction import fusion_sell_threshold, normalize_buy_threshold

    buy = buy_threshold if buy_threshold is not None else SCORE_ENTER
    buy = normalize_buy_threshold(float(buy))
    sell = sell_threshold if sell_threshold is not None else fusion_sell_threshold(buy)
    if score <= 0.0:
        return False
    if not risk_on and score > sell - 5:
        return False
    return score <= sell and score < buy


def _hold_eligible_short(
    score: float,
    risk_on: bool,
    hold_threshold: float | None = None,
    buy_threshold: float | None = None,
) -> bool:
    """Keep short while score stays below the upper hysteresis band."""
    buy = buy_threshold if buy_threshold is not None else SCORE_BUY
    hold = hold_threshold if hold_threshold is not None else SCORE_HOLD
    upper = max(float(buy), 100.0 - float(hold))
    if not risk_on and score > buy:
        return False
    return score <= upper


def _entry_eligible(
    score: float,
    risk_on: bool,
    hold_threshold: float | None = None,
    buy_threshold: float | None = None,
) -> bool:
    enter = buy_threshold if buy_threshold is not None else SCORE_ENTER
    if not risk_on and score < enter + 5:
        return False
    return score >= enter


def _hold_eligible(
    score: float,
    risk_on: bool,
    hold_threshold: float | None = None,
    buy_threshold: float | None = None,
) -> bool:
    """Lower bar to keep a position (hysteresis)."""
    exit_th = min(SCORE_EXIT, hold_threshold if hold_threshold is not None else SCORE_HOLD)
    buy = buy_threshold if buy_threshold is not None else SCORE_BUY
    if not risk_on and score < buy:
        return False
    return score >= exit_th





def _turnover_cost(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    commission_bps: float,
    slippage_bps: float = 0.0,
) -> float:
    """Стоимость ребаланса: bps * sum(|delta_w|)."""
    cost_bps = float(commission_bps) + float(slippage_bps)
    if cost_bps <= 0:
        return 0.0
    tickers = set(old_weights) | set(new_weights)
    turnover = sum(abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0)) for t in tickers)
    return (cost_bps / 10_000.0) * turnover


def _apply_rebalance_band(
    held_weights: dict[str, float],
    target_weights: dict[str, float],
    band: float,
) -> dict[str, float]:
    """No-trade band: keep the prior weight for held positions whose target drifts
    less than ``band``. New entries and full exits always go through; only small
    vol-target re-weights on continuing positions are suppressed (kills daily churn).
    """
    if band <= 0.0:
        return target_weights
    out: dict[str, float] = {}
    for t, target_w in target_weights.items():
        prev_w = held_weights.get(t, 0.0)
        if prev_w > 0.0 and abs(target_w - prev_w) < band:
            out[t] = prev_w
        else:
            out[t] = target_w
    return out





def _vol_target_weights(

    picks: list,

    score_col: str,

    exposure_cap: float = 1.0,

    bench_vol_ratio: float = 1.0,

) -> dict[str, float]:

    """

    Moreira & Muir (2017): inverse-vol sizing + regime exposure cap.

    Portfolio scaler reduces gross exposure when benchmark vol is elevated.

    """

    if not picks:
        return {}

    if bench_vol_ratio > TAIL_VOL_RATIO_FREEZE:
        return {}



    # vol-managed leverage: scale down when market vol > median

    vol_scaler = min(1.0, 1.0 / max(bench_vol_ratio, 0.75))

    gross_cap = exposure_cap * vol_scaler



    raw: dict[str, float] = {}

    for row in picks:

        vol = row.get("vol_ann", 0.25)

        if vol is None or (isinstance(vol, float) and math.isnan(vol)) or vol < 0.05:

            vol = 0.25

        sc = float(row.get(score_col, row.get("score", 50)))

        boost = 0.75 + 0.5 * (sc / 100)

        side = float(row.get("position_side", 1) or 1)
        sign = -1.0 if side < 0 else 1.0
        raw[row["ticker"]] = sign * (TARGET_VOL_ANN / vol) * boost



    total = sum(abs(v) for v in raw.values())

    if total <= 0:

        return {}



    weights = {t: min(MAX_WEIGHT, abs(v) / total * gross_cap) * (1.0 if v >= 0 else -1.0) for t, v in raw.items()}

    s = sum(abs(w) for w in weights.values())

    if s > gross_cap and s > 0:

        weights = {t: w * gross_cap / s for t, w in weights.items()}

    return weights





def _bar_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> frozenset[pd.Timestamp]:
    """Last actual bar timestamp in each rebalance period (not resample label)."""
    if len(index) == 0:
        return frozenset()
    pos = pd.Series(np.arange(len(index)), index=index)
    last_pos = pos.resample(freq).max().dropna().astype(int)
    return frozenset(index[i] for i in last_pos)


def _rebalance_picks(
    sub: pd.DataFrame,
    bench_col: str,
    score_col: str,
    use_dynamic_thresholds: bool,
    *,
    allow_short: bool = False,
) -> list:
    picks = []
    for _, row in sub.iterrows():
        sc = row.get(score_col, row.get("score"))
        hold_th = row.get("hold_threshold") if use_dynamic_thresholds else None
        buy_th = row.get("buy_threshold") if use_dynamic_thresholds else None
        sell_th = row.get("sell_threshold") if use_dynamic_thresholds else None
        risk_on = bool(row.get("risk_on", True))
        score_f = float(sc)
        if _entry_eligible(score_f, risk_on, hold_th, buy_th):
            picks.append(row)
        elif allow_short and _short_entry_eligible(score_f, risk_on, sell_th, buy_th):
            row = row.copy()
            row["position_side"] = -1
            picks.append(row)
    return picks


def _snapshot_should_replace(prev: pd.Series | None, new: pd.Series) -> bool:
    """Keep the strongest recent score; late zero bars must not erase intraday entries."""
    if prev is None:
        return True
    new_sc = float(new.get("score", 0) or 0)
    old_sc = float(prev.get("score", 0) or 0)
    if new_sc <= 0.0 and old_sc > 0.0:
        return False
    return True


def _precompute_snapshots(
    sig: pd.DataFrame, bench_col: str
) -> tuple[np.ndarray, list[tuple[pd.DataFrame, float, float]]]:
    """
    For each unique signal date, the latest row per ticker as of that date
    (carried forward), plus the benchmark's exposure_cap / vol_ratio.

    Returns (sorted date array, snapshots) so the backtest can map each bar to
    its snapshot via searchsorted instead of an O(n) groupby per bar.
    """
    s = sig.sort_values("date")
    uniq = s["date"].drop_duplicates().to_numpy()
    snaps: list[tuple[pd.DataFrame, float, float]] = []
    cur: dict[str, pd.Series] = {}
    for d in uniq:
        day_rows = s[s["date"].to_numpy() == d]
        for _, r in day_rows.iterrows():
            ticker = r["ticker"]
            if _snapshot_should_replace(cur.get(ticker), r):
                cur[ticker] = r
        sub_df = pd.DataFrame(list(cur.values()))
        ec, bvr = 1.0, 1.0
        if bench_col in cur:
            ec = float(cur[bench_col].get("exposure_cap", 1.0) or 1.0)
            bvr = float(cur[bench_col].get("vol_ratio", 1.0) or 1.0)
        snaps.append((sub_df, ec, bvr))
    return uniq, snaps


def _signal_snapshot(sig: pd.DataFrame, dt: pd.Timestamp, bench_col: str) -> tuple[pd.DataFrame, float, float]:
    sub = sig[sig["date"] <= dt].sort_values("date").groupby("ticker", as_index=False).tail(1)
    regime_row = sub[sub["ticker"] == bench_col]
    exposure_cap = 1.0
    bench_vr = 1.0
    if not regime_row.empty:
        exposure_cap = float(regime_row.iloc[0].get("exposure_cap", 1.0) or 1.0)
        bench_vr = float(regime_row.iloc[0].get("vol_ratio", 1.0) or 1.0)
    return sub, exposure_cap, bench_vr


def _resolve_exit_horizon(sub: pd.DataFrame, ticker: str, default_horizon: int) -> int:
    """Per-entry holding horizon: regime-adaptive ``exit_horizon`` if present."""
    if "exit_horizon" not in getattr(sub, "columns", []):
        return int(default_horizon)
    row_df = sub[sub["ticker"] == ticker]
    if row_df.empty:
        return int(default_horizon)
    val = row_df.iloc[0].get("exit_horizon")
    try:
        h = int(val)
    except (TypeError, ValueError):
        return int(default_horizon)
    return h if h > 0 else int(default_horizon)


def _ticker_stop_loss_bps(
    ticker: str,
    sub: pd.DataFrame,
    default_bps: float,
) -> float:
    """Per-ticker stop from the latest signal snapshot, else the policy default."""
    if sub is None or sub.empty or "stop_loss_bps" not in sub.columns:
        return float(default_bps)
    row_df = sub[sub["ticker"] == ticker]
    if row_df.empty:
        return float(default_bps)
    try:
        val = float(row_df.iloc[0]["stop_loss_bps"])
    except (TypeError, ValueError):
        return float(default_bps)
    return val if val > 0 else float(default_bps)


def _ticker_take_profit_bps(
    ticker: str,
    sub: pd.DataFrame,
    default_bps: float,
) -> float:
    """Per-ticker take-profit from the latest signal snapshot, else policy default."""
    if sub is None or sub.empty or "take_profit_bps" not in sub.columns:
        return float(default_bps)
    row_df = sub[sub["ticker"] == ticker]
    if row_df.empty:
        return float(default_bps)
    try:
        val = float(row_df.iloc[0]["take_profit_bps"])
    except (TypeError, ValueError):
        return float(default_bps)
    return val if val > 0 else float(default_bps)


def _apply_take_profit_exits(
    weights: dict[str, float],
    entry_price: dict[str, float],
    bar_i: int,
    prices_arr: np.ndarray,
    col_idx: dict[str, int],
    sub: pd.DataFrame,
    take_profit_bps: float,
    baseline_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], int]:
    """Exit when favorable move from entry price exceeds ``take_profit_bps`` (gross)."""
    default_tp = float(take_profit_bps)
    if default_tp <= 0 and (sub is None or "take_profit_bps" not in getattr(sub, "columns", [])):
        return dict(weights), 0
    out = dict(weights)
    exits = 0
    if bar_i < 0 or bar_i >= len(prices_arr):
        return out, 0
    row_px = prices_arr[bar_i]
    for t in list(out):
        ep = entry_price.get(t)
        if ep is None or ep <= 0:
            continue
        ci = col_idx.get(t)
        if ci is None:
            continue
        px = row_px[ci]
        if not np.isfinite(px):
            continue
        tp_bps = _ticker_take_profit_bps(t, sub, default_tp)
        if tp_bps <= 0:
            continue
        w = float(out.get(t, 0.0))
        move = float(px) / float(ep) - 1.0
        hit = (move >= tp_bps / 10_000.0) if w >= 0 else (move <= -tp_bps / 10_000.0)
        if hit:
            floor_w = float(baseline_weights.get(t, 0.0)) if baseline_weights else 0.0
            if floor_w > _WEIGHT_EPS:
                out[t] = floor_w
            else:
                del out[t]
            exits += 1
    return out, exits


def _apply_stop_loss_exits(
    weights: dict[str, float],
    entry_price: dict[str, float],
    bar_i: int,
    prices_arr: np.ndarray,
    col_idx: dict[str, int],
    sub: pd.DataFrame,
    stop_loss_bps: float,
    baseline_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], int]:
    """Exit when adverse move from entry price exceeds ``stop_loss_bps`` (gross)."""
    default_sl = float(stop_loss_bps)
    if default_sl <= 0 and (sub is None or "stop_loss_bps" not in getattr(sub, "columns", [])):
        return dict(weights), 0
    out = dict(weights)
    exits = 0
    if bar_i < 0 or bar_i >= len(prices_arr):
        return out, 0
    row_px = prices_arr[bar_i]
    for t in list(out):
        ep = entry_price.get(t)
        if ep is None or ep <= 0:
            continue
        ci = col_idx.get(t)
        if ci is None:
            continue
        px = row_px[ci]
        if not np.isfinite(px):
            continue
        sl_bps = _ticker_stop_loss_bps(t, sub, default_sl)
        if sl_bps <= 0:
            continue
        w = float(out.get(t, 0.0))
        move = float(px) / float(ep) - 1.0
        hit = (move <= -sl_bps / 10_000.0) if w >= 0 else (move >= sl_bps / 10_000.0)
        if hit:
            floor_w = float(baseline_weights.get(t, 0.0)) if baseline_weights else 0.0
            if floor_w > _WEIGHT_EPS:
                out[t] = floor_w
            else:
                del out[t]
            exits += 1
    return out, exits


_WEIGHT_EPS = 1e-8


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= _WEIGHT_EPS:
        return {}
    return {t: max(0.0, float(w)) / total for t, w in weights.items()}


def _normalize_signed_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(abs(float(v)) for v in weights.values())
    if total <= _WEIGHT_EPS:
        return {}
    return {t: float(w) / total for t, w in weights.items()}


def _baseline_with_pick_tilt(
    baseline: dict[str, float],
    picks: list,
    tactical_budget: float,
) -> dict[str, float]:
    """Keep a core EW book; shift ``tactical_budget`` toward active pick names."""
    budget = min(max(float(tactical_budget), 0.0), 1.0)
    out = dict(baseline)
    if not picks or budget <= 0.0:
        return _normalize_weights(out)
    pick_set = {str(row["ticker"]) for row in picks}
    boost = budget / len(picks)
    for t in pick_set:
        out[t] = out.get(t, 0.0) + boost
    others = [t for t in out if t not in pick_set]
    osum = sum(out[t] for t in others)
    cut = boost * len(picks)
    if osum > _WEIGHT_EPS:
        for t in others:
            out[t] = max(0.0, out[t] - cut * (out[t] / osum))
    return _normalize_weights(out)


def _apply_signal_exits(
    weights: dict[str, float],
    entry_idx: dict[str, int],
    bar_i: int,
    sub: pd.DataFrame,
    score_col: str,
    use_dynamic_thresholds: bool,
    horizon_bars: int,
    bench_col: str,
    entry_horizon: dict[str, int] | None = None,
    min_hold_bars: int = 0,
    partial_sell_frac: float | None = None,
    baseline_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Medallion-style exit: signal dead or forecast horizon expired (no hedge).

    The holding horizon is the one captured *at entry* (``entry_horizon``), so a
    regime-adaptive horizon stays fixed for the life of the trade and never peeks
    at later bars. Falls back to the scalar ``horizon_bars`` when unknown.

    ``min_hold_bars``: suppress score-dead exits until the position has been held
    at least this long (horizon exits are unaffected).

    ``partial_sell_frac``: when set (e.g. 0.2), score-dead exits trim the position
  by this fraction instead of closing fully. Horizon exits remain full closes.
    """
    out = dict(weights)
    min_hold = max(0, int(min_hold_bars))
    trim = float(partial_sell_frac) if partial_sell_frac is not None else 0.0
    trim = min(max(trim, 0.0), 1.0)
    if sub is None or sub.empty or "ticker" not in sub.columns:
        for t in list(out):
            held_bars = bar_i - entry_idx.get(t, bar_i)
            horizon = int((entry_horizon or {}).get(t, horizon_bars))
            if held_bars >= horizon:
                if baseline_weights and t in baseline_weights:
                    out[t] = float(baseline_weights[t])
                else:
                    del out[t]
        return out
    for t in list(out):
        row_df = sub[sub["ticker"] == t]
        if row_df.empty:
            del out[t]
            continue
        row = row_df.iloc[0]
        sc = row.get(score_col, row.get("score"))
        hold_th = row.get("hold_threshold") if use_dynamic_thresholds else None
        buy_th = row.get("buy_threshold") if use_dynamic_thresholds else None
        sell_th = row.get("sell_threshold") if use_dynamic_thresholds else None
        w = float(out.get(t, 0.0))
        if w < 0:
            eligible = _hold_eligible_short(float(sc), bool(row.get("risk_on", True)), hold_th, buy_th)
        else:
            eligible = _hold_eligible(float(sc), bool(row.get("risk_on", True)), hold_th, buy_th)
        held_bars = bar_i - entry_idx.get(t, bar_i)
        horizon = int((entry_horizon or {}).get(t, horizon_bars))
        if held_bars >= horizon:
            if baseline_weights and t in baseline_weights:
                out[t] = float(baseline_weights[t])
            else:
                del out[t]
        elif not eligible and held_bars >= min_hold:
            floor_w = float(baseline_weights.get(t, 0.0)) if baseline_weights else 0.0
            if trim > 0.0:
                nw = max(out[t] * (1.0 - trim), floor_w)
                if nw <= _WEIGHT_EPS:
                    del out[t]
                else:
                    out[t] = nw
            elif baseline_weights and t in baseline_weights:
                out[t] = float(baseline_weights[t])
            else:
                del out[t]
    return out


def _deploy_cash_on_buy_picks(
    held_weights: dict[str, float],
    picks: list,
) -> dict[str, float]:
    """Invest free cash (USD) into buy-signal names — restores correlation after partial sells.

    Model: cash sits idle until a buy signal; each pick receives an equal share of
    available cash on top of whatever is already held (weights sum to <= 1).
    """
    out = {t: max(0.0, float(w)) for t, w in held_weights.items() if float(w) > _WEIGHT_EPS}
    cash = max(0.0, 1.0 - sum(out.values()))
    if cash <= _WEIGHT_EPS or not picks:
        return out
    pick_tickers = [str(row["ticker"]) for row in picks]
    per = cash / len(pick_tickers)
    for t in pick_tickers:
        out[t] = out.get(t, 0.0) + per
    total = sum(out.values())
    if total > 1.0 + _WEIGHT_EPS:
        out = {t: w / total for t, w in out.items()}
    return out


def _merge_partial_sell_rebalance(
    held_weights: dict[str, float],
    pick_weights: dict[str, float],
) -> dict[str, float]:
    """Keep trimmed holdings; full-size picks on buy signals."""
    merged = dict(held_weights)
    for t, tw in pick_weights.items():
        merged[t] = tw
    return merged


def _count_signal_exits(
    before: dict[str, float],
    after: dict[str, float],
    *,
    partial_sell_frac: float | None = None,
) -> int:
    """Count full closes and partial trims triggered by signal exits."""
    trim = float(partial_sell_frac) if partial_sell_frac is not None else 0.0
    if trim <= 0.0:
        return sum(1 for t in before if t not in after)
    n = 0
    for t in before:
        if t not in after:
            n += 1
        elif after[t] < before[t] - _WEIGHT_EPS:
            n += 1
    return n


def _update_entry_idx(
    old_w: dict[str, float],
    new_w: dict[str, float],
    entry_idx: dict[str, int],
    bar_i: int,
) -> dict[str, int]:
    updated = dict(entry_idx)
    for t in new_w:
        if new_w[t] > 0 and old_w.get(t, 0.0) <= 0:
            updated[t] = bar_i
    for t in list(updated):
        if t not in new_w or new_w.get(t, 0.0) <= 0:
            del updated[t]
    return updated


def _bars_per_year() -> float:
    return float(BARS_PER_YEAR)


def _calendar_performance_stats(
    strategy: pd.Series,
    benchmark: pd.Series,
    exposure_frac: float,
    signal_exits: int,
    horizon_bars: int = FORECAST_HORIZON_BARS,
) -> dict:
    """Return metrics normalized by calendar time (not rebalance periods)."""
    if strategy.empty or len(strategy) < 2:
        return {}
    cal_days = max((strategy.index[-1] - strategy.index[0]).days, 1)
    years_cal = cal_days / 365.25
    total_s = strategy.iloc[-1] / strategy.iloc[0] - 1
    total_b = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    cagr = (1 + total_s) ** (1 / years_cal) - 1 if years_cal > 0 else 0.0
    cagr_b = (1 + total_b) ** (1 / years_cal) - 1 if years_cal > 0 else 0.0
    bps_per_cal_day = total_s / cal_days * 10_000

    sr = strategy.pct_change().dropna()
    periods_per_year = 365.25
    vol_ann = float(sr.std()) * math.sqrt(periods_per_year) if len(sr) > 1 else 0.0
    sharpe_cal = (float(sr.mean()) * periods_per_year - 0.04) / vol_ann if vol_ann > 1e-12 else 0.0

    return {
        "calendar_days": cal_days,
        "cagr_pct": round(cagr * 100, 2),
        "benchmark_cagr_pct": round(cagr_b * 100, 2),
        "bps_per_calendar_day": round(bps_per_cal_day, 3),
        "sharpe_bar_annualized": round(sharpe_cal, 2),
        "avg_exposure_pct": round(exposure_frac * 100, 1),
        "signal_exit_count": signal_exits,
        "forecast_horizon_bars": int(horizon_bars),
    }


def run_backtest_signal_exit(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    score_col: str = "score",
    use_dynamic_thresholds: bool = False,
    use_vol_targeting: bool = True,
    period_start: str | pd.Timestamp | None = None,
    period_end: str | pd.Timestamp | None = None,
    commission_bps: float = COMMISSION_BPS_PER_SIDE,
    horizon_bars: int | None = None,
    slippage_bps: float = 0.0,
    stop_loss_bps: float | None = None,
    take_profit_bps: float | None = None,
    *,
    min_hold_bars: int | None = None,
    rebalance_band: float | None = None,
    hold_entry_weight: bool | None = None,
    reentry_cooldown_bars: int | None = None,
    equal_weight: bool = False,
    partial_sell_frac: float | None = None,
    baseline_weights: dict[str, float] | None = None,
    tactical_budget: float = 0.0,
    keep_holdings_on_empty_picks: bool = False,
    merge_new_picks_only: bool = False,
    buy_only: bool = False,
    allow_short: bool | None = None,
) -> dict:
    """
    Bar-by-bar backtest with Medallion-style exits:
    - close when adverse move from entry >= ``stop_loss_bps`` (price stop)
    - close when favorable move from entry >= ``take_profit_bps`` (price target)
    - close when score < threshold (signal dead; optional partial trim)
    - close when bars_held >= forecast horizon
    Full rebalance still on REBALANCE_FREQ dates.

    ``buy_only``: never reduce or exit holdings — only add on buy picks (dip buying).
    """
    horizon_bars = horizon_bars or FORECAST_HORIZON_BARS
    sl_bps = float(
        stop_loss_bps
        if stop_loss_bps is not None
        else getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0)
    )
    tp_bps = float(
        take_profit_bps
        if take_profit_bps is not None
        else getattr(_cfg, "FUSION_TAKE_PROFIT_BPS", 0.0)
    )
    bench_col = BENCHMARK if BENCHMARK in prices.columns else prices.columns[0]
    bar_rets = prices.pct_change(fill_method=None)
    rebalance_dates = _bar_rebalance_dates(prices.index, _cfg.REBALANCE_FREQ)
    rebalance_band = float(
        rebalance_band
        if rebalance_band is not None
        else getattr(_cfg, "BACKTEST_REBALANCE_BAND", 0.0)
    )
    hold_entry_weight = bool(
        hold_entry_weight
        if hold_entry_weight is not None
        else getattr(_cfg, "BACKTEST_HOLD_ENTRY_WEIGHT", False)
    )
    reweight_freq = getattr(_cfg, "BACKTEST_REWEIGHT_FREQ", _cfg.REBALANCE_FREQ)
    reweight_dates = (
        _bar_rebalance_dates(prices.index, reweight_freq) if hold_entry_weight else rebalance_dates
    )
    min_hold_bars = int(
        min_hold_bars
        if min_hold_bars is not None
        else getattr(_cfg, "BACKTEST_MIN_HOLD_BARS", 0)
    )
    reentry_cooldown = int(
        reentry_cooldown_bars
        if reentry_cooldown_bars is not None
        else getattr(_cfg, "BACKTEST_REENTRY_COOLDOWN_BARS", 0)
    )
    partial_trim = (
        float(partial_sell_frac)
        if partial_sell_frac is not None
        else 0.0
    )
    partial_trim = min(max(partial_trim, 0.0), 1.0)
    use_partial_sell = partial_trim > 0.0
    core_weights = (
        _normalize_weights({str(k): float(v) for k, v in baseline_weights.items()})
        if baseline_weights
        else None
    )
    tac_budget = min(max(float(tactical_budget), 0.0), 1.0)
    allow_short = bool(
        allow_short
        if allow_short is not None
        else getattr(_cfg, "FUSION_ALLOW_SHORT", False)
    )

    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])

    bars = prices.index[1:]
    if period_start is not None:
        bars = bars[bars >= pd.Timestamp(period_start)]
    if period_end is not None:
        bars = bars[bars <= pd.Timestamp(period_end)]
    if len(bars) < 2:
        return {"equity": pd.DataFrame(), "benchmark": pd.DataFrame(), "stats": {}, "holdings": []}

    portfolio_value = 1.0
    bench_value = 1.0
    prev_weights: dict[str, float] = dict(core_weights) if core_weights else {}
    entry_idx: dict[str, int] = {t: 0 for t in prev_weights}
    entry_price: dict[str, float] = {}
    entry_horizon: dict[str, int] = {}
    last_exit_bar: dict[str, int] = {}
    total_commission = 0.0
    signal_exits = 0
    stop_loss_exits = 0
    take_profit_exits = 0
    exposure_sum = 0.0
    buy_cost_num: dict[str, float] = {}
    buy_w_sum: dict[str, float] = {}
    equity_curve = []
    bench_curve = []
    holdings_log = []
    bar_index = {ts: i for i, ts in enumerate(prices.index)}

    # --- Fast paths (critical for intraday: up to 10^5 bars) ---
    # Returns as a numpy matrix indexed by bar position + column index
    rets_arr = bar_rets.to_numpy()
    prices_arr = prices.to_numpy(dtype=float)
    col_idx = {c: i for i, c in enumerate(bar_rets.columns)}
    bench_ci = col_idx.get(bench_col)
    # Per-session signal snapshots: latest row per ticker, advanced by a pointer
    snap_dates, snapshots = _precompute_snapshots(sig, bench_col)

    cur_k = -2
    sub: pd.DataFrame = pd.DataFrame()
    exposure_cap = 1.0
    bench_vr = 1.0

    for dt in bars:
        bi = bar_index[dt]
        dt64 = np.datetime64(dt)
        row = rets_arr[bi]
        port_ret = 0.0
        for t, w in prev_weights.items():
            ci = col_idx.get(t)
            if ci is not None:
                r = row[ci]
                if r == r:  # not NaN
                    port_ret += w * r
        portfolio_value *= 1 + port_ret
        if bench_ci is not None:
            br = row[bench_ci]
            if br == br:
                bench_value *= 1 + br

        # advance to the snapshot applicable as of this bar (causal)
        k = int(np.searchsorted(snap_dates, dt64, side="right")) - 1 if len(snap_dates) else -1
        if k != cur_k:
            cur_k = k
            if k >= 0:
                sub, exposure_cap, bench_vr = snapshots[k]
            else:
                sub, exposure_cap, bench_vr = pd.DataFrame(), 1.0, 1.0

        after_tp, tp_n = _apply_take_profit_exits(
            prev_weights, entry_price, bi, prices_arr, col_idx, sub, tp_bps,
            baseline_weights=core_weights,
        )
        take_profit_exits += tp_n
        after_sl, sl_n = _apply_stop_loss_exits(
            after_tp, entry_price, bi, prices_arr, col_idx, sub, sl_bps,
            baseline_weights=core_weights,
        )
        stop_loss_exits += sl_n
        if buy_only:
            exited_weights = dict(after_sl)
        else:
            exited_weights = _apply_signal_exits(
                after_sl, entry_idx, bi, sub, score_col,
                use_dynamic_thresholds, horizon_bars, bench_col,
                entry_horizon,
                min_hold_bars=min_hold_bars,
                partial_sell_frac=partial_trim if use_partial_sell else None,
                baseline_weights=core_weights,
            )
        if not buy_only:
            for t in prev_weights:
                if prev_weights.get(t, 0.0) > 0 and t not in exited_weights:
                    last_exit_bar[t] = bi
            signal_exits += _count_signal_exits(
                after_sl, exited_weights,
                partial_sell_frac=partial_trim if use_partial_sell else None,
            )
        new_weights = exited_weights
        cost = _turnover_cost(prev_weights, exited_weights, commission_bps, slippage_bps=slippage_bps)

        if dt in rebalance_dates:
            picks = _rebalance_picks(
                sub, bench_col, score_col, use_dynamic_thresholds, allow_short=allow_short,
            )
            if reentry_cooldown > 0:
                picks = [
                    row for row in picks
                    if bi - last_exit_bar.get(row["ticker"], -10**9) >= reentry_cooldown
                ]
            if use_vol_targeting:
                target_weights = _vol_target_weights(picks, score_col, exposure_cap, bench_vr)
            elif core_weights is not None:
                target_weights = _baseline_with_pick_tilt(core_weights, picks, tac_budget)
            elif picks:
                if merge_new_picks_only and equal_weight:
                    target_weights = _deploy_cash_on_buy_picks(exited_weights, picks)
                elif equal_weight:
                    w = 1.0 / len(picks)
                    target_weights = {
                        row["ticker"]: w * (1.0 if float(row.get("position_side", 1) or 1) >= 0 else -1.0)
                        for row in picks
                    }
                else:
                    tmp = {}
                    for row in picks:
                        sc = row.get(score_col, row.get("score"))
                        vol = row.get("vol_ann", 0.25)
                        side = float(row.get("position_side", 1) or 1)
                        sign = -1.0 if side < 0 else 1.0
                        tmp[row["ticker"]] = sign * _legacy_weight(float(sc), vol, len(picks))
                    wsum = sum(abs(v) for v in tmp.values())
                    if wsum > 1.0:
                        tmp = {k: v / wsum for k, v in tmp.items()}
                    target_weights = tmp
            elif keep_holdings_on_empty_picks:
                target_weights = dict(exited_weights)
            else:
                target_weights = dict(core_weights) if core_weights is not None else {}
            if use_partial_sell and target_weights:
                target_weights = _merge_partial_sell_rebalance(exited_weights, target_weights)
            if hold_entry_weight and dt not in reweight_dates:
                # Keep existing holdings at their current (entry) weight; only
                # size NEW names from the vol-target. Eliminates daily re-weight
                # churn on an economically unchanged book.
                merged = dict(exited_weights)
                for t, tw in target_weights.items():
                    if exited_weights.get(t, 0.0) <= 0.0:
                        merged[t] = tw
                target_weights = merged
            new_weights = _apply_rebalance_band(exited_weights, target_weights, rebalance_band)
            cost += _turnover_cost(exited_weights, new_weights, commission_bps, slippage_bps=slippage_bps)
            if buy_only:
                for t, nw in new_weights.items():
                    dw = nw - exited_weights.get(t, 0.0)
                    if dw <= _WEIGHT_EPS:
                        continue
                    ci = col_idx.get(t)
                    if ci is None:
                        continue
                    px = prices_arr[bi, ci]
                    if np.isfinite(px):
                        buy_cost_num[t] = buy_cost_num.get(t, 0.0) + dw * float(px)
                        buy_w_sum[t] = buy_w_sum.get(t, 0.0) + dw

        if cost > 0:
            portfolio_value *= 1.0 - cost
            total_commission += cost
        for t in new_weights:
            nw = new_weights.get(t, 0.0)
            pw = exited_weights.get(t, 0.0)
            if abs(nw) > _WEIGHT_EPS and abs(pw) <= _WEIGHT_EPS:
                entry_horizon[t] = _resolve_exit_horizon(sub, t, horizon_bars)
                ci = col_idx.get(t)
                if ci is not None:
                    px = prices_arr[bi, ci]
                    if np.isfinite(px):
                        entry_price[t] = float(px)
        for t in list(entry_horizon):
            if abs(new_weights.get(t, 0.0)) <= _WEIGHT_EPS:
                del entry_horizon[t]
        for t in list(entry_price):
            if abs(new_weights.get(t, 0.0)) <= _WEIGHT_EPS:
                del entry_price[t]
        entry_idx = _update_entry_idx(exited_weights, new_weights, entry_idx, bi)
        prev_weights = new_weights
        exposure_sum += sum(abs(w) for w in prev_weights.values())

        if dt in rebalance_dates:
            holdings_log.append({
                "date": dt,
                "weights": dict(prev_weights),
                "n": len(prev_weights),
                "commission_pct": round(cost * 100, 4),
            })
        # Mark-to-market equity every bar (daily path must not look flat between rebalances).
        equity_curve.append({"date": dt, "value": portfolio_value})
        bench_curve.append({"date": dt, "value": bench_value})

    eq = pd.DataFrame(equity_curve).set_index("date") if equity_curve else pd.DataFrame()
    bq = pd.DataFrame(bench_curve).set_index("date") if bench_curve else pd.DataFrame()
    exposure_frac = exposure_sum / max(len(bars), 1)

    stats = _performance_stats(eq["value"], bq["value"]) if not eq.empty else {}
    if equity_curve:
        eq_daily = pd.DataFrame(equity_curve).set_index("date")["value"].resample("D").last().ffill()
        bq_daily = pd.DataFrame(bench_curve).set_index("date")["value"].resample("D").last().ffill()
        aligned = pd.concat([eq_daily, bq_daily], axis=1, join="inner").dropna()
        if not aligned.empty:
            stats.update(_calendar_performance_stats(
                aligned.iloc[:, 0], aligned.iloc[:, 1], exposure_frac, signal_exits, horizon_bars,
            ))

    stats["avg_exposure_pct"] = round(exposure_frac * 100, 4)
    stats["use_signal_exit"] = True
    stats["use_vol_targeting"] = use_vol_targeting
    stats["stop_loss_bps"] = sl_bps
    stats["take_profit_bps"] = tp_bps
    stats["stop_loss_exit_count"] = int(stop_loss_exits)
    stats["take_profit_exit_count"] = int(take_profit_exits)
    stats["commission_bps"] = commission_bps
    stats["slippage_bps"] = slippage_bps
    stats["total_commission_pct"] = round(total_commission * 100, 3)
    if use_partial_sell:
        stats["partial_sell_frac"] = partial_trim
    if buy_only:
        stats["buy_only"] = True
        discounts: list[float] = []
        for t, ws in buy_w_sum.items():
            if ws <= _WEIGHT_EPS:
                continue
            avg_buy = buy_cost_num.get(t, 0.0) / ws
            ci = col_idx.get(t)
            if ci is None:
                continue
            px_row = prices_arr[:, ci]
            valid = px_row[np.isfinite(px_row)]
            if len(valid) == 0 or avg_buy <= 0:
                continue
            twap = float(np.mean(valid))
            if twap > 0:
                discounts.append((1.0 - avg_buy / twap) * 100.0)
        if discounts:
            stats["avg_entry_discount_vs_twap_pct"] = round(float(np.mean(discounts)), 2)
    if equity_curve:
        stats["period_start"] = str(equity_curve[0]["date"].date())
        stats["period_end"] = str(equity_curve[-1]["date"].date())

    return {
        "equity": eq,
        "benchmark": bq,
        "stats": stats,
        "holdings": holdings_log,
        "period_start": equity_curve[0]["date"] if equity_curve else None,
        "period_end": equity_curve[-1]["date"] if equity_curve else None,
    }



def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    score_col: str = "score",
    use_dynamic_thresholds: bool = False,
    use_vol_targeting: bool = True,
    use_signal_exit: bool = False,
    period_start: str | pd.Timestamp | None = None,
    period_end: str | pd.Timestamp | None = None,
    commission_bps: float = COMMISSION_BPS_PER_SIDE,
) -> dict:
    if use_signal_exit:
        return run_backtest_signal_exit(
            prices, signals, score_col, use_dynamic_thresholds, use_vol_targeting,
            period_start, period_end, commission_bps,
        )

    bench_col = BENCHMARK if BENCHMARK in prices.columns else prices.columns[0]

    monthly_px = prices.resample(_cfg.REBALANCE_FREQ).last()
    monthly_rets = monthly_px.pct_change(fill_method=None)



    sig = signals.copy()

    sig["date"] = pd.to_datetime(sig["date"])



    portfolio_value = 1.0

    bench_value = 1.0

    equity_curve = []

    bench_curve = []

    holdings_log = []

    prev_weights: dict[str, float] = {}

    total_commission = 0.0



    rebalance_dates = list(monthly_rets.index[1:])

    if period_start is not None:

        ps = pd.Timestamp(period_start)

        rebalance_dates = [d for d in rebalance_dates if d >= ps]

    if period_end is not None:

        pe = pd.Timestamp(period_end)

        rebalance_dates = [d for d in rebalance_dates if d <= pe]



    period_start_dt = rebalance_dates[0] if rebalance_dates else None

    period_end_dt = rebalance_dates[-1] if rebalance_dates else None



    for dt in rebalance_dates:

        if not pd.isna(monthly_rets[bench_col].loc[dt]):

            bench_value *= 1 + monthly_rets[bench_col].loc[dt]



        day_ret = 0.0

        if prev_weights:

            for t, w in prev_weights.items():

                r = monthly_rets[t].loc[dt] if t in monthly_rets.columns else np.nan

                if not pd.isna(r):

                    day_ret += w * r

        portfolio_value *= 1 + day_ret



        sub = sig[sig["date"] <= dt].sort_values("date").groupby("ticker", as_index=False).tail(1)

        regime_row = sub[sub["ticker"] == bench_col]

        exposure_cap = 1.0

        bench_vr = 1.0

        if not regime_row.empty:

            exposure_cap = float(regime_row.iloc[0].get("exposure_cap", 1.0) or 1.0)

            bench_vr = float(regime_row.iloc[0].get("vol_ratio", 1.0) or 1.0)



        picks = []
        for _, row in sub.iterrows():
            if row["ticker"] == bench_col:
                continue
            sc = row.get(score_col, row.get("score"))
            hold_th = row.get("hold_threshold") if use_dynamic_thresholds else None
            buy_th = row.get("buy_threshold") if use_dynamic_thresholds else None
            if _entry_eligible(float(sc), bool(row.get("risk_on", True)), hold_th, buy_th):
                picks.append(row)

        if use_vol_targeting:

            new_weights = _vol_target_weights(picks, score_col, exposure_cap, bench_vr)

        else:

            new_weights = {}

            if picks:

                tmp = {}

                for row in picks:

                    sc = row.get(score_col, row.get("score"))

                    vol = row.get("vol_ann", 0.25)

                    w = _legacy_weight(float(sc), vol, len(picks))

                    tmp[row["ticker"]] = w

                wsum = sum(tmp.values())

                if wsum > 1.0:

                    tmp = {k: v / wsum for k, v in tmp.items()}

                new_weights = tmp



        cost = _turnover_cost(prev_weights, new_weights, commission_bps)

        if cost > 0:

            portfolio_value *= 1.0 - cost

            total_commission += cost

        prev_weights = new_weights



        holdings_log.append({
            "date": dt,
            "weights": dict(prev_weights),
            "n": len(prev_weights),
            "commission_pct": round(cost * 100, 4),
        })



        equity_curve.append({"date": dt, "value": portfolio_value})

        bench_curve.append({"date": dt, "value": bench_value})



    eq = pd.DataFrame(equity_curve).set_index("date")

    bq = pd.DataFrame(bench_curve).set_index("date")

    stats = _performance_stats(eq["value"], bq["value"])

    if period_start_dt is not None and period_end_dt is not None:

        stats["period_start"] = str(period_start_dt.date())

        stats["period_end"] = str(period_end_dt.date())

    stats["use_vol_targeting"] = use_vol_targeting

    stats["commission_bps"] = commission_bps

    stats["total_commission_pct"] = round(total_commission * 100, 3)

    if not eq.empty:
        eq_d = eq["value"].resample("D").last().ffill()
        bq_d = bq["value"].resample("D").last().ffill()
        aligned = pd.concat([eq_d, bq_d], axis=1, join="inner").dropna()
        if not aligned.empty:
            exp = np.mean([sum(h["weights"].values()) for h in holdings_log]) if holdings_log else 0
            stats.update(_calendar_performance_stats(aligned.iloc[:, 0], aligned.iloc[:, 1], exp, 0))

    return {

        "equity": eq,

        "benchmark": bq,

        "stats": stats,

        "holdings": holdings_log,

        "period_start": period_start_dt,

        "period_end": period_end_dt,

    }





def _legacy_weight(score: float, vol_ann: float, n_names: int) -> float:

    if vol_ann is None or math.isnan(vol_ann) or vol_ann < 0.05:

        vol_ann = 0.25

    raw = TARGET_VOL_ANN / vol_ann / max(n_names, 1)

    raw *= 0.8 + 0.4 * (score / 100)

    return min(MAX_WEIGHT, max(0.0, raw))





def backtest_per_ticker(

    prices: pd.DataFrame,

    signals: pd.DataFrame,

    holdings_log: list[dict],

    score_col: str = "score",

    use_dynamic_thresholds: bool = True,

    name_map: dict[str, str] | None = None,

) -> list[dict]:

    """

    Per-ticker results: buy-and-hold vs strategy participation over backtest window.

    """

    monthly_px = prices.resample(_cfg.REBALANCE_FREQ).last()
    monthly_rets = monthly_px.pct_change(fill_method=None)

    sig = signals.copy()

    sig["date"] = pd.to_datetime(sig["date"])



    dates = [h["date"] for h in holdings_log]

    if not dates:

        return []



    period_start = pd.Timestamp(dates[0])

    period_end = pd.Timestamp(dates[-1])

    n_months = len(dates)



    # find first valid price at or before period_start per ticker

    results = []

    for ticker in prices.columns:

        if ticker not in monthly_px.columns:

            continue

        px = monthly_px[ticker].dropna()

        if px.empty:

            continue

        valid_dates = px.index[px.index >= period_start]

        if valid_dates.empty:

            continue

        p0_date = valid_dates[0]

        p1_date = px.index[px.index <= period_end][-1]

        p0 = px.loc[p0_date]

        p1 = px.loc[p1_date]

        if p0 <= 0 or pd.isna(p0) or pd.isna(p1):

            continue

        bh_return_pct = round((p1 / p0 - 1) * 100, 2)



        months_held = 0

        weight_sum = 0.0

        contrib = 0.0

        wins = 0

        held_periods = 0



        for h in holdings_log:

            dt = pd.Timestamp(h["date"])

            w = h["weights"].get(ticker, 0.0)

            if w > 0:

                months_held += 1

                weight_sum += w

                if dt in monthly_rets.index and ticker in monthly_rets.columns:

                    r = monthly_rets.loc[dt, ticker]

                    if pd.notna(r):

                        contrib += w * r

                        held_periods += 1

                        if r > 0:

                            wins += 1



        avg_weight = round(weight_sum / months_held, 3) if months_held else 0.0

        hit_rate = round(wins / held_periods * 100, 1) if held_periods else None

        participation_pct = round(months_held / n_months * 100, 1)



        # standalone ticker strategy: hold when eligible else cash

        strat_val = 1.0

        for dt in dates:

            snap = sig[(sig["ticker"] == ticker) & (sig["date"] <= dt)].tail(1)

            if snap.empty:

                continue

            row = snap.iloc[0]

            sc = row.get(score_col, row.get("score"))

            hold_th = row.get("hold_threshold") if use_dynamic_thresholds else None

            buy_th = row.get("buy_threshold") if use_dynamic_thresholds else None

            in_pos = _eligible(float(sc), bool(row.get("risk_on", True)), hold_th, buy_th)

            if in_pos and dt in monthly_rets.index:

                r = monthly_rets.loc[dt, ticker]

                if pd.notna(r):

                    strat_val *= 1 + r



        strat_return_pct = round((strat_val - 1) * 100, 2)



        last_snap = sig[sig["ticker"] == ticker].sort_values("date").tail(1)

        last_score = float(last_snap.iloc[0][score_col]) if not last_snap.empty else None

        last_dec = last_snap.iloc[0].get("decision_dynamic") if not last_snap.empty else None



        results.append(

            {

                "ticker": ticker,

                "name": (name_map or {}).get(ticker, ticker),

                "period_start": str(p0_date.date()),

                "period_end": str(p1_date.date()),

                "buy_hold_return_pct": bh_return_pct,

                "strategy_return_pct": strat_return_pct,

                "portfolio_contrib_pct": round(contrib * 100, 2),

                "months_held": months_held,

                "participation_pct": participation_pct,

                "avg_weight": avg_weight,

                "hit_rate_pct": hit_rate,

                "last_score": last_score,

                "last_decision": last_dec,

            }

        )



    return sorted(results, key=lambda x: x["buy_hold_return_pct"], reverse=True)





def _performance_stats(strategy: pd.Series, benchmark: pd.Series) -> dict:
    sr = strategy.pct_change().dropna()
    br = benchmark.pct_change().dropna()
    aligned = pd.concat([sr, br], axis=1, join="inner").dropna()
    if aligned.empty:
        return {}

    s, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    # Intraday: use configured bars/year (RTH-aware). Daily/monthly: infer from spacing.
    idx = strategy.index
    if _cfg.IS_INTRADAY:
        ppy = float(_cfg.PERIODS_PER_YEAR)
    elif len(idx) > 2 and hasattr(idx, "to_series"):
        med_days = idx.to_series().diff().dt.days.median()
        ppy = 365.25 / med_days if med_days and med_days > 0 else float(_cfg.PERIODS_PER_YEAR)
    else:
        ppy = float(_cfg.PERIODS_PER_YEAR)
    years = max(len(s) / ppy, 0.1)
    total_s = strategy.iloc[-1] / strategy.iloc[0] - 1
    total_b = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    vol_s = s.std() * math.sqrt(ppy)
    sharpe = (s.mean() * ppy - 0.04) / vol_s if vol_s > 0 else 0
    cummax = strategy.cummax()
    dd = (strategy / cummax - 1).min()
    return {
        "total_return_pct": round(total_s * 100, 2),
        "benchmark_return_pct": round(total_b * 100, 2),
        "excess_return_pct": round((total_s - total_b) * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(dd * 100, 2),
        "years": round(years, 1),
        "rebalance_freq": _cfg.REBALANCE_FREQ,
        "periods_per_year": round(ppy, 1),
    }


