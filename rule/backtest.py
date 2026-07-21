"""Per-ticker and portfolio backtests for the rule strategy."""

from __future__ import annotations

import math

import pandas as pd

import config
from data_platform.universe import commission_bps_for_ticker
from simulation.engine import run_backtest_signal_exit

from rule.config import (
    RULE_BUY_ONLY,
    RULE_EQUAL_WEIGHT,
    RULE_FORECAST_HORIZON_BARS,
    RULE_MIN_HOLD_BARS,
    RULE_PARTIAL_SELL_FRAC,
    RULE_REBALANCE_BAND,
    RULE_REENTRY_COOLDOWN_BARS,
    RULE_STOP_LOSS_BPS,
    RULE_USE_VOL_TARGETING,
)


def _benchmark_return_pct(close: pd.Series, period_start, period_end) -> float | None:
    s = close.dropna()
    if period_start is not None:
        s = s[s.index >= pd.Timestamp(period_start)]
    if period_end is not None:
        s = s[s.index <= pd.Timestamp(period_end)]
    if len(s) < 2:
        return None
    p0, p1 = float(s.iloc[0]), float(s.iloc[-1])
    if p0 <= 0:
        return None
    return round((p1 / p0 - 1.0) * 100.0, 2)


def equal_weight_universe_return_pct(
    prices: pd.DataFrame,
    period_start,
    period_end,
) -> float | None:
    """Equal-weight buy-and-hold across all columns (fair portfolio benchmark)."""
    px = prices.dropna(how="all")
    if px.empty or len(px.columns) < 1:
        return None
    rets = px.pct_change(fill_method=None).mean(axis=1)
    eq = (1.0 + rets.fillna(0.0)).cumprod()
    s = eq.dropna()
    if period_start is not None:
        s = s[s.index >= pd.Timestamp(period_start)]
    if period_end is not None:
        s = s[s.index <= pd.Timestamp(period_end)]
    if len(s) < 2:
        return None
    p0, p1 = float(s.iloc[0]), float(s.iloc[-1])
    if p0 <= 0:
        return None
    return round((p1 / p0 - 1.0) * 100.0, 2)


def run_portfolio_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    tickers: list[str],
) -> dict:
    """Combined book: equal weight across active names when RULE_EQUAL_WEIGHT."""
    comm = float(sum(commission_bps_for_ticker(t) for t in tickers) / max(len(tickers), 1))
    return run_backtest_signal_exit(
        prices,
        signals,
        score_col="score",
        use_dynamic_thresholds=False,
        use_vol_targeting=RULE_USE_VOL_TARGETING,
        equal_weight=RULE_EQUAL_WEIGHT,
        commission_bps=comm,
        horizon_bars=RULE_FORECAST_HORIZON_BARS,
        stop_loss_bps=RULE_STOP_LOSS_BPS,
        min_hold_bars=RULE_MIN_HOLD_BARS,
        rebalance_band=RULE_REBALANCE_BAND,
        hold_entry_weight=False,
        reentry_cooldown_bars=RULE_REENTRY_COOLDOWN_BARS,
        partial_sell_frac=RULE_PARTIAL_SELL_FRAC,
        keep_holdings_on_empty_picks=True,
        merge_new_picks_only=True,
        buy_only=RULE_BUY_ONLY,
    )


def run_per_ticker_backtests(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    tickers: list[str],
    *,
    include_equity: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], dict[str, dict]]:
    """Independent equal-weight backtest per instrument (100% when in position)."""
    results: dict[str, dict] = {}
    equities: dict[str, dict] = {}
    for sym in tickers:
        sym_u = str(sym).upper()
        if sym_u not in prices.columns:
            results[sym_u] = {"skipped": True, "reason": "no_prices"}
            continue
        sig = signals[signals["ticker"].astype(str).str.upper() == sym_u]
        if sig.empty:
            results[sym_u] = {"skipped": True, "reason": "no_signals"}
            continue
        px = prices[[sym_u]]
        comm = float(commission_bps_for_ticker(sym_u))
        bt = run_backtest_signal_exit(
            px,
            sig,
            score_col="score",
            use_dynamic_thresholds=False,
            use_vol_targeting=False,
            equal_weight=True,
            commission_bps=comm,
            horizon_bars=RULE_FORECAST_HORIZON_BARS,
            stop_loss_bps=RULE_STOP_LOSS_BPS,
            min_hold_bars=RULE_MIN_HOLD_BARS,
            rebalance_band=RULE_REBALANCE_BAND,
            hold_entry_weight=False,
            reentry_cooldown_bars=RULE_REENTRY_COOLDOWN_BARS,
            partial_sell_frac=RULE_PARTIAL_SELL_FRAC,
            keep_holdings_on_empty_picks=True,
            merge_new_picks_only=True,
            buy_only=RULE_BUY_ONLY,
        )
        stats = dict(bt.get("stats") or {})
        ps = stats.get("period_start")
        pe = stats.get("period_end")
        bh = _benchmark_return_pct(px[sym_u], ps, pe)
        ret = stats.get("total_return_pct")
        results[sym_u] = {
            "total_return_pct": ret,
            "sharpe": stats.get("sharpe"),
            "max_drawdown_pct": stats.get("max_drawdown_pct"),
            "signal_exit_count": stats.get("signal_exit_count"),
            "stop_loss_exit_count": stats.get("stop_loss_exit_count"),
            "avg_exposure_pct": stats.get("avg_exposure_pct"),
            "n_signals": int(len(sig)),
            "benchmark_return_pct": bh,
            "excess_return_pct": round(float(ret) - float(bh), 2)
            if ret is not None and bh is not None
            else None,
            "period_start": str(ps) if ps is not None else None,
            "period_end": str(pe) if pe is not None else None,
        }
        if include_equity:
            equities[sym_u] = {
                "equity": bt.get("equity"),
                "benchmark": bt.get("benchmark"),
            }
    if include_equity:
        return results, equities
    return results


def expected_move_pct(vol_ann: float, lookback_bars: int) -> float:
    """Typical |move| in percent over lookback from annualized vol."""
    lb = max(int(lookback_bars), 1)
    vol = max(float(vol_ann), 1e-6)
    frac = vol * math.sqrt(lb / float(config.BARS_PER_YEAR))
    return round(frac * 100.0, 4)
