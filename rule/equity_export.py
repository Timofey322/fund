"""Export rebased equity curves + CAPM beta for the web UI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import BENCHMARK, OUT_DIR
from rule.config import RULE_INITIAL_CAPITAL_USD
from simulation.benchmark_alpha import comparative_alpha

RULE_EQUITY_DIR = OUT_DIR / "rule" / "equity"
# Keep every trading day for daily rule books (~4k points); downsample only intraday-scale series.
MAX_CHART_POINTS = 8000


def _resolve_equity_series(equity: pd.DataFrame | pd.Series | None) -> pd.Series:
    """Datetime-indexed strategy/benchmark levels (no rebase)."""
    if equity is None:
        return pd.Series(dtype=float)
    if isinstance(equity, pd.Series):
        s = equity.astype(float).dropna()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        return s[~s.index.duplicated(keep="last")].sort_index()
    if equity.empty or "value" not in equity.columns:
        return pd.Series(dtype=float)
    if isinstance(equity.index, pd.DatetimeIndex):
        s = equity["value"].astype(float).dropna()
        return s[~s.index.duplicated(keep="last")].sort_index()
    if "bar_time" in equity.columns:
        idx = pd.to_datetime(equity["bar_time"])
    elif "date" in equity.columns:
        idx = pd.to_datetime(equity["date"])
    else:
        return pd.Series(dtype=float)
    s = pd.Series(equity["value"].astype(float).values, index=idx)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _to_series(equity: pd.DataFrame | pd.Series | None) -> pd.Series:
    if equity is None:
        return pd.Series(dtype=float)
    if isinstance(equity, pd.Series):
        s = equity.astype(float).dropna()
        return s[~s.index.duplicated(keep="last")].sort_index()
    if equity.empty:
        return pd.Series(dtype=float)

    if "value" in equity.columns:
        if isinstance(equity.index, pd.DatetimeIndex):
            idx = equity.index
        elif "bar_time" in equity.columns:
            idx = pd.to_datetime(equity["bar_time"])
        elif "date" in equity.columns:
            idx = pd.to_datetime(equity["date"])
        else:
            return pd.Series(dtype=float)
        s = pd.Series(equity["value"].astype(float).values, index=idx)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.dropna()

    if len(equity.columns) == 1 and isinstance(equity.index, pd.DatetimeIndex):
        s = equity.iloc[:, 0].astype(float).dropna()
        return s[~s.index.duplicated(keep="last")].sort_index()

    return pd.Series(dtype=float)


def equal_weight_benchmark_series(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight buy-and-hold index from wide close matrix."""
    px = prices.dropna(how="all")
    if px.empty:
        return pd.Series(dtype=float)
    rets = px.pct_change(fill_method=None).mean(axis=1)
    return (1.0 + rets.fillna(0.0)).cumprod()


def _downsample(s: pd.Series, max_points: int = MAX_CHART_POINTS) -> pd.Series:
    if len(s) <= max_points:
        return s
    idx = np.linspace(0, len(s) - 1, max_points, dtype=int)
    return s.iloc[idx]


def _rebase(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty or float(s.iloc[0]) <= 0:
        return s
    return s / float(s.iloc[0])


def _downsample_aligned(
    strat: pd.Series,
    bench: pd.Series | None,
    max_points: int = MAX_CHART_POINTS,
) -> tuple[pd.Series, pd.Series]:
    """Rebase and downsample strategy/benchmark on shared timestamps."""
    s = _rebase(strat)
    if s.empty:
        return s, pd.Series(dtype=float)
    b = pd.Series(dtype=float)
    if bench is not None and not bench.empty:
        b = _rebase(bench.reindex(s.index).ffill()).reindex(s.index).ffill()
    if len(s) <= max_points:
        return s, b
    idx = np.linspace(0, len(s) - 1, max_points, dtype=int)
    return s.iloc[idx], b.iloc[idx] if not b.empty else pd.Series(dtype=float)


def _to_dollar_equity(s: pd.Series, capital: float = RULE_INITIAL_CAPITAL_USD) -> pd.Series:
    """Convert normalized backtest equity (start≈1) to USD portfolio value."""
    s = s.dropna()
    if s.empty or float(s.iloc[0]) <= 0:
        return s
    return s / float(s.iloc[0]) * float(capital)


def _ticker_price_series(prices: pd.DataFrame, ticker: str, index: pd.DatetimeIndex) -> pd.Series:
    sym = str(ticker).upper()
    col = sym if sym in prices.columns else next((c for c in prices.columns if str(c).upper() == sym), None)
    if col is None:
        return pd.Series(dtype=float)
    return prices[col].astype(float).reindex(index).ffill().dropna()


def _downsample_aligned_raw(
    strat: pd.Series,
    bench: pd.Series | None,
    max_points: int = MAX_CHART_POINTS,
) -> tuple[pd.Series, pd.Series]:
    """Downsample dollar (or level) series on shared timestamps — no rebase."""
    s = strat.dropna()
    if s.empty:
        return s, pd.Series(dtype=float)
    b = pd.Series(dtype=float)
    if bench is not None and not bench.empty:
        b = bench.reindex(s.index).ffill().reindex(s.index)
    if len(s) <= max_points:
        return s, b
    idx = np.linspace(0, len(s) - 1, max_points, dtype=int)
    return s.iloc[idx], b.iloc[idx] if not b.empty else pd.Series(dtype=float)


def _aligned_dollar_payload(strat: pd.Series, bench: pd.Series | None) -> tuple[dict, dict]:
    s, b = _downsample_aligned_raw(strat, bench)
    if s.empty:
        return {"t": [], "v": []}, {"t": [], "v": []}
    t = [ts.isoformat() for ts in s.index]
    strat_out = {"t": t, "v": [round(float(x), 2) for x in s.values]}
    if b.empty:
        return strat_out, {"t": [], "v": []}
    bench_out = {"t": t, "v": [round(float(x), 2) for x in b.values]}
    return strat_out, bench_out


def _series_payload(s: pd.Series) -> dict:
    strat_out, _ = _aligned_dollar_payload(_to_dollar_equity(s), None)
    return strat_out


def _return_pct(s: pd.Series) -> float | None:
    s = s.dropna()
    if len(s) < 2 or float(s.iloc[0]) <= 0:
        return None
    return round((float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) * 100.0, 2)


def build_equity_chart_payload(
    strategy: pd.DataFrame | pd.Series | None,
    benchmark: pd.DataFrame | pd.Series | None,
    *,
    ticker: str | None = None,
    benchmark_ticker: str = BENCHMARK,
    label: str,
    prices: pd.DataFrame | None = None,
    initial_capital: float = RULE_INITIAL_CAPITAL_USD,
) -> dict:
    strat_raw = _resolve_equity_series(strategy)
    strat = _to_dollar_equity(strat_raw, initial_capital)
    bench_raw = _resolve_equity_series(benchmark)
    value_mode = "portfolio_dollars"
    strategy_label = "Портфель, USD"
    benchmark_label = "EW B&H, USD"

    if ticker and prices is not None and not strat.empty:
        sym = str(ticker).upper()
        price = _ticker_price_series(prices, sym, strat.index)
        if not price.empty:
            bench_raw = price
            bench = price
            value_mode = "ticker_dollars"
            strategy_label = "Стоимость позиции, USD"
            benchmark_label = f"Цена {sym}, USD"
            benchmark_ticker = sym
            # Alpha vs buy-and-hold dollars (same capital), not vs raw price level.
            bench_for_alpha = _to_dollar_equity(price / float(price.iloc[0]), initial_capital)
        else:
            bench = _to_dollar_equity(bench_raw, initial_capital) if not bench_raw.empty else bench_raw
            bench_for_alpha = bench
    elif bench_raw.empty and prices is not None and not strat.empty:
        ew = equal_weight_benchmark_series(prices)
        bench_raw = ew.reindex(strat.index).ffill()
        bench = _to_dollar_equity(bench_raw, initial_capital)
        benchmark_ticker = "EW universe"
        bench_for_alpha = bench
    else:
        bench = _to_dollar_equity(bench_raw, initial_capital) if not bench_raw.empty else bench_raw
        bench_for_alpha = bench

    if not bench.empty and not strat.empty:
        common = strat.index.intersection(bench.index)
        if len(common) >= max(10, int(0.5 * min(len(strat), len(bench)))):
            strat = strat.loc[common]
            bench = bench.loc[common]
            if value_mode == "ticker_dollars" and prices is not None and ticker:
                alpha_bench = _to_dollar_equity(
                    _ticker_price_series(prices, ticker, common) / float(
                        _ticker_price_series(prices, ticker, common).iloc[0]
                    ),
                    initial_capital,
                )
            else:
                alpha_bench = bench_for_alpha.reindex(common).ffill() if not bench_for_alpha.empty else bench
        else:
            bench = bench.reindex(strat.index).ffill().bfill()
            alpha_bench = bench_for_alpha.reindex(strat.index).ffill() if not bench_for_alpha.empty else bench
    else:
        alpha_bench = bench

    alpha_stats = (
        comparative_alpha(strat, alpha_bench, label=label)
        if len(strat) >= 10 and len(alpha_bench) >= 10
        else {}
    )

    strat_payload, bench_payload = _aligned_dollar_payload(strat, bench if not bench.empty else None)

    ret_strategy = _return_pct(strat)
    ret_bench = _return_pct(alpha_bench if value_mode == "ticker_dollars" else bench)

    return {
        "ticker": ticker,
        "benchmark_ticker": benchmark_ticker,
        "label": label,
        "value_mode": value_mode,
        "strategy_label": strategy_label,
        "benchmark_label": benchmark_label,
        "initial_capital_usd": initial_capital,
        "series": {
            "strategy": strat_payload,
            "benchmark": bench_payload,
        },
        "return_pct": {
            "strategy": ret_strategy,
            "benchmark": ret_bench,
            "excess": round(ret_strategy - ret_bench, 2)
            if ret_strategy is not None and ret_bench is not None
            else None,
        },
        "beta": alpha_stats.get("beta"),
        "alpha_annualized_pct": alpha_stats.get("alpha_annualized_pct"),
        "correlation": alpha_stats.get("correlation"),
        "r_squared": alpha_stats.get("r_squared"),
        "information_ratio": alpha_stats.get("information_ratio"),
    }


def write_equity_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def export_rule_equity_curves(
    portfolio_equity: pd.DataFrame | None,
    portfolio_benchmark: pd.DataFrame | None,
    per_ticker_equity: dict[str, dict],
    *,
    benchmark_ticker: str = BENCHMARK,
    prices: pd.DataFrame | None = None,
) -> dict:
    """Write portfolio + per-ticker equity JSON; return web paths."""
    RULE_EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    port = build_equity_chart_payload(
        portfolio_equity,
        portfolio_benchmark,
        ticker=None,
        benchmark_ticker=benchmark_ticker,
        label="portfolio_vs_benchmark",
        prices=prices,
    )
    port_path = RULE_EQUITY_DIR / "portfolio.json"
    write_equity_json(port, port_path)
    paths["portfolio"] = "/output/rule/equity/portfolio.json"

    for sym, frames in sorted(per_ticker_equity.items()):
        sym_u = str(sym).upper()
        payload = build_equity_chart_payload(
            frames.get("equity"),
            frames.get("benchmark"),
            ticker=sym_u,
            benchmark_ticker=sym_u,
            label=f"{sym_u}_vs_buy_hold",
            prices=prices,
        )
        sym_path = RULE_EQUITY_DIR / f"{sym_u}.json"
        write_equity_json(payload, sym_path)
        paths[sym_u] = f"/output/rule/equity/{sym_u}.json"

    return paths
