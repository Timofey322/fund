"""QuantStats report integration for stitched backtests."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulation.benchmark_alpha import equal_weight_benchmark
from config import OUT_DIR

QUANTSTATS_DIR = OUT_DIR / "quantstats"
QUANTSTATS_HTML = QUANTSTATS_DIR / "fusion_quantstats.html"
QUANTSTATS_METRICS = QUANTSTATS_DIR / "fusion_quantstats_metrics.json"


def _daily_returns_from_equity(equity: pd.DataFrame | pd.Series | None) -> pd.Series:
    if equity is None:
        return pd.Series(dtype=float)
    if isinstance(equity, pd.Series):
        s = equity.astype(float).dropna()
    elif not equity.empty and "value" in equity.columns:
        s = equity["value"].astype(float).dropna()
    else:
        return pd.Series(dtype=float)
    if s.empty:
        return pd.Series(dtype=float)
    s.index = pd.to_datetime(s.index)
    daily = s.resample("D").last().ffill()
    return daily.pct_change().replace([np.inf, -np.inf], np.nan).dropna().rename("strategy")


def _benchmark_daily_returns(prices: pd.DataFrame, symbols: list[str], index: pd.Index) -> pd.Series:
    if prices is None or prices.empty or len(index) == 0:
        return pd.Series(dtype=float)
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    ew = equal_weight_benchmark(px, symbols)
    if ew.empty:
        first = symbols[0] if symbols and symbols[0] in px.columns else px.columns[0]
        ew = px[first].astype(float)
    daily = ew.resample("D").last().ffill()
    rets = daily.pct_change().replace([np.inf, -np.inf], np.nan).dropna().rename("benchmark")
    return rets.reindex(index).dropna()


def _basic_metrics(strategy: pd.Series, benchmark: pd.Series) -> dict[str, Any]:
    aligned = pd.concat([strategy, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        aligned = pd.DataFrame({"strategy": strategy})
    sr = aligned["strategy"].astype(float)
    bench = aligned["benchmark"].astype(float) if "benchmark" in aligned.columns else pd.Series(dtype=float)
    total = float((1.0 + sr).prod() - 1.0) if len(sr) else 0.0
    vol = float(sr.std() * np.sqrt(365.25)) if len(sr) > 1 else 0.0
    sharpe = float(sr.mean() * 365.25 / vol) if vol > 1e-12 else 0.0
    max_dd = float(((1.0 + sr).cumprod() / (1.0 + sr).cumprod().cummax() - 1.0).min()) if len(sr) else 0.0
    out: dict[str, Any] = {
        "n_days": int(len(sr)),
        "total_return_pct": round(total * 100.0, 3),
        "volatility_ann_pct": round(vol * 100.0, 3),
        "sharpe_ann": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 3),
    }
    if not bench.empty:
        b_total = float((1.0 + bench).prod() - 1.0)
        out["benchmark_return_pct"] = round(b_total * 100.0, 3)
        out["excess_return_pct"] = round((total - b_total) * 100.0, 3)
    return out


def write_quantstats_report(
    backtest: dict,
    prices: pd.DataFrame,
    symbols: list[str],
    *,
    html_path: Path | None = None,
    metrics_path: Path | None = None,
    title: str = "Fusion stitched walk-forward backtest",
) -> dict[str, Any]:
    """Write QuantStats HTML + metrics JSON, with graceful fallback."""
    html = html_path or QUANTSTATS_HTML
    metrics_file = metrics_path or QUANTSTATS_METRICS
    strategy = _daily_returns_from_equity(backtest.get("equity"))
    benchmark = _benchmark_daily_returns(prices, symbols, strategy.index)
    metrics = _basic_metrics(strategy, benchmark)
    metrics.update({"html_path": str(html), "metrics_path": str(metrics_file)})

    QUANTSTATS_DIR.mkdir(parents=True, exist_ok=True)
    if strategy.empty:
        metrics["quantstats_status"] = "skipped_empty_returns"
    else:
        try:
            import quantstats as qs  # type: ignore

            qs.extend_pandas()
            bench_arg = benchmark if not benchmark.empty else None
            # quantstats ships seaborn/pandas calls that emit FutureWarnings on
            # newer versions; the rendering itself is fine, so silence the noise.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                qs.reports.html(strategy, benchmark=bench_arg, output=str(html), title=title)
            metrics["quantstats_status"] = "ok"
        except Exception as exc:  # dependency missing or report rendering issue
            metrics["quantstats_status"] = "skipped"
            metrics["quantstats_error"] = str(exc)

    metrics_file.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics

