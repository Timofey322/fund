"""Export chart datasets for client-side SVG rendering (no PNG dependency on web)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import BARS_PER_YEAR, OUT_DIR
from rule.equity_export import (
    MAX_CHART_POINTS,
    _rebase,
    equal_weight_benchmark_series,
)
from rule.plots import _equity_series
from rule.web_export import RULE_PLOT_TITLES
from simulation.trade_survival import block_bootstrap_paths, survival_simulation

RULE_CHARTS_PATH = OUT_DIR / "rule" / "charts.json"


def _downsample_times_values(times: list[str], *series: list[float]) -> tuple[list[str], list[list[float]]]:
    n = len(times)
    if n <= MAX_CHART_POINTS:
        return times, [list(s) for s in series]
    idx = np.linspace(0, n - 1, MAX_CHART_POINTS, dtype=int)
    t_out = [times[i] for i in idx]
    s_out = [[float(s[i]) for i in idx] for s in series]
    return t_out, s_out


def _line_chart(
    title: str,
    times: pd.DatetimeIndex,
    series: list[dict],
    *,
    y_label: str = "",
    zero_line: bool = False,
) -> dict:
    t = [ts.isoformat() for ts in times]
    payloads = [list(s["values"]) for s in series]
    t_ds, vals_ds = _downsample_times_values(t, *payloads)
    out_series = []
    for spec, vals in zip(series, vals_ds):
        out_series.append({"name": spec["name"], "values": [round(v, 6) for v in vals], "color": spec.get("color")})
    return {
        "type": "line",
        "title": title,
        "times": t_ds,
        "series": out_series,
        "yLabel": y_label,
        "zeroLine": zero_line,
    }


def _histogram_chart(
    title: str,
    values: np.ndarray,
    *,
    bins: int = 40,
    x_label: str = "",
    vlines: list[dict] | None = None,
    subtitle: str | None = None,
) -> dict:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {"type": "histogram", "title": title, "binEdges": [], "counts": [], "xLabel": x_label}
    counts, edges = np.histogram(vals, bins=bins)
    return {
        "type": "histogram",
        "title": title,
        "subtitle": subtitle,
        "binEdges": [round(float(x), 4) for x in edges],
        "counts": [int(c) for c in counts],
        "xLabel": x_label,
        "vlines": vlines or [],
    }


def _grouped_bar(title: str, labels: list[str], strat: list[float], bench: list[float], *, y_label: str = "%") -> dict:
    return {
        "type": "grouped_bar",
        "title": title,
        "labels": labels,
        "series": [
            {"name": "Strategy", "values": [round(v, 2) for v in strat], "color": "#c9a227"},
            {"name": "Buy & hold", "values": [round(v, 2) for v in bench], "color": "#64748b"},
        ],
        "yLabel": y_label,
    }


def _bar_chart(title: str, labels: list[str], values: list[float], *, y_label: str = "%") -> dict:
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in values]
    return {
        "type": "bar",
        "title": title,
        "labels": labels,
        "values": [round(v, 2) for v in values],
        "colors": colors,
        "yLabel": y_label,
    }


def _sharpe_bar(title: str, labels: list[str], values: list[float]) -> dict:
    colors = ["#2563eb" if v >= 0 else "#dc2626" for v in values]
    return {
        "type": "bar",
        "title": title,
        "labels": labels,
        "values": [round(v, 2) for v in values],
        "colors": colors,
        "yLabel": "Sharpe",
        "refLines": [{"value": 0, "label": "0"}, {"value": 1, "label": "1"}],
    }


def _stacked_bar(title: str, rows: list[dict]) -> dict:
    labels = [str(r["ticker"]) for r in rows]
    dump = [float(r.get("pct_dump_buy") or 0.0) for r in rows]
    rally = [float(r.get("pct_rally_sell") or 0.0) for r in rows]
    other = [max(0.0, 100.0 - d - s) for d, s in zip(dump, rally)]
    return {
        "type": "stacked_bar",
        "title": title,
        "labels": labels,
        "segments": [
            {"name": "Dump buy", "values": [round(v, 1) for v in dump], "color": "#c9a227"},
            {"name": "Rally sell", "values": [round(v, 1) for v in rally], "color": "#b83d5e"},
            {"name": "Neutral", "values": [round(v, 1) for v in other], "color": "#334155"},
        ],
        "yLabel": "% of bars",
    }


def _heatmap_chart(title: str, eq: pd.Series) -> dict:
    s = eq.dropna().astype(float)
    if s.empty:
        return {"type": "heatmap", "title": title, "years": [], "months": [], "values": []}
    monthly = s.resample("ME").last().pct_change().dropna() * 100.0
    if monthly.empty:
        return {"type": "heatmap", "title": title, "years": [], "months": [], "values": []}
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = sorted({int(ts.year) for ts in monthly.index})
    grid: list[list[float | None]] = []
    for year in years:
        row: list[float | None] = []
        for m in range(1, 13):
            hits = monthly[(monthly.index.year == year) & (monthly.index.month == m)]
            row.append(round(float(hits.iloc[-1]), 2) if len(hits) else None)
        grid.append(row)
    return {
        "type": "heatmap",
        "title": title,
        "years": years,
        "months": month_names,
        "values": grid,
    }


def export_rule_charts(
    report: dict,
    equity: pd.DataFrame | None,
    *,
    prices: pd.DataFrame | None = None,
    n_mc_paths: int = 2000,
    mc_block_bars: int = 12,
) -> dict[str, dict]:
    """Build chart specs keyed by plot id; write ``charts.json``."""
    charts: dict[str, dict] = {}
    stats = report.get("backtest") or {}
    per_bt = report.get("per_ticker_backtest") or {}
    per_sig = report.get("per_ticker_signals") or []

    eq = _equity_series(equity)
    if not eq.empty:
        eq_rb = _rebase(eq)
        bench = pd.Series(dtype=float)
        if prices is not None and not prices.empty:
            ew = equal_weight_benchmark_series(prices)
            bench = _rebase(ew.reindex(eq_rb.index).ffill()).reindex(eq_rb.index).ffill()

        series = [{"name": "Strategy", "values": eq_rb.values.tolist(), "color": "#c9a227"}]
        if not bench.empty:
            series.append({"name": "EW universe", "values": bench.values.tolist(), "color": "#64748b"})
        charts["rule_equity_curve"] = _line_chart(
            RULE_PLOT_TITLES["rule_equity_curve"],
            eq_rb.index,
            series,
            y_label="Rebased NAV",
        )

        rets = eq_rb.pct_change().dropna().values * 100.0
        charts["rule_return_density"] = _histogram_chart(
            RULE_PLOT_TITLES["rule_return_density"],
            rets,
            x_label="Daily return, %",
        )

        peak = eq_rb.cummax()
        dd = (eq_rb / peak - 1.0) * 100.0
        charts["rule_underwater_drawdown"] = _line_chart(
            RULE_PLOT_TITLES["rule_underwater_drawdown"],
            dd.index,
            [{"name": "Drawdown", "values": dd.values.tolist(), "color": "#b83d5e"}],
            y_label="Drawdown, %",
            zero_line=True,
        )

        roll = eq_rb.pct_change().rolling(126, min_periods=40)
        rs = (roll.mean() / roll.std()) * math.sqrt(float(BARS_PER_YEAR))
        rs = rs.replace([np.inf, -np.inf], np.nan).dropna()
        if not rs.empty:
            charts["rule_rolling_sharpe"] = _line_chart(
                RULE_PLOT_TITLES["rule_rolling_sharpe"],
                rs.index,
                [{"name": "Rolling Sharpe", "values": rs.values.tolist(), "color": "#2563eb"}],
                y_label="Sharpe (ann.)",
                zero_line=True,
            )

        charts["rule_monthly_returns"] = _heatmap_chart(RULE_PLOT_TITLES["rule_monthly_returns"], eq_rb)

        bar_rets = eq.pct_change().dropna().values
        bar_rets = bar_rets[np.isfinite(bar_rets)]
        if len(bar_rets) >= 20:
            paths = block_bootstrap_paths(bar_rets, n_mc_paths, len(bar_rets), mc_block_bars)
            terminal = paths[:, -1]
            surv = survival_simulation(eq, n_paths=min(n_mc_paths, 500), block_bars=mc_block_bars)
            vlines = [{"value": 1.0, "label": "Break-even", "color": "#000000"}]
            p50 = surv.get("terminal_wealth_p50")
            if p50 is not None:
                vlines.append({"value": float(p50), "label": f"p50={p50}", "color": "#c9a227"})
            charts["rule_monte_carlo_terminal"] = _histogram_chart(
                RULE_PLOT_TITLES["rule_monte_carlo_terminal"],
                terminal,
                bins=40,
                x_label="Terminal wealth (× start)",
                vlines=vlines,
                subtitle=(
                    f"Survival={surv.get('survival_rate')} | P(loss)={surv.get('prob_terminal_loss')}"
                ),
            )

    rows = [
        (sym, st)
        for sym, st in sorted(per_bt.items())
        if isinstance(st, dict) and not st.get("skipped")
    ]
    if rows:
        labels = [r[0] for r in rows]
        strat = [float(r[1].get("total_return_pct") or 0.0) for r in rows]
        bench_vals = [float(r[1].get("benchmark_return_pct") or 0.0) for r in rows]
        excess = [float(r[1].get("excess_return_pct") or 0.0) for r in rows]
        sharpes = [float(r[1]["sharpe"]) for r in rows if r[1].get("sharpe") is not None]
        sh_labels = [r[0] for r in rows if r[1].get("sharpe") is not None]

        charts["rule_per_ticker_returns"] = _grouped_bar(
            RULE_PLOT_TITLES["rule_per_ticker_returns"], labels, strat, bench_vals,
        )
        charts["rule_per_ticker_excess"] = _bar_chart(
            RULE_PLOT_TITLES["rule_per_ticker_excess"], labels, excess,
        )
        if sharpes:
            charts["rule_per_ticker_sharpe"] = _sharpe_bar(
                RULE_PLOT_TITLES["rule_per_ticker_sharpe"], sh_labels, sharpes,
            )

    sig_rows = [r for r in per_sig if r.get("ticker")]
    if sig_rows:
        charts["rule_signal_distribution"] = _stacked_bar(
            RULE_PLOT_TITLES["rule_signal_distribution"], sig_rows,
        )

    RULE_CHARTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": report.get("generated_at"),
        "backtest": {
            "total_return_pct": stats.get("total_return_pct"),
            "benchmark_return_pct": stats.get("benchmark_return_pct"),
            "excess_return_pct": stats.get("excess_return_pct"),
            "sharpe": stats.get("sharpe"),
            "max_drawdown_pct": stats.get("max_drawdown_pct"),
        },
        "charts": charts,
    }
    RULE_CHARTS_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return charts


def chart_manifest_plots(charts: dict[str, dict]) -> list[dict]:
    """Plot entries for web manifest (SVG-only, no PNG path)."""
    return [
        {"id": cid, "title": spec.get("title") or RULE_PLOT_TITLES.get(cid, cid), "path": ""}
        for cid, spec in sorted(charts.items())
    ]
