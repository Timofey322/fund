"""Quantitative hedge-fund desk reports: risk, attribution, and PM-facing artifacts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as _cfg
from config import OUT_DIR
from reporting.plots import (
    _slice_buy_hold_return,
    _slice_ew_return,
    _strategy_equity_series,
    compute_fold_anomaly_table,
    plot_fold_benchmark_comparison,
    write_fold_benchmark_markdown,
)

DESK_RISK_JSON = OUT_DIR / "desk_risk_summary.json"
DESK_ATTRIBUTION_JSON = OUT_DIR / "desk_walk_forward_attribution.json"
DESK_TRADE_JSON = OUT_DIR / "desk_trade_analytics.json"
DESK_PERF_MD = OUT_DIR / "desk_performance_summary.md"


def desk_go_no_go(
    report: dict,
    *,
    decile_audit: dict | None = None,
) -> dict:
    """Desk pre-flight: go when any per-ticker tradeable set is non-empty.

    Portfolio-level decile is diagnostic only — does not veto a partial live book.
    """
    reasons: list[str] = []
    notes: list[str] = []
    audit = decile_audit or report.get("decile_audit") or {}
    impulse = (report.get("impulse_optimization") or {}).get("best") or {}
    bt = report.get("backtest_walk_forward_oos") or {}
    ml = report.get("ml_diagnostics") or {}
    tradeable_tickers = list(
        audit.get("tradeable_tickers") or impulse.get("tradeable_tickers") or []
    )

    if impulse.get("disable_trading") and not tradeable_tickers:
        reasons.append("impulse_disable_trading")
    if impulse.get("decile_gate_blocked") and not tradeable_tickers:
        reasons.append("decile_gate_blocked")
    if not tradeable_tickers:
        if not audit.get("tradeable", False):
            reasons.extend(audit.get("reasons") or ["decile_not_tradeable"])
        else:
            reasons.append("empty_tradeable_set")
    elif not audit.get("tradeable", False):
        notes.append(f"partial_trade_{len(tradeable_tickers)}_symbols")

    top_net = audit.get("top_decile_net_bps")
    if top_net is not None and tradeable_tickers:
        notes.append(f"portfolio_top_decile_net_bps={top_net}")
    elif top_net is not None and not tradeable_tickers:
        reasons.append(f"top_decile_net_bps={top_net}")

    ml_top = ml.get("top_decile_target_fwd_ret_bps")
    if ml_top is not None and not tradeable_tickers:
        reasons.append(f"ml_top_decile_gross_bps={ml_top}")

    folds = report.get("walk_forward_folds") or []
    active_folds = [f for f in folds if not f.get("skipped")]
    neg_obj = sum(
        1 for f in active_folds
        if ((f.get("threshold_optimization") or {}).get("cv") or {}).get("objective", 0) is not None
        and float(((f.get("threshold_optimization") or {}).get("cv") or {}).get("objective", 0)) < float(
            getattr(_cfg, "FUSION_THRESHOLD_NO_TRADE_OBJECTIVE", -2.0)
        )
    )
    if active_folds and neg_obj == len(active_folds) and not tradeable_tickers:
        reasons.append("all_fold_threshold_objectives_negative")

    tradeable = bool(tradeable_tickers) and not (
        bool(impulse.get("disable_trading")) and not tradeable_tickers
    )
    return {
        "tradeable": tradeable,
        "reasons": sorted(set(reasons)),
        "notes": sorted(set(notes)),
        "top_decile_net_bps": top_net,
        "tradeable_tickers": tradeable_tickers,
        "disabled_tickers": impulse.get("disabled_tickers") or [],
        "oos_total_return_pct": bt.get("total_return_pct"),
        "disable_trading": bool(impulse.get("disable_trading")) and not tradeable_tickers,
    }


def _bars_per_year() -> float:
    return float(getattr(_cfg, "BARS_PER_YEAR", getattr(_cfg, "BARS_PER_DAY", 288) * 365))


def _equity_bar_returns(equity: pd.DataFrame | pd.Series | None) -> pd.Series:
    strat = _strategy_equity_series(equity)
    if strat.empty:
        return pd.Series(dtype=float)
    return strat.pct_change().dropna()


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min() * 100.0)


def _underwater_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity / peak - 1.0) * 100.0


def _rolling_sharpe(returns: pd.Series, window: int, *, annualize: float) -> pd.Series:
    if returns.empty or window < 3:
        return pd.Series(dtype=float)
    roll_mean = returns.rolling(window, min_periods=max(window // 3, 5)).mean()
    roll_std = returns.rolling(window, min_periods=max(window // 3, 5)).std(ddof=1)
    sharpe = roll_mean / roll_std.replace(0.0, np.nan) * np.sqrt(annualize)
    return sharpe.replace([np.inf, -np.inf], np.nan).dropna()


def plot_rolling_sharpe(
    equity: pd.DataFrame | pd.Series | None,
    out_path: Path,
    *,
    title: str = "OOS rolling Sharpe (annualized)",
    window_bars: int | None = None,
) -> Path:
    """Rolling annualized Sharpe from strategy bar returns."""
    rets = _equity_bar_returns(equity)
    win = window_bars or max(int(getattr(_cfg, "BARS_PER_DAY", 288)) * 30, 20)
    annualize = _bars_per_year()
    roll = _rolling_sharpe(rets, win, annualize=annualize)

    fig, ax = plt.subplots(figsize=(12, 4))
    if roll.empty:
        ax.text(0.5, 0.5, "Insufficient data for rolling Sharpe", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(roll.index, roll.values, lw=1.6, color="#2563eb", label="Rolling Sharpe")
        ax.axhline(0.0, color="black", ls=":", lw=0.8)
        ax.axhline(1.0, color="#16a34a", ls="--", lw=0.7, alpha=0.6, label="Sharpe = 1")
        ax.fill_between(roll.index, 0, roll.values, where=roll.values >= 0, alpha=0.15, color="#16a34a")
        ax.fill_between(roll.index, 0, roll.values, where=roll.values < 0, alpha=0.15, color="#dc2626")
        ax.legend(loc="best", fontsize=9)

    ax.set_title(f"{title}\n(window ≈ {win} bars)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Annualized Sharpe")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_underwater_drawdown(
    equity: pd.DataFrame | pd.Series | None,
    out_path: Path,
    *,
    title: str = "OOS underwater equity (drawdown %)",
) -> Path:
    """Underwater plot: percent drawdown from running peak."""
    strat = _strategy_equity_series(equity)
    fig, ax = plt.subplots(figsize=(12, 4))
    if strat.empty:
        ax.text(0.5, 0.5, "No equity series", ha="center", va="center", transform=ax.transAxes)
    else:
        uw = _underwater_series(strat)
        ax.fill_between(uw.index, 0, uw.values, color="#dc2626", alpha=0.45)
        ax.plot(uw.index, uw.values, lw=1.2, color="#991b1b")
        mdd = float(uw.min())
        ax.axhline(mdd, color="black", ls="--", lw=0.8, alpha=0.7, label=f"Max DD = {mdd:.1f}%")
        ax.legend(loc="lower left", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown, %")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _trade_analytics(trade_returns: np.ndarray) -> dict:
    rets = np.asarray(trade_returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return {
            "n_trades": 0,
            "win_rate": None,
            "avg_win_bps": None,
            "avg_loss_bps": None,
            "expectancy_bps": None,
            "profit_factor": None,
            "median_bps": None,
        }
    bps = rets * 10_000.0
    wins = bps[bps > 0]
    losses = bps[bps <= 0]
    win_rate = float(len(wins) / len(bps))
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    gross_win = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 1e-9 else None
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss
    return {
        "n_trades": int(len(bps)),
        "win_rate": round(win_rate, 4),
        "avg_win_bps": round(avg_win, 2) if len(wins) else None,
        "avg_loss_bps": round(avg_loss, 2) if len(losses) else None,
        "expectancy_bps": round(expectancy, 2),
        "profit_factor": round(pf, 3) if pf is not None else None,
        "median_bps": round(float(np.median(bps)), 2),
        "sum_bps": round(float(np.sum(bps)), 1),
    }


def plot_trade_pnl_distribution(
    trade_returns: np.ndarray,
    out_path: Path,
    *,
    title: str = "Trade PnL distribution (net bps)",
) -> Path:
    """Single desk chart: full trade PnL with win/loss split and expectancy."""
    analytics = _trade_analytics(trade_returns)
    bps = np.asarray(trade_returns, dtype=float) * 10_000.0
    bps = bps[np.isfinite(bps)]

    fig, ax = plt.subplots(figsize=(11, 5))
    if len(bps) < 3:
        ax.text(0.5, 0.5, "Insufficient trades", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    else:
        lo, hi = np.percentile(bps, [2, 98])
        if np.isclose(lo, hi):
            lo, hi = lo - 5.0, hi + 5.0
        wins = bps[bps > 0]
        losses = bps[bps <= 0]
        bins = np.linspace(lo, hi, 40)
        ax.hist(wins, bins=bins, density=True, alpha=0.55, color="#16a34a", label=f"Wins (n={len(wins)})")
        ax.hist(losses, bins=bins, density=True, alpha=0.55, color="#dc2626", label=f"Losses (n={len(losses)})")
        ax.axvline(0.0, color="black", ls=":", lw=1)
        exp = analytics.get("expectancy_bps")
        if exp is not None:
            ax.axvline(exp, color="#2563eb", ls="--", lw=1.4, label=f"E[trade] = {exp:.1f} bps")
        subtitle = (
            f"n={analytics['n_trades']} | WR={100 * (analytics['win_rate'] or 0):.1f}% | "
            f"PF={analytics['profit_factor']}"
        )
        ax.set_title(f"{title}\n{subtitle}")
        ax.legend(loc="best", fontsize=9)

    ax.set_xlabel("Net return per trade, bps")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_monthly_returns_heatmap(
    equity: pd.DataFrame | pd.Series | None,
    out_path: Path,
    *,
    title: str = "Monthly returns heatmap (%)",
) -> Path:
    """Calendar heatmap of month-over-month strategy returns."""
    strat = _strategy_equity_series(equity)
    fig, ax = plt.subplots(figsize=(12, 5))
    if strat.empty or len(strat) < 10:
        ax.text(0.5, 0.5, "Insufficient equity for monthly heatmap", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    else:
        monthly = strat.resample("ME").last().pct_change().dropna() * 100.0
        if monthly.empty:
            ax.text(0.5, 0.5, "No monthly buckets", ha="center", va="center", transform=ax.transAxes)
        else:
            df = pd.DataFrame({
                "year": monthly.index.year,
                "month": monthly.index.month,
                "ret": monthly.values,
            })
            pivot = df.pivot(index="year", columns="month", values="ret")
            vmax = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())), 1.0)
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(12))
            ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(y) for y in pivot.index])
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.iloc[i, j]
                    if pd.notna(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="black")
            fig.colorbar(im, ax=ax, label="Return, %", shrink=0.85)
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_walk_forward_attribution_table(
    report: dict,
    prices: pd.DataFrame | None,
    *,
    fold_anomaly_rows: list[dict] | None = None,
) -> list[dict]:
    """Per-fold PM attribution: strategy return vs benchmarks + model diagnostics."""
    symbols = list(report.get("tickers") or report.get("symbols") or [])
    folds = report.get("walk_forward_folds") or report.get("folds") or []
    anomaly_by_fold = {int(r["fold"]): r for r in (fold_anomaly_rows or []) if r.get("fold") is not None}
    rows: list[dict] = []

    for fold in folds:
        if fold.get("skipped"):
            rows.append({
                "fold": fold.get("fold"),
                "test_start": str(fold.get("test_start", ""))[:10],
                "test_end": str(fold.get("test_end", ""))[:10],
                "skipped": True,
            })
            continue

        fnum = int(fold.get("fold", 0))
        test_start = fold.get("test_start")
        test_end = fold.get("test_end")
        oos = fold.get("oos_metrics") or {}
        policy = fold.get("trading_policy") or {}
        thresh = fold.get("threshold_optimization") or {}
        anom = anomaly_by_fold.get(fnum, {})

        row: dict = {
            "fold": fnum,
            "test_start": str(test_start)[:10],
            "test_end": str(test_end)[:10],
            "skipped": False,
            "strategy_return_pct": anom.get("equity_return_pct"),
            "n_trades": anom.get("n_trades"),
            "avg_net_bps": anom.get("avg_net_bps"),
            "oos_auc": oos.get("auc") or fold.get("auc"),
            "oos_log_loss": oos.get("log_loss") or fold.get("log_loss"),
            "policy_objective": (thresh.get("cv") or {}).get("objective"),
            "buy_threshold": policy.get("buy_threshold"),
            "min_edge_bps": policy.get("min_expected_edge_bps"),
        }
        if prices is not None and test_start and test_end:
            ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
            btc = _slice_buy_hold_return(prices, "BTC", ts, te)
            if btc is not None:
                row["benchmark_btc_pct"] = round(btc, 3)
            if "ETH" in symbols:
                eth = _slice_buy_hold_return(prices, "ETH", ts, te)
                if eth is not None:
                    row["benchmark_eth_pct"] = round(eth, 3)
            if len(symbols) >= 2:
                ew = _slice_ew_return(prices, symbols, ts, te)
                if ew is not None:
                    row["benchmark_ew_pct"] = round(ew, 3)
            strat = row.get("strategy_return_pct")
            if strat is not None and row.get("benchmark_btc_pct") is not None:
                row["excess_vs_btc_pct"] = round(float(strat) - float(row["benchmark_btc_pct"]), 3)
            if strat is not None and row.get("benchmark_ew_pct") is not None:
                row["excess_vs_ew_pct"] = round(float(strat) - float(row["benchmark_ew_pct"]), 3)
        rows.append(row)
    return rows


def build_desk_risk_summary(
    equity: pd.DataFrame | pd.Series | None,
    trade_returns: np.ndarray,
    stats: dict | None,
    *,
    mc_report: dict | None = None,
) -> dict:
    """Desk risk envelope: drawdown, vol, tail, MC survival."""
    strat = _strategy_equity_series(equity)
    rets = _equity_bar_returns(equity)
    annualize = _bars_per_year()

    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_bars": int(len(strat)),
        "total_return_pct": (stats or {}).get("total_return_pct"),
        "sharpe_annualized": (stats or {}).get("sharpe") or (stats or {}).get("sharpe_bar_annualized"),
        "max_drawdown_pct": (stats or {}).get("max_drawdown_pct"),
        "active_rebalances": (stats or {}).get("active_rebalances"),
    }

    if not strat.empty:
        summary["max_drawdown_pct"] = summary.get("max_drawdown_pct") or round(_max_drawdown_pct(strat), 3)
    if not rets.empty:
        vol = float(rets.std(ddof=1) * np.sqrt(annualize) * 100.0)
        summary["realized_vol_annualized_pct"] = round(vol, 3)
        summary["skew"] = round(float(sp_stats.skew(rets, nan_policy="omit")), 3)
        summary["kurtosis_excess"] = round(float(sp_stats.kurtosis(rets, nan_policy="omit")), 3)
        summary["var_95_pct"] = round(float(np.percentile(rets, 5) * 100.0), 4)
        summary["cvar_95_pct"] = round(float(rets[rets <= np.percentile(rets, 5)].mean() * 100.0), 4)

    summary["trade_analytics"] = _trade_analytics(trade_returns)

    if mc_report:
        surv = mc_report.get("survival") or {}
        summary["monte_carlo"] = {
            "survival_rate": surv.get("survival_rate"),
            "terminal_wealth_p05": surv.get("terminal_wealth_p05"),
            "terminal_wealth_p50": surv.get("terminal_wealth_p50"),
            "terminal_wealth_p95": surv.get("terminal_wealth_p95"),
        }
    return summary


def write_desk_performance_summary(
    path: Path,
    *,
    risk: dict,
    attribution: list[dict],
    report: dict,
) -> Path:
    """Short PM markdown brief."""
    bt = report.get("backtest_walk_forward_oos") or {}
    lines = [
        "# Desk performance summary",
        "",
        f"_Generated {risk.get('generated_at', '')}_",
        "",
        "## Portfolio risk",
        "",
        f"- **OOS total return:** {risk.get('total_return_pct')}%",
        f"- **Sharpe (ann.):** {risk.get('sharpe_annualized')}",
        f"- **Max drawdown:** {risk.get('max_drawdown_pct')}%",
        f"- **Realized vol (ann.):** {risk.get('realized_vol_annualized_pct')}%",
        f"- **VaR 95% (bar):** {risk.get('var_95_pct')}%",
        "",
        "## Trade economics",
        "",
    ]
    ta = risk.get("trade_analytics") or {}
    lines.extend([
        f"- **Trades:** {ta.get('n_trades')}",
        f"- **Win rate:** {100 * (ta.get('win_rate') or 0):.1f}%",
        f"- **Expectancy:** {ta.get('expectancy_bps')} bps/trade",
        f"- **Profit factor:** {ta.get('profit_factor')}",
        "",
        "## Walk-forward attribution",
        "",
        "| Fold | Period | Strat % | BTC % | Excess vs BTC | AUC | Trades |",
        "|------|--------|---------|-------|---------------|-----|--------|",
    ])
    for row in attribution:
        if row.get("skipped"):
            continue
        lines.append(
            f"| {row.get('fold')} | {row.get('test_start')} → {row.get('test_end')} | "
            f"{row.get('strategy_return_pct')} | {row.get('benchmark_btc_pct')} | "
            f"{row.get('excess_vs_btc_pct')} | {row.get('oos_auc')} | {row.get('n_trades')} |"
        )
    mc = risk.get("monte_carlo") or {}
    if mc:
        lines.extend([
            "",
            "## Monte Carlo stress",
            "",
            f"- **Survival rate:** {mc.get('survival_rate')}",
            f"- **Terminal wealth p50:** {mc.get('terminal_wealth_p50')}",
        ])
    desk_charts = [
        ("desk_rolling_sharpe.png", "Rolling Sharpe"),
        ("desk_underwater_drawdown.png", "Underwater drawdown"),
        ("desk_trade_pnl_distribution.png", "Trade PnL distribution"),
        ("desk_monthly_returns_heatmap.png", "Monthly returns heatmap"),
        ("desk_wf_fold_attribution.png", "Walk-forward fold attribution"),
        ("fusion_equity_vs_benchmarks.png", "Strategy vs benchmarks"),
        ("fusion_capital_over_time.png", "Capital over time"),
        ("equity_curve_oos.png", "OOS equity curve"),
        ("backtest_return_density.png", "Return density"),
    ]
    chart_lines: list[str] = []
    for fname, title in desk_charts:
        p = OUT_DIR / "plots" / fname
        if p.is_file():
            chart_lines.extend([f"### {title}", "", f"![{title}](/output/plots/{fname})", ""])
    if chart_lines:
        lines.extend(["", "## Charts", ""] + chart_lines)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_quant_desk_plots(
    equity: pd.DataFrame | pd.Series | None,
    trade_returns: np.ndarray,
    plots_dir: Path,
) -> dict[str, str]:
    """Desk-standard plots replacing retail TP/SL and rolling-excess charts."""
    paths: dict[str, str] = {}
    paths["rolling_sharpe"] = str(plot_rolling_sharpe(
        equity, plots_dir / "desk_rolling_sharpe.png",
        title="Fusion OOS: rolling Sharpe (annualized)",
    ))
    paths["underwater_drawdown"] = str(plot_underwater_drawdown(
        equity, plots_dir / "desk_underwater_drawdown.png",
        title="Fusion OOS: underwater drawdown",
    ))
    paths["trade_pnl_distribution"] = str(plot_trade_pnl_distribution(
        trade_returns, plots_dir / "desk_trade_pnl_distribution.png",
    ))
    paths["monthly_returns_heatmap"] = str(plot_monthly_returns_heatmap(
        equity, plots_dir / "desk_monthly_returns_heatmap.png",
        title="Fusion OOS: monthly returns heatmap",
    ))
    return paths


def write_quant_desk_bundle(
    *,
    equity: pd.DataFrame | pd.Series | None,
    prices: pd.DataFrame,
    symbols: list[str],
    trade_returns: np.ndarray,
    report: dict,
    plots_dir: Path | None = None,
    fold_signals: pd.DataFrame | None = None,
    fold_map: pd.DataFrame | None = None,
    commission_bps: float = 10.0,
    slippage_bps: float = 0.0,
    mc_report: dict | None = None,
) -> dict[str, str]:
    """Write quant desk plots + JSON/MD artifacts."""
    out_plots = plots_dir or (OUT_DIR / "plots")
    paths = write_quant_desk_plots(equity, trade_returns, out_plots)

    anomaly_rows = compute_fold_anomaly_table(
        equity, fold_signals, fold_map,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    attribution = build_walk_forward_attribution_table(report, prices, fold_anomaly_rows=anomaly_rows)
    attr_plot = plot_fold_benchmark_comparison(
        attribution,
        out_plots / "desk_wf_fold_attribution.png",
        title="Walk-forward fold attribution: strategy vs benchmarks",
    )
    paths["wf_fold_attribution"] = str(attr_plot)

    risk = build_desk_risk_summary(equity, trade_returns, report.get("backtest_walk_forward_oos"), mc_report=mc_report)
    trade_analytics = risk.get("trade_analytics") or _trade_analytics(trade_returns)
    trade_json = {"trade_analytics": trade_analytics}

    DESK_RISK_JSON.parent.mkdir(parents=True, exist_ok=True)
    DESK_RISK_JSON.write_text(json.dumps(risk, indent=2, default=str), encoding="utf-8")
    DESK_ATTRIBUTION_JSON.write_text(json.dumps({"folds": attribution}, indent=2, default=str), encoding="utf-8")
    DESK_TRADE_JSON.write_text(json.dumps(trade_json, indent=2, default=str), encoding="utf-8")
    write_desk_performance_summary(
        DESK_PERF_MD,
        risk={**risk, "trade_analytics": trade_analytics},
        attribution=attribution,
        report=report,
    )
    write_fold_benchmark_markdown(
        attribution,
        OUT_DIR / "desk_wf_fold_attribution.md",
        summary={"n_folds": len([r for r in attribution if not r.get("skipped")])},
    )

    paths["desk_risk_summary"] = str(DESK_RISK_JSON)
    paths["desk_walk_forward_attribution"] = str(DESK_ATTRIBUTION_JSON)
    paths["desk_trade_analytics"] = str(DESK_TRADE_JSON)
    paths["desk_performance_summary"] = str(DESK_PERF_MD)
    return paths
