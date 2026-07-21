"""Rule strategy pipeline: factor scores -> backtest -> report (no ML)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from config import apply_bar_timeframe
import config as fund_config
from data_platform.bars import load_closes
from rule.backtest import (
    equal_weight_universe_return_pct,
    run_per_ticker_backtests,
    run_portfolio_backtest,
)
from rule.config import (
    RULE_BAR_TIMEFRAME,
    RULE_EQUITY_CACHE,
    RULE_EXHAUSTION_MIN,
    RULE_HISTORY_DAYS,
    RULE_MONTE_CARLO_PATH,
    RULE_NAME,
    RULE_NEUTRAL_SCORE,
    RULE_NW_BANDWIDTH,
    RULE_NW_BAND_MULT,
    RULE_NW_LOOKBACK,
    RULE_NW_TOUCH_MAX,
    RULE_HMM_GROWTH_MIN,
    RULE_PARTIAL_SELL_FRAC,
    RULE_REBALANCE_FREQ,
    RULE_REPORT_PATH,
    RULE_SCORE_ENTER,
    RULE_SCORE_EXIT,
    RULE_SUMMARY_PATH,
    RULE_BUY_ONLY,
)
from rule.nw_hmm_signals import build_nw_hmm_buy_signal_frame
from rule.web_export import export_rule_web
from simulation.trade_survival import survival_simulation


def _build_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """NW kernel fair value + HMM regime — buy-only entries."""
    sig = build_nw_hmm_buy_signal_frame(prices)
    if sig.empty:
        return sig
    sig["buy_threshold"] = float(RULE_SCORE_ENTER)
    sig["hold_threshold"] = float(RULE_SCORE_EXIT)
    return sig


def run_rule_pipeline(tickers: list[str]) -> dict:
    """
    End-to-end rule strategy: load bars -> HMM signals -> equal-weight backtest -> artifacts.
    """
    apply_bar_timeframe(RULE_BAR_TIMEFRAME)
    fund_config.REBALANCE_FREQ = RULE_REBALANCE_FREQ
    fund_config.SCORE_ENTER = float(RULE_SCORE_ENTER)
    fund_config.SCORE_EXIT = float(RULE_SCORE_EXIT)
    tickers = [str(t).upper() for t in tickers]
    print(f"rule: loading closes for {len(tickers)} tickers ({RULE_HISTORY_DAYS}d)...", flush=True)
    prices = load_closes(tickers, RULE_BAR_TIMEFRAME)
    print(f"rule: timeframe {RULE_BAR_TIMEFRAME}", flush=True)
    if prices.empty:
        raise RuntimeError("rule: no price data -- run data stage first")

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"rule: warning -- missing columns: {missing}", flush=True)

    signals = _build_signals(prices)
    if signals.empty:
        raise RuntimeError("rule: signal frame empty")

    print(f"rule: portfolio backtest (equal weight) | {len(signals):,} signal rows", flush=True)
    bt = run_portfolio_backtest(prices, signals, [c for c in prices.columns])

    print("rule: per-ticker backtests...", flush=True)
    per_ticker_bt, per_ticker_equity = run_per_ticker_backtests(
        prices, signals, tickers, include_equity=True,
    )

    stats = bt.get("stats") or {}
    ps, pe = stats.get("period_start"), stats.get("period_end")
    ew_bh = equal_weight_universe_return_pct(prices, ps, pe)
    ret = stats.get("total_return_pct")
    if ret is not None and ew_bh is not None:
        stats["benchmark_return_pct"] = ew_bh
        stats["excess_return_pct"] = round(float(ret) - float(ew_bh), 2)
        stats["benchmark_label"] = "equal_weight_universe_bh"
    equity = bt.get("equity")
    portfolio_benchmark = bt.get("benchmark")
    equity_curves: dict[str, str] = {}
    plots: list[dict] = []
    chart_data = "/output/rule/charts.json"
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        RULE_EQUITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        equity.to_parquet(RULE_EQUITY_CACHE, index=True)

    mc_report: dict = {"skipped": True, "reason": "no_equity"}
    if isinstance(equity, pd.DataFrame) and not equity.empty and "value" in equity.columns:
        from rule.plots import _equity_series

        eq_series = _equity_series(equity)
        if len(eq_series) > 20:
            from config import MONTE_CARLO_BLOCK_BARS, MONTE_CARLO_MAX_DD_THRESHOLD, MONTE_CARLO_PATHS

            mc = survival_simulation(
                eq_series,
                n_paths=int(MONTE_CARLO_PATHS),
                block_bars=int(MONTE_CARLO_BLOCK_BARS),
                max_dd_threshold=float(MONTE_CARLO_MAX_DD_THRESHOLD),
            )
            mc_report = {"n_paths": MONTE_CARLO_PATHS, "equity_bars": len(eq_series), "survival": mc}
            RULE_MONTE_CARLO_PATH.write_text(json.dumps(mc_report, indent=2, default=str), encoding="utf-8")

    per_ticker_sig: list[dict] = []
    if "ticker" in signals.columns:
        for sym in sorted(signals["ticker"].unique()):
            sub = signals[signals["ticker"] == sym]
            side_col = sub.get("signal_side", pd.Series(dtype=str))
            per_ticker_sig.append({
                "ticker": sym,
                "n_signals": int(len(sub)),
                "mean_score": round(float(sub["score"].mean()), 2) if len(sub) else None,
                "mean_nw_dev_below": round(float(sub["nw_dev_below"].mean()), 4)
                if "nw_dev_below" in sub.columns
                else None,
                "pct_touch_lower": round(float(sub["nw_touches_lower"].mean()) * 100, 1)
                if "nw_touches_lower" in sub.columns
                else None,
                "pct_enter": round(float((sub["score"] >= RULE_SCORE_ENTER).mean()) * 100, 1) if len(sub) else None,
                "pct_buy_nw_hmm": round(float((side_col == "buy_nw_hmm").mean()) * 100, 1) if len(sub) else None,
            })

    if isinstance(equity, pd.DataFrame) and not equity.empty:
        report_stub = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "backtest": stats,
            "per_ticker_backtest": per_ticker_bt,
            "per_ticker_signals": per_ticker_sig,
        }
        try:
            from rule.chart_export import chart_manifest_plots, export_rule_charts

            chart_specs = export_rule_charts(report_stub, equity, prices=prices)
            plots = chart_manifest_plots(chart_specs)
        except Exception as exc:
            print(f"rule: chart export warning -- {exc}", flush=True)

        try:
            from rule.equity_export import (
                _resolve_equity_series,
                equal_weight_benchmark_series,
                export_rule_equity_curves,
            )

            eq_series = _resolve_equity_series(equity)
            chart_benchmark = equal_weight_benchmark_series(prices)
            if not eq_series.empty:
                chart_benchmark = chart_benchmark.reindex(eq_series.index).ffill()
            equity_curves = export_rule_equity_curves(
                equity,
                chart_benchmark,
                per_ticker_equity,
                benchmark_ticker="EW universe",
                prices=prices,
            )
        except Exception as exc:
            print(f"rule: equity export warning -- {exc}", flush=True)

    report = {
        "strategy": RULE_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "bar_timeframe": RULE_BAR_TIMEFRAME,
        "signal_rows": int(len(signals)),
        "backtest": {
            "total_return_pct": stats.get("total_return_pct"),
            "benchmark_return_pct": stats.get("benchmark_return_pct"),
            "excess_return_pct": stats.get("excess_return_pct"),
            "sharpe": stats.get("sharpe"),
            "max_drawdown_pct": stats.get("max_drawdown_pct"),
            "signal_exit_count": stats.get("signal_exit_count"),
            "stop_loss_exit_count": stats.get("stop_loss_exit_count"),
            "avg_exposure_pct": stats.get("avg_exposure_pct"),
            "avg_entry_discount_vs_twap_pct": stats.get("avg_entry_discount_vs_twap_pct"),
            "buy_only": stats.get("buy_only"),
            "period_start": str(stats.get("period_start")) if stats.get("period_start") else None,
            "period_end": str(stats.get("period_end")) if stats.get("period_end") else None,
            "allocation": "equal_weight",
        },
        "parameters": {
            "score_enter": RULE_SCORE_ENTER,
            "score_exit": RULE_SCORE_EXIT,
            "neutral_score": RULE_NEUTRAL_SCORE,
            "exhaustion_min": RULE_EXHAUSTION_MIN,
            "history_days": RULE_HISTORY_DAYS,
            "buy_only": RULE_BUY_ONLY,
            "partial_sell_frac": RULE_PARTIAL_SELL_FRAC,
            "nw_lookback": RULE_NW_LOOKBACK,
            "nw_bandwidth": RULE_NW_BANDWIDTH,
            "nw_band_mult": RULE_NW_BAND_MULT,
            "nw_touch_max": RULE_NW_TOUCH_MAX,
            "hmm_growth_min": RULE_HMM_GROWTH_MIN,
            "entry_rule": "touch NW lower envelope + HMM P(growth) high",
            "equal_weight": True,
            "vol_targeting": False,
        },
        "per_ticker_signals": per_ticker_sig,
        "per_ticker_backtest": per_ticker_bt,
        "monte_carlo": mc_report,
        "plots": plots,
        "chart_data": chart_data,
        "equity_curves": equity_curves,
        "artifacts": {
            "report": str(RULE_REPORT_PATH),
            "equity": str(RULE_EQUITY_CACHE) if RULE_EQUITY_CACHE.is_file() else None,
            "monte_carlo": str(RULE_MONTE_CARLO_PATH) if RULE_MONTE_CARLO_PATH.is_file() else None,
        },
    }

    RULE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULE_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_summary(report)
    web_path = export_rule_web(report)
    print(
        f"rule: return={stats.get('total_return_pct')}% sharpe={stats.get('sharpe')} "
        f"-> {RULE_REPORT_PATH.name} | web {web_path.name}",
        flush=True,
    )
    return report


def _write_summary(report: dict) -> None:
    bt = report.get("backtest") or {}
    mc = (report.get("monte_carlo") or {}).get("survival") or {}
    params = report.get("parameters") or {}
    per_bt = report.get("per_ticker_backtest") or {}

    lines = [
        "# HMM + NW kernel buy strategy summary",
        "",
        f"_Generated {report.get('generated_at')}_",
        "",
        "## Logic",
        "",
        "- **Fair value:** Nadaraya-Watson (Gaussian kernel regression) on daily closes",
        "- **Buy:** касание нижней волны NW-envelope + HMM ожидает рост (mean-reversion bounce)",
        "- **Sell:** никогда — только покупки, позиция держится",
        f"- **History:** {params.get('history_days')} calendar days (~20y target)",
        "- **Allocation:** equal weight; invest free USD cash on entry",
        "",
        "## Portfolio backtest",
        "",
        f"- **Return:** {bt.get('total_return_pct')}%",
        f"- **Benchmark (EW universe B&H):** {bt.get('benchmark_return_pct')}%",
        f"- **Alpha (excess):** {bt.get('excess_return_pct')}%",
        f"- **Sharpe:** {bt.get('sharpe')}",
        f"- **Max DD:** {bt.get('max_drawdown_pct')}%",
        f"- **Avg entry discount vs TWAP:** {bt.get('avg_entry_discount_vs_twap_pct')}%",
        f"- **Signal exits (must be 0):** {bt.get('signal_exit_count')}",
        "",
        "## Per instrument",
        "",
        "| Ticker | Return % | B&H % | Excess % | Sharpe |",
        "|--------|----------|-------|----------|--------|",
    ]
    for sym in sorted(per_bt.keys()):
        s = per_bt[sym]
        if s.get("skipped"):
            continue
        lines.append(
            f"| {sym} | {s.get('total_return_pct')} | {s.get('benchmark_return_pct')} | "
            f"{s.get('excess_return_pct')} | {s.get('sharpe')} |"
        )
    lines.extend([
        "",
        "## Monte Carlo",
        "",
        f"- **Survival:** {mc.get('survival_rate')}",
        f"- **P(terminal loss):** {mc.get('prob_terminal_loss')}",
    ])
    plots = report.get("plots") or {}
    if plots:
        lines.extend(["", "## Charts", ""])
        for entry in sorted(plots, key=lambda p: p.get("id", "")):
            cid = entry.get("id", "")
            title = entry.get("title") or cid.replace("_", " ").title()
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"<!-- chart:{cid} -->")
            lines.append("")
    lines.append("")
    RULE_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
