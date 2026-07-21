"""Export dashboard manifest from output/ artifacts for the web frontend."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "fund") not in sys.path:
    sys.path.insert(0, str(ROOT / "fund"))

from config import OUT_DIR

WEB_MANIFEST = ROOT / "web" / "public" / "manifest.json"
PLOTS_DIR = OUT_DIR / "plots"

PLOT_TITLES: dict[str, str] = {
    "desk_rolling_sharpe": "Rolling Sharpe (annualized)",
    "desk_underwater_drawdown": "Underwater drawdown",
    "desk_trade_pnl_distribution": "Trade PnL & expectancy",
    "desk_monthly_returns_heatmap": "Monthly returns heatmap",
    "desk_wf_fold_attribution": "Walk-forward fold attribution",
    "fusion_equity_vs_benchmarks": "Strategy vs benchmarks",
    "fusion_capital_over_time": "Capital over time",
    "equity_curve_oos": "OOS equity curve",
    "backtest_return_density": "Return density (legacy)",
    "fusion_fold_anomaly": "Fold anomaly audit (research)",
}


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _plot_entries() -> list[dict]:
    if not PLOTS_DIR.is_dir():
        return []
    desk_first = {
        "desk_rolling_sharpe",
        "desk_underwater_drawdown",
        "desk_trade_pnl_distribution",
        "desk_monthly_returns_heatmap",
        "desk_wf_fold_attribution",
        "fusion_equity_vs_benchmarks",
        "fusion_capital_over_time",
        "equity_curve_oos",
    }

    def sort_key(p: Path) -> tuple[int, float]:
        priority = 0 if p.stem in desk_first else 1
        return (priority, -p.stat().st_mtime)

    entries = []
    for p in sorted(PLOTS_DIR.glob("*.png"), key=sort_key):
        entries.append({
            "id": p.stem,
            "title": PLOT_TITLES.get(p.stem, p.stem.replace("_", " ").title()),
            "path": f"/output/plots/{p.name}",
            "mtime": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
        })
    return entries


def _report_links() -> list[dict]:
    links = [
        ("desk_performance_summary.md", "Desk PM brief"),
        ("desk_risk_summary.json", "Risk envelope"),
        ("desk_walk_forward_attribution.json", "WF attribution"),
        ("desk_trade_analytics.json", "Trade economics"),
        ("desk_wf_fold_attribution.md", "WF attribution table"),
        ("fusion_pipeline_report.json", "Fusion pipeline report"),
        ("monte_carlo_report.json", "Monte Carlo stress"),
        ("quantstats/fusion_quantstats.html", "QuantStats tearsheet"),
        ("fusion_fold_anomaly.json", "Fold anomaly audit (research)"),
        ("morning_summary.md", "Morning summary"),
        ("research/index.json", "Instrument research index"),
    ]
    out = []
    for rel, title in links:
        p = OUT_DIR / rel
        if p.is_file():
            out.append({"title": title, "path": f"/output/{rel.replace(chr(92), '/')}"})
    research_dir = OUT_DIR / "research"
    if research_dir.is_dir():
        for p in sorted(research_dir.glob("*.html")):
            sym = p.stem.upper()
            if sym == "INDEX":
                continue
            out.append({
                "title": f"Research · {sym}",
                "path": f"/output/research/{p.name}",
            })
    return out


def _instrument_entries(fusion: dict) -> list[dict]:
    index_path = OUT_DIR / "research" / "index.json"
    if index_path.is_file():
        try:
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            return idx.get("instruments") or []
        except (OSError, json.JSONDecodeError):
            pass
    tickers = fusion.get("tickers") or fusion.get("symbols") or []
    try:
        from reporting.ticker_research import build_ticker_research

        return [build_ticker_research(str(t), fusion) for t in tickers]
    except Exception:
        return []


def _export_ml_equity_chart() -> str | None:
    """Export fusion portfolio equity JSON for the web chart."""
    cache = OUT_DIR / "cache" / "fusion_bt_equity.parquet"
    if not cache.is_file():
        return None
    try:
        import pandas as pd
        from rule.equity_export import build_equity_chart_payload, write_equity_json

        eq_df = pd.read_parquet(cache)
        bench_df = None
        if "benchmark" in eq_df.columns:
            bench_df = eq_df[["bar_time" if "bar_time" in eq_df.columns else eq_df.columns[0], "benchmark"]].rename(
                columns={"benchmark": "value"}
            )
        payload = build_equity_chart_payload(
            eq_df, bench_df, ticker=None, label="ml_fusion_portfolio",
        )
        out = OUT_DIR / "ml" / "equity" / "portfolio.json"
        write_equity_json(payload, out)
        return "/output/ml/equity/portfolio.json"
    except Exception:
        return None


def build_manifest() -> dict:
    fusion = _read_json(OUT_DIR / "fusion_pipeline_report.json") or {}
    mc = _read_json(OUT_DIR / "monte_carlo_report.json") or {}
    desk_risk = _read_json(OUT_DIR / "desk_risk_summary.json") or {}
    desk_trade = _read_json(OUT_DIR / "desk_trade_analytics.json") or {}
    bt = fusion.get("backtest_walk_forward_oos") or {}
    surv = (mc.get("survival") or {}) if mc else {}
    folds = fusion.get("walk_forward_folds") or []
    ta = desk_trade.get("trade_analytics") or desk_risk.get("trade_analytics") or {}
    equity_chart = _export_ml_equity_chart()
    bench_ret = None
    excess_ret = None
    if equity_chart:
        eq_payload = _read_json(OUT_DIR / "ml" / "equity" / "portfolio.json")
        ret = (eq_payload or {}).get("return_pct") or {}
        bench_ret = ret.get("benchmark")
        excess_ret = ret.get("excess")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "equity_chart": equity_chart,
        "metrics": {
            "walk_forward_mode": fusion.get("walk_forward_mode") or fusion.get("mode"),
            "wf_test_months": fusion.get("wf_test_months"),
            "wf_train_days": fusion.get("wf_train_days"),
            "n_folds": len(folds),
            "oos_auc": fusion.get("oos_auc"),
            "total_return_pct": desk_risk.get("total_return_pct") or bt.get("total_return_pct"),
            "benchmark_return_pct": bench_ret,
            "excess_return_pct": excess_ret,
            "max_drawdown_pct": desk_risk.get("max_drawdown_pct") or bt.get("max_drawdown_pct"),
            "expectancy_bps": ta.get("expectancy_bps"),
            "win_rate": ta.get("win_rate"),
            "active_rebalances": desk_risk.get("active_rebalances") or bt.get("active_rebalances"),
            "survival_rate": surv.get("survival_rate"),
            "terminal_wealth_p50": surv.get("terminal_wealth_p50"),
            "tickers": fusion.get("tickers") or fusion.get("symbols"),
            "go_no_go": fusion.get("go_no_go") or {},
        },
        "plots": _plot_entries(),
        "reports": _report_links(),
        "instruments": _instrument_entries(fusion),
    }


def export_manifest(path: Path | None = None) -> Path:
    out = path or WEB_MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest()
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = export_manifest()
    print(f"Dashboard manifest -> {p}")
