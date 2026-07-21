"""Export rule strategy results for the web frontend."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB_PUBLIC = ROOT / "web" / "public"
WEB_RULE_MANIFEST = WEB_PUBLIC / "rule_manifest.json"

from config import DISPLAY_NAMES, OUT_DIR, TRADFI_DISPLAY_NAMES  # noqa: E402
from rule.config import RULE_WEB_DIR, RULE_WEB_MANIFEST  # noqa: E402


def _display_name(ticker: str) -> str:
    return TRADFI_DISPLAY_NAMES.get(ticker) or DISPLAY_NAMES.get(ticker) or ticker


RULE_PLOT_TITLES: dict[str, str] = {
    "rule_equity_curve": "Portfolio equity curve",
    "rule_return_density": "Return distribution (histogram + KDE)",
    "rule_underwater_drawdown": "Underwater drawdown",
    "rule_rolling_sharpe": "Rolling Sharpe (annualized)",
    "rule_monthly_returns": "Monthly returns heatmap",
    "rule_monte_carlo_terminal": "Monte Carlo terminal wealth",
    "rule_per_ticker_returns": "Strategy vs buy & hold by ticker",
    "rule_per_ticker_excess": "Excess return by ticker",
    "rule_per_ticker_sharpe": "Sharpe by ticker",
    "rule_signal_distribution": "Signal mix by instrument",
}


def _web_plot_path(path: str) -> str:
    if path.startswith("/output/"):
        return path
    try:
        rel = Path(path).resolve().relative_to(OUT_DIR.resolve())
        return f"/output/{rel.as_posix()}"
    except (ValueError, OSError):
        name = Path(path).name
        if name.startswith("rule_"):
            return f"/output/plots/rule/{name}"
        return f"/output/plots/{name}"


def _verdict_from_return(ret: float | None) -> str:
    if ret is None:
        return "SKIP"
    if ret > 0.5:
        return "TRADE"
    if ret > -0.5:
        return "WATCH"
    return "SKIP"


def build_rule_web_manifest(report: dict) -> dict:
    """Build web manifest from rule pipeline report."""
    bt = report.get("backtest") or {}
    params = report.get("parameters") or {}
    per_bt = report.get("per_ticker_backtest") or {}
    per_sig = {r["ticker"]: r for r in (report.get("per_ticker_signals") or [])}
    plots = report.get("plots") or {}
    plot_entries: list[dict] = []
    if isinstance(plots, list):
        plot_entries = plots
    else:
        plot_entries = [
            {
                "id": k,
                "title": RULE_PLOT_TITLES.get(k, k.replace("_", " ").title()),
                "path": _web_plot_path(v) if v else "",
            }
            for k, v in sorted(plots.items())
            if k
        ]

    instruments = []
    for sym in sorted(report.get("tickers") or []):
        sym_u = str(sym).upper()
        stats = per_bt.get(sym_u) or {}
        sig = per_sig.get(sym_u) or {}
        ret = stats.get("total_return_pct")
        sym_equity_path = (report.get("equity_curves") or {}).get(sym_u)
        instruments.append(
            {
                "ticker": sym_u,
                "display_name": _display_name(sym_u),
                "asset_class": "tradfi",
                "verdict": _verdict_from_return(ret if isinstance(ret, (int, float)) else None),
                "tradeable": bool(isinstance(ret, (int, float)) and ret > 0),
                "equity_chart": sym_equity_path,
                "backtest": {
                    "total_return_pct": stats.get("total_return_pct"),
                    "sharpe": stats.get("sharpe"),
                    "max_drawdown_pct": stats.get("max_drawdown_pct"),
                    "benchmark_return_pct": stats.get("benchmark_return_pct"),
                    "excess_return_pct": stats.get("excess_return_pct"),
                    "n_signals": stats.get("n_signals"),
                    "avg_exposure_pct": stats.get("avg_exposure_pct"),
                },
                "signals": {
                    "pct_dump_buy": sig.get("pct_dump_buy"),
                    "pct_rally_sell": sig.get("pct_rally_sell"),
                    "pct_enter": sig.get("pct_enter"),
                    "mean_impulse_thr_pct": sig.get("mean_impulse_thr_pct"),
                },
                "research_json": f"/output/rule/{sym_u}.json",
            }
        )

    return {
        "strategy": report.get("strategy"),
        "generated_at": report.get("generated_at") or datetime.now(timezone.utc).isoformat(),
        "description": (
            "Только покупки. Вход: касание нижней волны Nadaraya-Watson envelope "
            "на низкой цене + HMM ожидает рост. Продаж нет — держим позицию."
        ),
        "metrics": {
            "total_return_pct": bt.get("total_return_pct"),
            "benchmark_return_pct": bt.get("benchmark_return_pct"),
            "excess_return_pct": bt.get("excess_return_pct"),
            "sharpe": bt.get("sharpe"),
            "max_drawdown_pct": bt.get("max_drawdown_pct"),
            "avg_exposure_pct": bt.get("avg_exposure_pct"),
            "period_start": bt.get("period_start"),
            "period_end": bt.get("period_end"),
            "tickers": report.get("tickers"),
            "allocation": "equal_weight",
            "survival_rate": (report.get("monte_carlo") or {}).get("survival", {}).get("survival_rate"),
        },
        "parameters": params,
        "instruments": instruments,
        "equity_chart": (report.get("equity_curves") or {}).get("portfolio"),
        "chart_data": report.get("chart_data") or "/output/rule/charts.json",
        "plots": plot_entries,
        "reports": [
            {"title": "Rule pipeline report", "path": "/output/rule_pipeline_report.json"},
            {"title": "Rule summary", "path": "/output/rule_summary.md"},
        ],
    }


def export_rule_web(report: dict) -> Path:
    """Write rule manifest to output/ and web/public/."""
    RULE_WEB_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_rule_web_manifest(report)

    for inst in manifest.get("instruments") or []:
        sym = inst["ticker"]
        detail = {**inst, "strategy": manifest.get("strategy"), "parameters": manifest.get("parameters")}
        chart_path = RULE_WEB_DIR / "equity" / f"{sym}.json"
        if chart_path.is_file():
            try:
                chart = json.loads(chart_path.read_text(encoding="utf-8"))
                detail["beta"] = chart.get("beta")
                detail["alpha_annualized_pct"] = chart.get("alpha_annualized_pct")
                detail["correlation"] = chart.get("correlation")
                detail["benchmark_ticker"] = chart.get("benchmark_ticker")
            except (OSError, json.JSONDecodeError):
                pass
        (RULE_WEB_DIR / f"{sym}.json").write_text(
            json.dumps(detail, indent=2, default=str), encoding="utf-8"
        )

    RULE_WEB_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RULE_WEB_MANIFEST, WEB_RULE_MANIFEST)
    return WEB_RULE_MANIFEST
