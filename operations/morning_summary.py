"""Write a one-page morning brief after pipeline completion."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import OUT_DIR

MORNING_SUMMARY = OUT_DIR / "morning_summary.md"


def write_morning_summary() -> Path:
    fusion_path = OUT_DIR / "fusion_pipeline_report.json"
    mc_path = OUT_DIR / "monte_carlo_report.json"
    desk_risk_path = OUT_DIR / "desk_risk_summary.json"

    fusion = {}
    mc = {}
    desk = {}
    if fusion_path.is_file():
        try:
            fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if mc_path.is_file():
        try:
            mc = json.loads(mc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if desk_risk_path.is_file():
        try:
            desk = json.loads(desk_risk_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    bt = fusion.get("backtest_walk_forward_oos") or {}
    go = fusion.get("go_no_go") or {}
    decile = fusion.get("decile_audit") or {}
    impulse = (fusion.get("impulse_optimization") or {}).get("best") or {}
    surv = (mc.get("survival") or {}) if mc else {}
    ta = desk.get("trade_analytics") or {}

    lines = [
        "# Morning pipeline summary",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Go / no-go",
        "",
        f"- **Tradeable:** {go.get('tradeable', 'n/a')}",
        f"- **Tradeable tickers:** {', '.join(go.get('tradeable_tickers') or []) or '—'}",
        f"- **Disabled tickers:** {', '.join(go.get('disabled_tickers') or []) or '—'}",
        f"- **Disable trading:** {go.get('disable_trading', impulse.get('disable_trading'))}",
        f"- **Reasons:** {', '.join(go.get('reasons') or []) or '—'}",
        "",
        "## OOS performance",
        "",
        f"- **Total return:** {bt.get('total_return_pct')}%",
        f"- **Sharpe:** {bt.get('sharpe') or bt.get('sharpe_bar_annualized')}",
        f"- **Max drawdown:** {bt.get('max_drawdown_pct')}%",
        f"- **OOS AUC:** {fusion.get('oos_auc')}",
        f"- **Folds:** {len(fusion.get('walk_forward_folds') or [])}",
        "",
        "## Decile audit",
        "",
        f"- **Top decile net bps:** {decile.get('top_decile_net_bps')}",
        f"- **Monotonic:** {decile.get('monotonic')}",
        "",
    ]
    by_t = decile.get("by_ticker") or {}
    if by_t:
        lines.append("### Per ticker")
        lines.append("")
        for sym, aud in sorted(by_t.items()):
            side = aud.get("active_side") or aud.get("side") or "long"
            lines.append(
                f"- **{sym}:** net={aud.get('top_decile_net_bps')} bps | "
                f"side={side} | tradeable={aud.get('tradeable')} | "
                f"gross={aud.get('top_decile_gross_bps')}"
            )
        lines.append("")
    pt_bt = fusion.get("per_ticker_backtest") or {}
    pt_bt_gated = fusion.get("per_ticker_backtest_gated") or {}
    if pt_bt:
        lines.append("## Per-ticker backtest (OOS)")
        lines.append("")
        lines.append("_Ungated = solo symbol; gated = live book tradeable set only._")
        lines.append("")
        for sym, stats in sorted(pt_bt.items()):
            gated = pt_bt_gated.get(sym) or {}
            if stats.get("skipped") and gated.get("skipped"):
                lines.append(f"- **{sym}:** skipped ({stats.get('reason', 'n/a')})")
                continue
            lines.append(
                f"- **{sym}:** ungated return={stats.get('total_return_pct')}% | "
                f"gated return={gated.get('total_return_pct')}% | "
                f"trades={stats.get('n_trades', stats.get('n_signals'))} | "
                f"max_dd={stats.get('max_drawdown_pct')}%"
            )
        lines.append("")
    lines.extend([
        "## Trade economics",
        "",
        f"- **Trades:** {ta.get('n_trades')}",
        f"- **Win rate:** {ta.get('win_rate')}",
        f"- **Expectancy bps:** {ta.get('expectancy_bps')}",
        "",
        "## Monte Carlo",
        "",
        f"- **Survival rate:** {surv.get('survival_rate')}",
        f"- **Terminal wealth p50:** {surv.get('terminal_wealth_p50')}",
        "",
        "## Artifacts",
        "",
        f"- Report: `{fusion_path}`",
        f"- Desk brief: `{OUT_DIR / 'desk_performance_summary.md'}`",
        f"- Dashboard manifest: `web/public/manifest.json`",
        "",
    ])
    MORNING_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    MORNING_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    return MORNING_SUMMARY
