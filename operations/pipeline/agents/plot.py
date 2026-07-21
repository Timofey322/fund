"""Plot role: ret_z explainer + system diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from operations.pipeline.agents.base import Agent
from operations.checkpoint import PipelineCheckpoint, PipelineContext
from reporting.plots import STANDARD_EQUITY_PLOT, write_standard_oos_plots
from strategy.pipeline import REPORT_PATH, regenerate_fusion_oos_plots
from reporting.diagnostics import plot_system_overview


def _fusion_disable_trading() -> bool:
    if not REPORT_PATH.is_file():
        return False
    try:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    best = (report.get("impulse_optimization") or {}).get("best") or {}
    return bool(best.get("disable_trading"))


class PlotAgent(Agent):
    name = "plot"

    def run(self, ctx: PipelineContext, ckpt: PipelineCheckpoint | None = None) -> PipelineContext:
        if ckpt:
            cached = ckpt.load(ctx.run_id, self.name)
            if cached:
                ctx.artifacts["plot"] = cached
                return ctx

        all_paths: list[str] = []
        for sym in ctx.tickers or ["BTC"]:
            try:
                all_paths.extend(str(p) for p in plot_system_overview(sym))
            except Exception as exc:
                print(f"    plot: {sym} diagnostics skipped — {exc}", flush=True)
        symbol = ctx.tickers[0] if ctx.tickers else "BTC"
        paths = all_paths
        oos_plots: dict[str, str] = {}
        if REPORT_PATH.is_file():
            if _fusion_disable_trading():
                oos_plots = write_standard_oos_plots(
                    {"equity": None, "stats": {"total_return_pct": 0.0, "sharpe": 0.0}},
                    [],
                    title_equity="Fusion OOS equity (no-trade)",
                    title_density="Fusion OOS return density (no-trade)",
                )
            else:
                try:
                    oos_plots = regenerate_fusion_oos_plots(ctx.tickers)
                except Exception as exc:
                    print(f"    plot: fusion OOS regen skipped: {exc}", flush=True)
        payload = {
            "symbol": symbol,
            "plots": [str(p) for p in paths],
            "oos_plots": oos_plots,
        }
        ctx.artifacts["plot"] = payload
        if ckpt:
            ckpt.save(ctx.run_id, self.name, payload)
        return ctx
