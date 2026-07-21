"""Pipeline agents for the rule (non-ML) strategy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from operations.checkpoint import PipelineCheckpoint, PipelineContext
from operations.pipeline.agents.base import Agent
from operations.pipeline.agents.data import DataAgent
from rule.config import RULE_REPORT_PATH
from rule.pipeline import run_rule_pipeline


class RuleStrategyAgent(Agent):
    """Factor-score backtest — replaces FusionAgent for rule pipeline."""

    name = "rule"

    def run(self, ctx: PipelineContext, ckpt: PipelineCheckpoint | None = None) -> PipelineContext:
        if ckpt:
            cached = ckpt.load(ctx.run_id, self.name)
            if cached and cached.get("report_path"):
                ctx.artifacts["rule"] = cached
                ctx.artifacts["rule_report"] = _load_report(cached.get("report_path"))
                return ctx

        report = run_rule_pipeline(ctx.tickers)
        bt = report.get("backtest") or {}
        payload = {
            "report_path": str(RULE_REPORT_PATH),
            "total_return_pct": bt.get("total_return_pct"),
            "sharpe": bt.get("sharpe"),
            "signal_rows": report.get("signal_rows"),
        }
        ctx.artifacts["rule"] = payload
        ctx.artifacts["rule_report"] = report
        if ckpt:
            ckpt.save(ctx.run_id, self.name, payload)
        return ctx


class RulePlotAgent(Agent):
    """OOS equity + return density plots for rule backtest."""

    name = "rule_plot"

    def run(self, ctx: PipelineContext, ckpt: PipelineCheckpoint | None = None) -> PipelineContext:
        if ckpt:
            cached = ckpt.load(ctx.run_id, self.name)
            if cached:
                ctx.artifacts["rule_plot"] = cached
                return ctx

        from rule.chart_export import RULE_CHARTS_PATH, chart_manifest_plots, export_rule_charts
        from rule.config import RULE_EQUITY_CACHE

        report = ctx.artifacts.get("rule_report") or _load_report(str(RULE_REPORT_PATH))
        plots: list[dict] = (report or {}).get("plots") or []
        if not plots and RULE_EQUITY_CACHE.is_file():
            import pandas as pd

            eq = pd.read_parquet(RULE_EQUITY_CACHE)
            stub = report or {}
            specs = export_rule_charts(stub, eq)
            plots = chart_manifest_plots(specs)
        elif not plots and RULE_CHARTS_PATH.is_file():
            import json

            payload = json.loads(RULE_CHARTS_PATH.read_text(encoding="utf-8"))
            plots = chart_manifest_plots(payload.get("charts") or {})
        payload = {"plots": plots}
        ctx.artifacts["rule_plot"] = payload
        if ckpt:
            ckpt.save(ctx.run_id, self.name, payload)
        return ctx


DEFAULT_RULE_PIPELINE: list[Agent] = [
    DataAgent(),
    RuleStrategyAgent(),
    RulePlotAgent(),
]


def _load_report(path: str | None) -> dict | None:
    if not path or not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
