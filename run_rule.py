"""
Rule-based index strategy CLI (no ML).

  python run_rule.py verify
  python run_rule.py pipeline run --checkpoint

Pipeline: data → rule → rule_plot
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FLOW_DEFAULT_TICKERS
from operations.checkpoint import PipelineCheckpoint, PipelineContext
from operations.pipeline.agents.base import Agent
from operations.pipeline.agents.data import DataAgent
from rule.agents import DEFAULT_RULE_PIPELINE, RulePlotAgent, RuleStrategyAgent
from rule.config import RULE_BAR_TIMEFRAME, RULE_DEFAULT_TICKERS, RULE_HISTORY_DAYS


class RuleDataAgent(DataAgent):
    """Daily ETF history (~20 years) for rule backtests and equity charts."""

    def __init__(self, *, skip_download: bool = False, force: bool = False):
        super().__init__(
            days=RULE_HISTORY_DAYS,
            skip_download=skip_download,
            force=force,
            timeframe=RULE_BAR_TIMEFRAME,
        )


def _parse_tickers(raw: str) -> list[str]:
    from data_platform.universe import parse_tickers

    return parse_tickers(raw)


def _run_agents(
    ctx: PipelineContext,
    agents: list[Agent],
    ckpt: PipelineCheckpoint | None,
    *,
    start_idx: int = 0,
) -> PipelineContext:
    import time as _time

    run_list = agents[start_idx:]
    for i, agent in enumerate(run_list, start=1):
        print(f"-> [{i}/{len(run_list)}] {agent.name}", flush=True)
        t0 = _time.monotonic()
        ctx = agent.run(ctx, ckpt)
        print(f"<- [{i}/{len(run_list)}] {agent.name} done in {_time.monotonic() - t0:.1f}s", flush=True)
    return ctx


def cmd_verify(args) -> int:
    from operations.verify_pipeline import run_module_checks

    tickers = _parse_tickers(args.tickers)
    ok, results = run_module_checks(
        tickers,
        modules=["config", "data", "data.ohlcv", "market.bars"],
        fail_fast=getattr(args, "fail_fast", False),
    )
    if not ok:
        failed = [r.module for r in results if not r.ok]
        print(f"Failed: {', '.join(failed)}")
    return 0 if ok else 1


def cmd_pipeline_run(args) -> int:
    from common.runtime import configure_pipeline_runtime

    configure_pipeline_runtime()
    tickers = _parse_tickers(args.tickers)

    if not getattr(args, "skip_verify", False):
        from operations.verify_pipeline import run_module_checks

        ok, _ = run_module_checks(
            tickers,
            modules=["config", "data", "data.ohlcv", "market.bars"],
            fail_fast=True,
        )
        if not ok:
            print("Aborted — fix data cache or use --skip-verify", flush=True)
            return 1

    ctx = PipelineContext.new(tickers, run_id=args.run_id)
    ckpt = PipelineCheckpoint() if args.checkpoint else None
    agents = _pipeline_agents(args)

    start_idx = 0
    if ckpt and args.resume:
        last = ckpt.last_completed_step(ctx.run_id)
        order = [a.name for a in agents]
        if last and last in order:
            start_idx = order.index(last) + 1
            print(f"Resume after '{last}': {order[start_idx:]}")

    ctx = _run_agents(ctx, agents, ckpt, start_idx=start_idx)

    print("\n=== Rule pipeline done ===")
    rule = ctx.artifacts.get("rule") or {}
    for k, v in rule.items():
        print(f"  {k}: {v}")
    if ckpt:
        print(f"  checkpoint: {ckpt.db_path} | run_id={ctx.run_id}")
    return 0


def _pipeline_agents(args) -> list[Agent]:
    force = bool(getattr(args, "force_download", False))
    skip_dl = bool(getattr(args, "skip_download", False))
    data = RuleDataAgent(skip_download=skip_dl, force=force)
    return [data, RuleStrategyAgent(), RulePlotAgent()]


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Rule-based index strategy (no ML)")
    sub = p.add_subparsers(dest="group", required=True)

    pl = sub.add_parser("pipeline")
    pl_sub = pl.add_subparsers(dest="pipeline_cmd", required=True)
    run = pl_sub.add_parser("run")
    run.add_argument("--tickers", default=",".join(RULE_DEFAULT_TICKERS))
    run.add_argument("--checkpoint", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--run-id", default=None)
    run.add_argument("--skip-download", action="store_true")
    run.add_argument("--force-download", action="store_true")
    run.add_argument("--days", type=int, default=None)
    run.add_argument("--skip-verify", action="store_true")

    vrf = sub.add_parser("verify")
    vrf.add_argument("--tickers", default=",".join(RULE_DEFAULT_TICKERS))
    vrf.add_argument("--fail-fast", action="store_true")

    args = p.parse_args()
    if args.group == "pipeline" and args.pipeline_cmd == "run":
        return cmd_pipeline_run(args)
    if args.group == "verify":
        return cmd_verify(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
