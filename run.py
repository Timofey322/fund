"""
Crypto fusion pipeline CLI (ML strategy).

  python run.py verify
  python run.py pipeline run --checkpoint

Pipeline: data → hmm → fusion → plot → monte_carlo

Rule-based (no ML) index strategy:
  python run_rule.py pipeline run --checkpoint
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FLOW_DEFAULT_TICKERS, FUSION_HISTORY_DAYS
from operations.checkpoint import PipelineCheckpoint, PipelineContext
from operations.pipeline import DEFAULT_PIPELINE, DataAgent, FusionAgent, HmmAgent, MonteCarloAgent, PlotAgent
from operations.pipeline.agents.base import Agent


def _parse_tickers(raw: str) -> list[str]:
    from data_platform.universe import parse_tickers

    return parse_tickers(raw)


def _run_agents(ctx: PipelineContext, agents: list[Agent], ckpt: PipelineCheckpoint | None, *, start_idx: int = 0) -> PipelineContext:
    import time as _time

    run_list = agents[start_idx:]
    total_stages = len(run_list)
    for i, agent in enumerate(run_list, start=1):
        print(f"-> [{i}/{total_stages}] {agent.name}", flush=True)
        _t0 = _time.monotonic()
        ctx = agent.run(ctx, ckpt)
        print(f"<- [{i}/{total_stages}] {agent.name} done in {_time.monotonic() - _t0:.1f}s", flush=True)
    return ctx


def cmd_verify(args) -> int:
    from operations.verify_pipeline import run_module_checks

    tickers = _parse_tickers(args.tickers)
    ok, results = run_module_checks(
        tickers,
        modules=getattr(args, "modules", None),
        fail_fast=getattr(args, "fail_fast", False),
    )
    if not ok:
        failed = [r.module for r in results if not r.ok]
        print(f"Failed modules: {', '.join(failed)}")
    return 0 if ok else 1


def _run_target_opt(ctx: PipelineContext, *, apply: bool = True, scoring_mode: str | None = None) -> None:
    """Sweep per-instrument label specs; persist to cache and invalidate panel."""
    import config as cfg
    from config import OUT_DIR
    from strategy.pipeline import PANEL_CACHE_PATH, build_fusion_panel
    from strategy.target_opt import optimize_targets_per_instrument, save_target_optimization

    print("-> target-opt: building base panel (global labels)...", flush=True)
    prev = bool(getattr(cfg, "FUSION_IGNORE_APPLIED_TARGETS", False))
    cfg.FUSION_IGNORE_APPLIED_TARGETS = True
    try:
        hmm_frame = ctx.artifacts.get("hmm_frame")
        panel = build_fusion_panel(
            ctx.tickers,
            hybrid=True,
            tick_only=bool(getattr(cfg, "FUSION_TICK_ONLY_CRYPTO", True)),
            hmm_frame=hmm_frame,
        )
    finally:
        cfg.FUSION_IGNORE_APPLIED_TARGETS = prev

    if panel.empty:
        raise RuntimeError("target-opt: fusion panel empty")

    from strategy.leakage_guard import panel_for_causal_target_opt

    panel_full_rows = len(panel)
    panel, causal_meta = panel_for_causal_target_opt(panel)
    print(
        f"    causal target-opt panel: {len(panel):,}/{panel_full_rows:,} rows "
        f"(strictly before OOS {causal_meta['oos_cutoff']})",
        flush=True,
    )
    mode = scoring_mode or getattr(cfg, "TARGET_OPT_SCORING_MODE", "entry")
    result = optimize_targets_per_instrument(panel, ctx.tickers, scoring_mode=mode)
    result["causal_metadata"] = causal_meta
    path = save_target_optimization(result, applied=apply)
    n = len(result.get("per_symbol") or {})
    print(f"    target-opt saved: {path} ({n} symbols)", flush=True)

    for stale in (
        PANEL_CACHE_PATH,
        OUT_DIR / "cache" / "fusion_panel.parquet",
        OUT_DIR / "cache" / "fusion_panel_v7.parquet",
    ):
        if stale.is_file():
            stale.unlink()
            print(f"    removed stale panel: {stale.name}", flush=True)


def cmd_target_opt(args) -> int:
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
            print("target-opt aborted — fix data cache first", flush=True)
            return 1

    ctx = PipelineContext.new(tickers)
    agents_pre = [
        DataAgent(days=getattr(args, "days", None) or FUSION_HISTORY_DAYS, skip_download=args.skip_download),
        HmmAgent(),
    ]
    for agent in agents_pre:
        print(f"-> {agent.name}", flush=True)
        ctx = agent.run(ctx, None)

    loaded = (ctx.artifacts.get("data") or {}).get("loaded") or {}
    bad = [t for t in tickers if not loaded.get(t)]
    if bad:
        print(f"target-opt aborted — data not loaded for: {bad}", flush=True)
        return 1

    _run_target_opt(ctx, apply=bool(args.apply), scoring_mode=getattr(args, "scoring_mode", None))
    return 0


def cmd_pipeline_run(args) -> int:
    from common.runtime import configure_pipeline_runtime

    configure_pipeline_runtime()
    if not getattr(args, "skip_verify", False):
        from operations.verify_pipeline import run_module_checks

        tickers = _parse_tickers(args.tickers)
        ok, _ = run_module_checks(tickers, fail_fast=True)
        if not ok:
            print("Pipeline aborted — fix verify failures or use --skip-verify", flush=True)
            return 1

    ctx = PipelineContext.new(_parse_tickers(args.tickers), run_id=args.run_id)
    ckpt = PipelineCheckpoint() if args.checkpoint else None
    agents = _pipeline_agents(args)

    start_idx = 0
    if ckpt and args.resume:
        last = ckpt.last_completed_step(ctx.run_id)
        order = [a.name for a in agents]
        if last and last in order:
            start_idx = order.index(last) + 1
            print(f"Resume after step '{last}': {order[start_idx:]}")
    elif ckpt and not args.run_id:
        print(f"New run_id: {ctx.run_id}")

    if getattr(args, "target_opt", False):
        pre = [a for a in agents if a.name in ("data", "hmm")]
        rest = [a for a in agents if a.name not in ("data", "hmm")]
        pre_start = start_idx if start_idx < len(pre) else len(pre)
        if pre_start < len(pre):
            ctx = _run_agents(ctx, pre, ckpt, start_idx=pre_start)
        _run_target_opt(ctx, apply=True)
        ctx = _run_agents(ctx, rest, ckpt, start_idx=0)
    else:
        ctx = _run_agents(ctx, agents, ckpt, start_idx=start_idx)

    print("\n=== Done ===")
    for k, v in ctx.artifacts.items():
        if k.endswith("_frame") or k.endswith("_report"):
            continue
        print(f"  {k}: {v}")
    if ckpt:
        print(f"  checkpoint: {ckpt.db_path} | run_id={ctx.run_id}")

    try:
        from operations.export_dashboard import export_manifest
        from operations.morning_summary import write_morning_summary
        from reporting.ticker_research import write_ticker_research_bundle

        write_ticker_research_bundle(ctx.tickers)
        manifest = export_manifest()
        summary = write_morning_summary()
        print(f"  dashboard manifest: {manifest}", flush=True)
        print(f"  morning summary: {summary}", flush=True)
    except Exception as exc:
        print(f"  post-run export skipped: {exc}", flush=True)

    return 0


def _pipeline_agents(args) -> list[Agent]:
    agents: list[Agent] = list(DEFAULT_PIPELINE)

    force_dl = bool(getattr(args, "force_download", False))
    dl_days = getattr(args, "days", None) or FUSION_HISTORY_DAYS
    if args.skip_download:
        agents = [
            DataAgent(days=dl_days, skip_download=True) if a.name == "data" else a
            for a in agents
        ]
    else:
        agents = [
            DataAgent(days=dl_days, skip_download=False, force=force_dl)
            if a.name == "data"
            else a
            for a in agents
        ]

    wf_mode = getattr(args, "wf_mode", None) or __import__("config").FUSION_WF_MODE
    agents = [
        FusionAgent(
            max_oos_sessions=getattr(args, "max_oos_sessions", 120),
            wf_mode=wf_mode,
            train_days=getattr(args, "train_days", None),
            backtest_years=getattr(args, "backtest_years", None),
            test_months=getattr(args, "test_months", None),
            max_folds=getattr(args, "max_folds", None),
        )
        if a.name == "fusion"
        else a
        for a in agents
    ]
    return agents


def cmd_checkpoint_list(args) -> int:
    ckpt = PipelineCheckpoint()
    for row in ckpt.list_runs(args.limit):
        print(f"{row['run_id']}  {row['updated_at']}")
    return 0


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Crypto fusion pipeline")
    sub = p.add_subparsers(dest="group", required=True)

    pl = sub.add_parser("pipeline", help="Run full pipeline")
    pl_sub = pl.add_subparsers(dest="pipeline_cmd", required=True)
    run = pl_sub.add_parser("run")
    run.add_argument("--tickers", default=",".join(FLOW_DEFAULT_TICKERS))
    run.add_argument("--checkpoint", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--run-id", default=None)
    run.add_argument("--skip-download", action="store_true")
    run.add_argument("--force-download", action="store_true", help="Re-download bar cache")
    run.add_argument("--days", type=int, default=None, help="History window for DataAgent (days)")
    run.add_argument("--wf-mode", default=None, choices=("adaptive", "monthly4y", "legacy"), help="Fusion walk-forward mode")
    run.add_argument("--train-days", type=int, default=None, help="Monthly WF trailing train window in days")
    run.add_argument("--backtest-years", type=int, default=None, help="Monthly WF stitched OOS years")
    run.add_argument("--test-months", type=int, default=None, help="Monthly WF test window length")
    run.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Cap monthly walk-forward folds (default: all, e.g. ~48 for 4y/1m)",
    )
    run.add_argument("--max-oos-sessions", type=int, default=120, help="Legacy WF max OOS sessions")
    run.add_argument("--skip-verify", action="store_true", help="Skip pre-flight module checks")
    run.add_argument(
        "--target-opt",
        action="store_true",
        help="Per-instrument label optimization before fusion (recommended)",
    )

    to = sub.add_parser("target-opt", help="Optimize ML target per instrument")
    to.add_argument("--tickers", default=",".join(FLOW_DEFAULT_TICKERS))
    to.add_argument("--apply", action="store_true", help="Activate specs at runtime (default)")
    to.add_argument("--skip-download", action="store_true")
    to.add_argument("--days", type=int, default=None)
    to.add_argument("--skip-verify", action="store_true")
    to.add_argument(
        "--scoring-mode",
        choices=("entry", "direction"),
        default=None,
        help="Target scoring mode (default: config TARGET_OPT_SCORING_MODE)",
    )
    to.set_defaults(apply=True)

    vrf = sub.add_parser("verify", help="Pre-flight: test each pipeline module on real cache")
    vrf.add_argument("--tickers", default=",".join(FLOW_DEFAULT_TICKERS))
    vrf.add_argument("--module", action="append", dest="modules", help="Only run checks for this module")
    vrf.add_argument("--fail-fast", action="store_true")

    ck = sub.add_parser("checkpoint")
    ck_sub = ck.add_subparsers(dest="ck_cmd", required=True)
    ls = ck_sub.add_parser("list")
    ls.add_argument("--limit", type=int, default=10)

    args = p.parse_args()

    if args.group == "pipeline" and args.pipeline_cmd == "run":
        return cmd_pipeline_run(args)
    if args.group == "target-opt":
        return cmd_target_opt(args)
    if args.group == "verify":
        return cmd_verify(args)
    if args.group == "checkpoint" and args.ck_cmd == "list":
        return cmd_checkpoint_list(args)

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
