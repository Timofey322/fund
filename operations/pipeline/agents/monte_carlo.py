"""Monte Carlo survival simulation on fusion OOS equity (final pipeline stage)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

import config as _cfg
from simulation.trade_survival import survival_simulation
from operations.checkpoint import PipelineCheckpoint, PipelineContext
from config import OUT_DIR
from operations.pipeline.agents.base import Agent

MONTE_CARLO_REPORT_PATH = OUT_DIR / "monte_carlo_report.json"
EQUITY_CACHE = OUT_DIR / "cache" / "fusion_bt_equity.parquet"


class MonteCarloAgent(Agent):
    name = "monte_carlo"

    def __init__(self, *, n_paths: int | None = None):
        self.n_paths = int(n_paths or getattr(_cfg, "MONTE_CARLO_PATHS", 2000))

    def run(self, ctx: PipelineContext, ckpt: PipelineCheckpoint | None = None) -> PipelineContext:
        if ckpt:
            cached = ckpt.load(ctx.run_id, self.name)
            if cached and cached.get("report_path"):
                ctx.artifacts["monte_carlo"] = cached
                return ctx

        fusion = ctx.artifacts.get("fusion") or {}
        if fusion.get("disable_trading"):
            payload = {"skipped": True, "reason": "trading_disabled"}
            ctx.artifacts["monte_carlo"] = payload
            if ckpt:
                ckpt.save(ctx.run_id, self.name, payload)
            return ctx

        if not EQUITY_CACHE.is_file():
            oos_path = None
            for candidate in (
                OUT_DIR / "cache" / "fusion_oos_adaptive.parquet",
                OUT_DIR / "cache" / "fusion_oos_monthly4y.parquet",
            ):
                if candidate.is_file():
                    oos_path = candidate
                    break
            if oos_path is not None:
                oos = pd.read_parquet(oos_path, columns=["bar_time"])
                idx = pd.DatetimeIndex(pd.to_datetime(oos["bar_time"]).unique()).sort_values()
                eq_df = pd.DataFrame({"value": np.ones(len(idx), dtype=float)}, index=idx)
                eq_df.index.name = "bar_time"
                eq_df = eq_df.reset_index()
                print("    Monte Carlo: flat equity from OOS index (no trade cache)", flush=True)
            else:
                raise RuntimeError(f"Equity cache missing: {EQUITY_CACHE} — run FusionAgent first")
        else:
            eq_df = pd.read_parquet(EQUITY_CACHE)
        if eq_df.empty or "value" not in eq_df.columns:
            raise RuntimeError("Equity cache empty or missing value column")

        time_col = "bar_time" if "bar_time" in eq_df.columns else eq_df.columns[0]
        eq = eq_df.set_index(pd.to_datetime(eq_df[time_col]))["value"].sort_index()
        hold = int(getattr(_cfg, "FWD_HORIZON_BARS", 12))
        block = int(getattr(_cfg, "MONTE_CARLO_BLOCK_BARS", hold))
        max_dd = float(getattr(_cfg, "MONTE_CARLO_MAX_DD_THRESHOLD", 0.20))

        print(
            f"    Monte Carlo: {self.n_paths} paths | {len(eq)} equity bars | block={block}",
            flush=True,
        )
        survival = survival_simulation(
            eq,
            n_paths=self.n_paths,
            block_bars=block,
            max_dd_threshold=max_dd,
        )
        report = {
            "n_paths": self.n_paths,
            "equity_bars": int(len(eq)),
            "period_start": str(eq.index.min().date()) if len(eq) else None,
            "period_end": str(eq.index.max().date()) if len(eq) else None,
            "survival": survival,
            "backtest_return_pct": fusion.get("backtest_return"),
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        MONTE_CARLO_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        payload = {
            "report_path": str(MONTE_CARLO_REPORT_PATH),
            "survival_rate": survival.get("survival_rate"),
            "prob_terminal_loss": survival.get("prob_terminal_loss"),
            "terminal_wealth_p50": survival.get("terminal_wealth_p50"),
        }
        ctx.artifacts["monte_carlo"] = payload
        if ckpt:
            ckpt.save(ctx.run_id, self.name, payload)
        return ctx
