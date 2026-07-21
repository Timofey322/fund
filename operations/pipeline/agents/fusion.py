"""Fusion role: walk-forward OOS + impulse grid + backtest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd

from operations.pipeline.agents.base import Agent
from operations.checkpoint import PipelineCheckpoint, PipelineContext
import config as _cfg
from config import COMMISSION_BPS_PER_SIDE
from strategy.pipeline import (
    CALENDAR_WF_MODES,
    ML_OPT_PATH,
    PANEL_CACHE_PATH,
    REPORT_PATH,
    build_fusion_panel,
    load_ml_optimize_artifact,
    train_fusion_pipeline,
)


class FusionAgent(Agent):
    name = "fusion"

    def __init__(
        self,
        *,
        max_oos_sessions: int | None = 120,
        wf_mode: str | None = None,
        train_days: int | None = None,
        backtest_years: int | None = None,
        test_months: int | None = None,
        max_folds: int | None = None,
    ):
        self.max_oos_sessions = max_oos_sessions
        self.wf_mode = wf_mode or getattr(_cfg, "FUSION_WF_MODE", "adaptive")
        self.train_days = train_days or int(getattr(_cfg, "FUSION_WF_TRAIN_DAYS", 365))
        self.backtest_years = backtest_years or int(getattr(_cfg, "FUSION_WF_BACKTEST_YEARS", 4))
        self.test_months = test_months or int(getattr(_cfg, "FUSION_WF_TEST_MONTHS", 6))
        self.max_folds = (
            int(max_folds)
            if max_folds is not None
            else getattr(_cfg, "FUSION_WF_MAX_FOLDS", None)
        )

    def run(self, ctx: PipelineContext, ckpt: PipelineCheckpoint | None = None) -> PipelineContext:
        if ckpt:
            cached = ckpt.load(ctx.run_id, self.name)
            if cached and cached.get("report_path"):
                ctx.artifacts["fusion"] = cached
                return ctx

        use_cached_ml = (
            self.wf_mode.lower() not in CALENDAR_WF_MODES
            and bool(getattr(_cfg, "FUSION_MONTHLY_OPTIMIZE_MODEL", False))
        )
        ml_opt = (ctx.artifacts.get("ml_optimize") or load_ml_optimize_artifact()) if use_cached_ml else None
        if ml_opt and "ml_optimize" not in ctx.artifacts:
            ctx.artifacts["ml_optimize"] = ml_opt
        panel = self._load_panel(ctx, ml_opt)
        if panel.empty:
            raise RuntimeError("Fusion panel empty — run DataAgent first")

        from data_platform.universe import needs_tick_flow

        use_tick_only = bool(getattr(_cfg, "FUSION_TICK_ONLY_CRYPTO", True)) and any(
            needs_tick_flow(str(t)) for t in ctx.tickers
        )
        report = train_fusion_pipeline(
            panel,
            ctx.tickers,
            tick_only=use_tick_only,
            commission_bps=COMMISSION_BPS_PER_SIDE,
            max_oos_sessions=self.max_oos_sessions,
            model_name=ml_opt.get("model_name") if ml_opt else None,
            model_params=(ml_opt.get("model_params") or ml_opt.get("gbm_params")) if ml_opt else None,
            gbm_cv=ml_opt.get("cv") if ml_opt else None,
            feat_cols=ml_opt.get("feat_cols") if ml_opt else None,
            wf_mode=self.wf_mode,
            train_days=self.train_days,
            backtest_years=self.backtest_years,
            test_months=self.test_months,
            max_folds=self.max_folds,
        )
        impulse = report.get("impulse_optimization", {}).get("best", {})
        payload = {
            "report_path": str(REPORT_PATH),
            "oos_auc": report.get("oos_auc"),
            "oos_log_loss": report.get("oos_log_loss"),
            "oos_rows": report.get("oos_rows"),
            "backtest_return": report.get("backtest_walk_forward_oos", {}).get("total_return_pct"),
            "disable_trading": bool(impulse.get("disable_trading")),
        }
        ctx.artifacts["fusion"] = payload
        ctx.artifacts["fusion_report"] = report
        if ckpt:
            ckpt.save(ctx.run_id, self.name, payload)
        return ctx

    @staticmethod
    def _panel_needs_rebuild(panel: pd.DataFrame) -> bool:
        """True when active label mode expects columns missing from cache."""
        from research.labels.trade import TARGET_ENTRY

        if getattr(_cfg, "FUSION_USE_BALANCED_ENTRY", False):
            from research.labels.balanced import TARGET_SL_BPS, TARGET_TP_BPS

            need = (TARGET_ENTRY, TARGET_TP_BPS, TARGET_SL_BPS)
            if any(c not in panel.columns for c in need):
                return True
            return int(panel[TARGET_ENTRY].notna().sum()) == 0
        from research.labels.trade import TARGET_ENTRY_SHORT

        need = (TARGET_ENTRY, TARGET_ENTRY_SHORT, "target_tp_bps", "target_sl_bps")
        if any(c not in panel.columns for c in need):
            return True
        if int(panel[TARGET_ENTRY].notna().sum()) == 0:
            return True
        if getattr(_cfg, "FUSION_IGNORE_APPLIED_TARGETS", False):
            return False
        try:
            from strategy.target_opt import per_instrument_specs

            tradeable_only = not bool(getattr(_cfg, "FUSION_APPLY_TARGET_CACHE", True))
            if not per_instrument_specs(tradeable_only=tradeable_only):
                return False
        except Exception:
            return False
        if TARGET_ENTRY not in panel.columns:
            return True
        return int(panel[TARGET_ENTRY].notna().sum()) == 0

    @staticmethod
    def _isolated_panel_stale(tickers: list[str]) -> bool:
        from config import BAR_TIMEFRAME, HMM_REGIME_CACHE_PATH
        from data_platform.bars import bars_cache_path
        from strategy.panel_paths import panel_cache_path

        for sym in tickers:
            path = panel_cache_path(sym)
            if not path.is_file():
                return True
            panel_mtime = path.stat().st_mtime
            bar_path = bars_cache_path(sym, BAR_TIMEFRAME)
            if bar_path.is_file() and bar_path.stat().st_mtime > panel_mtime:
                return True
            if HMM_REGIME_CACHE_PATH.is_file() and HMM_REGIME_CACHE_PATH.stat().st_mtime > panel_mtime:
                return True
        return False

    @staticmethod
    def _panel_cache_stale(tickers: list[str]) -> bool:
        if not PANEL_CACHE_PATH.is_file():
            return True
        panel_mtime = PANEL_CACHE_PATH.stat().st_mtime
        from config import BAR_TIMEFRAME, HMM_REGIME_CACHE_PATH
        from data_platform.bars import bars_cache_path
        from strategy.target_opt import TARGET_OPT_PATH

        for sym in tickers:
            bar_path = bars_cache_path(sym, BAR_TIMEFRAME)
            if bar_path.is_file() and bar_path.stat().st_mtime > panel_mtime:
                return True
        if HMM_REGIME_CACHE_PATH.is_file() and HMM_REGIME_CACHE_PATH.stat().st_mtime > panel_mtime:
            return True
        if TARGET_OPT_PATH.is_file() and TARGET_OPT_PATH.stat().st_mtime > panel_mtime:
            return True
        return False

    @staticmethod
    def _panel_covers_tickers(panel: pd.DataFrame, tickers: list[str]) -> bool:
        if panel.empty or "ticker" not in panel.columns:
            return False
        have = {str(t) for t in panel["ticker"].dropna().unique()}
        return set(tickers).issubset(have)

    @staticmethod
    def _load_panel(ctx: PipelineContext, ml_opt: dict | None) -> pd.DataFrame:
        from strategy.panel_paths import load_panels, panel_isolated_enabled

        if panel_isolated_enabled():
            if not FusionAgent._isolated_panel_stale(ctx.tickers):
                panel = FusionAgent._finalize_panel(load_panels(ctx.tickers))
                if FusionAgent._panel_covers_tickers(panel, ctx.tickers) and not FusionAgent._panel_needs_rebuild(panel):
                    print("    panel: per-instrument cache hit", flush=True)
                    return panel
            print("    building per-instrument fusion panels...", flush=True)
            hmm_frame = ctx.artifacts.get("hmm_frame")
            if hmm_frame is None:
                from config import HMM_REGIME_CACHE_PATH

                if HMM_REGIME_CACHE_PATH.is_file():
                    hmm_frame = pd.read_parquet(HMM_REGIME_CACHE_PATH)
            panel = build_fusion_panel(
                ctx.tickers,
                hybrid=True,
                tick_only=bool(getattr(_cfg, "FUSION_TICK_ONLY_CRYPTO", True)),
                hmm_frame=hmm_frame,
            )
            return FusionAgent._finalize_panel(panel)

        def _cached_panel(path: Path) -> pd.DataFrame | None:
            if not path.is_file():
                return None
            panel = FusionAgent._finalize_panel(pd.read_parquet(path))
            if not FusionAgent._panel_covers_tickers(panel, ctx.tickers):
                print(
                    f"    panel cache missing tickers "
                    f"{set(ctx.tickers) - set(panel['ticker'].unique())} — rebuilding",
                    flush=True,
                )
                return None
            if FusionAgent._panel_needs_rebuild(panel):
                print("    panel cache missing label_entry — rebuilding", flush=True)
                return None
            return panel

        if ml_opt:
            cache = Path(ml_opt.get("panel_cache") or PANEL_CACHE_PATH)
            panel = _cached_panel(cache)
            if panel is not None:
                return panel
        if PANEL_CACHE_PATH.is_file() and not FusionAgent._panel_cache_stale(ctx.tickers):
            panel = _cached_panel(PANEL_CACHE_PATH)
            if panel is not None:
                return panel
        elif PANEL_CACHE_PATH.is_file() and FusionAgent._panel_cache_stale(ctx.tickers):
            print("    panel cache stale (bars/HMM/targets newer) — rebuilding", flush=True)
        legacy = PANEL_CACHE_PATH.parent / "fusion_panel.parquet"
        if legacy.is_file() and not PANEL_CACHE_PATH.is_file():
            legacy_panel = FusionAgent._finalize_panel(pd.read_parquet(legacy))
            if FusionAgent._panel_covers_tickers(legacy_panel, ctx.tickers):
                print(f"    panel fallback: {legacy.name} -> will cache as {PANEL_CACHE_PATH.name}", flush=True)
                if not legacy_panel.empty:
                    PANEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    legacy_panel.to_parquet(PANEL_CACHE_PATH, index=False)
                return legacy_panel
            print(f"    legacy panel missing tickers — rebuilding for {ctx.tickers}", flush=True)
        print(f"    building fusion panel (no cache at {PANEL_CACHE_PATH.name})...", flush=True)
        hmm_frame = ctx.artifacts.get("hmm_frame")
        if hmm_frame is None:
            from config import HMM_REGIME_CACHE_PATH

            if HMM_REGIME_CACHE_PATH.is_file():
                hmm_frame = pd.read_parquet(HMM_REGIME_CACHE_PATH)
        panel = build_fusion_panel(
            ctx.tickers,
            hybrid=True,
            tick_only=bool(getattr(_cfg, "FUSION_TICK_ONLY_CRYPTO", True)),
            hmm_frame=hmm_frame,
        )
        if not panel.empty:
            PANEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            panel = FusionAgent._finalize_panel(panel)
            panel.to_parquet(PANEL_CACHE_PATH, index=False)
            print(f"    panel cached: {PANEL_CACHE_PATH.name} ({len(panel):,} rows)", flush=True)
        return panel

    @staticmethod
    def _finalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
        from research.features.advanced_ts import ML_ADVANCED_TS_COLS
        from research.features.registry import DERIVED_ML_COLS, attach_fusion_derived_features

        if panel.empty:
            return panel
        need = [c for c in (*DERIVED_ML_COLS, *ML_ADVANCED_TS_COLS) if c not in panel.columns]
        ret_broken = False
        if "ret_24" in panel.columns and "ticker" in panel.columns:
            sample = pd.to_numeric(panel["ret_24"], errors="coerce")
            ret_broken = bool(sample.abs().quantile(0.999) > 1.0)
        if need or ret_broken:
            panel = attach_fusion_derived_features(panel)
        return panel
