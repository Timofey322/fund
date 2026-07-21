"""
Pre-flight checks for each pipeline module on real cached Binance data.

Run before a full pipeline to catch broken imports, missing cache, or regressions:

    python run.py verify --tickers BTC,ETH,SOL,ETHBTC
    python verify_pipeline.py

Pipeline run also calls verify by default (use --skip-verify to bypass).
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class ModuleCheck:
    module: str
    ok: bool
    message: str
    detail: dict = field(default_factory=dict)


CheckFn = Callable[[list[str]], ModuleCheck]


def _fail(module: str, message: str, **detail) -> ModuleCheck:
    return ModuleCheck(module=module, ok=False, message=message, detail=dict(detail))


def _ok(module: str, message: str, **detail) -> ModuleCheck:
    return ModuleCheck(module=module, ok=True, message=message, detail=dict(detail))


def check_config(_tickers: list[str]) -> ModuleCheck:
    try:
        import config as cfg

        required = (
            "OUT_DIR",
            "BAR_TIMEFRAME",
            "FUSION_WF_MODE",
            "PANEL_CACHE_VERSION",
            "HMM_REGIME_CACHE_PATH",
            "MONTE_CARLO_PATHS",
        )
        missing = [k for k in required if not hasattr(cfg, k)]
        if missing:
            return _fail("config", f"missing attributes: {missing}")
        cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
        return _ok(
            "config",
            f"wf_mode={cfg.FUSION_WF_MODE} panel_v={cfg.PANEL_CACHE_VERSION}",
            out_dir=str(cfg.OUT_DIR),
        )
    except Exception as exc:
        return _fail("config", str(exc), traceback=traceback.format_exc())


def check_data_cache(tickers: list[str]) -> ModuleCheck:
    from config import BAR_TIMEFRAME
    from data_platform.binance import load_slim_panel, bars_flow_slim_path
    from data_platform.bars import bars_cache_path, load_ohlcv
    from data_platform.universe import is_crypto_symbol, is_tradfi_symbol

    stats: dict[str, int] = {}
    missing: list[str] = []
    for sym in tickers:
        if is_crypto_symbol(sym):
            path = bars_flow_slim_path(sym)
            if not path.is_file():
                missing.append(sym)
                continue
            df = load_slim_panel(sym)
            min_rows = 10_000
        elif is_tradfi_symbol(sym):
            path = bars_cache_path(sym, BAR_TIMEFRAME)
            if not path.is_file():
                missing.append(sym)
                continue
            df = load_ohlcv(sym, BAR_TIMEFRAME)
            min_rows = 2_000
        else:
            missing.append(sym)
            continue
        if df.empty or len(df) < min_rows:
            missing.append(sym)
        else:
            stats[sym] = len(df)
    if missing:
        return _fail(
            "data",
            f"missing or thin cache for: {missing} — run pipeline without --skip-download",
            paths={t: str(bars_flow_slim_path(t) if is_crypto_symbol(t) else bars_cache_path(t, BAR_TIMEFRAME)) for t in tickers},
        )
    return _ok("data", f"{len(tickers)} symbols loaded", rows=stats)


def check_ohlcv_cache(tickers: list[str]) -> ModuleCheck:
    from config import BAR_TIMEFRAME
    from data_platform.binance import load_crypto_ohlcv
    from data_platform.bars import load_ohlcv
    from data_platform.universe import is_crypto_symbol, is_tradfi_symbol

    stats: dict[str, int] = {}
    missing: list[str] = []
    for sym in tickers:
        if is_crypto_symbol(sym):
            df = load_crypto_ohlcv(sym, BAR_TIMEFRAME)
            min_rows = 5_000
        elif is_tradfi_symbol(sym):
            df = load_ohlcv(sym, BAR_TIMEFRAME)
            min_rows = 2_000
        else:
            missing.append(sym)
            continue
        if df.empty or len(df) < min_rows:
            missing.append(sym)
        else:
            stats[sym] = len(df)
    if missing:
        return _fail("data.ohlcv", f"OHLCV cache missing/thin: {missing}")
    return _ok("data.ohlcv", "OHLCV bars for HMM", rows=stats)


def check_prices(tickers: list[str]) -> ModuleCheck:
    from config import BAR_TIMEFRAME
    from data_platform.bars import load_closes

    prices = load_closes(tickers, BAR_TIMEFRAME)
    if prices.empty:
        return _fail("market.bars", "load_closes returned empty frame")
    thin = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if thin:
        return _fail("market.bars", f"no close series for {thin}")
    return _ok("market.bars", f"{len(prices):,} bars x {len(prices.columns)} symbols")


def check_hmm_regime(tickers: list[str]) -> ModuleCheck:
    from config import HMM_REGIME_CACHE_PATH, BAR_TIMEFRAME
    from data_platform.bars import load_closes
    from common.naming import COL_PROB_HMM_IMPULSE, COL_PROB_HMM_STRESS

    if HMM_REGIME_CACHE_PATH.is_file():
        import pandas as pd

        hmm = pd.read_parquet(HMM_REGIME_CACHE_PATH)
        need = {COL_PROB_HMM_IMPULSE, COL_PROB_HMM_STRESS, "bar_time"}
        if hmm.empty or not need.issubset(hmm.columns):
            return _fail("hmm", f"cache invalid: {HMM_REGIME_CACHE_PATH.name}")
        return _ok("hmm", f"regime cache {len(hmm):,} rows", path=str(HMM_REGIME_CACHE_PATH))

    prices = load_closes(tickers[:2], BAR_TIMEFRAME).tail(4_000)
    if prices.empty:
        return _fail("hmm", "no prices to probe HMM — run hmm stage first")
    try:
        from research.regime.hmm import build_hmm_regime_frame

        regime = build_hmm_regime_frame(prices)
        if regime.empty:
            return _fail("hmm", "build_hmm_regime_frame returned empty on price probe")
        return _ok("hmm", f"probe build OK ({len(regime)} rows, cache will be built in hmm stage)")
    except Exception as exc:
        return _fail("hmm", str(exc), traceback=traceback.format_exc())


def _load_panel_for_verify() -> tuple["pd.DataFrame", str]:
    import pandas as pd
    from config import OUT_DIR, PANEL_CACHE_VERSION
    from operations.pipeline.agents.fusion import FusionAgent

    v2 = OUT_DIR / "cache" / f"fusion_panel_v{PANEL_CACHE_VERSION}.parquet"
    legacy = OUT_DIR / "cache" / "fusion_panel.parquet"
    for path in (v2, legacy):
        if path.is_file():
            return FusionAgent._finalize_panel(pd.read_parquet(path)), path.name
    return pd.DataFrame(), ""


def check_fusion_panel(tickers: list[str]) -> ModuleCheck:
    from research.features.registry import active_ml_feature_cols

    panel, source = _load_panel_for_verify()
    if panel.empty:
        from config import PANEL_CACHE_VERSION

        return _ok(
            "fusion.panel",
            f"no fusion_panel_v{PANEL_CACHE_VERSION} yet — fusion stage will build it",
        )
    feat = [c for c in active_ml_feature_cols() if c in panel.columns]
    if len(feat) < 20:
        return _fail("fusion.panel", f"too few ML features in panel: {len(feat)}")
    sym_ok = all(s in panel["ticker"].unique() for s in tickers)
    if not sym_ok:
        have = sorted(panel["ticker"].unique().tolist())
        return _fail("fusion.panel", f"panel missing tickers; have {have}")
    return _ok(
        "fusion.panel",
        f"{len(panel):,} rows from {source}",
        features=len(feat),
        sessions=int(panel["session"].nunique()),
    )


def check_ml_per_fold(tickers: list[str]) -> ModuleCheck:
    from strategy.pipeline import optimize_fusion_model_on_train_slice, resolve_ml_feature_cols, _default_entry_target

    panel, _ = _load_panel_for_verify()
    if panel.empty:
        return _ok("fusion.ml_opt", "skipped — panel builds in fusion stage")

    panel = panel.sort_values("bar_time").tail(40_000)
    feat = resolve_ml_feature_cols(panel)
    target = _default_entry_target(panel)
    if target not in panel.columns:
        return _fail("fusion.ml_opt", f"target {target} missing in panel")

    try:
        opt = optimize_fusion_model_on_train_slice(
            panel.dropna(subset=feat + [target]),
            feat,
            target,
            n_trials=3,
            max_train_rows=8_000,
            fold_meta={"fold": 0, "verify": True},
        )
        cv = opt.get("cv") or {}
        return _ok(
            "fusion.ml_opt",
            f"composite={cv.get('composite')} auc={cv.get('auc')}",
            model=opt.get("model_name"),
            optimizer=opt.get("optimizer"),
            n_trials=opt.get("n_trials"),
        )
    except Exception as exc:
        return _fail("fusion.ml_opt", str(exc), traceback=traceback.format_exc())


def check_threshold_per_fold(tickers: list[str]) -> ModuleCheck:
    from config import COMMISSION_BPS_PER_SIDE, BAR_TIMEFRAME
    from models.entry_model import DEFAULT_LIGHTGBM_PARAMS, make_entry_classifier
    from strategy.pipeline import resolve_ml_feature_cols, _default_entry_target
    from strategy.threshold_opt import optimize_trading_policy_on_train
    from data_platform.bars import load_closes

    panel, _ = _load_panel_for_verify()
    if panel.empty:
        return _ok("fusion.threshold_opt", "skipped — panel builds in fusion stage")

    panel = panel.sort_values("bar_time").tail(30_000)
    feat = resolve_ml_feature_cols(panel)
    target = _default_entry_target(panel)
    work = panel.dropna(subset=feat + [target]).tail(12_000)
    if len(work) < 2_000 or work[target].nunique() < 2:
        return _fail("fusion.threshold_opt", "insufficient labeled rows in panel tail")

    prices = load_closes(tickers, BAR_TIMEFRAME)
    params = dict(DEFAULT_LIGHTGBM_PARAMS)
    clf = make_entry_classifier("lightgbm", params)
    clf.fit(work[feat].iloc[:8_000], work[target].iloc[:8_000].astype(int))
    scored = work.iloc[8_000:].copy()
    scored["ml_proba"] = clf.predict_proba(scored[feat])[:, 1]
    scored["ml_base_rate"] = float(work[target].mean())

    try:
        import config as cfg

        old_trials = getattr(cfg, "FUSION_THRESHOLD_OPTUNA_TRIALS", 50)
        cfg.FUSION_THRESHOLD_OPTUNA_TRIALS = 4
        th = optimize_trading_policy_on_train(
            scored,
            prices,
            commission_bps=COMMISSION_BPS_PER_SIDE,
            fold_meta={"fold": 0, "verify": True},
        )
        cfg.FUSION_THRESHOLD_OPTUNA_TRIALS = old_trials
        cv = th.get("cv") or {}
        bp = th.get("best_params") or {}
        return _ok(
            "fusion.threshold_opt",
            f"objective={cv.get('objective')} buy={bp.get('buy_threshold')} "
            f"edge={bp.get('min_expected_edge_bps')} sl={bp.get('stop_loss_bps')}",
            optimizer=th.get("optimizer"),
            n_trials=th.get("n_trials"),
            signal_rows=cv.get("signal_rows"),
        )
    except Exception as exc:
        return _fail("fusion.threshold_opt", str(exc), traceback=traceback.format_exc())


def check_walk_forward_smoke(tickers: list[str]) -> ModuleCheck:
    import pandas as pd
    from config import BAR_TIMEFRAME, COMMISSION_BPS_PER_SIDE
    from models.entry_model import DEFAULT_LIGHTGBM_PARAMS
    from strategy.pipeline import resolve_ml_feature_cols, walk_forward_fusion_oos_monthly
    from data_platform.bars import load_closes

    panel, _ = _load_panel_for_verify()
    if panel.empty:
        return _ok("fusion.walk_forward", "skipped — panel builds in fusion stage")

    panel = panel.sort_values("bar_time")
    cutoff = panel["bar_time"].max() - pd.Timedelta(days=450)
    slice_panel = panel[panel["bar_time"] >= cutoff].copy()
    feat = resolve_ml_feature_cols(slice_panel)
    params = dict(DEFAULT_LIGHTGBM_PARAMS)
    prices = load_closes(tickers, BAR_TIMEFRAME)

    try:
        oos, folds = walk_forward_fusion_oos_monthly(
            slice_panel,
            feat,
            params,
            train_days=90,
            backtest_years=1,
            test_months=1,
            min_train_rows=3_000,
            optimize_per_fold=False,
            optimize_thresholds_per_fold=False,
        )
        active = [f for f in folds if not f.get("skipped")]
        if oos.empty or not active:
            return _fail("fusion.walk_forward", "smoke WF produced no OOS rows")
        return _ok(
            "fusion.walk_forward",
            f"{len(oos):,} OOS rows | {len(active)} folds",
            oos_auc=round(float(oos["ml_proba"].mean()), 4),
        )
    except Exception as exc:
        return _fail("fusion.walk_forward", str(exc), traceback=traceback.format_exc())


def check_backtest(tickers: list[str]) -> ModuleCheck:
    from config import BAR_TIMEFRAME, COMMISSION_BPS_PER_SIDE, OUT_DIR
    from simulation.engine import run_backtest_signal_exit
    from strategy.pipeline import _default_monthly_impulse_fallback, _fusion_signal_frame
    from data_platform.bars import load_closes
    import pandas as pd

    oos_path = OUT_DIR / "cache" / "fusion_oos_monthly4y.parquet"
    if not oos_path.is_file():
        return _ok("backtest", "skipped — no OOS cache yet (built during fusion)")

    oos = pd.read_parquet(oos_path).tail(20_000)
    if oos.empty or "ml_proba" not in oos.columns:
        return _fail("backtest", "OOS cache missing ml_proba")

    prices = load_closes(tickers, BAR_TIMEFRAME)
    params = _default_monthly_impulse_fallback(10.0)
    sig = _fusion_signal_frame(oos, prices, params)
    if sig.empty:
        return _ok("backtest", "signal frame empty on OOS tail (thresholds may be strict)")

    bt = run_backtest_signal_exit(
        prices,
        sig,
        score_col="score",
        use_dynamic_thresholds=True,
        commission_bps=COMMISSION_BPS_PER_SIDE,
        period_start=pd.Timestamp(oos["bar_time"].min()),
        period_end=pd.Timestamp(oos["bar_time"].max()),
    )
    stats = bt.get("stats") or {}
    return _ok(
        "backtest",
        f"return={stats.get('total_return_pct')}% signals={stats.get('signal_exit_count')}",
        exposure=stats.get("avg_exposure_pct"),
    )


def check_quantstats(_tickers: list[str]) -> ModuleCheck:
    try:
        from reporting.quantstats_report import write_quantstats_report

        return _ok("quantstats", "import OK", fn=write_quantstats_report.__name__)
    except Exception as exc:
        return _fail("quantstats", str(exc))


def check_monte_carlo(_tickers: list[str]) -> ModuleCheck:
    from config import OUT_DIR
    from simulation.trade_survival import survival_simulation
    import pandas as pd

    eq_path = OUT_DIR / "cache" / "fusion_bt_equity.parquet"
    if not eq_path.is_file():
        return _ok("monte_carlo", "skipped — no equity cache (built during fusion)")

    eq_df = pd.read_parquet(eq_path)
    if eq_df.empty or "value" not in eq_df.columns:
        return _fail("monte_carlo", "equity cache invalid")

    col = "bar_time" if "bar_time" in eq_df.columns else eq_df.columns[0]
    eq = eq_df.set_index(pd.to_datetime(eq_df[col]))["value"]
    sim = survival_simulation(eq, n_paths=200, block_bars=12)
    if sim.get("error"):
        return _fail("monte_carlo", sim["error"])
    return _ok(
        "monte_carlo",
        f"survival={sim.get('survival_rate')} p_loss={sim.get('prob_terminal_loss')}",
        paths=sim.get("n_paths"),
    )


def check_plot(_tickers: list[str]) -> ModuleCheck:
    try:
        from reporting.diagnostics import plot_system_overview
        from reporting.plots import write_standard_oos_plots

        return _ok("plot", "imports OK", functions="plot_system_overview, write_standard_oos_plots")
    except Exception as exc:
        return _fail("plot", str(exc))


def check_agents(tickers: list[str]) -> ModuleCheck:
    from operations.checkpoint import PipelineContext
    from operations.pipeline import DEFAULT_PIPELINE

    names = [a.name for a in DEFAULT_PIPELINE]
    expected = ["data", "hmm", "fusion", "plot", "monte_carlo"]
    if names != expected:
        return _fail("pipeline.agents", f"order mismatch: {names} != {expected}")

    ctx = PipelineContext.new(tickers)
    from operations.pipeline.agents.data import DataAgent

    agent = DataAgent(days=30, skip_download=True)
    ctx = agent.run(ctx, None)
    loaded = (ctx.artifacts.get("data") or {}).get("loaded") or {}
    if not all(loaded.get(t) for t in tickers):
        return _fail("pipeline.agents", "DataAgent dry-run failed", loaded=loaded)
    return _ok("pipeline.agents", f"chain {' -> '.join(names)} | DataAgent OK")


# Ordered checks — mirrors pipeline stages.
MODULE_CHECKS: list[tuple[str, CheckFn]] = [
    ("config", check_config),
    ("data", check_data_cache),
    ("data.ohlcv", check_ohlcv_cache),
    ("market.bars", check_prices),
    ("hmm", check_hmm_regime),
    ("fusion.panel", check_fusion_panel),
    ("fusion.ml_opt", check_ml_per_fold),
    ("fusion.threshold_opt", check_threshold_per_fold),
    ("fusion.walk_forward", check_walk_forward_smoke),
    ("backtest", check_backtest),
    ("quantstats", check_quantstats),
    ("monte_carlo", check_monte_carlo),
    ("plot", check_plot),
    ("pipeline.agents", check_agents),
]


def run_module_checks(
    tickers: list[str],
    *,
    modules: list[str] | None = None,
    fail_fast: bool = False,
) -> tuple[bool, list[ModuleCheck]]:
    """Run all or selected module checks. Returns (all_ok, results)."""
    selected = {m.lower() for m in modules} if modules else None
    results: list[ModuleCheck] = []
    all_ok = True

    print("\n=== Pipeline module verify (real data) ===\n")
    for label, fn in MODULE_CHECKS:
        if selected and label.lower() not in selected and not any(label.lower().startswith(s) for s in selected):
            continue
        try:
            result = fn(tickers)
        except Exception as exc:
            result = _fail(label, f"unexpected: {exc}", traceback=traceback.format_exc())
        results.append(result)
        mark = "OK" if result.ok else "FAIL"
        print(f"  [{mark:4}] {result.module:<22} {result.message}")
        if not result.ok:
            all_ok = False
            if fail_fast:
                break

    n_ok = sum(1 for r in results if r.ok)
    print(f"\n=== {n_ok}/{len(results)} checks passed ===\n")
    return all_ok, results


def main(argv: list[str] | None = None) -> int:
    import argparse
    from config import FLOW_DEFAULT_TICKERS

    p = argparse.ArgumentParser(description="Verify pipeline modules on real cached data")
    p.add_argument("--tickers", default=",".join(FLOW_DEFAULT_TICKERS))
    p.add_argument("--module", action="append", dest="modules", help="Run only this module (repeatable)")
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args(argv)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    ok, _ = run_module_checks(tickers, modules=args.modules, fail_fast=args.fail_fast)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
