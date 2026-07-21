"""
HMM market context + impulse features + ML entry model.

Decision layers:
  1. HMM (bar): P(impulse), P(mean_revert), P(stress) — trade context / risk gate
  2. ML (bar): LightGBM P(entry after costs) on flow + impulse + HMM features
  3. Impulse strength (optimized): scales conviction / position sizing
"""

from __future__ import annotations

import json
import math as _math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as _cfg
from data_platform.bars import load_closes, load_ohlcv
from config import BAR_TIMEFRAME, HMM_FREQUENCY, OUT_DIR
from research.features.entry_ml import FWD_HORIZON_BARS, _engineer_features, _session_key
from research.features.registry import (
    FUSION_FEATURE_COLS,
    PANEL_FEATURE_COLS,
    aggregate_importance_by_group,
    attach_fusion_derived_features,
    feature_group_catalog,
    merge_fold_importances,
    resolve_ml_feature_cols,
)
from research.labels.trade import DEFAULT_SLIPPAGE_BPS, TARGET_12_AFTER_COSTS, TARGET_20_AFTER_COSTS, TARGET_TRIPLE_BARRIER_20, attach_trade_targets
from research.regime.hmm import build_hmm_regime_frame
from research.features.impulse import IMPULSE_COLS, attach_impulse_features
from common.naming import COL_PROB_HMM_STRESS, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_IMPULSE
from research.features.volume_profile import VP_COLS, VP_HMM_COLS, attach_volume_profile, attach_vp_hmm_features
from research.features.hybrid_flow import filter_tick_only
from common.timeframe import filter_session
from data_platform.binance import load_slim_panel
from research.features.hmm_context import HMM_XGB_FEATURE_COLS, attach_hmm_xgb_features

REPORT_PATH = OUT_DIR / "fusion_pipeline_report.json"
_panel_ver = int(getattr(_cfg, "PANEL_CACHE_VERSION", 1))
PANEL_CACHE_PATH = OUT_DIR / "cache" / f"fusion_panel_v{_panel_ver}.parquet"
ML_OPT_PATH = OUT_DIR / "cache" / "ml_optimization.json"
MONTHLY_FOLD_OPT_PATH = OUT_DIR / "cache" / "monthly_fold_optimizations.json"
OPTIMIZATION_SUMMARY_PATH = OUT_DIR / "cache" / "optimization_summary.json"
OPTIMIZATION_TRIALS_PARQUET_PATH = OUT_DIR / "cache" / "optimization_trials_flat.parquet"
GBM_OPT_PATH = ML_OPT_PATH  # backward compat


def load_ml_optimize_artifact() -> dict | None:
    """Load last LightGBM optimization result written by ml_optimize stage."""
    if not ML_OPT_PATH.is_file():
        return None
    try:
        return json.loads(ML_OPT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_ml_optimize_artifact(result: dict, *, panel_cache: str | Path | None = None) -> Path:
    """Persist ML optimization result for fusion stage and CLI inspection."""
    payload = {
        **result,
        "model_params": result.get("model_params") or result.get("gbm_params"),
        "gbm_params": result.get("model_params") or result.get("gbm_params"),
        "panel_cache": str(panel_cache or PANEL_CACHE_PATH),
        "report_path": str(ML_OPT_PATH),
    }
    ML_OPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ML_OPT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return ML_OPT_PATH


CALENDAR_WF_MODES = frozenset({
    "monthly4y", "monthly", "monthly_walk_forward", "adaptive", "semiannual_adaptive",
})


def is_calendar_wf_mode(mode: str | None = None) -> bool:
    m = (mode or getattr(_cfg, "FUSION_WF_MODE", "adaptive")).lower()
    return m in CALENDAR_WF_MODES


def is_adaptive_wf_mode(mode: str | None = None) -> bool:
    m = (mode or getattr(_cfg, "FUSION_WF_MODE", "adaptive")).lower()
    return m in ("adaptive", "semiannual_adaptive")


def causal_model_opt_cutoff(
    panel: pd.DataFrame,
    *,
    train_days: int = 365,
    backtest_years: int = 4,
    test_months: int = 1,
) -> pd.Timestamp | None:
    """First monthly OOS test_start — training for global hyperparameter tune must end here."""
    windows = monthly_walk_forward_windows(
        panel,
        train_days=train_days,
        backtest_years=backtest_years,
        test_months=test_months,
    )
    if not windows:
        return None
    return pd.Timestamp(windows[0]["test_start"])


def optimize_fusion_model_causal(
    panel: pd.DataFrame,
    *,
    tick_only: bool = True,
    target_col: str | None = None,
    train_days: int | None = None,
    backtest_years: int | None = None,
    test_months: int | None = None,
    model_candidates: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Purged CV hyperparameter search on data strictly before the stitched OOS window."""
    td = int(train_days or getattr(_cfg, "FUSION_WF_TRAIN_DAYS", 365))
    by = int(backtest_years or getattr(_cfg, "FUSION_WF_BACKTEST_YEARS", 4))
    tm = int(test_months or getattr(_cfg, "FUSION_WF_TEST_MONTHS", 1))
    cutoff = causal_model_opt_cutoff(panel, train_days=td, backtest_years=by, test_months=tm)
    work = panel.copy()
    if cutoff is not None and "bar_time" in work.columns:
        work["bar_time"] = pd.to_datetime(work["bar_time"])
        train_panel = work[work["bar_time"] < cutoff]
        if len(train_panel) >= int(getattr(_cfg, "FUSION_WF_MIN_TRAIN_ROWS", 20_000)):
            print(
                f"    causal ML optimize: {len(train_panel):,} bars before OOS {cutoff.date()}",
                flush=True,
            )
            work = train_panel
        else:
            print(
                f"    causal ML optimize: insufficient pre-OOS rows ({len(train_panel):,}), using full panel",
                flush=True,
            )
    result = optimize_fusion_model(
        work,
        tick_only=tick_only,
        target_col=target_col,
        model_candidates=model_candidates,
    )
    if cutoff is not None:
        result["causal"] = True
        result["causal_cutoff"] = str(cutoff.date())
        result["n_bars_opt"] = int(len(work))
    return result


def optimize_lightgbm(
    panel: pd.DataFrame,
    *,
    tick_only: bool = True,
    target_col: str | None = None,
    causal: bool = False,
    train_days: int | None = None,
    backtest_years: int | None = None,
    test_months: int | None = None,
) -> dict:
    """Purged session CV over LightGBM hyperparameter grid."""
    from config import FUSION_ENTRY_MODEL

    if causal:
        return optimize_fusion_model_causal(
            panel,
            tick_only=tick_only,
            target_col=target_col,
            train_days=train_days,
            backtest_years=backtest_years,
            test_months=test_months,
            model_candidates=(FUSION_ENTRY_MODEL,),
        )
    return optimize_fusion_model(
        panel,
        tick_only=tick_only,
        target_col=target_col,
        model_candidates=(FUSION_ENTRY_MODEL,),
    )

HMM_CONTEXT_COLS = [
    COL_PROB_HMM_IMPULSE,
    COL_PROB_HMM_MEAN_REVERT,
    COL_PROB_HMM_STRESS,
    "hmm_confidence",
    "hmm_risk_on",
    "hmm_vol_ratio",
]

HMM_MODEL_COLS = HMM_CONTEXT_COLS + HMM_XGB_FEATURE_COLS


def _default_entry_target(panel: pd.DataFrame) -> str:
    """Resolve the training target column.

    Uses the unified per-instrument ``label_entry`` when per-instrument targets are
    active and the column is present, else the global ``FUSION_ENTRY_TARGET``.
    """
    from research.labels.trade import TARGET_ENTRY

    if getattr(_cfg, "FUSION_IGNORE_APPLIED_TARGETS", False):
        return str(getattr(_cfg, "FUSION_ENTRY_TARGET", TARGET_12_AFTER_COSTS))
    if getattr(_cfg, "USE_PER_INSTRUMENT_TARGETS", False) or _per_instrument_active():
        if TARGET_ENTRY in panel.columns and panel[TARGET_ENTRY].notna().any():
            return TARGET_ENTRY
        # Specs active but cache/panel stale — still prefer label_entry if fwd exists.
        if "fwd_ret_entry" in panel.columns and panel["fwd_ret_entry"].notna().any():
            return TARGET_ENTRY
    return str(getattr(_cfg, "FUSION_ENTRY_TARGET", TARGET_12_AFTER_COSTS))


def _per_instrument_active() -> bool:
    if getattr(_cfg, "FUSION_IGNORE_APPLIED_TARGETS", False):
        return False
    try:
        from strategy.target_opt import per_instrument_specs

        tradeable_only = not bool(getattr(_cfg, "FUSION_APPLY_TARGET_CACHE", True))
        return bool(per_instrument_specs(tradeable_only=tradeable_only))
    except Exception:
        return False


def _entry_target_summary() -> dict:
    """Active per-instrument target specs + resolved holding default for the report."""
    try:
        from strategy.target_opt import per_instrument_specs

        tradeable_only = not bool(getattr(_cfg, "FUSION_APPLY_TARGET_CACHE", True))
        specs = per_instrument_specs(tradeable_only=tradeable_only)
    except Exception:
        specs = {}
    return {
        "enabled": bool(specs),
        "specs": specs,
        "hold_default_bars": _fusion_hold_default(),
    }


def _fusion_hold_default() -> int:
    """Default holding horizon (bars), aligned to applied per-instrument targets."""
    try:
        from strategy.target_opt import applied_hold_default

        return int(applied_hold_default(FWD_HORIZON_BARS))
    except Exception:
        return int(FWD_HORIZON_BARS)


def _expected_move_bps(hold_bars: int | None = None, *, ticker: str | None = None) -> float:
    """Expected per-trade move (bps), aligned to per-symbol label TP when available."""
    if ticker:
        try:
            from strategy.target_opt import ticker_threshold_bps

            thr = ticker_threshold_bps(str(ticker))
            if thr is not None:
                return float(thr)
        except Exception:
            pass
    base = float(getattr(_cfg, "FUSION_EXPECTED_MOVE_BPS", 30.0))
    hold = int(hold_bars or _fusion_hold_default())
    ref = max(int(FWD_HORIZON_BARS), 1)
    if hold <= ref:
        return base
    return base * _math.sqrt(hold / ref)


DEFAULT_IMPULSE_WEIGHTS = {
    "w_ml": 0.30,
    "w_mom": 0.20,
    "w_nw": 0.15,
    "w_flow": 0.15,
    "w_vp": 0.20,
}

IMPULSE_GRID = {
    "w_ml": (0.35, 0.45),
    "w_mom": (0.20,),
    "w_nw": (0.15,),
    "stress_max": (0.30, 0.45, 0.55),
    "hmm_impulse_min": (0.05, 0.20),
    "hmm_confidence_min": (0.25, 0.35),
    "hmm_entropy_max": (0.90, 1.05),
    "allow_mean_revert": (True, False),
    "impulse_min": (0.10, 0.20),
    "min_expected_edge_bps": (6.0, 10.0, 12.0),
    "gain": (120, 160),
    "hold_threshold": (48, 52),
    "buy_threshold": (55, 60),
}

# Populated once per optimize_impulse_params() call for process-pool workers (Windows).
_GRID_WORKER_CTX: dict = {}


def _col_num(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def hmm_allows_trade_mask(
    df: pd.DataFrame,
    *,
    stress_max: float,
    hmm_impulse_min: float,
    confidence_min: float | None = None,
    entropy_max: float | None = None,
    allow_mean_revert: bool = True,
) -> pd.Series:
    """Vectorized HMM gate (same rules as ``hmm_allows_trade`` per row)."""
    p_stress = _col_num(df, COL_PROB_HMM_STRESS, 0.34)
    p_impulse = _col_num(df, COL_PROB_HMM_IMPULSE, 0.33)
    p_mean_revert = _col_num(df, COL_PROB_HMM_MEAN_REVERT, 0.33)
    conf = _col_num(df, "hmm_confidence", 0.33)
    entropy = _col_num(df, "hmm_prob_entropy", 0.0)
    if "hmm_risk_on" in df.columns:
        risk_on = df["hmm_risk_on"].fillna(True).astype(bool)
    elif "risk_on" in df.columns:
        risk_on = df["risk_on"].fillna(True).astype(bool)
    else:
        risk_on = pd.Series(True, index=df.index)

    gate = p_stress <= float(stress_max)
    if confidence_min is not None:
        gate &= conf >= float(confidence_min)
    if entropy_max is not None:
        gate &= entropy <= float(entropy_max)
    gate &= risk_on | (p_impulse >= float(hmm_impulse_min) + 0.1)
    gate &= (p_impulse >= float(hmm_impulse_min)) | (
        bool(allow_mean_revert) & (p_mean_revert >= 0.35)
    )
    return gate


def compute_impulse_strength_series(
    df: pd.DataFrame,
    ml_proba: pd.Series,
    weights: dict | None = None,
    baseline_proba: float | pd.Series = 0.5,
) -> pd.Series:
    """Vectorized impulse strength (same formula as ``compute_impulse_strength``)."""
    w = {**DEFAULT_IMPULSE_WEIGHTS, **(weights or {})}
    w_sum = w.get("w_ml", 0) + w.get("w_mom", 0) + w.get("w_nw", 0) + w.get("w_flow", 0) + w.get("w_vp", 0)
    if w_sum <= 0:
        w_sum = 1.0

    if isinstance(baseline_proba, pd.Series):
        base = np.clip(baseline_proba.astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    else:
        base = np.clip(float(baseline_proba), 1e-6, 1 - 1e-6)

    proba = ml_proba.astype(float).to_numpy()
    ml_edge = np.maximum(0.0, (proba - base) / np.maximum(1.0 - base, 1e-6))
    mom = np.minimum(np.abs(_col_num(df, "roc_6", 0.0).to_numpy()) * 50.0, 1.0)
    nw_ext = np.abs(_col_num(df, "nw_env_pos", 0.5).to_numpy() - 0.5) * 2.0
    flow = np.abs(_col_num(df, "vol_imbalance", 0.0).to_numpy())
    power = _col_num(df, "power_shift", 0.0).to_numpy()
    flow = np.minimum(1.0, flow + power * 2.0)
    vp_sig = np.minimum(1.0, np.abs(_col_num(df, "vp_hmm_composite", 0.0).to_numpy()))

    raw = (
        w.get("w_ml", 0) * ml_edge
        + w.get("w_mom", 0) * mom
        + w.get("w_nw", 0) * nw_ext
        + w.get("w_flow", 0) * flow
        + w.get("w_vp", 0) * vp_sig
    ) / w_sum

    p_stress = _col_num(df, COL_PROB_HMM_STRESS, 0.34).to_numpy()
    conf = _col_num(df, "hmm_confidence", 0.33).to_numpy()
    return pd.Series(raw * (1.0 - p_stress) * (0.5 + 0.5 * conf), index=df.index)


def _expected_edge_bps_series(
    ml_proba: pd.Series,
    impulse_strength: pd.Series,
    baseline_proba: float | pd.Series = 0.5,
    *,
    hold_bars: int | None = None,
    ticker: str | None = None,
) -> pd.Series:
    move = _expected_move_bps(hold_bars, ticker=ticker)
    if isinstance(baseline_proba, pd.Series):
        base = np.clip(baseline_proba.astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    else:
        base = np.full(len(ml_proba), np.clip(float(baseline_proba), 1e-6, 1 - 1e-6))
    proba = ml_proba.astype(float).to_numpy()
    imp = np.clip(impulse_strength.astype(float).to_numpy(), 0.0, 1.5)
    directional_edge = (proba - base) / np.maximum(1.0 - base, 1e-6)
    return pd.Series(directional_edge * move * (0.5 + imp), index=ml_proba.index)


def fusion_entry_score_from_components(
    proba: np.ndarray,
    impulse: np.ndarray,
) -> np.ndarray:
    """Map ML probability + impulse to 0–100 entry score (aligned with buy_threshold)."""
    proba_arr = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    imp_arr = np.clip(np.asarray(impulse, dtype=float), 0.0, 1.5)
    threshold_score = 50.0 + 50.0 * proba_arr
    boost = 0.5 + 0.5 * np.minimum(imp_arr, 1.0)
    return np.clip(threshold_score * boost, 0.0, 100.0)


def calibrate_min_expected_edge_bps(
    scored_panel: pd.DataFrame,
    commission_bps: float,
    *,
    slippage_bps: float | None = None,
) -> float:
    """Fee-aware floor + quantile on gated expected edge (bps).

    ``scored_panel`` must be train or validation rows with ``ml_proba`` — not
    stitched OOS test months used for final backtest reporting.
    """
    from strategy.edge_gate import heuristic_gate_floor_bps, resolve_min_expected_edge_bps

    slip = DEFAULT_SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    cap_bps = float(getattr(_cfg, "FUSION_EDGE_CALIBRATION_CAP_BPS", 35.0))
    quantile = float(getattr(_cfg, "FUSION_EDGE_CALIBRATION_QUANTILE", 0.35))
    default_bps = float(getattr(_cfg, "FUSION_MIN_EXPECTED_EDGE_BPS", 8.0))
    floor_bps = heuristic_gate_floor_bps(commission_bps, slip)

    if scored_panel.empty or "ml_proba" not in scored_panel.columns:
        return resolve_min_expected_edge_bps(
            default_bps, commission_bps=commission_bps, slippage_bps=slip,
        )

    probe_params = {
        **DEFAULT_IMPULSE_WEIGHTS,
        "stress_max": 0.55,
        "hmm_impulse_min": 0.05,
        "impulse_min": 0.0,
        "gain": 80,
    }
    fused = apply_fusion_scores(scored_panel, probe_params)
    gated = fused[fused["hmm_gate"]] if "hmm_gate" in fused.columns else fused
    if gated.empty:
        return resolve_min_expected_edge_bps(
            default_bps, commission_bps=commission_bps, slippage_bps=slip,
        )

    edges = gated["expected_edge_bps"].astype(float)
    # Magnitude for long+short: signed gate compares |edge| to min_edge.
    edges = edges[np.isfinite(edges)].abs()
    if len(edges) < 30:
        return resolve_min_expected_edge_bps(
            default_bps, commission_bps=commission_bps, slippage_bps=slip,
        )

    q_edge = float(np.quantile(edges, quantile))
    calibrated = min(max(floor_bps, q_edge, default_bps * 0.5), cap_bps)
    from strategy.edge_gate import cap_calibrated_edge_to_panel, panel_abs_edge_stats

    panel_max, panel_q65, _ = panel_abs_edge_stats(edges)
    calibrated = cap_calibrated_edge_to_panel(
        calibrated,
        panel_max_edge_bps=panel_max,
        panel_q65_edge_bps=panel_q65,
        commission_bps=commission_bps,
        slippage_bps=slip,
    )
    return resolve_min_expected_edge_bps(
        calibrated,
        commission_bps=commission_bps,
        slippage_bps=slip,
        calibrated=calibrated,
    )


def _trade_constraint_metrics(sig: pd.DataFrame, stats: dict, holdings: list[dict]) -> dict:
    """Metrics used to prevent optimizers from selecting all-cash parameter sets."""
    active_rebalances = sum(1 for h in holdings if h.get("n", 0) > 0)
    avg_exposure = float(stats.get("avg_exposure_pct", 0.0) or 0.0)
    if avg_exposure <= 0.0 and active_rebalances > 0:
        weight_totals = [
            sum(float(v) for v in (h.get("weights") or {}).values())
            for h in holdings
            if h.get("n", 0) > 0
        ]
        if weight_totals:
            avg_exposure = float(np.mean(weight_totals)) * 100.0
    signal_rows = len(sig)
    if not sig.empty and {"score", "buy_threshold", "risk_on"}.issubset(sig.columns):
        risk = sig["risk_on"].fillna(True).astype(bool)
        score = pd.to_numeric(sig["score"], errors="coerce").fillna(0.0)
        buy = pd.to_numeric(sig["buy_threshold"], errors="coerce").fillna(0.0)
        sell = (
            pd.to_numeric(sig["sell_threshold"], errors="coerce").fillna(100.0 - buy)
            if "sell_threshold" in sig.columns
            else (100.0 - buy)
        )
        long_elig = risk & (score > 0.0) & (score >= buy)
        short_elig = risk & (score > 0.0) & (score <= sell)
        signal_rows = int((long_elig | short_elig).sum())
    return {
        "signal_rows": int(signal_rows),
        "active_rebalances": int(active_rebalances),
        "avg_exposure_pct": avg_exposure,
    }


def _passes_trade_constraints(metrics: dict) -> bool:
    return (
        metrics.get("signal_rows", 0) >= int(getattr(_cfg, "FUSION_MIN_SIGNAL_ROWS", 20))
        and metrics.get("active_rebalances", 0) >= int(getattr(_cfg, "FUSION_MIN_ACTIVE_REBALANCES", 1))
        and metrics.get("avg_exposure_pct", 0.0) >= float(getattr(_cfg, "FUSION_MIN_AVG_EXPOSURE_PCT", 0.1))
    )


def _constraint_penalty(metrics: dict) -> float:
    """Soft penalty used only for fallback ranking when no candidate passes hard constraints."""
    min_rows = max(int(getattr(_cfg, "FUSION_MIN_SIGNAL_ROWS", 20)), 1)
    min_reb = max(int(getattr(_cfg, "FUSION_MIN_ACTIVE_REBALANCES", 1)), 1)
    min_exp = max(float(getattr(_cfg, "FUSION_MIN_AVG_EXPOSURE_PCT", 0.1)), 1e-9)
    shortfall = (
        max(0.0, 1.0 - metrics.get("signal_rows", 0) / min_rows)
        + max(0.0, 1.0 - metrics.get("active_rebalances", 0) / min_reb)
        + max(0.0, 1.0 - metrics.get("avg_exposure_pct", 0.0) / min_exp)
    )
    return float(getattr(_cfg, "FUSION_EXPOSURE_PENALTY", 5.0)) * shortfall


def attach_hmm_to_bars(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    hmm_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge HMM regime frame onto bar-level panel (causal merge_asof)."""
    if panel.empty:
        return panel
    hmm = hmm_frame
    if hmm is None or hmm.empty:
        cache_path = getattr(_cfg, "HMM_REGIME_CACHE_PATH", None)
        if cache_path is not None and Path(cache_path).is_file():
            hmm = pd.read_parquet(cache_path)
            print(f"    HMM regime from cache ({len(hmm):,} rows)", flush=True)
    if hmm is None or hmm.empty:
        hmm = build_hmm_regime_frame(prices)
    if hmm.empty:
        print("    WARNING: HMM regime empty — using neutral defaults (check HMM stage)", flush=True)
        for c in HMM_CONTEXT_COLS:
            panel[c] = 0.0 if "prob" not in c else 0.33
        panel[COL_PROB_HMM_STRESS] = 0.34
        panel[COL_PROB_HMM_IMPULSE] = 0.33
        panel[COL_PROB_HMM_MEAN_REVERT] = 0.33
        panel["hmm_confidence"] = 0.33
        panel["hmm_risk_on"] = 1.0
        panel["hmm_vol_ratio"] = 1.0
        return panel

    use_bar = HMM_FREQUENCY == "bar" and "bar_time" in hmm.columns
    time_col = "bar_time" if use_bar else "date"
    hmm = hmm.sort_values(time_col)
    use = hmm[
        [time_col, COL_PROB_HMM_IMPULSE, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_STRESS,
         "hmm_confidence", "risk_on", "vol_ratio"]
    ].rename(columns={"risk_on": "hmm_risk_on", "vol_ratio": "hmm_vol_ratio"})
    use[time_col] = pd.to_datetime(use[time_col], errors="coerce")
    use = use.dropna(subset=[time_col]).sort_values(time_col)

    out = panel.sort_values("bar_time").copy()
    out["bar_time"] = pd.to_datetime(out["bar_time"], errors="coerce")
    out = out.dropna(subset=["bar_time"])
    merged = pd.merge_asof(
        out,
        use,
        left_on="bar_time",
        right_on=time_col,
        direction="backward",
    )
    for c in HMM_CONTEXT_COLS:
        if c not in merged.columns:
            merged[c] = 0.0
    merged[COL_PROB_HMM_IMPULSE] = pd.to_numeric(merged[COL_PROB_HMM_IMPULSE], errors="coerce").fillna(1.0 / 3.0)
    merged[COL_PROB_HMM_MEAN_REVERT] = pd.to_numeric(merged[COL_PROB_HMM_MEAN_REVERT], errors="coerce").fillna(1.0 / 3.0)
    merged[COL_PROB_HMM_STRESS] = pd.to_numeric(merged[COL_PROB_HMM_STRESS], errors="coerce").fillna(1.0 / 3.0)
    merged["hmm_confidence"] = pd.to_numeric(merged["hmm_confidence"], errors="coerce").fillna(1.0 / 3.0)
    merged["hmm_risk_on"] = pd.to_numeric(merged["hmm_risk_on"], errors="coerce").fillna(1.0)
    merged["hmm_vol_ratio"] = pd.to_numeric(merged["hmm_vol_ratio"], errors="coerce").fillna(1.0)
    merged = attach_vp_hmm_features(merged)
    merged = attach_hmm_xgb_features(merged)
    merged = attach_fusion_derived_features(merged)
    drop_cols = [time_col] if time_col != "bar_time" else []
    return merged.drop(columns=drop_cols, errors="ignore")


def split_train_test(
    panel: pd.DataFrame,
    train_frac: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split by bar_time (no shuffle)."""
    if panel.empty:
        return panel, panel
    panel = panel.sort_values("bar_time")
    cut = panel["bar_time"].quantile(train_frac)
    train = panel[panel["bar_time"] <= cut].copy()
    test = panel[panel["bar_time"] > cut].copy()
    if test.empty or train.empty:
        mid = max(1, int(len(panel) * train_frac))
        train, test = panel.iloc[:mid].copy(), panel.iloc[mid:].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _calibrate_probabilities(
    y_val: pd.Series | np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y_val, dtype=float)
    pv = np.clip(np.asarray(p_val, dtype=float), 1e-6, 1 - 1e-6)
    pt = np.clip(np.asarray(p_test, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return pt
    try:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(pv, y)
        return np.clip(iso.predict(pt), 1e-6, 1 - 1e-6)
    except Exception:
        return pt


def _filter_panel_tick_only(panel: pd.DataFrame) -> pd.DataFrame:
    """Require tick flow for crypto rows; keep candle-imputed tradfi rows."""
    if panel.empty or "ticker" not in panel.columns:
        return panel
    from data_platform.universe import needs_tick_flow

    tickers = panel["ticker"].astype(str)
    tradfi = panel[~tickers.map(needs_tick_flow)]
    crypto = filter_tick_only(panel[tickers.map(needs_tick_flow)])
    if tradfi.empty:
        return crypto.reset_index(drop=True)
    if crypto.empty:
        return tradfi.reset_index(drop=True)
    return pd.concat([tradfi, crypto], ignore_index=True).sort_values("bar_time").reset_index(drop=True)


def _align_panel_common_window(panel: pd.DataFrame) -> pd.DataFrame:
    """Align end dates across tickers; allow different start dates (tradfi has shorter 5m history)."""
    if panel.empty or "ticker" not in panel.columns or "bar_time" not in panel.columns:
        return panel
    bounds = panel.groupby("ticker")["bar_time"].agg(["min", "max"])
    if bounds.empty:
        return panel
    common_end = bounds["max"].min()
    out = panel[panel["bar_time"] <= common_end].copy()
    return out.reset_index(drop=True)


def _build_symbol_panel_rows(
    sym: str,
    *,
    timeframe: str,
    hybrid: bool,
    tick_only_crypto: bool,
    entry_specs: dict[str, dict],
    baseline_horizon: int,
) -> pd.DataFrame | None:
    """Feature engineering for one symbol (process-pool safe)."""
    from data_platform.binance import load_slim_panel
    from data_platform.universe import needs_tick_flow
    from research.features.advanced_ts import attach_ml_advanced_ts_features
    from research.features.registry import FUSION_FEATURE_COLS, PANEL_FEATURE_COLS

    use_tick = tick_only_crypto and needs_tick_flow(sym)
    if use_tick:
        raw = load_slim_panel(sym)
    else:
        raw = load_ohlcv(sym, timeframe)
    if raw.empty:
        return None
    df = filter_session(raw, symbol=sym)
    if df.empty:
        return None
    feat = _engineer_features(df, symbol=sym if hybrid else None)
    feat = attach_impulse_features(feat)
    feat = attach_volume_profile(feat, session=_session_key(feat.index))
    feat = attach_ml_advanced_ts_features(feat)
    from research.features.structure import attach_structure_features

    feat = attach_structure_features(feat)
    feat = attach_trade_targets(feat, baseline_horizon=baseline_horizon)
    if getattr(_cfg, "FUSION_USE_BALANCED_ENTRY", False):
        from research.labels.balanced import attach_balanced_entry_label

        feat = attach_balanced_entry_label(feat, close_col="close", symbol=sym)
    else:
        from research.labels.trade import attach_economic_entry_labels, resolve_entry_spec

        spec = entry_specs.get(sym.upper()) or resolve_entry_spec(sym)
        feat = attach_economic_entry_labels(feat, close_col="close", symbol=sym, spec=spec)
    feat["ticker"] = sym
    feat["bar_time"] = feat.index
    from common.timeframe import session_id as _session_id_for_symbol

    feat["session"] = _session_id_for_symbol(feat.index, symbol=sym)

    from research.labels.balanced import (
        BALANCED_HORIZON,
        BALANCED_POSITIVE_RATE,
        TARGET_SL_BPS,
        TARGET_TP_BPS,
    )
    from research.labels.trade import (
        ENTRY_LABEL_HORIZON,
        ENTRY_LONG_POSITIVE_RATE,
        ENTRY_SHORT_POSITIVE_RATE,
        TARGET_DIRECTION,
        TARGET_ENTRY_SHORT,
    )

    extra_cols = [
        "buy_count", "sell_count", "count_imbalance", "flow_source", "tick_imbalance",
        TARGET_TP_BPS, TARGET_SL_BPS,
        BALANCED_HORIZON, BALANCED_POSITIVE_RATE,
        ENTRY_LABEL_HORIZON, ENTRY_LONG_POSITIVE_RATE, ENTRY_SHORT_POSITIVE_RATE,
        "direction_flat_rate", "direction_long_rate", "direction_short_rate",
    ]
    sub = feat[[c for c in set(list(FUSION_FEATURE_COLS) + list(PANEL_FEATURE_COLS) + extra_cols + [
        "fwd_ret", "label", "fwd_ret_12", "label_12_positive", "fwd_ret_20",
        "label_12_after_costs", "label_20_1pct", "label_20_after_costs",
        "label_entry", "fwd_ret_entry", TARGET_ENTRY_SHORT, TARGET_DIRECTION,
        TARGET_TRIPLE_BARRIER_20,
        "future_mae_20", "future_mfe_20", "future_mae_48",
        "future_mfe_48", "future_mae_96", "future_mfe_96", "best_hold_bucket",
        "ticker", "session", "bar_time", "close", "volume",
    ]) if c in feat.columns]]
    from research.features.advanced_ts import CROSS_SECTIONAL_COLS
    from research.features.registry import active_ml_feature_cols
    from research.features.structure import STRUCTURE_CS_COLS

    ml_req = [
        c for c in active_ml_feature_cols()
        if c in sub.columns and c not in CROSS_SECTIONAL_COLS and c not in STRUCTURE_CS_COLS
    ]
    req = ml_req + ["fwd_ret"]
    sub = sub.dropna(subset=req)
    sub = sub[sub["volume"] > 0]
    if use_tick:
        sub = filter_tick_only(sub)
    if sub.empty:
        return None
    return sub


def _build_symbol_panel_worker(payload: dict) -> pd.DataFrame | None:
    return _build_symbol_panel_rows(
        payload["sym"],
        timeframe=payload["timeframe"],
        hybrid=payload["hybrid"],
        tick_only_crypto=payload["tick_only_crypto"],
        entry_specs=payload["entry_specs"],
        baseline_horizon=payload["baseline_horizon"],
    )


def build_fusion_panel(
    symbols: list[str],
    *,
    timeframe: str = BAR_TIMEFRAME,
    hybrid: bool = True,
    tick_only: bool = False,
    hmm_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """OHLCV + hybrid flow + impulse + HMM context + forward labels."""
    from common.parallel import resolve_worker_count
    from strategy.panel_paths import load_panels, panel_isolated_enabled, save_panel

    if panel_isolated_enabled():
        cached = load_panels(symbols)
        if not cached.empty and cached["ticker"].astype(str).str.upper().nunique() >= len(symbols):
            have = set(cached["ticker"].astype(str).str.upper().unique())
            if have >= {str(s).upper() for s in symbols}:
                panel = _align_panel_common_window(cached)
                prices = load_closes(symbols, timeframe)
                if not prices.empty:
                    panel = attach_hmm_to_bars(panel, prices, hmm_frame=hmm_frame)
                elif any(c in panel.columns for c in VP_COLS):
                    panel = attach_vp_hmm_features(panel)
                if tick_only and bool(getattr(_cfg, "FUSION_TICK_ONLY_CRYPTO", True)):
                    panel = _filter_panel_tick_only(panel)
                return panel

    prices = load_closes(symbols, timeframe)
    rows: list[pd.DataFrame] = []
    h = FWD_HORIZON_BARS

    entry_specs: dict[str, dict] = {}
    if not getattr(_cfg, "FUSION_IGNORE_APPLIED_TARGETS", False):
        try:
            from strategy.target_opt import per_instrument_specs

            tradeable_only = not bool(getattr(_cfg, "FUSION_APPLY_TARGET_CACHE", True))
            entry_specs = per_instrument_specs(tradeable_only=tradeable_only)
        except Exception:
            entry_specs = {}

    tick_only_crypto = bool(tick_only)
    workers = resolve_worker_count(
        "FUSION_PANEL_BUILD_WORKERS",
        cap=len(symbols),
        env_var="FUSION_PANEL_BUILD_WORKERS",
    )
    build_kwargs = {
        "timeframe": timeframe,
        "hybrid": hybrid,
        "tick_only_crypto": tick_only_crypto,
        "entry_specs": entry_specs,
        "baseline_horizon": h,
    }

    if workers <= 1 or len(symbols) <= 1:
        for sym in symbols:
            sub = _build_symbol_panel_rows(sym, **build_kwargs)
            if sub is not None:
                if panel_isolated_enabled():
                    save_panel(sym, sub)
                rows.append(sub)
    else:
        print(f"    panel build: parallel workers={workers}", flush=True)
        payloads = [{"sym": sym, **build_kwargs} for sym in symbols]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_build_symbol_panel_worker, p) for p in payloads]
            for fut in as_completed(futures):
                sub = fut.result()
                if sub is not None:
                    if panel_isolated_enabled() and "ticker" in sub.columns:
                        for sym in sub["ticker"].astype(str).str.upper().unique():
                            save_panel(sym, sub[sub["ticker"].astype(str).str.upper() == sym])
                    rows.append(sub)

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows).sort_values("bar_time").reset_index(drop=True)
    panel = _align_panel_common_window(panel)
    from research.features.registry import attach_cross_sectional_features

    panel = attach_cross_sectional_features(panel)
    if not prices.empty:
        panel = attach_hmm_to_bars(panel, prices, hmm_frame=hmm_frame)
    elif any(c in panel.columns for c in VP_COLS):
        panel = attach_vp_hmm_features(panel)
    if tick_only_crypto:
        panel = _filter_panel_tick_only(panel)
    return panel


def hmm_allows_trade(
    row: pd.Series,
    *,
    stress_max: float,
    hmm_impulse_min: float,
    confidence_min: float | None = None,
    entropy_max: float | None = None,
    allow_mean_revert: bool = True,
) -> bool:
    """HMM context gate: avoid stress, require impulse or mean-revert opportunity."""
    return bool(
        hmm_allows_trade_mask(
            row.to_frame().T,
            stress_max=stress_max,
            hmm_impulse_min=hmm_impulse_min,
            confidence_min=confidence_min,
            entropy_max=entropy_max,
            allow_mean_revert=allow_mean_revert,
        ).iloc[0]
    )


def _expected_edge_bps(
    ml_proba: float,
    impulse_strength: float,
    *,
    baseline_proba: float = 0.5,
    expected_move_bps: float | None = None,
) -> float:
    """Conservative expected directional edge estimate before fees/slippage."""
    move = float(expected_move_bps) if expected_move_bps else _expected_move_bps()
    base = min(max(float(baseline_proba), 1e-6), 1 - 1e-6)
    # Signed edge: below-baseline probabilities imply a *negative* expectation, so
    # the downstream `expected_edge_bps >= min_edge` gate can actually filter them.
    # Previously this was clamped to >= 0, which made the gate inert and dressed a
    # one-sided heuristic up as an expectation.
    directional_edge = (float(ml_proba) - base) / max(1.0 - base, 1e-6)
    conviction = max(0.0, min(float(impulse_strength), 1.5))
    return float(directional_edge * move * (0.5 + conviction))


def compute_impulse_strength(
    row: pd.Series,
    ml_proba: float,
    weights: dict | None = None,
    baseline_proba: float = 0.5,
) -> float:
    """Scalar impulse strength (delegates to vectorized implementation)."""
    s = compute_impulse_strength_series(
        row.to_frame().T,
        pd.Series([ml_proba]),
        weights,
        baseline_proba=baseline_proba,
    )
    return float(s.iloc[0])


def apply_fusion_scores(
    oos: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Add ml_proba-based score, impulse_strength, hmm_gate to OOS panel (vectorized)."""
    out = oos.copy()
    stress_max = params.get("stress_max", 0.35)
    hmm_impulse_min = params.get("hmm_impulse_min", 0.25)
    confidence_min = params.get("hmm_confidence_min", getattr(_cfg, "FUSION_HMM_CONFIDENCE_MIN", None))
    entropy_max = params.get("hmm_entropy_max", getattr(_cfg, "FUSION_HMM_ENTROPY_MAX", None))
    allow_mean_revert = bool(params.get("allow_mean_revert", True))
    weights = {k: params[k] for k in DEFAULT_IMPULSE_WEIGHTS if k in params}

    proba = out["ml_proba"].astype(float)
    if "ml_base_rate" in out.columns:
        baseline = _col_num(out, "ml_base_rate", float(params.get("ml_base_rate", 0.5) or 0.5))
    else:
        baseline = float(params.get("ml_base_rate", 0.5) or 0.5)

    imp = compute_impulse_strength_series(out, proba, weights, baseline_proba=baseline)
    hard_gate = bool(getattr(_cfg, "FUSION_HMM_HARD_GATE", False))
    if hard_gate:
        gate = hmm_allows_trade_mask(
            out,
            stress_max=stress_max,
            hmm_impulse_min=hmm_impulse_min,
            confidence_min=confidence_min,
            entropy_max=entropy_max,
            allow_mean_revert=allow_mean_revert,
        )
    else:
        gate = pd.Series(True, index=out.index)

    from strategy.fusion_direction import fusion_allow_short, fusion_signed_scores, signed_expected_edge_bps

    buy_th = float(params.get("buy_threshold", 55))
    hold_default = int(params.get("hold_bars") or _fusion_hold_default())
    if fusion_allow_short() and "ml_proba_short" in out.columns:
        proba_short = out["ml_proba_short"].astype(float)
        fusion, side = fusion_signed_scores(
            proba.to_numpy(),
            proba_short.to_numpy(),
            imp.to_numpy(),
            baseline=baseline.to_numpy() if isinstance(baseline, pd.Series) else baseline,
            buy_threshold=buy_th,
        )
        out["position_side"] = side
        move = _expected_move_bps(hold_default)
        out["expected_edge_bps"] = signed_expected_edge_bps(
            proba.to_numpy(),
            proba_short.to_numpy(),
            imp.to_numpy(),
            baseline=baseline.to_numpy() if isinstance(baseline, pd.Series) else baseline,
            move_bps=move,
        )
    else:
        fusion = fusion_entry_score_from_components(proba.to_numpy(), imp.to_numpy())
        out["position_side"] = np.where(proba.to_numpy() >= 0.5, 1, 0)
        if "ticker" in out.columns:
            try:
                from strategy.target_opt import ticker_hold_horizon_bars

                holds = out["ticker"].map(
                    lambda t: int(ticker_hold_horizon_bars(str(t), hold_default))
                )
                edge_parts: list[pd.Series] = []
                for hold_val in holds.dropna().unique():
                    mask = holds == hold_val
                    tickers = out.loc[mask, "ticker"].astype(str)
                    for tick in tickers.unique():
                        tmask = mask & (out["ticker"].astype(str) == tick)
                        edge_parts.append(
                            _expected_edge_bps_series(
                                proba[tmask],
                                imp[tmask],
                                baseline_proba=baseline[tmask] if isinstance(baseline, pd.Series) else baseline,
                                hold_bars=int(hold_val),
                                ticker=str(tick),
                            )
                        )
                out["expected_edge_bps"] = pd.concat(edge_parts).sort_index().to_numpy()
            except Exception:
                out["expected_edge_bps"] = _expected_edge_bps_series(
                    proba, imp, baseline_proba=baseline, hold_bars=hold_default,
                ).to_numpy()
        else:
            out["expected_edge_bps"] = _expected_edge_bps_series(
                proba, imp, baseline_proba=baseline, hold_bars=hold_default,
            ).to_numpy()

    if hard_gate:
        fusion = np.where(gate.to_numpy(), fusion, 0.0)

    out["impulse_strength"] = imp.to_numpy()
    out["hmm_gate"] = gate.to_numpy()
    out["fusion_score"] = fusion
    return out


def _impulse_grid_axes(calibrated_min_edge: float | None = None) -> list:
    if calibrated_min_edge is not None:
        lo = max(4.0, round(calibrated_min_edge - 2.0, 1))
        mid = round(calibrated_min_edge, 1)
        hi = min(15.0, round(calibrated_min_edge + 3.0, 1))
        edge_axis = tuple(sorted({lo, mid, hi}))
    else:
        edge_axis = IMPULSE_GRID.get(
            "min_expected_edge_bps",
            (getattr(_cfg, "FUSION_MIN_EXPECTED_EDGE_BPS", 8.0),),
        )
    return [
        IMPULSE_GRID["w_ml"],
        IMPULSE_GRID["w_mom"],
        IMPULSE_GRID["w_nw"],
        IMPULSE_GRID["stress_max"],
        IMPULSE_GRID["hmm_impulse_min"],
        IMPULSE_GRID.get("hmm_confidence_min", (getattr(_cfg, "FUSION_HMM_CONFIDENCE_MIN", 0.30),)),
        IMPULSE_GRID.get("hmm_entropy_max", (getattr(_cfg, "FUSION_HMM_ENTROPY_MAX", 1.05),)),
        IMPULSE_GRID.get("allow_mean_revert", (True,)),
        IMPULSE_GRID["impulse_min"],
        edge_axis,
        IMPULSE_GRID.get("gain", (80,)),
        IMPULSE_GRID.get("hold_threshold", (50,)),
        IMPULSE_GRID.get("buy_threshold", (55,)),
    ]


def _params_from_grid_combo(combo: tuple) -> dict | None:
    (
        w_ml, w_mom, w_nw, stress_max, hmm_impulse_min, hmm_confidence_min,
        hmm_entropy_max, allow_mean_revert, impulse_min, min_expected_edge_bps,
        gain, hold_threshold, buy_threshold,
    ) = combo
    w_flow = max(0.05, 1.0 - w_ml - w_mom - w_nw - 0.15)
    w_vp = 0.15
    if w_flow < 0:
        return None
    return {
        "w_ml": w_ml, "w_mom": w_mom, "w_nw": w_nw, "w_flow": w_flow, "w_vp": w_vp,
        "stress_max": stress_max, "hmm_impulse_min": hmm_impulse_min,
        "hmm_confidence_min": hmm_confidence_min, "hmm_entropy_max": hmm_entropy_max,
        "allow_mean_revert": allow_mean_revert,
        "impulse_min": impulse_min, "min_expected_edge_bps": min_expected_edge_bps, "gain": gain,
        "hold_threshold": hold_threshold, "buy_threshold": buy_threshold,
    }


def _init_grid_worker(ctx: dict) -> None:
    global _GRID_WORKER_CTX
    _GRID_WORKER_CTX = ctx


def _fusion_grid_workers() -> int:
    env = os.environ.get("FUSION_GRID_WORKERS")
    if env:
        return max(1, int(env))
    cfg_val = getattr(_cfg, "FUSION_GRID_WORKERS", None)
    if cfg_val is not None:
        return max(1, int(cfg_val))
    return max(1, (os.cpu_count() or 4) - 1)


def _evaluate_impulse_combo(combo: tuple) -> dict | None:
    """Score one impulse-grid candidate on purged session CV (worker-safe)."""
    ctx = _GRID_WORKER_CTX
    params = _params_from_grid_combo(combo)
    if params is None:
        return None

    prices = ctx["prices"]
    oos = ctx["oos"]
    folds = ctx["folds"]
    commission_bps = ctx["commission_bps"]

    fold_sharpes: list[float] = []
    fold_returns: list[float] = []
    fold_signal_rows: list[int] = []
    fold_active_rebalances: list[int] = []
    fold_exposures: list[float] = []
    fold_trade_net_bps: list[float] = []

    from simulation.engine import run_backtest_signal_exit
    from simulation.entry_signals import active_entry_signals, deoverlap_signals, trade_returns_from_signals
    from strategy.objective import fusion_cv_objective

    for _train_s, val_s in folds:
        val = oos[oos["session"].isin(val_s)]
        if val.empty:
            continue
        sig = _fusion_signal_frame(val, prices, params)
        if sig.empty:
            fold_signal_rows.append(0)
            fold_active_rebalances.append(0)
            fold_exposures.append(0.0)
            fold_returns.append(0.0)
            fold_sharpes.append(float("nan"))
            continue
        val_start = pd.Timestamp(val["bar_time"].min())
        val_end = pd.Timestamp(val["bar_time"].max())
        bt = run_backtest_signal_exit(
            prices, sig, score_col="score",
            use_dynamic_thresholds=True, use_vol_targeting=True,
            commission_bps=commission_bps,
            horizon_bars=int(ctx.get("hold_default", FWD_HORIZON_BARS)),
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
            stop_loss_bps=float(params.get("stop_loss_bps", getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0))),
            period_start=val_start,
            period_end=val_end,
        )
        stats = bt.get("stats") or {}
        metrics = _trade_constraint_metrics(sig, stats, bt.get("holdings", []))
        fold_signal_rows.append(metrics["signal_rows"])
        fold_active_rebalances.append(metrics["active_rebalances"])
        fold_exposures.append(metrics["avg_exposure_pct"])
        fold_returns.append(float(stats.get("total_return_pct") or 0.0))
        fold_sharpes.append(
            float(stats.get("sharpe_bar_annualized") or stats.get("sharpe") or 0.0)
        )
        entry_sig = active_entry_signals(sig)
        ev_sig = deoverlap_signals(entry_sig, prices, int(ctx.get("hold_default", FWD_HORIZON_BARS)))
        trade_rets = trade_returns_from_signals(ev_sig, commission_bps)
        fold_trade_net_bps.append(float(np.mean(trade_rets)) * 10_000.0 if len(trade_rets) else 0.0)

    if not fold_sharpes:
        return None

    valid_sharpes = [float(s) for s in fold_sharpes if np.isfinite(s)]
    if not valid_sharpes:
        return None
    sharpe = float(np.mean(valid_sharpes))
    ret = float(np.mean(fold_returns)) if fold_returns else 0.0
    constraint_metrics = {
        "signal_rows": int(np.mean(fold_signal_rows)) if fold_signal_rows else 0,
        "active_rebalances": int(np.mean(fold_active_rebalances)) if fold_active_rebalances else 0,
        "avg_exposure_pct": float(np.mean(fold_exposures)) if fold_exposures else 0.0,
    }
    constraints_ok = _passes_trade_constraints(constraint_metrics)
    penalty = 0.0 if constraints_ok else _constraint_penalty(constraint_metrics)
    mean_trade_net = float(np.mean(fold_trade_net_bps)) if fold_trade_net_bps else 0.0
    objective = fusion_cv_objective(
        sharpe=sharpe,
        mean_return_pct=ret,
        active_rebalances=constraint_metrics["active_rebalances"],
        mean_trade_net_bps=mean_trade_net,
        constraint_penalty=penalty,
        fold_days=90.0,
    )
    return {
        **params,
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(ret, 2),
        "n_folds": len(fold_sharpes),
        **constraint_metrics,
        "constraints_ok": constraints_ok,
        "constraint_penalty": round(penalty, 3),
        "return_component": round(ret_component, 3),
        "turnover_penalty": round(turnover_penalty, 3),
        "negative_return_penalty": round(negative_return_penalty, 3),
        "objective": round(objective, 3),
    }


def _compact_impulse_grid_axes(calibrated_edge: float) -> tuple[tuple, ...]:
    """Small monthly-WF grid that is feasible on multi-year OOS."""
    edge = max(float(calibrated_edge), float(getattr(_cfg, "FUSION_MIN_EXPECTED_EDGE_BPS", 8.0)))
    return (
        (0.35, 0.45),       # w_ml
        (0.20,),            # w_mom
        (0.15,),            # w_nw
        (0.45, 0.55),       # stress_max
        (0.05, 0.20),       # hmm_impulse_min
        (0.25,),            # hmm_confidence_min
        (1.05,),            # hmm_entropy_max
        (False,),           # allow_mean_revert
        (0.10, 0.20),       # impulse_min
        (edge,),            # min_expected_edge_bps
        (120,),             # gain
        (48,),              # hold_threshold
        (55, 60),           # buy_threshold
    )


def optimize_impulse_params(
    prices: pd.DataFrame,
    oos: pd.DataFrame,
    *,
    commission_bps: float,
    grid_profile: str = "legacy",
    calibration_panel: pd.DataFrame | None = None,
) -> tuple[dict, list[dict]]:
    """Purged session CV on walk-forward OOS — no arbitrary time split.

    ``calibration_panel`` must be train-only rows (with ``ml_proba``) for edge
    floor calibration; defaults to ``oos`` when omitted (legacy callers).
    """
    from research.features.entry_ml import _purged_session_folds

    if oos.empty or "ml_proba" not in oos.columns:
        best = {**DEFAULT_IMPULSE_WEIGHTS, "stress_max": 0.35, "hmm_impulse_min": 0.25,
                "impulse_min": 0.25, "gain": 80, "sharpe": 0.0}
        return best, []

    sessions = sorted(oos["session"].unique())
    folds = _purged_session_folds(sessions)
    print(f"    impulse grid: purged CV on {len(oos):,} OOS bars | {len(folds)} folds...", flush=True)
    results: list[dict] = []
    best: dict | None = None
    best_objective = -999.0
    fallback_best: dict | None = None
    fallback_objective = -999.0

    from operations.progress import ProgressReporter

    cal_src = calibration_panel if calibration_panel is not None else oos
    calibrated_edge = calibrate_min_expected_edge_bps(cal_src, commission_bps)
    print(f"    calibrated min_expected_edge_bps={calibrated_edge:.1f}", flush=True)
    use_compact = grid_profile in ("monthly4y", "monthly") and bool(getattr(_cfg, "FUSION_MONTHLY_COMPACT_GRID", True))
    skip_grid = grid_profile in ("monthly4y", "monthly") and bool(getattr(_cfg, "FUSION_MONTHLY_SKIP_GRID", True))
    if skip_grid:
        fixed = {
            **DEFAULT_IMPULSE_WEIGHTS,
            "w_ml": 0.45,
            "w_mom": 0.20,
            "w_nw": 0.15,
            "w_flow": 0.05,
            "w_vp": 0.15,
            "stress_max": 0.55,
            "hmm_impulse_min": 0.05,
            "hmm_confidence_min": 0.20,
            "hmm_entropy_max": 1.05,
            "allow_mean_revert": True,
            "impulse_min": 0.05,
            "min_expected_edge_bps": calibrated_edge,
            "gain": 100,
            "hold_threshold": 45,
            "buy_threshold": 50,
            "stop_loss_bps": float(getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0)),
            "disable_trading": False,
            "grid_skipped": True,
            "grid_profile": "fixed_monthly4y",
            "calibrated_min_expected_edge_bps": calibrated_edge,
        }
        print("    impulse grid skipped: fixed monthly4y policy", flush=True)
        return fixed, []
    axes = _compact_impulse_grid_axes(calibrated_edge) if use_compact else _impulse_grid_axes(calibrated_edge)
    combos = [c for c in product(*axes) if _params_from_grid_combo(c) is not None]
    workers = _fusion_grid_workers()
    if use_compact:
        workers = int(getattr(_cfg, "FUSION_MONTHLY_GRID_WORKERS", 1))
    profile_label = "compact" if use_compact else "full"
    print(f"    impulse grid: {len(combos):,} candidates | workers={workers} | profile={profile_label}", flush=True)
    _rep = ProgressReporter(len(combos), "impulse grid")

    worker_ctx = {
        "prices": prices,
        "oos": oos,
        "folds": folds,
        "commission_bps": commission_bps,
        "hold_default": _fusion_hold_default(),
    }

    def _consume(rec: dict | None) -> None:
        nonlocal best, best_objective, fallback_best, fallback_objective
        if rec is None:
            return
        results.append(rec)
        objective = float(rec["objective"])
        if objective > fallback_objective:
            fallback_objective = objective
            fallback_best = rec
        if rec.get("constraints_ok") and objective > 0 and objective > best_objective:
            best_objective = objective
            best = rec

    if workers <= 1:
        _init_grid_worker(worker_ctx)
        for combo in combos:
            _rep.update()
            _consume(_evaluate_impulse_combo(combo))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_grid_worker,
            initargs=(worker_ctx,),
        ) as pool:
            futures = {pool.submit(_evaluate_impulse_combo, combo): combo for combo in combos}
            for fut in as_completed(futures):
                _rep.update()
                try:
                    _consume(fut.result())
                except Exception as exc:
                    print(f"    impulse grid worker error: {exc}", flush=True)

    _rep.close()
    if best is None:
        fb = fallback_best or {}
        min_rows = int(getattr(_cfg, "FUSION_MIN_SIGNAL_ROWS", 20))
        min_reb = int(getattr(_cfg, "FUSION_MIN_ACTIVE_REBALANCES", 2))
        if fb.get("signal_rows", 0) >= min_rows and fb.get("active_rebalances", 0) >= min_reb:
            best = {
                **fb,
                "disable_trading": False,
                "constraints_fallback": True,
                "calibrated_min_expected_edge_bps": calibrated_edge,
            }
        else:
            best = {
                **(fb or DEFAULT_IMPULSE_WEIGHTS),
                "disable_trading": True,
                "no_valid_candidate": True,
                "constraints_fallback": True,
                "calibrated_min_expected_edge_bps": calibrated_edge,
            }
    else:
        best = {
            **best,
            "constraints_fallback": False,
            "calibrated_min_expected_edge_bps": calibrated_edge,
        }
    return best, results


def _resolved_min_edge_bps(
    impulse_params: dict,
    *,
    commission_bps: float,
    ticker: str | None = None,
) -> float:
    from strategy.edge_gate import resolve_min_expected_edge_bps, resolve_ticker_min_edge_bps

    if ticker is not None:
        return resolve_ticker_min_edge_bps(ticker, impulse_params)
    requested = float(impulse_params.get(
        "min_expected_edge_bps", getattr(_cfg, "FUSION_MIN_EXPECTED_EDGE_BPS", 0.0),
    ))
    calibrated = impulse_params.get("calibrated_min_expected_edge_bps")
    return resolve_min_expected_edge_bps(
        requested,
        commission_bps=commission_bps,
        calibrated=float(calibrated) if calibrated is not None else None,
    )


def _entry_active_mask(fused: pd.DataFrame, impulse_params: dict) -> pd.Series:
    """Impulse + signed expected-edge gate (per-instrument min edge when ticker present)."""
    from strategy.edge_gate import resolve_ticker_min_edge_bps, signed_edge_active_mask
    from strategy.side_policy import side_policy_mask

    if fused.empty:
        return pd.Series(dtype=bool)

    impulse_min = float(impulse_params.get("impulse_min", 0.25))
    if "ticker" in fused.columns:
        min_edge = fused["ticker"].astype(str).map(
            lambda t: resolve_ticker_min_edge_bps(t, impulse_params)
        )
    else:
        comm = float(getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 1.1))
        min_edge = _resolved_min_edge_bps(impulse_params, commission_bps=comm)

    side = (
        fused["position_side"].fillna(0).astype(int)
        if "position_side" in fused.columns
        else pd.Series(1, index=fused.index, dtype=int)
    )
    edge_ok = signed_edge_active_mask(
        fused["expected_edge_bps"].astype(float),
        side,
        min_edge,
    )
    imp_ok = fused["impulse_strength"].astype(float).to_numpy() >= impulse_min
    active = np.asarray(edge_ok, dtype=bool) & np.asarray(imp_ok, dtype=bool)
    if "ticker" in fused.columns:
        active = active & side_policy_mask(fused["ticker"], side).to_numpy()
    return pd.Series(active, index=fused.index)


def _gated_entries(oos: pd.DataFrame, impulse_params: dict) -> pd.DataFrame:
    """OOS rows that pass the gate / impulse / edge filters (pre-horizon)."""
    fused = apply_fusion_scores(oos, impulse_params)
    hard_gate = bool(getattr(_cfg, "FUSION_HMM_HARD_GATE", False))
    mask = _entry_active_mask(fused, impulse_params)
    if hard_gate:
        mask &= fused["hmm_gate"]
    return fused[mask].copy()


def _select_adaptive_horizons(
    oos: pd.DataFrame,
    prices: pd.DataFrame,
    impulse_params: dict,
    *,
    commission_bps: float,
) -> tuple[dict[str, int] | None, dict]:
    """Pick a per-regime holding horizon by de-overlapped after-cost EV.

    Disabled (returns ``None``) when the optimizer chose no-trade or the flag
    ``FUSION_ADAPTIVE_HORIZON`` is off, so the backtest keeps the fixed horizon.
    """
    detail = {"enabled": False, "regime_horizons": None}
    if not bool(getattr(_cfg, "FUSION_ADAPTIVE_HORIZON", True)):
        return None, detail
    if impulse_params.get("disable_trading") or oos.empty:
        return None, detail

    from simulation.adaptive_horizon import select_regime_horizons

    entries = _gated_entries(oos, impulse_params)
    if entries.empty:
        return None, detail
    entries["date"] = pd.to_datetime(entries["bar_time"])
    regime_horizons, sel_detail = select_regime_horizons(
        entries, prices,
        commission_bps=commission_bps,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
        default_horizon=_fusion_hold_default(),
    )
    return regime_horizons, {"enabled": True, "regime_horizons": regime_horizons, **sel_detail}


def _fusion_signal_frame(
    oos: pd.DataFrame,
    prices: pd.DataFrame,
    impulse_params: dict,
    regime_horizons: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Apply fusion scores + gates and build bar signals for backtest."""
    from simulation.signal_frame import build_flow_signal_frame
    from strategy.instrument_adapter import per_ticker_exposure_budget
    from strategy.threshold_calibrator import resolve_ticker_policy

    if impulse_params.get("disable_trading"):
        return pd.DataFrame()
    fused = apply_fusion_scores(oos, impulse_params)
    if fused.empty:
        return pd.DataFrame()
    by_ticker = impulse_params.get("by_ticker") or {}
    if by_ticker and "ticker" in fused.columns:
        parts: list[pd.DataFrame] = []
        policy_by_ticker = impulse_params.get("by_ticker") or {}
        tradeable = impulse_params.get("tradeable_tickers")
        allowed = {str(t).upper() for t in tradeable} if tradeable else None
        budget = per_ticker_exposure_budget(list(tradeable or []))
        for ticker, grp in fused.groupby("ticker", sort=False):
            sym = str(ticker).upper()
            if allowed is not None and sym not in allowed:
                continue
            tpol = policy_by_ticker.get(sym) or {}
            pol = resolve_ticker_policy(impulse_params, sym)
            pol = {k: v for k, v in pol.items() if k not in ("by_ticker", "threshold_calibrator")}
            # SQ v2 keepers: soft-size, do not last-fold hard-skip.
            sq_keep = bool(pol.get("sq_soft_keep") or tpol.get("sq_soft_keep"))
            if (
                pol.get("signal_quality_ok") is False
                and not sq_keep
                and not bool(getattr(_cfg, "FUSION_DISABLE_QUALITY_GATE", False))
            ):
                continue
            from strategy.soft_sizing import soft_size_block_reason, soft_size_multiplier

            if soft_size_multiplier(pol) <= 0.0:
                reason = soft_size_block_reason(pol) or "exposure_cap=0"
                print(
                    f"      soft-size hard-zero [{sym}] — {reason}",
                    flush=True,
                )
                continue
            pol = dict(pol)
            pol["_exposure_budget"] = budget.get(sym, 1.0)
            part = _fusion_signal_frame(grp, prices, pol, regime_horizons)
            if not part.empty:
                parts.append(part)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)
    tradeable_list = list(impulse_params.get("tradeable_tickers") or [])
    tradeable = impulse_params.get("tradeable_tickers")
    if (
        tradeable
        and "ticker" in fused.columns
        and len(tradeable_list) > 1
        and bool(getattr(_cfg, "FUSION_PER_TICKER_EXPOSURE_BUDGET", True))
    ):
        budget = per_ticker_exposure_budget(tradeable_list)
        allowed = {str(t).upper() for t in tradeable}
        parts: list[pd.DataFrame] = []
        for ticker, grp in fused.groupby("ticker", sort=False):
            sym = str(ticker).upper()
            if sym not in allowed:
                continue
            pol = dict(impulse_params)
            pol["_exposure_budget"] = budget.get(sym, 1.0)
            pol["tradeable_tickers"] = [sym]
            px = prices[[sym]] if sym in prices.columns else prices
            part = _fusion_signal_frame(grp, px, pol, regime_horizons)
            if not part.empty:
                parts.append(part)
        if parts:
            return pd.concat(parts, ignore_index=True)
    if tradeable and "ticker" in fused.columns:
        allowed = {str(t).upper() for t in tradeable}
        mask_t = fused["ticker"].astype(str).str.upper().isin(allowed)
        fused = fused.copy()
        fused.loc[~mask_t, "fusion_score"] = 0.0
        if "position_side" in fused.columns:
            fused.loc[~mask_t, "position_side"] = 0
    hard_gate = bool(getattr(_cfg, "FUSION_HMM_HARD_GATE", False))
    active = _entry_active_mask(fused, impulse_params)
    if hard_gate:
        active &= fused["hmm_gate"].astype(bool)
    from strategy.soft_sizing import soft_size_multiplier

    budget_mult = float(impulse_params.get("_exposure_budget", 1.0))
    from strategy.tail_filters import apply_tail_entry_filters, tail_entry_skip_mask

    fused = fused.copy()
    tail_skip = tail_entry_skip_mask(fused)
    active &= ~tail_skip
    fused = apply_tail_entry_filters(fused)
    fused.loc[~active, "fusion_score"] = 0.0
    if "position_side" in fused.columns:
        fused.loc[~active, "position_side"] = 0
    # HMM gate already encodes stress/risk-on logic. Avoid applying the legacy
    # daily risk_on filter a second time in build_flow_signal_frame/backtest.
    fused["hmm_risk_on"] = True
    fused["risk_on"] = True
    exposure_cap = soft_size_multiplier(impulse_params) * budget_mult
    return build_flow_signal_frame(
        fused, prices,
        gain=impulse_params.get("gain", 80),
        hold_threshold=impulse_params.get("hold_threshold", 50),
        buy_threshold=impulse_params.get("buy_threshold", 55),
        score_col="fusion_score",
        regime_horizons=regime_horizons,
        default_horizon=_fusion_hold_default(),
        stop_loss_bps=float(impulse_params.get("stop_loss_bps", getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0))),
        exposure_cap=exposure_cap,
    )


def _target_forward_return_col(df: pd.DataFrame, target_col: str | None = None) -> str:
    """Forward-return column that matches the active target."""
    if target_col == "label_entry" and "fwd_ret_entry" in df.columns:
        return "fwd_ret_entry"
    if "fwd_ret_entry" in df.columns and target_col == TARGET_12_AFTER_COSTS:
        return "fwd_ret"
    return "fwd_ret_entry" if "fwd_ret_entry" in df.columns and target_col == "label_entry" else "fwd_ret"


def _signal_diagnostics(oos: pd.DataFrame, sig: pd.DataFrame, bt: dict) -> dict:
    """Compact OOS diagnostics for edge concentration and trading costs."""
    from simulation.entry_signals import active_entry_signals

    actionable = active_entry_signals(sig) if not sig.empty else sig
    out: dict = {
        "signal_rows": int(len(actionable)),
        "signal_frame_rows": int(len(sig)),
        "oos_rows": int(len(oos)),
        "signal_coverage_pct": round(float(len(actionable) / max(len(oos), 1)) * 100, 2),
    }
    ret_col = _target_forward_return_col(oos, (oos.attrs or {}).get("target_col"))
    if not oos.empty and "ml_proba" in oos.columns and ret_col in oos.columns:
        q = oos.copy()
        try:
            q["proba_decile"] = pd.qcut(q["ml_proba"], 10, labels=False, duplicates="drop")
            dec = []
            for d, part in q.groupby("proba_decile"):
                dec.append({
                    "decile": int(d),
                    "n": int(len(part)),
                    "mean_proba": round(float(part["ml_proba"].mean()), 4),
                    "hit_rate": round(float((part[ret_col] > 0).mean()), 4),
                    "mean_fwd_ret_bps": round(float(part[ret_col].mean()) * 10_000, 3),
                })
            out["probability_deciles"] = dec
        except ValueError:
            out["probability_deciles"] = []

    sig_ret_col = _target_forward_return_col(sig, (sig.attrs or {}).get("target_col"))
    if not sig.empty and sig_ret_col in sig.columns:
        regime_cols = {
            "impulse": COL_PROB_HMM_IMPULSE,
            "mean_revert": COL_PROB_HMM_MEAN_REVERT,
            "stress": COL_PROB_HMM_STRESS,
        }
        available = {name: col for name, col in regime_cols.items() if col in sig.columns}
        if available:
            s = sig.copy()
            s["dominant_hmm_regime"] = s[list(available.values())].idxmax(axis=1).map(
                {col: name for name, col in available.items()}
            )
            regimes = []
            for regime, part in s.groupby("dominant_hmm_regime"):
                regimes.append({
                    "regime": str(regime),
                    "n": int(len(part)),
                    "hit_rate": round(float((part[sig_ret_col] > 0).mean()), 4),
                    "mean_fwd_ret_bps": round(float(part[sig_ret_col].mean()) * 10_000, 3),
                    "mean_expected_edge_bps": round(float(part.get("expected_edge_bps", pd.Series([0.0])).mean()), 3),
                })
            out["signals_by_hmm_regime"] = regimes

    stats = bt.get("stats") or {}
    holdings = bt.get("holdings") or []
    total_commission = float(stats.get("total_commission_pct", 0.0) or 0.0)
    total_return = float(stats.get("total_return_pct", 0.0) or 0.0)
    out["costs"] = {
        "total_return_pct": round(total_return, 3),
        "gross_return_approx_pct": round(total_return + total_commission, 3),
        "total_commission_pct": round(total_commission, 3),
        "active_rebalances": int(sum(1 for h in holdings if h.get("n", 0) > 0)),
        "avg_names_held": round(float(np.mean([h.get("n", 0) for h in holdings])) if holdings else 0.0, 3),
        "signal_exit_count": int(stats.get("signal_exit_count", 0) or 0),
    }
    return out


def _feature_quality_report(panel: pd.DataFrame, feat_cols: list[str]) -> dict:
    """NaN/constant coverage to catch useless or broken ML inputs."""
    rows: list[dict] = []
    for col in feat_cols:
        if col not in panel.columns:
            rows.append({"feature": col, "missing": True})
            continue
        s = pd.to_numeric(panel[col], errors="coerce")
        finite = s[np.isfinite(s)]
        rows.append(
            {
                "feature": col,
                "nan_pct": round(float(s.isna().mean()) * 100.0, 3),
                "zero_pct": round(float((s.fillna(0.0) == 0.0).mean()) * 100.0, 3),
                "n_unique": int(finite.nunique()) if len(finite) else 0,
                "std": round(float(finite.std()) if len(finite) else 0.0, 6),
                "constant": bool(len(finite) == 0 or finite.nunique() <= 1 or float(finite.std()) < 1e-12),
            }
        )
    problem = [r for r in rows if r.get("missing") or r.get("constant") or float(r.get("nan_pct", 0.0)) > 25.0]
    return {"n_features": len(feat_cols), "problem_count": len(problem), "problems": problem[:30]}


def _monthly_oos_diagnostics(oos: pd.DataFrame, target_col: str, cv: dict) -> dict:
    """CV-vs-OOS and calibration drift summaries for stitched monthly OOS."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    if oos.empty or target_col not in oos.columns or "ml_proba" not in oos.columns:
        return {}
    y = oos[target_col].astype(int).to_numpy()
    p = np.clip(oos["ml_proba"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    auc = 0.5
    if len(np.unique(y)) > 1:
        try:
            auc = float(roc_auc_score(y, p))
        except Exception:
            auc = 0.5
    ll = float(log_loss(y, p)) if len(np.unique(y)) > 1 else 0.6931
    brier = float(brier_score_loss(y, p)) if len(y) else 0.25
    cv_auc = float(cv.get("auc", 0.5) or 0.5)
    cv_ll = float(cv.get("log_loss", 0.6931) or 0.6931)

    by_fold: list[dict] = []
    if "wf_fold" in oos.columns:
        for fold, part in oos.groupby("wf_fold", sort=True):
            yy = part[target_col].astype(int).to_numpy()
            pp = np.clip(part["ml_proba"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
            row = {
                "fold": int(fold),
                "n": int(len(part)),
                "mean_pred": round(float(np.mean(pp)), 4),
                "observed_freq": round(float(np.mean(yy)), 4),
                "calibration_gap": round(float(np.mean(yy) - np.mean(pp)), 4),
            }
            if len(np.unique(yy)) > 1:
                try:
                    row["auc"] = round(float(roc_auc_score(yy, pp)), 4)
                    row["log_loss"] = round(float(log_loss(yy, pp)), 4)
                except Exception:
                    pass
            by_fold.append(row)

    ret_col = _target_forward_return_col(oos, target_col)
    top_decile_edge = None
    if ret_col in oos.columns:
        cut = float(np.quantile(p, 0.9))
        top = oos[p >= cut]
        if not top.empty:
            top_decile_edge = round(float(top[ret_col].mean()) * 10_000.0, 3)

    return {
        "cv_auc": round(cv_auc, 4),
        "oos_auc": round(auc, 4),
        "auc_gap": round(float(auc - cv_auc), 4),
        "cv_log_loss": round(cv_ll, 4),
        "oos_log_loss": round(ll, 4),
        "log_loss_gap": round(float(ll - cv_ll), 4),
        "oos_brier": round(brier, 4),
        "top_decile_target_fwd_ret_bps": top_decile_edge,
        "monthly": by_fold[:80],
    }


def walk_forward_fusion_oos(
    panel: pd.DataFrame,
    feat_cols: list[str],
    params: dict,
    *,
    model_name: str = "lightgbm",
    max_oos_sessions: int | None = None,
    target_col: str = "label",
) -> pd.DataFrame:
    """Causal session-by-session ml_proba (no lookahead)."""
    from research.features.entry_ml import REOPT_EVERY_SESSIONS, TRAIN_SESSIONS_MIN, WARMUP_SESSIONS
    from models.entry_model import make_entry_classifier

    sessions = sorted(panel["session"].unique())
    start_i = WARMUP_SESSIONS
    if max_oos_sessions is not None and max_oos_sessions > 0:
        start_i = max(WARMUP_SESSIONS, len(sessions) - max_oos_sessions)

    parts: list[pd.DataFrame] = []
    cached = None
    cached_calibration = None
    cached_base_rate: float | None = None
    from operations.progress import ProgressReporter

    _rep = ProgressReporter(max(0, len(sessions) - start_i), "fusion walk-forward OOS")
    for i, sess in enumerate(sessions):
        if i < start_i:
            continue
        _rep.update()
        train_sess = sessions[max(0, i - TRAIN_SESSIONS_MIN): i]
        if len(train_sess) < TRAIN_SESSIONS_MIN // 2:
            continue
        tr = panel[panel["session"].isin(train_sess)].dropna(subset=feat_cols + [target_col])
        te = panel[panel["session"] == sess].dropna(subset=feat_cols + [target_col])
        if len(tr) < 100 or te.empty:
            continue
        if i % REOPT_EVERY_SESSIONS == 0 or cached is None:
            if tr[target_col].nunique() < 2:
                if cached is None:
                    continue
            else:
                cached = make_entry_classifier(model_name, params)
                val_cut = max(1, int(len(tr) * 0.85))
                tr_fit, va = tr.iloc[:val_cut], tr.iloc[val_cut:]
                cached.fit(tr_fit[feat_cols], tr_fit[target_col])
                raw_val = cached.predict_proba(va[feat_cols])[:, 1] if not va.empty else np.array([])
                cached_calibration = (va[target_col], raw_val)
                cached_base_rate = float(tr[target_col].mean())
        p_raw = cached.predict_proba(te[feat_cols])[:, 1]
        y_val, p_val = cached_calibration if cached_calibration is not None else (pd.Series(dtype=float), np.array([]))
        p = _calibrate_probabilities(y_val, p_val, p_raw)
        chunk = te.copy()
        chunk["ml_proba_raw"] = p_raw
        chunk["ml_proba"] = p
        chunk["ml_base_rate"] = float(cached_base_rate if cached_base_rate is not None else tr[target_col].mean())
        parts.append(chunk)

    _rep.close()
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def monthly_walk_forward_windows(
    panel: pd.DataFrame,
    *,
    train_days: int = 365,
    backtest_years: int = 4,
    test_months: int = 1,
) -> list[dict]:
    """Calendar windows: trailing train_days -> next test_months OOS.

    The first test month is the later of:
    - max(panel_date) - backtest_years, aligned to month start;
    - min(panel_date) + train_days, aligned to the next month start.

    All ranges are half-open: [start, end), so train never includes the tested
    month and each OOS bar appears in at most one monthly fold.
    """
    if panel.empty or "bar_time" not in panel.columns:
        return []
    dates = pd.to_datetime(panel["bar_time"])
    min_ts = pd.Timestamp(dates.min()).normalize()
    max_ts = pd.Timestamp(dates.max()).normalize()
    earliest = min_ts + pd.Timedelta(days=int(train_days))
    desired = max_ts - pd.DateOffset(years=int(backtest_years))
    start = max(pd.Timestamp(desired), pd.Timestamp(earliest)).normalize().replace(day=1)
    if start < earliest:
        start = start + pd.offsets.MonthBegin(1)
    end_limit = max_ts + pd.Timedelta(days=1)
    step = max(1, int(test_months))
    windows: list[dict] = []
    fold = 0
    test_start = pd.Timestamp(start)
    while test_start < end_limit:
        test_end = min(test_start + pd.DateOffset(months=step), end_limit)
        train_end = test_start
        train_start = train_end - pd.Timedelta(days=int(train_days))
        if train_start >= min_ts and test_end > test_start:
            windows.append(
                {
                    "fold": fold,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )
            fold += 1
        test_start = test_start + pd.DateOffset(months=step)
    return windows


def limit_walk_forward_windows(
    windows: list[dict],
    max_folds: int | None,
) -> list[dict]:
    """Keep the first ``max_folds`` chronological OOS windows (None = all)."""
    if max_folds is None:
        return windows
    cap = int(max_folds)
    if cap <= 0 or len(windows) <= cap:
        return windows
    return [{**w, "fold": i} for i, w in enumerate(windows[:cap])]


def _oos_flat_equity(
    prices: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    """Flat strategy equity on OOS bar index (for Monte Carlo / plots when no trades)."""
    if prices.empty:
        return pd.DataFrame()
    bars = prices.index[1:]
    bars = bars[(bars >= period_start) & (bars <= period_end)]
    if len(bars) < 1:
        return pd.DataFrame()
    return pd.DataFrame({"value": np.ones(len(bars), dtype=float)}, index=bars)


def _monthly_fold_feature_importance(clf, feat_cols: list[str]) -> dict[str, float]:
    model = getattr(clf, "named_steps", {}).get("clf") if hasattr(clf, "named_steps") else clf
    vals = getattr(model, "feature_importances_", None)
    if vals is None:
        coef = getattr(model, "coef_", None)
        if coef is not None:
            vals = np.abs(np.asarray(coef).ravel())
    if vals is None:
        return {}
    arr = np.asarray(vals, dtype=float)
    if len(arr) != len(feat_cols):
        return {}
    total = float(np.sum(np.abs(arr)))
    if total <= 0:
        return {}
    pairs = sorted(zip(feat_cols, arr / total), key=lambda x: abs(float(x[1])), reverse=True)
    return {k: round(float(v), 6) for k, v in pairs[:20]}


def _resolve_backtest_friction(commission_bps: float) -> tuple[float, float]:
    """Return (commission_bps, slippage_bps) for backtest — gross when configured."""
    from research.labels.trade import DEFAULT_SLIPPAGE_BPS

    if bool(getattr(_cfg, "FUSION_BACKTEST_GROSS_ONLY", False)):
        return 0.0, 0.0
    return float(commission_bps), float(DEFAULT_SLIPPAGE_BPS)


def _per_ticker_oos_backtests(
    oos: pd.DataFrame,
    prices: pd.DataFrame,
    wf_folds: list[dict],
    impulse_params: dict,
    symbols: list[str],
    *,
    regime_horizons: dict[str, int] | None,
    commission_bps: float,
    oos0: pd.Timestamp | None,
    oos1: pd.Timestamp | None,
    hold_default: int,
    monthly_policies: bool,
    gated: bool = False,
) -> dict[str, dict]:
    """Independent backtest per symbol (ungated solo by default; gated respects live book)."""
    from research.labels.trade import DEFAULT_SLIPPAGE_BPS
    from simulation.engine import run_backtest_signal_exit

    bt_commission, bt_slippage = _resolve_backtest_friction(commission_bps)
    live_tradeable = {str(t).upper() for t in (impulse_params.get("tradeable_tickers") or [])}
    results: dict[str, dict] = {}
    for sym in symbols:
        sym_u = str(sym).upper()
        if sym_u not in prices.columns:
            continue
        if gated and live_tradeable and sym_u not in live_tradeable:
            results[sym_u] = {
                "skipped": True,
                "reason": "not_in_live_tradeable_set",
                "gated": True,
                "total_return_pct": 0.0,
                "n_signals": 0,
            }
            continue
        oos_sym = (
            oos[oos["ticker"].astype(str).str.upper() == sym_u]
            if "ticker" in oos.columns
            else oos
        )
        if oos_sym.empty:
            results[sym_u] = {"skipped": True, "reason": "no_oos_rows"}
            continue
        pol = dict(impulse_params)
        pol["disable_trading"] = False
        if gated:
            pol["tradeable_tickers"] = sorted(live_tradeable) if live_tradeable else [sym_u]
            pol["disabled_tickers"] = [s for s in symbols if str(s).upper() not in pol["tradeable_tickers"]]
        else:
            pol["tradeable_tickers"] = [sym_u]
            pol["disabled_tickers"] = [s for s in symbols if str(s).upper() != sym_u]
        px = prices[[sym_u]]
        if monthly_policies and wf_folds:
            sig = _fusion_signal_frame_monthly(oos_sym, px, wf_folds, pol, regime_horizons)
        else:
            sig = _fusion_signal_frame(oos_sym, px, pol, regime_horizons)
        if sig.empty:
            results[sym_u] = {
                "total_return_pct": 0.0,
                "sharpe": 0.0,
                "n_signals": 0,
                "no_trades": True,
                "gated": gated,
            }
            continue
        bt = run_backtest_signal_exit(
            px,
            sig,
            score_col="score",
            use_dynamic_thresholds=True,
            use_vol_targeting=True,
            commission_bps=bt_commission,
            horizon_bars=hold_default,
            slippage_bps=bt_slippage,
            period_start=oos0,
            period_end=oos1,
        )
        stats = dict(bt.get("stats") or {})
        stats["n_signals"] = int(len(sig))
        stats["gated"] = gated
        results[sym_u] = stats
    return results


def _fusion_signal_frame_monthly(
    oos: pd.DataFrame,
    prices: pd.DataFrame,
    wf_folds: list[dict],
    fallback_params: dict,
    regime_horizons: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build stitched signals using per-fold trading policies from train optimization."""
    policy_by_fold: dict[int, dict] = {}
    horizon_by_fold: dict[int, dict[str, int] | None] = {}
    for fold in wf_folds:
        if fold.get("skipped"):
            continue
        fid = int(fold["fold"])
        pol = fold.get("trading_policy") or (fold.get("threshold_optimization") or {}).get("best_params")
        if pol:
            policy_by_fold[fid] = pol
        rh = fold.get("regime_horizons")
        if rh:
            horizon_by_fold[fid] = rh
    if "wf_fold" not in oos.columns or not policy_by_fold:
        return _fusion_signal_frame(oos, prices, fallback_params, regime_horizons)

    parts: list[pd.DataFrame] = []
    for fold_id, grp in oos.groupby("wf_fold", sort=True):
        params = dict(policy_by_fold.get(int(fold_id), fallback_params))
        # Portfolio-level live book overlays fold CV policies.
        if fallback_params.get("tradeable_tickers") is not None:
            params["tradeable_tickers"] = list(fallback_params["tradeable_tickers"])
        if fallback_params.get("disabled_tickers") is not None:
            params["disabled_tickers"] = list(fallback_params["disabled_tickers"])
        if "disable_trading" in fallback_params:
            params["disable_trading"] = bool(fallback_params["disable_trading"])
        fb_by = fallback_params.get("by_ticker") or {}
        if fb_by:
            fold_by = dict(params.get("by_ticker") or {})
            for sym, fb_pol in fb_by.items():
                merged = dict(fold_by.get(sym) or {})
                for k in ("sq_pass_rate", "sq_soft_keep", "ticker", "symbol"):
                    if k in fb_pol:
                        merged[k] = fb_pol[k]
                fold_by[str(sym).upper()] = merged
            params["by_ticker"] = fold_by
        fold_horizons = horizon_by_fold.get(int(fold_id), regime_horizons)
        sig = _fusion_signal_frame(grp, prices, params, fold_horizons)
        if not sig.empty:
            parts.append(sig)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _default_monthly_impulse_fallback(calibrated_edge: float = 12.0) -> dict:
    from strategy.edge_gate import edge_gate_floor_mode

    return {
        **DEFAULT_IMPULSE_WEIGHTS,
        "w_ml": 0.45,
        "w_mom": 0.20,
        "w_nw": 0.15,
        "w_flow": 0.05,
        "w_vp": 0.15,
        "stress_max": 0.55,
        "hmm_impulse_min": 0.05,
        "hmm_confidence_min": 0.20,
        "hmm_entropy_max": 1.05,
        "allow_mean_revert": True,
        "impulse_min": 0.05,
        "min_expected_edge_bps": calibrated_edge,
        "gain": 100,
        "hold_threshold": 49,
        "buy_threshold": 51,
        "stop_loss_bps": float(getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0)),
        "edge_floor_mode": edge_gate_floor_mode(),
        "disable_trading": False,
        "grid_profile": "fallback_monthly",
    }


def attach_execution_cost_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-ticker round-trip cost (bps) as an ML feature."""
    if panel.empty or "ticker" not in panel.columns:
        return panel
    if "rt_cost_bps" in panel.columns:
        return panel
    from simulation.execution_costs import round_trip_cost_bps_for_ticker

    out = panel.copy()
    out["rt_cost_bps"] = out["ticker"].astype(str).map(round_trip_cost_bps_for_ticker).astype(float)
    return out


def _with_cost_feature(feat_cols: list[str], panel: pd.DataFrame) -> list[str]:
    cols = list(feat_cols)
    if "rt_cost_bps" in panel.columns and "rt_cost_bps" not in cols:
        cols.append("rt_cost_bps")
    return cols


def walk_forward_fusion_oos_monthly(
    panel: pd.DataFrame,
    feat_cols: list[str],
    params: dict,
    *,
    model_name: str = "lightgbm",
    target_col: str = "label",
    train_days: int = 365,
    backtest_years: int = 4,
    test_months: int = 1,
    min_train_rows: int | None = None,
    calibration_fraction: float | None = None,
    optimize_per_fold: bool | None = None,
    optimize_thresholds_per_fold: bool | None = None,
    optimize_targets_per_fold: bool | None = None,
    prices: pd.DataFrame | None = None,
    commission_bps: float = 10.0,
    max_folds: int | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Calendar stitched OOS: trailing train window -> test block -> advance.

    Default (``adaptive`` mode): 12m train, 6m OOS, recency-weighted LightGBM with
    warm-start incremental trees between blocks. Hyperparameters re-optimized each block.
    """
    from models.adaptive_trainer import AdaptiveEntryModel, recency_sample_weights
    from models.asset_class_models import AssetClassModelBundle, asset_class_models_enabled
    from models.entry_model import make_entry_classifier
    from research.features.entry_ml import _auc
    from sklearn.metrics import log_loss

    adaptive = is_adaptive_wf_mode()

    if panel.empty:
        return pd.DataFrame(), []
    work = panel.copy()
    work["bar_time"] = pd.to_datetime(work["bar_time"])
    work = work.sort_values(["bar_time", "ticker"]).reset_index(drop=True)
    windows = monthly_walk_forward_windows(
        work,
        train_days=train_days,
        backtest_years=backtest_years,
        test_months=test_months,
    )
    cap = max_folds
    if cap is None:
        cap = getattr(_cfg, "FUSION_WF_MAX_FOLDS", None)
    n_total = len(windows)
    windows = limit_walk_forward_windows(windows, cap)
    if cap is not None and n_total > len(windows):
        print(
            f"    monthly WF: {len(windows)}/{n_total} folds (max_folds={int(cap)})",
            flush=True,
        )
    if not windows:
        return pd.DataFrame(), []

    min_rows = int(min_train_rows if min_train_rows is not None else getattr(_cfg, "FUSION_WF_MIN_TRAIN_ROWS", 20_000))
    cal_frac = float(
        calibration_fraction
        if calibration_fraction is not None
        else getattr(_cfg, "FUSION_WF_CALIBRATION_FRACTION", 0.15)
    )
    cal_frac = min(max(cal_frac, 0.0), 0.4)
    per_fold_opt = (
        bool(getattr(_cfg, "FUSION_MONTHLY_OPTIMIZE_PER_FOLD", True))
        if optimize_per_fold is None
        else bool(optimize_per_fold)
    )
    _th_mode = str(getattr(_cfg, "FUSION_THRESHOLD_POLICY_MODE", "calibrated")).lower()
    per_fold_th = (
        _th_mode != "off"
        and (
            bool(getattr(_cfg, "FUSION_MONTHLY_OPTIMIZE_THRESHOLDS_PER_FOLD", True))
            if optimize_thresholds_per_fold is None
            else bool(optimize_thresholds_per_fold)
        )
    )
    per_fold_target = (
        bool(getattr(_cfg, "FUSION_MONTHLY_OPTIMIZE_TARGETS_PER_FOLD", True))
        if optimize_targets_per_fold is None
        else bool(optimize_targets_per_fold)
    )
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    from operations.progress import ProgressReporter
    from strategy.leakage_guard import resolve_label_horizon_bars, split_fit_calibration, trim_label_embargo

    label_horizon = resolve_label_horizon_bars()
    progress_label = "fusion adaptive OOS" if adaptive else "fusion monthly OOS"
    if per_fold_opt or per_fold_th or per_fold_target:
        from common.runtime import log_fold_parallelism

        log_fold_parallelism()
    _rep = ProgressReporter(len(windows), progress_label)
    adaptive_model: AdaptiveEntryModel | None = None
    asset_bundle: AssetClassModelBundle | None = None
    ticker_bundle: "PerTickerModelBundle | None" = None
    tp_sl_bundle: "PerTickerTPSLRegressorBundle | None" = None
    from models.per_ticker_models import PerTickerModelBundle, per_ticker_models_enabled
    from models.tp_sl_regressor import PerTickerTPSLRegressorBundle, tp_sl_regressor_enabled

    use_per_ticker = per_ticker_models_enabled() and "ticker" in panel.columns
    use_tp_sl = tp_sl_regressor_enabled() and use_per_ticker
    use_asset_class = asset_class_models_enabled() and "ticker" in panel.columns
    use_multi_model = use_per_ticker or use_asset_class
    class_min_rows = int(getattr(_cfg, "FUSION_ASSET_CLASS_MIN_ROWS", 500))
    ticker_min_rows = int(getattr(_cfg, "FUSION_PER_TICKER_MIN_ROWS", 200))
    if use_per_ticker:
        print("    entry model: per-ticker models (Optuna + LGBM per symbol)", flush=True)
    elif use_asset_class:
        print("    entry model: asset-class models (crypto / tradfi)", flush=True)
    if use_tp_sl:
        print("    exit levels: per-ticker TP/SL regressors (MFE/MAE targets)", flush=True)
    if per_fold_target:
        print("    entry targets: per-fold grid search on train window (before model opt)", flush=True)
    from strategy.fold_diagnostics import append_fold_diagnostics, log_fold_diagnostics, diagnose_fold_slice

    diag_path = OUT_DIR / "fold_diagnostics.jsonl"
    if diag_path.exists():
        diag_path.unlink()
    print(f"    fold diagnostics -> {diag_path}", flush=True)
    halflife = float(getattr(_cfg, "FUSION_ADAPTIVE_RECENCY_HALFLIFE_DAYS", 120.0))
    for w in windows:
        _rep.update()
        from common.stage_log import stage_log

        stage_log(
            "walk-forward fold",
            fold=w["fold"],
            detail=(
                f"train {w['train_start'].date()}..{w['train_end'].date()} "
                f"-> test {w['test_start'].date()}..{w['test_end'].date()}"
            ),
        )
        tr = work[
            (work["bar_time"] >= w["train_start"])
            & (work["bar_time"] < w["train_end"])
        ].dropna(subset=feat_cols + [target_col])
        te = work[
            (work["bar_time"] >= w["test_start"])
            & (work["bar_time"] < w["test_end"])
        ].dropna(subset=feat_cols + [target_col])
        row = {
            **{k: str(v.date()) if hasattr(v, "date") else v for k, v in w.items()},
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "skipped": False,
        }
        if len(tr) < min_rows or te.empty or tr[target_col].nunique() < 2:
            row.update({"skipped": True, "reason": "insufficient_train_or_test"})
            fold_rows.append(row)
            continue

        fold_model_name = model_name
        fold_params = dict(params)
        fold_params_by_ticker: dict[str, dict] = {}
        fold_short_params_by_ticker: dict[str, dict] = {}
        max_label_h = int(getattr(_cfg, "TARGET_OPT_MAX_HORIZON_BARS", 96) or 96)
        embargo_h = max(int(label_horizon), max_label_h)
        tr_safe = trim_label_embargo(
            tr, test_start=w["test_start"], horizon_bars=embargo_h,
        )
        if len(tr_safe) < min_rows:
            row.update({"skipped": True, "reason": "insufficient_train_after_embargo"})
            fold_rows.append(row)
            continue

        fold_label_horizon = int(label_horizon)
        if per_fold_target:
            from strategy.target_opt import (
                max_horizon_from_specs,
                optimize_targets_on_fold_train,
                relabel_panel_entry_targets,
                specs_dict_from_optimization,
            )

            fold_meta = {
                "fold": int(w["fold"]),
                "train_start": str(w["train_start"].date()),
                "train_end": str(w["train_end"].date()),
                "test_start": str(w["test_start"].date()),
                "test_end": str(w["test_end"].date()),
            }
            th_target = optimize_targets_on_fold_train(
                tr_safe,
                feat_cols,
                sorted(tr_safe["ticker"].astype(str).str.upper().unique()),
                fold_meta=fold_meta,
            )
            row["target_optimization"] = th_target
            fold_specs = specs_dict_from_optimization(th_target)
            if fold_specs:
                tr_safe = relabel_panel_entry_targets(tr_safe, fold_specs)
                te = relabel_panel_entry_targets(te, fold_specs)
                fold_label_horizon = max_horizon_from_specs(fold_specs, label_horizon)
                tr_safe = trim_label_embargo(
                    tr_safe,
                    test_start=w["test_start"],
                    horizon_bars=fold_label_horizon,
                )
                tr_safe = tr_safe.dropna(subset=feat_cols + [target_col])
                te = te.dropna(subset=feat_cols + [target_col])
                row["fold_target_specs"] = fold_specs
                row["fold_label_horizon_bars"] = int(fold_label_horizon)
                if len(tr_safe) < min_rows or te.empty or tr_safe[target_col].nunique() < 2:
                    row.update({"skipped": True, "reason": "insufficient_train_after_target_relabel"})
                    fold_rows.append(row)
                    continue

        if per_fold_opt:
            fold_meta = {
                "fold": int(w["fold"]),
                "train_start": str(w["train_start"].date()),
                "train_end": str(w["train_end"].date()),
                "test_start": str(w["test_start"].date()),
                "test_end": str(w["test_end"].date()),
            }
            stage_log("entry model: Optuna search starting", fold=w["fold"])
            if use_per_ticker:
                from models.model_per_ticker_opt import optimize_fusion_model_per_ticker_on_train_slice

                opt = optimize_fusion_model_per_ticker_on_train_slice(
                    tr_safe,
                    feat_cols,
                    target_col,
                    model_name=model_name,
                    fold_meta=fold_meta,
                    label_horizon_bars=label_horizon,
                )
            else:
                opt = optimize_fusion_model_on_train_slice(
                    tr_safe,
                    feat_cols,
                    target_col,
                    model_name=model_name,
                    fold_meta=fold_meta,
                    label_horizon_bars=label_horizon,
                )
            row["model_optimization"] = opt
            fold_model_name = opt["model_name"]
            fold_params = dict(opt.get("model_params") or {})
            fold_params_by_ticker = {
                sym: dict(entry.get("model_params") or {})
                for sym, entry in (opt.get("per_ticker") or {}).items()
            }
            fold_short_params_by_ticker = {
                sym: dict(p)
                for sym, p in (opt.get("short_params_by_ticker") or {}).items()
            }
            if not fold_short_params_by_ticker:
                # Fall back to long HPs when short Optuna disabled / empty.
                fold_short_params_by_ticker = dict(fold_params_by_ticker)
            if int(w["fold"]) < 2 or int(w["fold"]) % 2 == 0:
                cv = opt.get("cv") or {}
                print(
                    f"      fold {w['fold']} opt: composite={cv.get('composite')} "
                    f"profit_net_bps={cv.get('top_decile_net_bps')} "
                    f"auc={cv.get('auc')} short_tuned={len(opt.get('short_params_by_ticker') or {})} "
                    f"params={fold_params}",
                    flush=True,
                )

        tr_fit, va = split_fit_calibration(
            tr_safe,
            cal_frac=cal_frac,
            horizon_bars=fold_label_horizon,
            test_start=w["test_start"],
            target_col=target_col,
        )
        if tr_fit.empty or tr_fit[target_col].nunique() < 2:
            row.update({"skipped": True, "reason": "insufficient_train_after_split"})
            fold_rows.append(row)
            continue

        ref_time = pd.Timestamp(w["train_end"]) - pd.Timedelta(seconds=1)
        fit_weights = recency_sample_weights(
            tr_fit["bar_time"], ref_time, halflife_days=halflife,
        )
        y_fit = tr_fit[target_col].astype(int)

        if use_per_ticker:
            train_mode = (
                "per-ticker adaptive warm-start"
                if adaptive and ticker_bundle is not None
                else ("per-ticker adaptive initial" if adaptive else "per-ticker full refit")
            )
        elif use_asset_class:
            train_mode = (
                "asset-class adaptive warm-start"
                if adaptive and asset_bundle is not None
                else ("asset-class adaptive initial" if adaptive else "asset-class full refit")
            )
        else:
            train_mode = "adaptive warm-start" if adaptive and adaptive_model is not None else (
                "adaptive initial" if adaptive else "full refit"
            )
        stage_log(
            f"entry model: training ({train_mode})",
            fold=w["fold"],
            detail=f"{len(tr_fit):,} fit rows",
        )
        if use_per_ticker:
            prev_ticker_bundle = ticker_bundle
            ticker_bundle = PerTickerModelBundle(fold_model_name)
            ticker_bundle.fit(
                tr_fit,
                feat_cols,
                target_col,
                sample_weight=fit_weights,
                params=fold_params,
                params_by_ticker=fold_params_by_ticker,
                adaptive=adaptive,
                prev=prev_ticker_bundle,
                min_rows=ticker_min_rows,
            )
            row["training_mode"] = train_mode
            row["adaptive_training"] = ticker_bundle.training_state()
            row["per_ticker_models"] = sorted(ticker_bundle.models)
            clf = ticker_bundle
            if bool(getattr(_cfg, "FUSION_ALLOW_SHORT", False)) and "label_entry_short" in tr_fit.columns:
                ticker_bundle.fit_short(
                    tr_fit,
                    feat_cols,
                    sample_weight=fit_weights,
                    params=fold_params,
                    params_by_ticker=fold_short_params_by_ticker or fold_params_by_ticker,
                    min_rows=ticker_min_rows,
                )
                row["per_ticker_short_models"] = sorted(ticker_bundle.short_models)
            if use_tp_sl:
                prev_tp_sl = tp_sl_bundle
                tp_sl_bundle = PerTickerTPSLRegressorBundle()
                tp_sl_bundle.fit(
                    tr_fit,
                    feat_cols,
                    sample_weight=fit_weights,
                    params=fold_params,
                    params_by_ticker=fold_params_by_ticker,
                    min_rows=ticker_min_rows,
                    fwd_ret_col=_target_forward_return_col(tr_fit, target_col),
                )
                row["tp_sl_regressor"] = tp_sl_bundle.training_state()
        elif use_asset_class:
            prev_bundle = asset_bundle
            asset_bundle = AssetClassModelBundle(fold_model_name)
            asset_bundle.fit(
                tr_fit,
                feat_cols,
                target_col,
                sample_weight=fit_weights,
                params=fold_params,
                adaptive=adaptive,
                prev=prev_bundle,
                min_rows=class_min_rows,
            )
            row["training_mode"] = train_mode
            row["adaptive_training"] = asset_bundle.training_state()
            row["asset_class_models"] = sorted(asset_bundle.models)
            clf = asset_bundle
        elif adaptive:
            if adaptive_model is None:
                adaptive_model = AdaptiveEntryModel(fold_model_name)
                adaptive_model.fit_initial(
                    tr_fit[feat_cols], y_fit, fit_weights, fold_params,
                )
                row["training_mode"] = "initial"
            else:
                adaptive_model.fit_incremental(
                    tr_fit[feat_cols], y_fit, fit_weights, fold_params,
                )
                row["training_mode"] = "incremental_warm_start"
            row["adaptive_training"] = adaptive_model.training_state()
            clf = adaptive_model
        else:
            clf = make_entry_classifier(fold_model_name, fold_params)
            clf.fit(tr_fit[feat_cols], y_fit, sample_weight=fit_weights)
            row["training_mode"] = "full_refit"
        stage_log(
            "OOS inference + probability calibration",
            fold=w["fold"],
            detail=f"{len(te):,} test rows",
        )
        if use_multi_model:
            p_raw = clf.predict_proba(te, feat_cols)[:, 1]
            p = np.asarray(p_raw, dtype=float).copy()
            va_policy = va
            if not va.empty:
                if use_per_ticker:
                    te_keys = te["ticker"].astype(str).str.upper()
                    va_keys = va["ticker"].astype(str).str.upper()
                    for sym in te_keys.unique():
                        te_m = te_keys.eq(sym).to_numpy()
                        va_m = va_keys.eq(sym).to_numpy()
                        if not te_m.any() or int(va_m.sum()) < 40:
                            continue
                        va_sub = va.loc[va_m]
                        cal_cut = max(20, len(va_sub) // 2)
                        va_cal = va_sub.iloc[:cal_cut]
                        raw_cal = clf.predict_proba(va_cal, feat_cols)[:, 1]
                        p[te_m] = _calibrate_probabilities(
                            va_cal[target_col], raw_cal, p_raw[te_m],
                        )
                    va_policy = va.iloc[0:0]
                else:
                    from models.asset_class_models import asset_class_series

                    te_cls = asset_class_series(te["ticker"])
                    va_cls = asset_class_series(va["ticker"])
                    for cls_name in te_cls.unique():
                        te_m = (te_cls == cls_name).to_numpy()
                        va_m = (va_cls == cls_name).to_numpy()
                        if not te_m.any() or int(va_m.sum()) < 40:
                            continue
                        va_sub = va.loc[va_m]
                        cal_cut = max(20, len(va_sub) // 2)
                        va_cal = va_sub.iloc[:cal_cut]
                        raw_cal = clf.predict_proba(va_cal, feat_cols)[:, 1]
                        p[te_m] = _calibrate_probabilities(
                            va_cal[target_col], raw_cal, p_raw[te_m],
                        )
                    va_policy = va.iloc[0:0]
        else:
            p_raw = clf.predict_proba(te[feat_cols])[:, 1]
            if not va.empty and len(va) >= 40:
                cal_cut = max(20, len(va) // 2)
                va_cal = va.iloc[:cal_cut]
                va_policy = va.iloc[cal_cut:]
                raw_cal = clf.predict_proba(va_cal[feat_cols])[:, 1]
                p = _calibrate_probabilities(va_cal[target_col], raw_cal, p_raw)
            elif not va.empty:
                raw_val = clf.predict_proba(va[feat_cols])[:, 1]
                p = _calibrate_probabilities(va[target_col], raw_val, p_raw)
                va_policy = va.iloc[0:0]
            else:
                p = p_raw
                va_policy = va

        fold_meta = {
            "fold": int(w["fold"]),
            "train_start": str(w["train_start"].date()),
            "train_end": str(w["train_end"].date()),
            "test_start": str(w["test_start"].date()),
            "test_end": str(w["test_end"].date()),
        }
        if per_fold_th and prices is not None and not prices.empty:
            policy_mode = _th_mode

            import time as _time
            min_train_rows = int(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_TRAIN_ROWS", 500))
            min_train_sessions = int(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_TRAIN_SESSIONS", 4))
            if len(tr_fit) >= min_train_rows and tr_fit["session"].nunique() >= min_train_sessions:
                _t_policy = _time.perf_counter()
                if policy_mode in ("fixed", "default", "fallback"):
                    edge0 = float(getattr(_cfg, "FUSION_MIN_EXPECTED_EDGE_BPS", 0.0))
                    fixed_pol = _default_monthly_impulse_fallback(edge0)
                    th_opt = {
                        "optimizer": "fixed",
                        "best_params": fixed_pol,
                        "cv": {
                            "source": "fixed_default_impulse",
                            "calibrated_min_expected_edge_bps": edge0,
                            "signal_rows": int(len(tr_fit)),
                        },
                        "trial_results": [],
                        "n_trials": 0,
                        **fold_meta,
                    }
                    stage_log(
                        "trading policy: fixed default impulse",
                        fold=w["fold"],
                        detail=f"buy={fixed_pol.get('buy_threshold')} edge={edge0}",
                    )
                else:
                    policy_scored = tr_fit.copy()
                    stage_log(
                        f"trading policy: {policy_mode}",
                        fold=w["fold"],
                        detail=f"{len(tr_fit):,} train rows",
                    )
                    if use_multi_model:
                        policy_scored["ml_proba"] = clf.predict_proba(tr_fit, feat_cols)[:, 1]
                    else:
                        policy_scored["ml_proba"] = clf.predict_proba(tr_fit[feat_cols])[:, 1]
                    if (
                        use_per_ticker
                        and bool(getattr(_cfg, "FUSION_ALLOW_SHORT", False))
                        and getattr(clf, "short_models", None)
                    ):
                        policy_scored["ml_proba_short"] = clf.predict_proba_short(tr_fit, feat_cols)[:, 1]
                    policy_scored["ml_base_rate"] = float(tr_fit[target_col].mean())
                    va_holdout_scored = None
                    if not va.empty:
                        va_holdout_scored = va.copy()
                        if use_multi_model:
                            va_holdout_scored["ml_proba"] = clf.predict_proba(va, feat_cols)[:, 1]
                        else:
                            va_holdout_scored["ml_proba"] = clf.predict_proba(va[feat_cols])[:, 1]
                        if (
                            use_per_ticker
                            and bool(getattr(_cfg, "FUSION_ALLOW_SHORT", False))
                            and getattr(clf, "short_models", None)
                        ):
                            va_holdout_scored["ml_proba_short"] = clf.predict_proba_short(va, feat_cols)[:, 1]
                    if policy_mode in ("optuna_per_ticker", "per_ticker_optuna"):
                        from strategy.threshold_opt import optimize_trading_policy_per_ticker_on_train

                        th_opt = optimize_trading_policy_per_ticker_on_train(
                            policy_scored,
                            prices,
                            commission_bps=commission_bps,
                            fold_meta=fold_meta,
                        )
                    elif policy_mode == "optuna":
                        from strategy.threshold_opt import optimize_trading_policy_on_train

                        th_opt = optimize_trading_policy_on_train(
                            policy_scored,
                            prices,
                            commission_bps=commission_bps,
                            fold_meta=fold_meta,
                        )
                    else:
                        from strategy.threshold_opt import build_calibrated_trading_policy

                        th_opt = build_calibrated_trading_policy(
                            policy_scored,
                            prices,
                            commission_bps=commission_bps,
                            fold_meta=fold_meta,
                            holdout=va_holdout_scored,
                        )
                row["threshold_optimization"] = th_opt
                row["trading_policy"] = th_opt["best_params"]
                cv_th = th_opt.get("cv") or {}
                sig_rows = int(cv_th.get("signal_rows") or 0)
                flags: list[str] = []
                if sig_rows <= 0:
                    flags.append("zero_cv_signals")
                obj = cv_th.get("objective")
                if obj is not None and float(obj) < float(
                    getattr(_cfg, "FUSION_THRESHOLD_NO_TRADE_OBJECTIVE", -2.0)
                ):
                    flags.append("negative_cv_objective")
                if cv_th.get("trade_anomaly"):
                    flags.append("threshold_trade_anomaly")
                row["trade_anomaly"] = {
                    "anomaly": bool(flags),
                    "flags": flags,
                    "cv_signal_rows": sig_rows,
                    "cv_objective": obj,
                }
                if int(w["fold"]) < 2 or int(w["fold"]) % 2 == 0:
                    cv_th = th_opt.get("cv") or {}
                    by_t = (cv_th.get("by_ticker") or th_opt.get("best_params", {}).get("by_ticker") or {})
                    extra = ""
                    if by_t:
                        extra = " | " + ", ".join(
                            f"{k}:edge={v.get('min_expected_edge_bps')} "
                            f"top_decile={v.get('cv_top_decile_net_bps', v.get('cv_net_bps'))}"
                            f"{'' if v.get('signal_quality_ok', True) else '(skip)'}"
                            for k, v in sorted(by_t.items())
                        )
                    print(
                        f"      fold {w['fold']} policy ({th_opt.get('optimizer', policy_mode)}): "
                        f"edge={th_opt['best_params'].get('min_expected_edge_bps')} "
                        f"buy={th_opt['best_params'].get('buy_threshold')} "
                        f"sell={th_opt['best_params'].get('sell_threshold')} "
                        f"signals={cv_th.get('signal_rows')} "
                        f"({_time.perf_counter() - _t_policy:.1f}s){extra}",
                        flush=True,
                    )
                pol = th_opt["best_params"]
                if (
                    bool(getattr(_cfg, "FUSION_ADAPTIVE_HORIZON", True))
                    and not pol.get("disable_trading")
                ):
                    stage_log("adaptive horizon selection", fold=w["fold"])
                    horizon_panel = tr_fit.copy()
                    if use_multi_model:
                        horizon_panel["ml_proba"] = clf.predict_proba(tr_fit, feat_cols)[:, 1]
                    else:
                        horizon_panel["ml_proba"] = clf.predict_proba(tr_fit[feat_cols])[:, 1]
                    fold_horizons, fold_horizon_detail = _select_adaptive_horizons(
                        horizon_panel,
                        prices,
                        pol,
                        commission_bps=commission_bps,
                    )
                    if fold_horizons:
                        row["regime_horizons"] = fold_horizons
                        row["adaptive_horizon"] = fold_horizon_detail
            else:
                row["trading_policy"] = _default_monthly_impulse_fallback(
                    float(calibrate_min_expected_edge_bps(tr_fit, commission_bps))
                )
                row["threshold_optimization"] = {
                    "optimizer": "fallback",
                    "best_params": row["trading_policy"],
                    "cv": {"source": "fallback_insufficient_calibration_holdout"},
                }
                pol = row["trading_policy"]
                if (
                    bool(getattr(_cfg, "FUSION_ADAPTIVE_HORIZON", True))
                    and prices is not None
                    and not pol.get("disable_trading")
                ):
                    stage_log("adaptive horizon selection", fold=w["fold"])
                    horizon_panel = tr_fit.copy()
                    if use_multi_model:
                        horizon_panel["ml_proba"] = clf.predict_proba(tr_fit, feat_cols)[:, 1]
                    else:
                        horizon_panel["ml_proba"] = clf.predict_proba(tr_fit[feat_cols])[:, 1]
                    fold_horizons, fold_horizon_detail = _select_adaptive_horizons(
                        horizon_panel,
                        prices,
                        pol,
                        commission_bps=0.0 if bool(getattr(_cfg, "FUSION_METRICS_GROSS_ONLY", False)) else commission_bps,
                    )
                    if fold_horizons:
                        row["regime_horizons"] = fold_horizons
                        row["adaptive_horizon"] = fold_horizon_detail

        chunk = te.copy()
        if use_tp_sl and tp_sl_bundle is not None:
            pred_tp, pred_sl = tp_sl_bundle.predict(te, feat_cols)
            chunk["pred_tp_bps"] = pred_tp
            chunk["pred_sl_bps"] = pred_sl
        chunk["ml_proba_raw"] = p_raw
        chunk["ml_proba"] = p
        if use_per_ticker and bool(getattr(_cfg, "FUSION_ALLOW_SHORT", False)) and hasattr(clf, "predict_proba_short"):
            chunk["ml_proba_short"] = clf.predict_proba_short(te, feat_cols)[:, 1]
        chunk["ml_base_rate"] = float(tr[target_col].mean())
        chunk["wf_fold"] = int(w["fold"])
        chunk["wf_train_start"] = w["train_start"]
        chunk["wf_train_end"] = w["train_end"]
        chunk["wf_test_start"] = w["test_start"]
        chunk["wf_test_end"] = w["test_end"]
        chunk["wf_model_params"] = json.dumps(fold_params, sort_keys=True, default=str)
        parts.append(chunk)

        y = te[target_col].astype(int).to_numpy()
        oos_metrics = {
            "positive_rate_train": round(float(tr[target_col].mean()), 4),
            "positive_rate_test": round(float(te[target_col].mean()), 4),
            "auc": round(_auc(y, p), 4) if len(np.unique(y)) > 1 else 0.5,
            "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4)
            if len(np.unique(y)) > 1
            else 0.6931,
            "mean_proba": round(float(np.mean(p)), 4),
            "model_name": fold_model_name,
            "model_params": fold_params,
            "top_features": _monthly_fold_feature_importance(clf, feat_cols),
        }
        row["oos_metrics"] = oos_metrics
        row.update(oos_metrics)

        # Per-iteration diagnosis: train (in-sample proba) vs OOS top-decile net.
        train_diag = tr_fit.copy()
        if use_multi_model:
            train_diag["ml_proba"] = clf.predict_proba(tr_fit, feat_cols)[:, 1]
        else:
            train_diag["ml_proba"] = clf.predict_proba(tr_fit[feat_cols])[:, 1]
        fold_diag = diagnose_fold_slice(train_diag, chunk, fold=int(w["fold"]))
        # Attach model-opt profit if present
        mo = row.get("model_optimization") or {}
        cv_mo = mo.get("cv") or {}
        fold_diag["model_opt"] = {
            "composite": cv_mo.get("composite"),
            "top_decile_net_bps": cv_mo.get("top_decile_net_bps"),
            "auc": cv_mo.get("auc"),
        }
        th = row.get("threshold_optimization") or {}
        by_t = (th.get("cv") or {}).get("by_ticker") or (th.get("best_params") or {}).get("by_ticker") or {}
        fold_diag["threshold_signal"] = {
            k: {
                "cv_top_decile": v.get("cv_top_decile_net_bps", v.get("cv_net_bps")),
                "signal_ok": v.get("signal_quality_ok"),
            }
            for k, v in by_t.items()
        }
        log_fold_diagnostics(fold_diag)
        append_fold_diagnostics(fold_diag, diag_path)
        row["fold_diagnostics"] = fold_diag
        fold_rows.append(row)

    _rep.close()
    from strategy.fold_diagnostics import summarize_fold_diagnostics

    summary = summarize_fold_diagnostics(diag_path)
    print(
        f"    fold diagnostics summary: folds={summary.get('folds')} "
        f"mean_oos_net={summary.get('mean_oos_net')} "
        f"portfolio_fails={summary.get('portfolio_bottlenecks')}",
        flush=True,
    )
    for sym, counts in sorted((summary.get("ticker_bottlenecks") or {}).items()):
        print(f"      {sym}: {counts}", flush=True)
    if not parts:
        return pd.DataFrame(), fold_rows
    return pd.concat(parts, ignore_index=True), fold_rows


def _score_entry_model_cv(
    work: pd.DataFrame,
    feat_cols: list[str],
    target: str,
    folds: list[tuple[set, set]],
    model_name: str,
    params: dict,
    *,
    commission_bps: float | None = None,
) -> dict | None:
    from models.entry_model import make_entry_classifier
    from models.model_selection import aggregate_cv_metrics, compute_fold_metrics, criteria_breakdown
    from models.per_ticker_models import _with_class_weight

    use_weights = is_adaptive_wf_mode()
    use_profit_w = bool(getattr(_cfg, "FUSION_OPTUNA_PROFIT_SAMPLE_WEIGHT", True))
    halflife = float(getattr(_cfg, "FUSION_ADAPTIVE_RECENCY_HALFLIFE_DAYS", 120.0))
    invert_fwd = str(target).lower().endswith("_short")
    fold_rows: list[dict[str, float]] = []
    for train_s, val_s in folds:
        tr = work[work["session"].isin(train_s)].dropna(subset=feat_cols + [target])
        va = work[work["session"].isin(val_s)].dropna(subset=feat_cols + [target])
        if len(tr) < 200 or len(va) < 50:
            continue
        if tr[target].nunique() < 2:
            continue
        fit_params = _with_class_weight(dict(params), tr[target])
        clf = make_entry_classifier(model_name, fit_params)
        fwd_col = _target_forward_return_col(tr, target)
        sample_w = None
        if use_weights and "bar_time" in tr.columns:
            from models.adaptive_trainer import recency_sample_weights

            ref = pd.Timestamp(tr["bar_time"].max())
            sample_w = recency_sample_weights(tr["bar_time"], ref, halflife_days=halflife)
        if use_profit_w and fwd_col in tr.columns:
            mag = np.abs(tr[fwd_col].astype(float).to_numpy())
            mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
            mean_mag = float(np.mean(mag)) if len(mag) else 0.0
            if mean_mag > 1e-12:
                profit_w = mag / mean_mag
                sample_w = (
                    profit_w
                    if sample_w is None
                    else np.asarray(sample_w, dtype=float) * profit_w
                )
        if sample_w is not None:
            clf.fit(tr[feat_cols], tr[target], sample_weight=np.asarray(sample_w, dtype=float))
        else:
            clf.fit(tr[feat_cols], tr[target])
        pr = clf.predict_proba(va[feat_cols])[:, 1]
        y = va[target].values
        fwd = va[fwd_col].values if fwd_col in va.columns else None
        if fwd is not None and invert_fwd:
            fwd = -np.asarray(fwd, dtype=float)
        tickers = va["ticker"].values if "ticker" in va.columns else None
        comm = commission_bps
        if comm is None:
            comm = float(getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 1.1))
        if bool(getattr(_cfg, "FUSION_METRICS_GROSS_ONLY", False)):
            comm = 0.0
        metrics = compute_fold_metrics(y, pr, fwd, commission_bps=comm, tickers=tickers)
        if fwd is not None:
            from models.profit_metrics import top_decile_stats

            pr_tr = clf.predict_proba(tr[feat_cols])[:, 1]
            fwd_tr = tr[fwd_col].values if fwd_col in tr.columns else None
            tickers_tr = tr["ticker"].values if "ticker" in tr.columns else None
            if fwd_tr is not None:
                if invert_fwd:
                    fwd_tr = -np.asarray(fwd_tr, dtype=float)
                train_st = top_decile_stats(
                    pr_tr,
                    fwd_tr,
                    commission_bps=float(comm),
                    tickers=tickers_tr,
                )
                train_net = train_st["top_decile_net_bps"]
                val_net = metrics.get("top_decile_net_bps")
                if np.isfinite(train_net) and val_net is not None:
                    metrics["train_top_decile_net_bps"] = round(float(train_net), 3)
                    metrics["train_oos_gap_bps"] = round(
                        max(0.0, float(train_net) - float(val_net)), 3,
                    )
        fold_rows.append(metrics)

    agg = aggregate_cv_metrics(fold_rows)
    if agg.get("composite", -1) < 0:
        return None

    row = {
        "params": params,
        "n_folds": len(fold_rows),
        "criteria": criteria_breakdown(agg),
        **agg,
    }
    return row


def optimize_fusion_model_on_train_slice(
    train: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    *,
    model_name: str | None = None,
    max_combos: int | None = None,
    max_train_rows: int | None = None,
    fold_meta: dict | None = None,
    n_trials: int | None = None,
    label_horizon_bars: int | None = None,
) -> dict:
    """Purged session CV on one trailing train window only (test/OOS rows excluded)."""
    from config import FUSION_ENTRY_MODEL
    from research.features.entry_ml import _purged_session_folds
    from models.entry_model import DEFAULT_LIGHTGBM_PARAMS, FUSION_ENTRY_MODEL_NAME
    from models.model_opt import optimize_lightgbm_on_train
    from strategy.target_opt import _subsample

    name = model_name or FUSION_ENTRY_MODEL
    if name.lower() != FUSION_ENTRY_MODEL_NAME:
        raise ValueError(f"Only {FUSION_ENTRY_MODEL_NAME} is supported, got {name!r}")

    work = train.dropna(subset=feat_cols + [target_col]).copy()
    if work.empty or work[target_col].nunique() < 2:
        fallback = dict(DEFAULT_LIGHTGBM_PARAMS)
        return {
            "optimizer": "optuna",
            "model_name": name,
            "model_params": fallback,
            "cv": {"composite": None, "source": "fallback_empty_train"},
            "trial_results": [],
            "grid_results": [],
            "leaderboard": [],
            "grid_size": 0,
            "n_trials": 0,
            **(fold_meta or {}),
        }

    row_cap = max_train_rows if max_train_rows is not None else getattr(_cfg, "FUSION_FOLD_OPT_MAX_TRAIN_ROWS", None)
    if row_cap and len(work) > int(row_cap):
        seed = int((fold_meta or {}).get("fold", 0))
        work = _subsample(work, int(row_cap), seed)

    from strategy.leakage_guard import resolve_label_horizon_bars

    horizon = int(label_horizon_bars or resolve_label_horizon_bars())
    sessions = sorted(work["session"].unique())
    folds = _purged_session_folds(sessions, max_label_horizon_bars=horizon)
    trials = n_trials
    if trials is None and max_combos is not None:
        trials = int(max_combos)

    return optimize_lightgbm_on_train(
        work,
        feat_cols,
        target_col,
        folds,
        n_trials=trials,
        fold_meta=fold_meta,
    )


def save_monthly_fold_optimizations(folds: list[dict], path: Path | None = None) -> Path:
    """Persist per-fold optimization + OOS metrics for audit."""
    from strategy.optimization_audit import (
        flatten_ml_trials,
        flatten_threshold_trials,
        summarize_fold_optimizations,
    )

    out = path or MONTHLY_FOLD_OPT_PATH
    top_frac = float(getattr(_cfg, "FUSION_OPTIMIZATION_SUMMARY_TOP_FRAC", 0.25))
    summary = summarize_fold_optimizations(folds, top_frac=top_frac)
    payload = {
        "schema_version": 2,
        "n_folds": len(folds),
        "n_optimized": sum(1 for f in folds if f.get("model_optimization")),
        "n_threshold_optimized": sum(1 for f in folds if f.get("threshold_optimization")),
        "save_all_trials": bool(getattr(_cfg, "FUSION_OPTIMIZATION_SAVE_ALL_TRIALS", True)),
        "optimization_summary": summary,
        "folds": folds,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    summary_path = OPTIMIZATION_SUMMARY_PATH
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": str(out),
                **summary,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    flat = flatten_threshold_trials(folds) + flatten_ml_trials(folds)
    if flat:
        pd.DataFrame(flat).to_parquet(OPTIMIZATION_TRIALS_PARQUET_PATH, index=False)

    return out


def optimize_fusion_model(
    panel: pd.DataFrame,
    *,
    tick_only: bool = True,
    target_col: str | None = None,
    model_candidates: tuple[str, ...] | list[str] | None = None,
    n_trials: int | None = None,
) -> dict:
    """Global LightGBM tune via Optuna on purged session CV (legacy / causal pre-OOS slice)."""
    from config import ENTRY_MODEL_CRITERIA_WEIGHTS, FUSION_ENTRY_MODEL

    work = _filter_panel_tick_only(panel) if tick_only else panel.copy()
    if work.empty:
        raise ValueError("Empty panel after tick-only filter — download slim tick data first")

    feat_cols = resolve_ml_feature_cols(work)
    if len(feat_cols) < 10:
        raise ValueError(f"Too few fusion features: {len(feat_cols)}")

    target = target_col or _default_entry_target(work)
    if target not in work.columns:
        target = "label"

    n_t = n_trials or getattr(_cfg, "FUSION_MODEL_OPTUNA_TRIALS", 20)
    print(
        f"    entry model Optuna: lightgbm | trials={n_t} | "
        f"criteria={len(ENTRY_MODEL_CRITERIA_WEIGHTS)}",
        flush=True,
    )
    opt = optimize_fusion_model_on_train_slice(
        work.dropna(subset=feat_cols + [target]),
        feat_cols,
        target,
        model_name=FUSION_ENTRY_MODEL,
        n_trials=n_t,
        fold_meta={"scope": "global"},
    )
    cv = opt.get("cv") or {}
    return {
        "optimizer": opt.get("optimizer", "optuna"),
        "model_name": opt.get("model_name", FUSION_ENTRY_MODEL),
        "model_params": opt.get("model_params") or {},
        "gbm_params": opt.get("model_params") or {},
        "cv": cv,
        "criteria": cv.get("criteria", {}),
        "criteria_weights": dict(ENTRY_MODEL_CRITERIA_WEIGHTS),
        "feat_cols": feat_cols,
        "target_col": target,
        "search_space": opt.get("search_space"),
        "trial_results": opt.get("trial_results") or [],
        "grid_results": opt.get("grid_results") or [],
        "leaderboard": opt.get("leaderboard") or [],
        "grid_size": opt.get("grid_size", 0),
        "n_trials": opt.get("n_trials", 0),
        "cv_folds": opt.get("cv_folds"),
        "n_bars": int(len(work)),
        "n_sessions": int(work["session"].nunique()),
        "model_candidates": (FUSION_ENTRY_MODEL,),
    }


def train_fusion_pipeline(
    panel: pd.DataFrame,
    symbols: list[str],
    *,
    tick_only: bool = True,
    commission_bps: float = 10.0,
    max_oos_sessions: int | None = None,
    gbm_params: dict | None = None,
    gbm_cv: dict | None = None,
    model_name: str | None = None,
    model_params: dict | None = None,
    feat_cols: list[str] | None = None,
    wf_mode: str | None = None,
    train_days: int | None = None,
    backtest_years: int | None = None,
    test_months: int | None = None,
    max_folds: int | None = None,
) -> dict:
    """
    Walk-forward OOS LightGBM (purged CV + log-loss) + impulse purged CV + full OOS backtest.

    No arbitrary 70/30 time split — all evaluation is walk-forward or purged session CV.
    """
    from config import FUSION_ENTRY_MODEL
    from research.features.entry_ml import _auc, _purged_session_folds
    from simulation.ml_calibration import calibration_bins, ml_likelihood_matrix
    from sklearn.metrics import log_loss
    from simulation.engine import run_backtest_signal_exit

    if tick_only:
        panel = _filter_panel_tick_only(panel)
    if panel.empty:
        raise ValueError("Empty panel after tick-only filter — download bar data first")

    panel = attach_execution_cost_features(panel)
    if feat_cols is None:
        feat_cols = resolve_ml_feature_cols(panel)
    feat_cols = _with_cost_feature(feat_cols, panel)
    if len(feat_cols) < 10:
        raise ValueError(f"Too few fusion features: {len(feat_cols)}")
    mode = (wf_mode or getattr(_cfg, "FUSION_WF_MODE", "adaptive")).lower()

    target_col = _default_entry_target(panel)
    if target_col not in panel.columns:
        target_col = "label"
    if target_col == "label_entry" and panel[target_col].notna().sum() == 0:
        raise ValueError(
            "Per-instrument targets are configured but label_entry is empty — "
            "delete output/cache/fusion_panel_v*.parquet and rebuild the panel"
        )

    n_sess = panel["session"].nunique()
    print(f"    fusion panel: {len(panel):,} bars | {n_sess} sessions", flush=True)

    sessions = sorted(panel["session"].unique())
    folds = _purged_session_folds(sessions)

    resolved_name = model_name or FUSION_ENTRY_MODEL
    resolved_params = model_params or gbm_params
    per_fold_opt = bool(getattr(_cfg, "FUSION_MONTHLY_OPTIMIZE_PER_FOLD", True))
    if resolved_params is None:
        if mode in CALENDAR_WF_MODES:
            if per_fold_opt:
                from models.entry_model import DEFAULT_LIGHTGBM_PARAMS

                resolved_name = FUSION_ENTRY_MODEL
                resolved_params = dict(DEFAULT_LIGHTGBM_PARAMS)
                best_m = {"source": "per_fold_adaptive" if is_adaptive_wf_mode(mode) else "per_fold_monthly", "composite": None}
                tm = int(test_months or getattr(_cfg, "FUSION_WF_TEST_MONTHS", 6))
                td = int(train_days or getattr(_cfg, "FUSION_WF_TRAIN_DAYS", 365))
                if is_adaptive_wf_mode(mode):
                    print(
                        f"    entry model: adaptive — {td}d rolling train, {tm}m OOS, "
                        "recency weights + warm-start",
                        flush=True,
                    )
                else:
                    print(
                        f"    entry model: per-fold optimization on each 1y train window "
                        f"(OOS {tm}m excluded)",
                        flush=True,
                    )
            else:
                cached_opt = load_ml_optimize_artifact()
                monthly_opt = bool(getattr(_cfg, "FUSION_MONTHLY_OPTIMIZE_MODEL", True))
                if cached_opt and (cached_opt.get("model_params") or cached_opt.get("gbm_params")):
                    resolved_name = cached_opt.get("model_name", FUSION_ENTRY_MODEL)
                    resolved_params = cached_opt.get("model_params") or cached_opt.get("gbm_params")
                    best_m = cached_opt.get("cv", {})
                    if feat_cols is None and cached_opt.get("feat_cols"):
                        feat_cols = [c for c in cached_opt["feat_cols"] if c in panel.columns]
                    src = "causal_cache" if cached_opt.get("causal") else "cache"
                    print(f"    entry model from ml_optimize ({src}): {ML_OPT_PATH.name}", flush=True)
                elif monthly_opt:
                    ml_opt = optimize_fusion_model_causal(
                        panel,
                        tick_only=False,
                        target_col=target_col,
                        train_days=int(train_days or getattr(_cfg, "FUSION_WF_TRAIN_DAYS", 365)),
                        backtest_years=int(backtest_years or getattr(_cfg, "FUSION_WF_BACKTEST_YEARS", 4)),
                        test_months=int(test_months or getattr(_cfg, "FUSION_WF_TEST_MONTHS", 1)),
                    )
                    save_ml_optimize_artifact(ml_opt)
                    resolved_name = ml_opt["model_name"]
                    resolved_params = ml_opt["model_params"]
                    best_m = ml_opt.get("cv", {})
                    if feat_cols is None and ml_opt.get("feat_cols"):
                        feat_cols = [c for c in ml_opt["feat_cols"] if c in panel.columns]
                    print(
                        f"    entry model: causal tune composite={best_m.get('composite')} "
                        f"auc={best_m.get('auc')}",
                        flush=True,
                    )
                else:
                    from models.entry_model import DEFAULT_LIGHTGBM_PARAMS

                    resolved_name = FUSION_ENTRY_MODEL
                    resolved_params = dict(DEFAULT_LIGHTGBM_PARAMS)
                    best_m = {
                        "composite": None,
                        "auc": None,
                        "log_loss": None,
                        "source": "fixed_default_no_global_tune",
                    }
                    print("    entry model: fixed default params (FUSION_MONTHLY_OPTIMIZE_MODEL=False)", flush=True)
        else:
            cached_opt = load_ml_optimize_artifact()
            if cached_opt and (cached_opt.get("model_params") or cached_opt.get("gbm_params")):
                resolved_name = cached_opt.get("model_name", FUSION_ENTRY_MODEL)
                resolved_params = cached_opt.get("model_params") or cached_opt.get("gbm_params")
                best_m = cached_opt.get("cv", {})
                if feat_cols is None and cached_opt.get("feat_cols"):
                    feat_cols = [c for c in cached_opt["feat_cols"] if c in panel.columns]
                print(f"    LightGBM from cache: {ML_OPT_PATH.name}", flush=True)
            else:
                ml_opt = optimize_lightgbm(panel, tick_only=False, target_col=target_col)
                resolved_name = ml_opt["model_name"]
                resolved_params = ml_opt["model_params"]
                best_m = ml_opt["cv"]
    else:
        best_m = gbm_cv or {"composite": -1.0, "auc": 0.5, "log_loss": 0.693}
        print(
            f"    entry model from ml_optimize: {resolved_name} "
            f"composite={best_m.get('composite')}",
            flush=True,
        )
    best_p = resolved_params

    prices = load_closes(symbols, BAR_TIMEFRAME)
    per_fold_th = bool(getattr(_cfg, "FUSION_MONTHLY_OPTIMIZE_THRESHOLDS_PER_FOLD", True))
    per_fold_target = bool(getattr(_cfg, "FUSION_MONTHLY_OPTIMIZE_TARGETS_PER_FOLD", True))

    wf_folds: list[dict] = []
    fold_opt_path: Path | None = None
    if mode in CALENDAR_WF_MODES:
        oos, wf_folds = walk_forward_fusion_oos_monthly(
            panel,
            feat_cols,
            best_p,
            model_name=resolved_name,
            target_col=target_col,
            train_days=int(train_days or getattr(_cfg, "FUSION_WF_TRAIN_DAYS", 365)),
            backtest_years=int(backtest_years or getattr(_cfg, "FUSION_WF_BACKTEST_YEARS", 4)),
            test_months=int(test_months or getattr(_cfg, "FUSION_WF_TEST_MONTHS", 1)),
            optimize_per_fold=per_fold_opt,
            optimize_thresholds_per_fold=per_fold_th,
            optimize_targets_per_fold=per_fold_target,
            prices=prices,
            commission_bps=commission_bps,
            max_folds=max_folds,
        )
        fold_opt_path = save_monthly_fold_optimizations(wf_folds)
        fold_label = "adaptive" if is_adaptive_wf_mode(mode) else "monthly"
        print(f"    {fold_label} fold optimizations: {fold_opt_path.name} ({len(wf_folds)} folds)", flush=True)
    else:
        oos = walk_forward_fusion_oos(
            panel,
            feat_cols,
            best_p,
            model_name=resolved_name,
            max_oos_sessions=max_oos_sessions,
            target_col=target_col,
        )
    print(f"    walk-forward OOS: {len(oos):,} rows | {oos['session'].nunique() if not oos.empty else 0} sessions", flush=True)
    cache_name = "fusion_oos_adaptive.parquet" if is_adaptive_wf_mode(mode) else (
        "fusion_oos_monthly4y.parquet" if mode in CALENDAR_WF_MODES else "fusion_oos.parquet"
    )
    oos_cache = OUT_DIR / "cache" / cache_name
    if not oos.empty:
        oos_cache.parent.mkdir(parents=True, exist_ok=True)
        oos.to_parquet(oos_cache, index=False)
    if oos.empty:
        raise ValueError("No walk-forward OOS predictions — need more tick sessions")
    oos.attrs["target_col"] = target_col

    oos0 = pd.Timestamp(oos["bar_time"].min()) if not oos.empty else None
    oos1 = pd.Timestamp(oos["bar_time"].max()) if not oos.empty else None

    oos_auc = _auc(oos[target_col].values, oos["ml_proba"].values) if oos[target_col].nunique() > 1 else 0.5
    y_oos = oos[target_col].values
    p_oos = oos["ml_proba"].values
    oos_ll = float(log_loss(y_oos, np.clip(p_oos, 1e-6, 1 - 1e-6))) if len(np.unique(y_oos)) > 1 else 0.693
    likelihood = ml_likelihood_matrix(y_oos, p_oos)
    calibration = calibration_bins(y_oos, p_oos)

    if mode in CALENDAR_WF_MODES and per_fold_th and wf_folds:
        cal_edges = [
            (f.get("threshold_optimization") or {}).get("cv", {}).get("calibrated_min_expected_edge_bps")
            for f in wf_folds
            if not f.get("skipped")
        ]
        cal_edges = [float(e) for e in cal_edges if e is not None]
        calibrated_edge = float(np.median(cal_edges)) if cal_edges else float(
            getattr(_cfg, "FUSION_MIN_EXPECTED_EDGE_BPS", 8.0)
        )
        best_impulse = _default_monthly_impulse_fallback(calibrated_edge)
        best_impulse["source"] = "per_fold_thresholds"
        best_impulse["grid_skipped"] = True
        impulse_grid: list[dict] = []
        print("    impulse: per-fold threshold policies from train CV", flush=True)
    else:
        cal_panel = None
        if oos0 is not None:
            cal_panel = panel[panel["bar_time"] < oos0] if "bar_time" in panel.columns else None
        best_impulse, impulse_grid = optimize_impulse_params(
            prices,
            oos,
            commission_bps=commission_bps,
            grid_profile=mode,
            calibration_panel=cal_panel,
        )

    from research.diagnostics.decile_audit import (
        decile_audit_by_ticker,
        decile_monotonicity_check_side_aware,
        filter_tradeable_by_signal_quality_detailed,
        resolve_tradeable_tickers,
    )

    ret_col = _target_forward_return_col(oos, target_col)
    from common.stage_log import stage_log
    from strategy.instrument_economics import instrument_economics_snapshot

    stage_log("decile gate audit on stitched OOS", detail=f"{len(oos):,} rows")
    decile_audit = decile_monotonicity_check_side_aware(
        oos, ret_col=ret_col,
        commission_bps=0.0 if bool(getattr(_cfg, "FUSION_METRICS_GROSS_ONLY", False)) else commission_bps,
    )
    decile_by_ticker = decile_audit_by_ticker(oos, ret_col=ret_col, side_aware=True)
    instrument_econ = {str(s).upper(): instrument_economics_snapshot(s) for s in symbols}
    for sym, audit in decile_by_ticker.items():
        econ = instrument_econ.get(str(sym).upper()) or {}
        audit["economics_floor_bps"] = econ.get("economics_floor_bps")
        audit["vol_ann"] = econ.get("vol_ann")
    disable_gate = bool(getattr(_cfg, "FUSION_DISABLE_QUALITY_GATE", False))
    sq_detail: dict[str, dict] = {}
    stitched_before_sq: list[str] = []
    if disable_gate:
        tradeable_syms = [str(s).upper() for s in symbols]
        gate_source = "disabled"
        sq_removed: list[str] = []
        decile_audit["gate_skipped"] = True
        stitched_before_sq = list(tradeable_syms)
    else:
        tradeable_syms, gate_source = resolve_tradeable_tickers(decile_by_ticker, wf_folds or [])
        stitched_before_sq = list(tradeable_syms)
        tradeable_syms, sq_removed, sq_detail = filter_tradeable_by_signal_quality_detailed(
            wf_folds or [], tradeable_syms
        )
    if sq_removed:
        gate_source = f"{gate_source}_sq_filtered"
        decile_audit["signal_quality_removed"] = sq_removed
    decile_audit["by_ticker"] = decile_by_ticker
    decile_audit["tradeable_tickers"] = tradeable_syms
    decile_audit["stitched_tradeable_before_sq"] = stitched_before_sq
    decile_audit["signal_quality_detail"] = sq_detail
    decile_audit["gate_source"] = gate_source
    decile_audit["instrument_economics"] = instrument_econ
    # Portfolio tradeable iff any per-ticker name remains.
    decile_audit["tradeable"] = bool(tradeable_syms)

    if disable_gate or tradeable_syms:
        best_impulse = dict(best_impulse)
        best_impulse["disable_trading"] = False
        best_impulse["tradeable_tickers"] = tradeable_syms
        best_impulse["disabled_tickers"] = [
            s for s in symbols if str(s).upper() not in set(tradeable_syms)
        ]
        # Attach SQ soft-size hints into per-ticker policies.
        by_t = dict(best_impulse.get("by_ticker") or {})
        for sym_u, det in sq_detail.items():
            pol = dict(by_t.get(sym_u) or {})
            pol["ticker"] = sym_u
            if det.get("pass_rate") is not None:
                pol["sq_pass_rate"] = det["pass_rate"]
            if det.get("soft_size") is not None:
                pol["sq_soft_keep"] = bool(det.get("ok"))
            by_t[sym_u] = pol
        best_impulse["by_ticker"] = by_t
        if disable_gate:
            best_impulse["quality_gate_skipped"] = True
        elif stitched_before_sq and set(tradeable_syms) != set(stitched_before_sq):
            best_impulse["decile_gate_partial"] = True
            best_impulse["decile_gate_reasons"] = [
                f"sq_filtered; trading {len(tradeable_syms)}/{len(stitched_before_sq)} "
                f"stitched={stitched_before_sq} live={tradeable_syms}",
            ]
        print(
            f"    decile gate ({gate_source}): per-ticker tradeable={tradeable_syms} "
            f"disabled={best_impulse.get('disabled_tickers', [])}",
            flush=True,
        )
    elif not tradeable_syms:
        best_impulse = dict(best_impulse)
        best_impulse["disable_trading"] = True
        best_impulse["decile_gate_blocked"] = True
        best_impulse["decile_gate_reasons"] = decile_audit.get("reasons", [])
        best_impulse["tradeable_tickers"] = []
        best_impulse["disabled_tickers"] = [str(s).upper() for s in symbols]
        print(
            f"    decile gate: no tradeable tickers — portfolio_net={decile_audit.get('top_decile_net_bps')} "
            f"stitched_before_sq={stitched_before_sq} reasons={decile_audit.get('reasons')}",
            flush=True,
        )

    has_per_fold_horizons = any(f.get("regime_horizons") for f in (wf_folds or []))
    if has_per_fold_horizons:
        regime_horizons = None
        horizon_detail = {"enabled": True, "mode": "per_fold_train_val"}
    else:
        regime_horizons = None
        horizon_detail = {
            "enabled": False,
            "reason": "adaptive_horizon_requires_per_fold_train_val",
        }

    hold_default = _fusion_hold_default()
    stage_log("building OOS entry signals", detail=f"per_fold_policies={bool(per_fold_th and wf_folds)}")
    if best_impulse.get("disable_trading"):
        sig_full = pd.DataFrame()
        stage_log("OOS signals skipped", detail="decile gate disable_trading")
    elif mode in CALENDAR_WF_MODES and per_fold_th and wf_folds:
        sig_full = _fusion_signal_frame_monthly(oos, prices, wf_folds, best_impulse, regime_horizons)
    else:
        sig_full = _fusion_signal_frame(oos, prices, best_impulse, regime_horizons)
    if not sig_full.empty:
        sig_full.attrs["target_col"] = target_col
    bt_commission, bt_slippage = _resolve_backtest_friction(commission_bps)
    stage_log(
        "OOS backtest (signal exit)",
        detail=f"{len(sig_full):,} signal bars | gross={bool(getattr(_cfg, 'FUSION_BACKTEST_GROSS_ONLY', False))}",
    )
    bt_full = run_backtest_signal_exit(
        prices, sig_full, score_col="score",
        use_dynamic_thresholds=True, use_vol_targeting=True,
        commission_bps=bt_commission,
        horizon_bars=hold_default,
        slippage_bps=bt_slippage,
        period_start=oos0, period_end=oos1,
    ) if not sig_full.empty else {"stats": {}, "equity": pd.DataFrame()}
    bt_net: dict | None = None
    if (
        not sig_full.empty
        and bool(getattr(_cfg, "FUSION_BACKTEST_GROSS_ONLY", False))
        and bool(getattr(_cfg, "FUSION_BACKTEST_REPORT_NET", False))
    ):
        from research.labels.trade import DEFAULT_SLIPPAGE_BPS

        bt_net = run_backtest_signal_exit(
            prices, sig_full, score_col="score",
            use_dynamic_thresholds=True, use_vol_targeting=True,
            commission_bps=commission_bps,
            horizon_bars=hold_default,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
            period_start=oos0, period_end=oos1,
        )
    eq = bt_full.get("equity")
    if (eq is None or eq.empty) and oos0 is not None and oos1 is not None:
        eq = _oos_flat_equity(prices, oos0, oos1)
        flat_stats = {
            "total_return_pct": 0.0,
            "sharpe": 0.0,
            "period_start": str(oos0.date()),
            "period_end": str(oos1.date()),
            "no_trades": True,
        }
        bt_full = {"equity": eq, "stats": flat_stats, "benchmark": pd.DataFrame()}
    monthly_policies = mode in CALENDAR_WF_MODES and per_fold_th and bool(wf_folds)
    per_ticker_backtest = _per_ticker_oos_backtests(
        oos,
        prices,
        wf_folds or [],
        best_impulse,
        symbols,
        regime_horizons=regime_horizons,
        commission_bps=commission_bps,
        oos0=oos0,
        oos1=oos1,
        hold_default=hold_default,
        monthly_policies=monthly_policies,
        gated=False,
    )
    per_ticker_backtest_gated = _per_ticker_oos_backtests(
        oos,
        prices,
        wf_folds or [],
        best_impulse,
        symbols,
        regime_horizons=regime_horizons,
        commission_bps=commission_bps,
        oos0=oos0,
        oos1=oos1,
        hold_default=hold_default,
        monthly_policies=monthly_policies,
        gated=True,
    )
    for sym, stats in sorted(per_ticker_backtest.items()):
        gated_stats = (per_ticker_backtest_gated or {}).get(sym) or {}
        print(
            f"    per-ticker backtest {sym}: ungated return={stats.get('total_return_pct')}% "
            f"gated return={gated_stats.get('total_return_pct')}% "
            f"trades={stats.get('n_trades', stats.get('n_signals'))}",
            flush=True,
        )
    # Counterfactual: equity as if we traded stitched-passers before SQ filter.
    stitched_diag_bt: dict = {}
    if stitched_before_sq:
        diag_pol = dict(best_impulse)
        diag_pol["disable_trading"] = False
        diag_pol["tradeable_tickers"] = list(stitched_before_sq)
        diag_pol["disabled_tickers"] = [
            s for s in symbols if str(s).upper() not in set(stitched_before_sq)
        ]
        diag_by = dict(diag_pol.get("by_ticker") or {})
        for sym_u in stitched_before_sq:
            pol = dict(diag_by.get(sym_u) or {})
            pol["ticker"] = sym_u
            pol["sq_soft_keep"] = True
            pol["sq_pass_rate"] = float(pol.get("sq_pass_rate") or 1.0)
            diag_by[sym_u] = pol
        diag_pol["by_ticker"] = diag_by
        if monthly_policies and wf_folds:
            diag_sig = _fusion_signal_frame_monthly(oos, prices, wf_folds, diag_pol, regime_horizons)
        else:
            diag_sig = _fusion_signal_frame(oos, prices, diag_pol, regime_horizons)
        if not diag_sig.empty:
            from simulation.engine import run_backtest_signal_exit as _rbs

            diag_bt = _rbs(
                prices,
                diag_sig,
                score_col="score",
                use_dynamic_thresholds=True,
                use_vol_targeting=True,
                commission_bps=commission_bps,
                horizon_bars=hold_default,
                period_start=oos0,
                period_end=oos1,
            )
            eq_diag = diag_bt.get("equity")
            eq_path = None
            try:
                from reporting.plots import plot_equity_curve

                plot_dir = OUT_DIR / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                eq_path = str(plot_dir / "equity_curve_stitched_before_sq.png")
                plot_equity_curve(
                    diag_bt,
                    Path(eq_path),
                    title="Stitched-before-SQ (diagnostic)",
                )
            except Exception:
                if eq_diag is not None and hasattr(eq_diag, "empty") and not eq_diag.empty:
                    try:
                        csv_path = OUT_DIR / "plots" / "equity_stitched_before_sq.csv"
                        csv_path.parent.mkdir(parents=True, exist_ok=True)
                        eq_diag.to_csv(csv_path)
                        eq_path = str(csv_path)
                    except Exception:
                        eq_path = None
            stitched_diag_bt = {
                "tradeable_tickers": list(stitched_before_sq),
                "stats": dict(diag_bt.get("stats") or {}),
                "n_signals": int(len(diag_sig)),
                "equity_path": eq_path,
            }
            print(
                f"    stitched-before-SQ diagnostic: return="
                f"{stitched_diag_bt['stats'].get('total_return_pct')}% "
                f"tickers={stitched_before_sq}",
                flush=True,
            )
    from strategy.instrument_adapter import instrument_registry

    instrument_adapters = {
        sym: ad.to_dict() for sym, ad in instrument_registry(symbols).items()
    }
    from simulation.benchmark_alpha import comparative_alpha, equal_weight_benchmark, per_symbol_alpha
    from reporting.quantstats_report import write_quantstats_report

    from models.feature_importance import build_and_save_feature_importance_report

    fi_report = build_and_save_feature_importance_report(wf_folds, feat_cols)
    alpha_report: dict = {}
    if eq is not None and not eq.empty and "value" in eq.columns:
        strat_eq = eq["value"]
        ew = equal_weight_benchmark(prices, symbols)
        if not ew.empty:
            alpha_report["vs_equal_weight"] = comparative_alpha(
                strat_eq, ew.reindex(strat_eq.index).ffill().dropna(),
                label="vs_equal_weight_BTC_ETH",
            )
        alpha_report["per_symbol"] = per_symbol_alpha(strat_eq, prices, symbols)
    quantstats_report = write_quantstats_report(
        bt_full,
        prices,
        symbols,
        title="Fusion stitched walk-forward backtest",
    )

    from reporting.plots import (
        plot_fold_anomaly_analysis,
        write_fusion_extended_oos_plots,
        write_standard_oos_plots,
    )
    from simulation.entry_signals import active_entry_signals, deoverlap_signals, trade_returns_from_signals

    plots_dir = OUT_DIR / "plots"
    plot_paths: dict[str, str] = {}
    if not sig_full.empty:
        entry_sig_full = active_entry_signals(sig_full)
        ev_sig_full = deoverlap_signals(entry_sig_full, prices, None)
        trade_rets = trade_returns_from_signals(ev_sig_full, commission_bps)
    else:
        ev_sig_full = None
        trade_rets = np.array([])
    cache_dir = OUT_DIR / "cache"
    if not best_impulse.get("disable_trading") and not sig_full.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        sig_full.to_parquet(cache_dir / "fusion_bt_signals.parquet", index=False)
    if not best_impulse.get("disable_trading") and eq is not None and not eq.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_eq = eq.reset_index()
        if "index" in out_eq.columns and "date" not in out_eq.columns:
            out_eq = out_eq.rename(columns={"index": "date"})
        out_eq.to_parquet(cache_dir / "fusion_bt_equity.parquet", index=False)
    oos_stats = bt_full.get("stats") or {}
    plot_paths.update(write_standard_oos_plots(
        bt_full,
        trade_rets,
        title_equity="Fusion OOS equity",
        title_density="Fusion OOS return density",
    ))
    plot_paths.update(write_fusion_extended_oos_plots(
        eq, prices, symbols, trade_rets, plots_dir,
        stats=oos_stats,
        oos_index=pd.DatetimeIndex(pd.to_datetime(oos["bar_time"])) if "bar_time" in oos.columns else None,
        start_capital=float(getattr(_cfg, "BACKTEST_START_CAPITAL", 10_000.0)),
    ))
    report = {
        "mode": "adaptive_walk_forward" if is_adaptive_wf_mode(mode) else (
            "monthly4y_walk_forward" if mode in CALENDAR_WF_MODES else "walk_forward_oos"
        ),
        "walk_forward_mode": mode,
        "walk_forward_max_folds": max_folds if max_folds is not None else getattr(_cfg, "FUSION_WF_MAX_FOLDS", None),
        "tick_only": tick_only,
        "validation": "purged_session_cv + walk_forward_oos",
        "walk_forward_folds": wf_folds,
        "tickers": symbols,
        "symbols": symbols,
        "features": feat_cols,
        "target": target_col,
        "n_features": len(feat_cols),
        "n_bars_total": len(panel),
        "n_sessions": n_sess,
        "cv": best_m,
        "oos_auc": round(oos_auc, 4),
        "oos_log_loss": round(oos_ll, 4),
        "likelihood_matrix": likelihood,
        "calibration": calibration,
        "oos_rows": len(oos),
        "oos_sessions": int(oos["session"].nunique()),
        "oos_period": [str(oos0.date()), str(oos1.date())] if oos0 is not None else [],
        "wf_train_days": int(train_days or getattr(_cfg, "FUSION_WF_TRAIN_DAYS", 365)),
        "wf_backtest_years": int(backtest_years or getattr(_cfg, "FUSION_WF_BACKTEST_YEARS", 4)),
        "wf_test_months": int(test_months or getattr(_cfg, "FUSION_WF_TEST_MONTHS", 1)),
        "entry_model": resolved_name,
        "model_name": resolved_name,
        "model_params": best_p,
        "gbm_params": best_p,
        "model_optimization": {
            "mode": "per_fold" if per_fold_opt else "causal_global",
            "enabled": per_fold_opt or bool(getattr(_cfg, "FUSION_MONTHLY_OPTIMIZE_MODEL", True))
            or mode not in CALENDAR_WF_MODES,
            "per_fold": per_fold_opt,
            "causal": bool((load_ml_optimize_artifact() or {}).get("causal")) and not per_fold_opt,
            "causal_cutoff": (load_ml_optimize_artifact() or {}).get("causal_cutoff"),
            "cv": best_m,
            "leaderboard_top5": (load_ml_optimize_artifact() or {}).get("leaderboard", [])[:5],
            "artifact_path": str(ML_OPT_PATH) if ML_OPT_PATH.is_file() else None,
            "monthly_fold_optimizations_path": str(fold_opt_path) if fold_opt_path else None,
            "optimization_summary_path": str(OPTIMIZATION_SUMMARY_PATH)
            if fold_opt_path and OPTIMIZATION_SUMMARY_PATH.is_file()
            else None,
            "optimization_trials_parquet_path": str(OPTIMIZATION_TRIALS_PARQUET_PATH)
            if fold_opt_path and OPTIMIZATION_TRIALS_PARQUET_PATH.is_file()
            else None,
            "fold_summary": [
                {
                    "fold": f.get("fold"),
                    "train_start": f.get("train_start"),
                    "train_end": f.get("train_end"),
                    "test_start": f.get("test_start"),
                    "test_end": f.get("test_end"),
                    "opt_composite": (f.get("model_optimization") or {}).get("cv", {}).get("composite"),
                    "opt_auc": (f.get("model_optimization") or {}).get("cv", {}).get("auc"),
                    "policy_objective": (f.get("threshold_optimization") or {}).get("cv", {}).get("objective"),
                    "buy_threshold": (f.get("trading_policy") or {}).get("buy_threshold"),
                    "min_edge_bps": (f.get("trading_policy") or {}).get("min_expected_edge_bps"),
                    "oos_auc": (f.get("oos_metrics") or f).get("auc"),
                    "oos_log_loss": (f.get("oos_metrics") or f).get("log_loss"),
                    "model_params": (f.get("model_optimization") or {}).get("model_params"),
                    "skipped": bool(f.get("skipped")),
                }
                for f in wf_folds
            ],
        },
        "impulse_optimization": {
            "best": best_impulse,
            "grid_size": len(impulse_grid),
            "cv_folds": len(folds),
            "per_fold_thresholds": per_fold_th and mode in CALENDAR_WF_MODES,
        },
        "backtest_walk_forward_oos": bt_full.get("stats", {}),
        "backtest_gross_only": bool(getattr(_cfg, "FUSION_BACKTEST_GROSS_ONLY", False)),
        "backtest_walk_forward_oos_net": (bt_net or {}).get("stats", {}) if bt_net else None,
        "adaptive_horizon": horizon_detail,
        "signal_diagnostics": _signal_diagnostics(oos, sig_full, bt_full),
        "ml_diagnostics": _monthly_oos_diagnostics(oos, target_col, best_m),
        "feature_quality": _feature_quality_report(panel, feat_cols),
        "feature_groups": feature_group_catalog(),
        "feature_importance_by_group": aggregate_importance_by_group(
            merge_fold_importances(wf_folds)
        ),
        "ml_feature_importance": fi_report,
        "fusion_policy": {
            "hmm_hard_gate": bool(getattr(_cfg, "FUSION_HMM_HARD_GATE", False)),
            "edge_floor_mode": str(getattr(_cfg, "FUSION_EDGE_FLOOR_MODE", "sl_plus_commission")),
            "edge_gate_floor_mode": str(getattr(_cfg, "FUSION_EDGE_GATE_FLOOR_MODE", "full_round_trip")),
            "stop_loss_bps_default": float(getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0)),
            "ignore_applied_targets": bool(getattr(_cfg, "FUSION_IGNORE_APPLIED_TARGETS", False)),
        },
        "comparative_alpha": alpha_report,
        "quantstats": quantstats_report,
        "monte_carlo_stage": "MonteCarloAgent",
        "fwd_horizon_bars": FWD_HORIZON_BARS,
        "hold_default_bars": hold_default,
        "expected_move_bps": round(_expected_move_bps(hold_default), 2),
        "per_instrument_targets": _entry_target_summary(),
        "instrument_adapters": instrument_adapters,
        "decile_audit": decile_audit,
        "per_ticker_backtest": per_ticker_backtest,
        "per_ticker_backtest_gated": per_ticker_backtest_gated,
        "stitched_before_sq_backtest": stitched_diag_bt or None,
        "plots": plot_paths,
    }
    from reporting.desk_reports import desk_go_no_go

    report["go_no_go"] = desk_go_no_go(report, decile_audit=decile_audit)
    fold_map = (
        oos[["bar_time", "wf_fold"]].copy()
        if "wf_fold" in oos.columns
        else pd.DataFrame(columns=["bar_time", "wf_fold"])
    )
    plot_paths.update(_fusion_desk_bundle(
        equity=eq,
        prices=prices,
        symbols=symbols,
        trade_returns=trade_rets,
        report=report,
        fold_map=fold_map,
        fold_signals=ev_sig_full,
        commission_bps=commission_bps,
    ))
    report["plots"] = plot_paths
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _fusion_desk_bundle(
    *,
    equity: "pd.DataFrame | None",
    prices: "pd.DataFrame",
    symbols: list[str],
    trade_returns: np.ndarray,
    report: dict,
    fold_map: "pd.DataFrame",
    fold_signals: "pd.DataFrame | None",
    commission_bps: float,
) -> dict[str, str]:
    from reporting.desk_reports import write_quant_desk_bundle

    mc_path = OUT_DIR / "monte_carlo_report.json"
    mc_report = None
    if mc_path.is_file():
        try:
            mc_report = json.loads(mc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            mc_report = None
    return write_quant_desk_bundle(
        equity=equity,
        prices=prices,
        symbols=symbols,
        trade_returns=trade_returns,
        report=report,
        fold_signals=fold_signals,
        fold_map=fold_map,
        commission_bps=commission_bps,
        slippage_bps=float((report.get("backtest_walk_forward_oos") or {}).get("slippage_bps", 0.0) or 0.0),
        mc_report=mc_report,
    )


def regenerate_fusion_oos_plots(
    symbols: list[str],
    *,
    commission_bps: float | None = None,
    max_oos_sessions: int | None = 120,
) -> dict[str, str]:
    """Re-run walk-forward OOS backtest + standard plots from saved fusion report (skips impulse grid)."""
    from config import FUSION_ENTRY_MODEL
    from simulation.engine import run_backtest_signal_exit
    from simulation.entry_signals import active_entry_signals, deoverlap_signals, trade_returns_from_signals
    from reporting.plots import (
        plot_fold_anomaly_analysis,
        write_fusion_extended_oos_plots,
        write_standard_oos_plots,
    )

    def _fold_map() -> "pd.DataFrame":
        p = _oos_cache_path()
        if not p.is_file():
            return pd.DataFrame(columns=["bar_time", "wf_fold"])
        try:
            return pd.read_parquet(p, columns=["bar_time", "wf_fold"])
        except (OSError, ValueError, KeyError):
            return pd.DataFrame(columns=["bar_time", "wf_fold"])

    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"Fusion report missing: {REPORT_PATH}")
    saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    saved.setdefault("tickers", symbols)
    saved.setdefault("symbols", symbols)
    comm = float(commission_bps if commission_bps is not None else getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 10.0))
    start_cap = float(getattr(_cfg, "BACKTEST_START_CAPITAL", 10_000.0))
    plots_dir = OUT_DIR / "plots"
    best_impulse = saved["impulse_optimization"]["best"]
    mode = saved.get("walk_forward_mode") or saved.get("mode") or ""
    calendar_wf = mode in CALENDAR_WF_MODES or mode in (
        "monthly4y_walk_forward", "adaptive_walk_forward",
    )

    def _oos_cache_path() -> Path:
        if mode in ("adaptive", "semiannual_adaptive", "adaptive_walk_forward"):
            p = OUT_DIR / "cache" / "fusion_oos_adaptive.parquet"
        elif calendar_wf:
            p = OUT_DIR / "cache" / "fusion_oos_monthly4y.parquet"
        else:
            p = OUT_DIR / "cache" / "fusion_oos.parquet"
        if calendar_wf and not p.is_file():
            alt = OUT_DIR / "cache" / "fusion_oos_adaptive.parquet"
            if alt.is_file():
                return alt
            p = OUT_DIR / "cache" / "fusion_oos.parquet"
        return p

    if best_impulse.get("disable_trading"):
        paths = write_standard_oos_plots(
            {"equity": None, "stats": {"total_return_pct": 0.0, "sharpe": 0.0}},
            [],
            title_equity="Fusion OOS equity (no-trade)",
            title_density="Fusion OOS return density (no-trade)",
        )
        oos_cache = _oos_cache_path()
        oos_idx = None
        prices = pd.DataFrame()
        if oos_cache.is_file():
            oos_df = pd.read_parquet(oos_cache, columns=["bar_time"])
            oos_idx = pd.DatetimeIndex(pd.to_datetime(oos_df["bar_time"]))
            prices = load_closes(symbols, BAR_TIMEFRAME)
        paths.update(write_fusion_extended_oos_plots(
            None, prices, symbols, np.array([]), plots_dir,
            stats={"total_return_pct": 0.0},
            oos_index=oos_idx,
            start_capital=start_cap,
        ))
        paths.update(_fusion_desk_bundle(
            equity=None,
            prices=prices,
            symbols=symbols,
            trade_returns=np.array([]),
            report=saved,
            fold_map=_fold_map(),
            fold_signals=None,
            commission_bps=comm,
        ))
        saved.setdefault("plots", {}).update(paths)
        REPORT_PATH.write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")
        return paths
    regime_horizons = (saved.get("adaptive_horizon") or {}).get("regime_horizons")
    oos_cache = _oos_cache_path()
    eq_cache = OUT_DIR / "cache" / "fusion_bt_equity.parquet"
    sig_cache = OUT_DIR / "cache" / "fusion_bt_signals.parquet"

    if eq_cache.is_file() and sig_cache.is_file() and saved.get("backtest_walk_forward_oos"):
        print("    regen plots: loading cached backtest equity + signals", flush=True)
        prices = load_closes(symbols, BAR_TIMEFRAME)
        eq = pd.read_parquet(eq_cache)
        if "date" in eq.columns:
            eq = eq.set_index("date")
        sig = pd.read_parquet(sig_cache)
        bt = {"equity": eq, "stats": saved.get("backtest_walk_forward_oos", {})}
        entry_sig = active_entry_signals(sig)
        ev_sig = deoverlap_signals(entry_sig, prices=prices, horizon_bars=None)
        trade_rets = trade_returns_from_signals(ev_sig, comm)
        oos_idx = None
        if oos_cache.is_file():
            oos_df = pd.read_parquet(oos_cache, columns=["bar_time"])
            oos_idx = pd.DatetimeIndex(pd.to_datetime(oos_df["bar_time"]))
        paths = write_standard_oos_plots(
            bt, trade_rets,
            title_equity="Fusion OOS equity",
            title_density="Fusion OOS return density",
        )
        paths.update(write_fusion_extended_oos_plots(
            eq, prices, symbols, trade_rets, plots_dir,
            stats=saved.get("backtest_walk_forward_oos", {}),
            oos_index=oos_idx,
            start_capital=start_cap,
        ))
        paths.update(_fusion_desk_bundle(
            equity=eq,
            prices=prices,
            symbols=symbols,
            trade_returns=trade_rets,
            report=saved,
            fold_map=_fold_map(),
            fold_signals=ev_sig,
            commission_bps=comm,
        ))
        anomaly = plot_fold_anomaly_analysis(
            eq, ev_sig, _fold_map(),
            plots_dir / "fusion_fold_anomaly.png",
            commission_bps=comm,
            slippage_bps=float(saved.get("backtest_walk_forward_oos", {}).get("slippage_bps", 0.0) or 0.0),
            title="Fusion per-fold anomaly analysis",
        )
        paths["fold_anomaly"] = anomaly["plot"]
        saved["fold_anomaly"] = {"json": anomaly["json"], "n_anomalies": anomaly["n_anomalies"], "folds": anomaly["folds"]}
        saved.setdefault("plots", {}).update(paths)
        REPORT_PATH.write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")
        return paths

    if oos_cache.is_file():
        print(f"    regen plots: loading cached OOS {oos_cache.name}", flush=True)
        oos = pd.read_parquet(oos_cache)
    else:
        print("    regen plots: rebuilding panel + walk-forward OOS (slow; run fusion once to cache)", flush=True)
        feat_cols = saved["features"]
        best_p = saved.get("model_params") or saved["gbm_params"]
        model_name = saved.get("entry_model") or saved.get("model_name", FUSION_ENTRY_MODEL)
        target_col = saved.get("target", TARGET_12_AFTER_COSTS)
        panel = build_fusion_panel(symbols, hybrid=True, tick_only=True)
        if panel.empty:
            raise ValueError("Fusion panel empty")
        oos = walk_forward_fusion_oos(
            panel, feat_cols, best_p,
            model_name=model_name,
            max_oos_sessions=max_oos_sessions or saved.get("oos_sessions"),
            target_col=target_col,
        )
    if oos.empty:
        raise ValueError("No walk-forward OOS rows")
    prices = load_closes(symbols, BAR_TIMEFRAME)
    oos0 = pd.Timestamp(oos["bar_time"].min())
    oos1 = pd.Timestamp(oos["bar_time"].max())
    wf_folds = saved.get("walk_forward_folds") or []
    per_fold_th = bool((saved.get("impulse_optimization") or {}).get("per_fold_thresholds"))
    if monthly_wf and per_fold_th and wf_folds:
        sig = _fusion_signal_frame_monthly(oos, prices, wf_folds, best_impulse, regime_horizons)
    else:
        sig = _fusion_signal_frame(oos, prices, best_impulse, regime_horizons)
    bt = run_backtest_signal_exit(
        prices, sig, score_col="score",
        use_dynamic_thresholds=True, use_vol_targeting=True,
        commission_bps=comm,
        horizon_bars=_fusion_hold_default(),
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
        period_start=oos0, period_end=oos1,
    ) if not sig.empty else {"stats": {}, "equity": pd.DataFrame()}
    if not sig.empty:
        entry_sig = active_entry_signals(sig)
        ev_sig = deoverlap_signals(entry_sig, prices, None)
        trade_rets = trade_returns_from_signals(ev_sig, comm)
    else:
        ev_sig = None
        trade_rets = np.array([])
    paths = write_standard_oos_plots(
        bt, trade_rets,
        title_equity="Fusion OOS equity",
        title_density="Fusion OOS return density",
    )
    paths.update(write_fusion_extended_oos_plots(
        bt.get("equity"), prices, symbols, trade_rets, plots_dir,
        stats=bt.get("stats") or {},
        oos_index=pd.DatetimeIndex(pd.to_datetime(oos["bar_time"])) if "bar_time" in oos.columns else None,
        start_capital=start_cap,
    ))
    fold_map = oos[["bar_time", "wf_fold"]] if "wf_fold" in oos.columns else _fold_map()
    paths.update(_fusion_desk_bundle(
        equity=bt.get("equity"),
        prices=prices,
        symbols=symbols,
        trade_returns=trade_rets,
        report=saved,
        fold_map=fold_map,
        fold_signals=ev_sig,
        commission_bps=comm,
    ))
    anomaly = plot_fold_anomaly_analysis(
        bt.get("equity"),
        ev_sig if not sig.empty else None,
        fold_map,
        plots_dir / "fusion_fold_anomaly.png",
        commission_bps=comm,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
        title="Fusion per-fold anomaly analysis",
    )
    paths["fold_anomaly"] = anomaly["plot"]
    saved["fold_anomaly"] = {"json": anomaly["json"], "n_anomalies": anomaly["n_anomalies"], "folds": anomaly["folds"]}
    saved.setdefault("plots", {}).update(paths)
    REPORT_PATH.write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")
    return paths
