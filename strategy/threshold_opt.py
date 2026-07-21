"""Per-fold trading-policy optimization on train window only (profitability objective)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as _cfg
from strategy.pipeline import (
    DEFAULT_IMPULSE_WEIGHTS,
    DEFAULT_SLIPPAGE_BPS,
    _fusion_hold_default,
    _fusion_signal_frame,
    _trade_constraint_metrics,
    calibrate_min_expected_edge_bps,
)
from strategy.edge_gate import (
    cap_calibrated_edge_to_panel,
    edge_gate_floor_mode,
    panel_abs_edge_stats,
    proportional_constraint_limits,
    resolve_min_expected_edge_bps,
    threshold_search_bounds,
)


POLICY_PARAM_KEYS = (
    "w_ml", "w_mom", "w_nw", "w_flow", "w_vp", "stress_max",
    "hmm_impulse_min", "hmm_confidence_min", "hmm_entropy_max",
    "allow_mean_revert", "buy_threshold", "sell_threshold", "min_expected_edge_bps",
    "impulse_min", "hold_threshold", "gain", "stop_loss_bps",
    "edge_floor_mode", "edge_floor_bps", "disable_trading",
    "calibrated_min_expected_edge_bps",
)


def _policy_best_params(finalized: dict) -> dict:
    """Extract tradable policy keys; preserve per-ticker overrides when present."""
    bp = {k: finalized[k] for k in POLICY_PARAM_KEYS if k in finalized}
    if finalized.get("by_ticker"):
        bp["by_ticker"] = finalized["by_ticker"]
        bp["threshold_calibrator"] = bool(finalized.get("threshold_calibrator", True))
    return bp


def _edge_floor_mode() -> str:
    return edge_gate_floor_mode()


def _threshold_opt_constraint_limits(
    val_rows: int = 0,
    val_sessions: int = 0,
) -> tuple[int, int, float]:
    if val_rows > 0 or val_sessions > 0:
        return proportional_constraint_limits(val_rows, val_sessions)
    return (
        int(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_SIGNAL_ROWS", 5)),
        int(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_ACTIVE_REBALANCES", 1)),
        float(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_AVG_EXPOSURE_PCT", 0.05)),
    )


def _passes_threshold_opt_constraints(
    metrics: dict,
    *,
    val_rows: int = 0,
    val_sessions: int = 0,
) -> bool:
    min_rows, min_reb, min_exp = _threshold_opt_constraint_limits(val_rows, val_sessions)
    return (
        metrics.get("signal_rows", 0) >= min_rows
        and metrics.get("active_rebalances", 0) >= min_reb
        and metrics.get("avg_exposure_pct", 0.0) >= min_exp
    )


def _threshold_opt_constraint_penalty(
    metrics: dict,
    *,
    val_rows: int = 0,
    val_sessions: int = 0,
) -> float:
    min_rows, min_reb, min_exp = _threshold_opt_constraint_limits(val_rows, val_sessions)
    min_rows = max(min_rows, 1)
    min_reb = max(min_reb, 1)
    min_exp = max(min_exp, 1e-9)
    shortfall = (
        max(0.0, 1.0 - metrics.get("signal_rows", 0) / min_rows)
        + max(0.0, 1.0 - metrics.get("active_rebalances", 0) / min_reb)
        + max(0.0, 1.0 - metrics.get("avg_exposure_pct", 0.0) / min_exp)
    )
    return float(getattr(_cfg, "FUSION_EXPOSURE_PENALTY", 5.0)) * shortfall


def _threshold_opt_gate_mode() -> str:
    return str(
        getattr(_cfg, "FUSION_THRESHOLD_OPT_EDGE_FLOOR_MODE", "commission_only")
    )


def _build_policy_params(
    *,
    buy_threshold: int,
    min_expected_edge_bps: float,
    impulse_min: float,
    hold_threshold: int,
    gain: int,
    calibrated_edge: float,
    w_ml: float = 0.45,
    commission_bps: float | None = None,
    stop_loss_bps: float | None = None,
    gate_mode: str | None = None,
) -> dict:
    comm = float(commission_bps if commission_bps is not None else getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 10.0))
    sl = float(
        stop_loss_bps
        if stop_loss_bps is not None
        else getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0)
    )
    floor_mode = gate_mode or _edge_floor_mode()
    from simulation.entry_signals import edge_floor_bps

    edge_floor = edge_floor_bps(comm, DEFAULT_SLIPPAGE_BPS, mode=floor_mode)
    requested = float(max(edge_floor, min_expected_edge_bps))
    cap = float(getattr(_cfg, "FUSION_EDGE_CALIBRATION_CAP_BPS", 60.0))
    opt_gate = _threshold_opt_gate_mode()
    if gate_mode == opt_gate:
        # Relaxed CV search: commission floor only — do not inflate via calibrated edge.
        resolved_edge = float(min(max(requested, edge_floor), cap))
    elif gate_mode and gate_mode != _edge_floor_mode():
        resolved_edge = float(min(max(requested, edge_floor), cap))
    else:
        resolved_edge = resolve_min_expected_edge_bps(
            requested,
            commission_bps=comm,
            calibrated=calibrated_edge,
        )
    from strategy.fusion_direction import resolve_trading_thresholds

    resolved = resolve_trading_thresholds(
        int(buy_threshold),
        int(hold_threshold),
    )
    return {
        **DEFAULT_IMPULSE_WEIGHTS,
        "w_ml": float(w_ml),
        "w_mom": 0.20,
        "w_nw": 0.15,
        "w_flow": 0.05,
        "w_vp": 0.15,
        "stress_max": 0.55,
        "hmm_impulse_min": 0.05,
        "hmm_confidence_min": 0.20,
        "hmm_entropy_max": 1.05,
        "allow_mean_revert": True,
        "buy_threshold": int(resolved["buy_threshold"]),
        "sell_threshold": float(resolved["sell_threshold"]),
        "min_expected_edge_bps": resolved_edge,
        "impulse_min": float(impulse_min),
        "hold_threshold": int(resolved["hold_threshold"]),
        "gain": int(gain),
        "stop_loss_bps": sl,
        "edge_floor_mode": edge_gate_floor_mode(),
        "disable_trading": False,
        "calibrated_min_expected_edge_bps": calibrated_edge,
        "edge_floor_bps": edge_floor,
    }


def _default_policy_params(*, calibrated_edge: float = 10.0) -> dict:
    return _build_policy_params(
        buy_threshold=51,
        min_expected_edge_bps=calibrated_edge,
        impulse_min=0.05,
        hold_threshold=49,
        gain=100,
        calibrated_edge=calibrated_edge,
    )


def _edge_bounds(
    calibrated_edge: float,
    *,
    commission_bps: float | None = None,
    stop_loss_bps: float | None = None,
    panel_max_edge_bps: float | None = None,
) -> tuple[float, float]:
    comm = float(commission_bps if commission_bps is not None else getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 10.0))
    return threshold_search_bounds(
        calibrated_edge,
        commission_bps=comm,
        panel_max_edge_bps=panel_max_edge_bps,
    )


def _search_space_spec(
    *,
    calibrated_edge: float,
    commission_bps: float | None = None,
    panel_max_edge_bps: float | None = None,
) -> dict:
    comm = float(commission_bps if commission_bps is not None else getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 10.0))
    default_sl = float(getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0))
    edge_lo, edge_hi = _edge_bounds(
        calibrated_edge,
        commission_bps=comm,
        stop_loss_bps=default_sl,
        panel_max_edge_bps=panel_max_edge_bps,
    )
    buy_lo, buy_hi = getattr(_cfg, "FUSION_THRESHOLD_BUY_RANGE", (35, 55))
    hold_lo, hold_hi = getattr(_cfg, "FUSION_THRESHOLD_HOLD_RANGE", (38, 52))
    gain_lo, gain_hi = getattr(_cfg, "FUSION_THRESHOLD_GAIN_RANGE", (70, 120))
    imp_lo, imp_hi = getattr(_cfg, "FUSION_THRESHOLD_IMPULSE_MIN_RANGE", (0.02, 0.12))
    w_ml_lo, w_ml_hi = getattr(_cfg, "FUSION_THRESHOLD_W_ML_RANGE", (0.35, 0.50))
    sl_lo, sl_hi = getattr(_cfg, "FUSION_THRESHOLD_STOP_LOSS_RANGE", (25, 45))
    return {
        "optimizer": "optuna",
        "sampler": "TPESampler",
        "gate_mode": _threshold_opt_gate_mode(),
        "calibrated_min_expected_edge_bps": round(calibrated_edge, 2),
        "params": {
            "stop_loss_bps": {"type": "float", "low": float(sl_lo), "high": float(sl_hi), "step": 1.0},
            "buy_threshold": {"type": "int", "low": int(buy_lo), "high": int(buy_hi)},
            "min_expected_edge_bps": {
                "type": "float",
                "low": edge_lo,
                "high": edge_hi,
                "step": 0.5,
                "dynamic_floor_from": "stop_loss_bps",
            },
            "impulse_min": {"type": "float", "low": float(imp_lo), "high": float(imp_hi), "step": 0.01},
            "hold_threshold": {"type": "int", "low": int(hold_lo), "high": int(hold_hi)},
            "gain": {"type": "int", "low": int(gain_lo), "high": int(gain_hi)},
            "w_ml": {"type": "float", "low": float(w_ml_lo), "high": float(w_ml_hi), "step": 0.05},
        },
        "fixed": {
            "w_mom": 0.20,
            "w_nw": 0.15,
            "w_flow": 0.05,
            "w_vp": 0.15,
            "stress_max": 0.55,
            "hmm_impulse_min": 0.05,
            "hmm_confidence_min": 0.20,
            "hmm_entropy_max": 1.05,
        },
    }


def _edge_bounds_from_search_space(
    search_space: dict,
    *,
    commission_bps: float | None = None,
    calibrated_edge: float | None = None,
    stop_loss_bps: float | None = None,
    panel_max_edge_bps: float | None = None,
) -> tuple[float, float]:
    """Edge search bounds for Optuna — prefer panel-capped spec over uncapped recompute."""
    spec = (search_space or {}).get("params", {}).get("min_expected_edge_bps")
    if spec and "low" in spec and "high" in spec:
        return float(spec["low"]), float(spec["high"])
    return _edge_bounds(
        float(calibrated_edge or 0.0),
        commission_bps=commission_bps,
        stop_loss_bps=stop_loss_bps,
        panel_max_edge_bps=panel_max_edge_bps,
    )


def _params_from_trial(trial, *, calibrated_edge: float, search_space: dict, commission_bps: float | None = None) -> dict:
    spec = search_space["params"]
    gate_mode = search_space.get("gate_mode")
    comm = float(commission_bps if commission_bps is not None else getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 10.0))
    sl_spec = spec["stop_loss_bps"]
    stop_loss_bps = trial.suggest_float(
        "stop_loss_bps", sl_spec["low"], sl_spec["high"], step=sl_spec.get("step", 1.0)
    )
    edge_lo, edge_hi = _edge_bounds_from_search_space(
        search_space,
        commission_bps=comm,
        calibrated_edge=calibrated_edge,
        stop_loss_bps=stop_loss_bps,
    )
    buy = spec["buy_threshold"]
    hold = spec["hold_threshold"]
    gain = spec["gain"]
    imp = spec["impulse_min"]
    w_ml = spec["w_ml"]
    buy_threshold = trial.suggest_int("buy_threshold", buy["low"], buy["high"])
    hold_threshold = trial.suggest_int("hold_threshold", int(hold["low"]), int(hold["high"]))
    from strategy.fusion_direction import normalize_buy_threshold, resolve_trading_thresholds

    bands = resolve_trading_thresholds(
        normalize_buy_threshold(buy_threshold),
        hold_threshold,
    )
    buy_threshold = int(bands["buy_threshold"])
    hold_threshold = int(bands["hold_threshold"])
    return _build_policy_params(
        buy_threshold=buy_threshold,
        min_expected_edge_bps=trial.suggest_float(
            "min_expected_edge_bps", edge_lo, edge_hi, step=0.5
        ),
        impulse_min=trial.suggest_float(
            "impulse_min", imp["low"], imp["high"], step=imp.get("step", 0.01)
        ),
        hold_threshold=hold_threshold,
        gain=trial.suggest_int("gain", gain["low"], gain["high"]),
        w_ml=trial.suggest_float("w_ml", w_ml["low"], w_ml["high"], step=w_ml.get("step", 0.05)),
        calibrated_edge=calibrated_edge,
        commission_bps=commission_bps,
        stop_loss_bps=stop_loss_bps,
        gate_mode=gate_mode,
    )


def _params_from_values(
    values: dict,
    *,
    calibrated_edge: float,
    commission_bps: float | None = None,
    gate_mode: str | None = None,
    search_space: dict | None = None,
) -> dict:
    comm = float(commission_bps if commission_bps is not None else getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 10.0))
    sl = float(values.get("stop_loss_bps", getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0)))
    edge_lo, edge_hi = _edge_bounds_from_search_space(
        search_space or {},
        commission_bps=comm,
        calibrated_edge=calibrated_edge,
        stop_loss_bps=sl,
    )
    min_edge = float(values["min_expected_edge_bps"])
    min_edge = max(edge_lo, min(min_edge, edge_hi))
    return _build_policy_params(
        buy_threshold=int(values["buy_threshold"]),
        min_expected_edge_bps=min_edge,
        impulse_min=float(values["impulse_min"]),
        hold_threshold=int(values["hold_threshold"]),
        gain=int(values["gain"]),
        w_ml=float(values.get("w_ml", 0.45)),
        calibrated_edge=calibrated_edge,
        commission_bps=commission_bps,
        stop_loss_bps=sl,
        gate_mode=gate_mode,
    )


def _threshold_opt_backtest_kwargs() -> dict:
    """Relaxed execution settings for short purged-CV folds in threshold search."""
    return {
        "min_hold_bars": int(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_HOLD_BARS", 12)),
        "rebalance_band": float(getattr(_cfg, "FUSION_THRESHOLD_OPT_REBALANCE_BAND", 0.02)),
        "hold_entry_weight": bool(getattr(_cfg, "FUSION_THRESHOLD_OPT_HOLD_ENTRY_WEIGHT", False)),
        "reentry_cooldown_bars": int(getattr(_cfg, "FUSION_THRESHOLD_OPT_REENTRY_COOLDOWN_BARS", 12)),
    }


def _panel_edge_ceiling(
    train: pd.DataFrame,
    commission_bps: float,
    *,
    max_rows: int = 25_000,
) -> tuple[float | None, float]:
    """Positive-edge ceiling on the policy panel: (max_bps, q65_bps)."""
    if train.empty or "ml_proba" not in train.columns:
        return None, 0.0
    from strategy.pipeline import apply_fusion_scores

    work = train
    if len(train) > max_rows:
        work = train.sample(n=max_rows, random_state=42)
    probe = _build_policy_params(
        buy_threshold=52,
        min_expected_edge_bps=0.0,
        impulse_min=0.0,
        hold_threshold=49,
        gain=60,
        calibrated_edge=0.0,
        w_ml=0.45,
        commission_bps=commission_bps,
        stop_loss_bps=float(getattr(_cfg, "FUSION_STOP_LOSS_BPS", 25.0)),
    )
    fused = apply_fusion_scores(work, probe)
    panel_max, panel_q65, n_pos = panel_abs_edge_stats(fused["expected_edge_bps"])
    return (panel_max if n_pos > 0 else None), panel_q65


def _evaluate_policy_cv(
    train: pd.DataFrame,
    prices: pd.DataFrame,
    params: dict,
    folds: list[tuple[set, set]],
    *,
    commission_bps: float,
) -> dict | None:
    from simulation.engine import run_backtest_signal_exit

    hold_default = _fusion_hold_default()
    fold_sharpes: list[float] = []
    fold_returns: list[float] = []
    fold_signal_rows: list[int] = []
    fold_active_rebalances: list[int] = []
    fold_exposures: list[float] = []
    fold_trade_net_bps: list[float] = []
    total_val_rows = 0
    total_val_sessions = 0

    from simulation.entry_signals import (
        active_entry_signals,
        deoverlap_signals,
        trade_returns_from_signals,
    )
    from strategy.objective import fusion_cv_objective

    for _train_s, val_s in folds:
        val = train[train["session"].isin(val_s)]
        if val.empty:
            continue
        total_val_rows += len(val)
        total_val_sessions += len(val_s)
        sig = _fusion_signal_frame(val, prices, params)
        if sig.empty:
            fold_signal_rows.append(0)
            fold_active_rebalances.append(0)
            fold_exposures.append(0.0)
            fold_returns.append(0.0)
            fold_sharpes.append(float("nan"))
            fold_trade_net_bps.append(0.0)
            continue
        val_start = pd.Timestamp(val["bar_time"].min())
        val_end = pd.Timestamp(val["bar_time"].max())
        bt = run_backtest_signal_exit(
            prices,
            sig,
            score_col="score",
            use_dynamic_thresholds=True,
            use_vol_targeting=True,
            commission_bps=commission_bps,
            horizon_bars=hold_default,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
            stop_loss_bps=float(params.get("stop_loss_bps", getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0))),
            period_start=val_start,
            period_end=val_end,
            **_threshold_opt_backtest_kwargs(),
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
        ev_sig = deoverlap_signals(entry_sig, prices, hold_default)
        trade_rets = trade_returns_from_signals(ev_sig, commission_bps)
        if len(trade_rets):
            fold_trade_net_bps.append(float(np.mean(trade_rets)) * 10_000.0)
        else:
            fold_trade_net_bps.append(0.0)

    if not fold_sharpes:
        return None

    valid_sharpes = [float(s) for s in fold_sharpes if np.isfinite(s)]
    if not valid_sharpes:
        return None
    sharpe = float(np.mean(valid_sharpes))
    ret = float(np.mean(fold_returns)) if fold_returns else 0.0
    constraint_metrics = {
        "signal_rows": round(float(np.mean(fold_signal_rows))) if fold_signal_rows else 0,
        "active_rebalances": round(float(np.mean(fold_active_rebalances))) if fold_active_rebalances else 0,
        "avg_exposure_pct": float(np.mean(fold_exposures)) if fold_exposures else 0.0,
    }
    constraints_ok = _passes_threshold_opt_constraints(
        constraint_metrics,
        val_rows=total_val_rows,
        val_sessions=total_val_sessions,
    )
    mean_trade_net = float(np.mean(fold_trade_net_bps)) if fold_trade_net_bps else 0.0
    if (
        bool(getattr(_cfg, "FUSION_THRESHOLD_OPT_REQUIRE_POSITIVE_TRADE_NET", True))
        and mean_trade_net < 0.0
    ):
        constraints_ok = False
    penalty = (
        0.0
        if constraints_ok
        else _threshold_opt_constraint_penalty(
            constraint_metrics,
            val_rows=total_val_rows,
            val_sessions=total_val_sessions,
        )
    )
    turnover_penalty = float(getattr(_cfg, "FUSION_TURNOVER_PENALTY", 0.02)) * constraint_metrics["active_rebalances"]
    n_cv_folds = max(1, len(fold_sharpes))
    avg_val_sessions = max(1.0, total_val_sessions / n_cv_folds)
    fold_days = max(7.0, avg_val_sessions)
    objective = fusion_cv_objective(
        sharpe=sharpe,
        mean_return_pct=ret,
        active_rebalances=constraint_metrics["active_rebalances"],
        mean_trade_net_bps=mean_trade_net,
        constraint_penalty=penalty,
        fold_days=fold_days,
        n_cv_folds=n_cv_folds,
    )
    return {
        **params,
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(ret, 2),
        "objective": round(objective, 3),
        "constraints_ok": constraints_ok,
        "mean_trade_net_bps": round(mean_trade_net, 2),
        **constraint_metrics,
    }


def _slim_trial_row(row: dict, *, trial_number: int | None = None) -> dict:
    out = {
        "trial_number": trial_number,
        "buy_threshold": row["buy_threshold"],
        "min_expected_edge_bps": row["min_expected_edge_bps"],
        "impulse_min": row["impulse_min"],
        "hold_threshold": row["hold_threshold"],
        "gain": row["gain"],
        "w_ml": row.get("w_ml"),
        "stop_loss_bps": row.get("stop_loss_bps"),
        "objective": row["objective"],
        "sharpe": row["sharpe"],
        "total_return_pct": row["total_return_pct"],
        "signal_rows": row["signal_rows"],
        "active_rebalances": row.get("active_rebalances"),
        "avg_exposure_pct": row.get("avg_exposure_pct"),
        "constraints_ok": row["constraints_ok"],
    }
    return out


def _trial_history_payload(trial_results: list[dict]) -> dict:
    from strategy.optimization_audit import summarize_numeric_params, top_fraction_rows

    top_frac = float(getattr(_cfg, "FUSION_OPTIMIZATION_SUMMARY_TOP_FRAC", 0.25))
    keys = (
        "buy_threshold", "min_expected_edge_bps", "impulse_min",
        "hold_threshold", "gain", "w_ml", "stop_loss_bps",
    )
    top_rows = top_fraction_rows(trial_results, score_key="objective", top_frac=top_frac)
    return {
        "all_trials": summarize_numeric_params(trial_results, keys),
        "top_trials": summarize_numeric_params(top_rows, keys),
        "n_trials_recorded": len(trial_results),
    }


def _finalize_threshold_winner(
    winner: dict,
    *,
    best: dict | None,
    calibrated_edge: float,
) -> dict:
    from strategy.objective import threshold_no_trade_objective

    obj = winner.get("objective")
    zero_signals = int(winner.get("signal_rows") or 0) <= 0
    obj_floor = threshold_no_trade_objective()
    below_floor = obj is not None and float(obj) < obj_floor
    return {
        **winner,
        "constraints_fallback": best is None,
        "disable_trading": False,
        "cv_zero_signals": zero_signals,
        "trade_anomaly": zero_signals or below_floor,
        "calibrated_min_expected_edge_bps": calibrated_edge,
    }


def _pack_threshold_result(
    *,
    optimizer: str,
    winner: dict,
    search_space: dict,
    trial_results: list[dict],
    folds: list,
    meta: dict,
    calibrated_edge: float,
    n_train: int,
    n_opt_rows: int,
    seed: int | None = None,
    optuna_study=None,
    had_constrained_best: bool = False,
) -> dict:
    save_all = bool(getattr(_cfg, "FUSION_OPTIMIZATION_SAVE_ALL_TRIALS", True))
    finalized = _finalize_threshold_winner(
        winner,
        best=winner if had_constrained_best else None,
        calibrated_edge=calibrated_edge,
    )
    leaderboard = sorted(trial_results, key=lambda r: r.get("objective", -999), reverse=True)[:10]
    recorded = trial_results if save_all else leaderboard
    param_stats = _trial_history_payload(trial_results) if trial_results else {}
    cv = {
        "objective": finalized.get("objective"),
        "sharpe": finalized.get("sharpe"),
        "total_return_pct": finalized.get("total_return_pct"),
        "signal_rows": finalized.get("signal_rows"),
        "constraints_ok": finalized.get("constraints_ok"),
        "calibrated_min_expected_edge_bps": calibrated_edge,
        "cv_zero_signals": finalized.get("cv_zero_signals"),
        "trade_anomaly": finalized.get("trade_anomaly"),
    }
    if optuna_study is not None:
        cv["optuna_best_value"] = optuna_study.best_value if optuna_study.trials else None
        cv["optuna_n_trials"] = len(optuna_study.trials)
        cv["optuna_best_trial"] = optuna_study.best_trial.number if optuna_study.best_trial else None
    return {
        "optimizer": optimizer,
        "best_params": _policy_best_params(finalized),
        "search_space": search_space,
        "cv": cv,
        "trial_results": recorded,
        "grid_results": recorded,
        "leaderboard": leaderboard,
        "param_stats": param_stats,
        "n_trials": len(trial_results),
        "n_trials_saved": len(recorded),
        "grid_size": len(recorded),
        "cv_folds": len(folds),
        "optuna_seed": seed,
        "n_train_rows": n_train,
        "n_opt_rows": n_opt_rows,
        **meta,
    }


def _range_midpoint(lo_hi: tuple[float, float] | tuple[int, int], *, as_int: bool = False):
    lo, hi = lo_hi
    mid = (float(lo) + float(hi)) / 2.0
    return int(round(mid)) if as_int else mid


def _fixed_policy_from_config(
    *,
    calibrated_edge: float,
    commission_bps: float,
) -> dict:
    buy_lo, buy_hi = getattr(_cfg, "FUSION_THRESHOLD_BUY_RANGE", (52, 58))
    hold_lo, hold_hi = getattr(_cfg, "FUSION_THRESHOLD_HOLD_RANGE", (40, 48))
    buy = _range_midpoint((buy_lo, buy_hi), as_int=True)
    hold = _range_midpoint((hold_lo, hold_hi), as_int=True)
    gain = _range_midpoint(getattr(_cfg, "FUSION_THRESHOLD_GAIN_RANGE", (60, 100)), as_int=True)
    impulse = _range_midpoint(getattr(_cfg, "FUSION_THRESHOLD_IMPULSE_MIN_RANGE", (0.01, 0.06)))
    w_ml = _range_midpoint(getattr(_cfg, "FUSION_THRESHOLD_W_ML_RANGE", (0.35, 0.55)))
    gate = _threshold_opt_gate_mode()
    from strategy.fusion_direction import resolve_trading_thresholds

    bands = resolve_trading_thresholds(int(buy), int(hold))
    return _build_policy_params(
        buy_threshold=int(bands["buy_threshold"]),
        min_expected_edge_bps=calibrated_edge,
        impulse_min=float(impulse),
        hold_threshold=int(bands["hold_threshold"]),
        gain=gain,
        w_ml=float(w_ml),
        calibrated_edge=calibrated_edge,
        commission_bps=commission_bps,
        gate_mode=gate,
    )


def _quick_signal_count(
    train: pd.DataFrame,
    prices: pd.DataFrame,
    params: dict,
    *,
    max_rows: int = 20_000,
) -> int:
    from simulation.entry_signals import active_entry_signals

    work = train
    if len(train) > max_rows:
        work = train.sample(n=max_rows, random_state=42)
    sig = _fusion_signal_frame(work, prices, params)
    if sig.empty:
        return 0
    return int(len(active_entry_signals(sig)))


def _calibrate_edge_on_train(
    train: pd.DataFrame,
    *,
    commission_bps: float,
) -> float:
    calibrated_edge = calibrate_min_expected_edge_bps(train, commission_bps)
    panel_max_edge, panel_q65 = _panel_edge_ceiling(train, commission_bps)
    if panel_max_edge is not None:
        calibrated_edge = cap_calibrated_edge_to_panel(
            calibrated_edge,
            panel_max_edge_bps=panel_max_edge,
            panel_q65_edge_bps=panel_q65,
            commission_bps=commission_bps,
        )
    return float(calibrated_edge)


def build_calibrated_trading_policy(
    train: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    commission_bps: float,
    fold_meta: dict | None = None,
    holdout: pd.DataFrame | None = None,
) -> dict:
    """Per-fold policy without Optuna: calibrate edge + fixed thresholds from config."""
    from common.stage_log import stage_log

    meta = fold_meta or {}
    fold_n = meta.get("fold", "?")
    if train.empty or "ml_proba" not in train.columns:
        fallback = _default_policy_params(calibrated_edge=10.0)
        return {
            "optimizer": "calibrated",
            "best_params": fallback,
            "search_space": _search_space_spec(calibrated_edge=10.0),
            "cv": {"objective": None, "source": "fallback_empty_train"},
            "trial_results": [],
            "grid_results": [],
            "leaderboard": [],
            "param_stats": {},
            "n_trials": 0,
            "grid_size": 0,
            **meta,
        }

    stage_log("trading policy: calibrating edge floor", fold=fold_n, detail=f"{len(train):,} train rows")
    calibrated_edge = _calibrate_edge_on_train(train, commission_bps=commission_bps)
    stage_log(
        "trading policy: building fixed thresholds",
        fold=fold_n,
        detail=f"edge={calibrated_edge:.1f} bps",
    )
    policy = _fixed_policy_from_config(
        calibrated_edge=calibrated_edge,
        commission_bps=commission_bps,
    )
    stage_log("trading policy: counting entry signals (sample)", fold=fold_n)
    signal_rows = _quick_signal_count(train, prices, policy)

    mode = str(getattr(_cfg, "FUSION_THRESHOLD_POLICY_MODE", "calibrated")).lower()
    by_ticker: dict[str, dict] = {}
    optimizer = "calibrated"
    if mode in ("per_ticker_calibrated", "per_ticker") and "ticker" in train.columns:
        from strategy.threshold_calibrator import fit_threshold_calibrator, merge_ticker_policies

        stage_log("threshold model: signal quality + policy per instrument", fold=fold_n)
        by_ticker = fit_threshold_calibrator(train, policy, fold=fold_n, holdout=holdout)
        if by_ticker:
            policy = merge_ticker_policies(policy, by_ticker)
            optimizer = "per_ticker_calibrated"
            nets = [
                float(v.get("cv_top_decile_net_bps", v.get("cv_net_bps")))
                for v in by_ticker.values()
                if v.get("cv_top_decile_net_bps", v.get("cv_net_bps")) is not None
                and np.isfinite(v.get("cv_top_decile_net_bps", v.get("cv_net_bps")))
            ]
            n_ok = sum(1 for v in by_ticker.values() if v.get("signal_quality_ok"))
            if nets:
                parts = [
                    f"{k}={v.get('cv_top_decile_net_bps', v.get('cv_net_bps'))}"
                    f"{'' if v.get('signal_quality_ok') else '(skip)'}"
                    for k, v in by_ticker.items()
                ]
                stage_log(
                    "threshold model summary",
                    fold=fold_n,
                    detail=(
                        f"mean_top_decile={np.mean(nets):.1f} bps "
                        f"signal_ok={n_ok}/{len(by_ticker)} | " + ", ".join(parts)
                    ),
                )

    winner = {
        **policy,
        "objective": None,
        "signal_rows": signal_rows,
        "constraints_ok": signal_rows > 0,
        "sharpe": None,
        "total_return_pct": None,
    }
    search_space = _search_space_spec(
        calibrated_edge=calibrated_edge,
        commission_bps=commission_bps,
    )
    cv_extra = {"by_ticker": by_ticker} if by_ticker else {}
    result = _pack_threshold_result(
        optimizer=optimizer,
        winner=winner,
        search_space=search_space,
        trial_results=[],
        folds=[],
        meta=meta,
        calibrated_edge=calibrated_edge,
        n_train=int(len(train)),
        n_opt_rows=int(len(train)),
        had_constrained_best=signal_rows > 0,
    )
    if cv_extra:
        result.setdefault("cv", {}).update(cv_extra)
    return result


def optimize_trading_policy_on_train(
    train: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    commission_bps: float,
    fold_meta: dict | None = None,
    ticker: str | None = None,
    n_trials: int | None = None,
) -> dict:
    """Purged session CV on train predictions only; test month never included."""
    from common.stage_log import stage_log
    from research.features.entry_ml import _purged_session_folds

    meta = dict(fold_meta or {})
    if ticker:
        meta["ticker"] = str(ticker).upper()
    fold_n = meta.get("fold", "?")
    sym = meta.get("ticker")

    if train.empty or "ml_proba" not in train.columns:
        fallback = _default_policy_params(calibrated_edge=10.0)
        return {
            "optimizer": "optuna",
            "best_params": fallback,
            "search_space": _search_space_spec(calibrated_edge=10.0),
            "cv": {"objective": None, "source": "fallback_empty_train"},
            "trial_results": [],
            "grid_results": [],
            "leaderboard": [],
            "param_stats": {},
            "n_trials": 0,
            "grid_size": 0,
            **meta,
        }

    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required for threshold optimization; install via requirements-quant.txt"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    n_train = int(len(train))
    row_cap = getattr(_cfg, "FUSION_THRESHOLD_OPT_MAX_TRAIN_ROWS", None)
    if row_cap is None:
        row_cap = getattr(_cfg, "FUSION_FOLD_OPT_MAX_TRAIN_ROWS", None)
    seed = int(getattr(_cfg, "FUSION_THRESHOLD_OPTUNA_SEED", 42)) + int(meta.get("fold", 0))
    if sym:
        seed += sum(ord(c) for c in str(sym)) % 997
    work = train
    if row_cap and n_train > int(row_cap):
        from strategy.target_opt import _subsample

        work = _subsample(train, int(row_cap), seed)
        tag = f"{fold_n}/{sym}" if sym else str(fold_n)
        print(
            f"      fold {tag} threshold opt: subsampled {n_train:,} -> {len(work):,} rows",
            flush=True,
        )

    calibrated_edge = _calibrate_edge_on_train(work, commission_bps=commission_bps)
    panel_max_edge, panel_q65 = _panel_edge_ceiling(work, commission_bps)
    search_space = _search_space_spec(
        calibrated_edge=calibrated_edge,
        commission_bps=commission_bps,
        panel_max_edge_bps=panel_max_edge,
    )
    from strategy.leakage_guard import resolve_label_horizon_bars

    horizon_bars = resolve_label_horizon_bars()
    sessions = sorted(work["session"].unique())
    folds = _purged_session_folds(sessions, max_label_horizon_bars=horizon_bars)
    if n_trials is None:
        n_trials = int(getattr(_cfg, "FUSION_THRESHOLD_OPTUNA_TRIALS", 25))
    progress_every = max(1, int(getattr(_cfg, "FUSION_THRESHOLD_OPT_PROGRESS_EVERY", 5)))
    scope = f"{sym} " if sym else ""
    stage_log(
        f"trading policy: Optuna {scope}threshold search",
        fold=fold_n,
        detail=f"{n_trials} trials, {len(work):,} rows",
    )

    trial_results: list[dict] = []
    best: dict | None = None
    best_objective = -999.0
    fallback: dict | None = None
    fallback_objective = -999.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best, best_objective, fallback, fallback_objective
        params = _params_from_trial(
            trial,
            calibrated_edge=calibrated_edge,
            search_space=search_space,
            commission_bps=commission_bps,
        )
        row = _evaluate_policy_cv(work, prices, params, folds, commission_bps=commission_bps)
        if row is None:
            raise optuna.TrialPruned()

        slim = _slim_trial_row(row, trial_number=trial.number)
        trial_results.append(slim)
        trial.set_user_attr("sharpe", row["sharpe"])
        trial.set_user_attr("total_return_pct", row["total_return_pct"])
        trial.set_user_attr("constraints_ok", row["constraints_ok"])
        trial.set_user_attr("signal_rows", row["signal_rows"])
        trial.set_user_attr("active_rebalances", row.get("active_rebalances"))
        trial.set_user_attr("avg_exposure_pct", row.get("avg_exposure_pct"))

        obj = float(row["objective"])
        if obj > fallback_objective:
            fallback_objective = obj
            fallback = row
        if row.get("constraints_ok") and obj > best_objective:
            best_objective = obj
            best = row
        return obj

    from common.optuna_progress import optuna_progress_callback
    from common.parallel import optuna_parallel_jobs_threshold

    progress_label = f"threshold/{sym}" if sym else "threshold"
    _on_progress = optuna_progress_callback(
        fold=meta.get("fold", "?"),
        label=progress_label,
        n_trials=n_trials,
        progress_every=progress_every,
        metric_name="objective",
        value_fmt=".3f",
    )

    n_jobs = optuna_parallel_jobs_threshold()
    sampler = optuna.samplers.TPESampler(
        seed=seed, multivariate=True, warn_independent_sampling=False,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
        callbacks=[_on_progress],
    )

    tag = f"{fold_n}/{sym}" if sym else str(fold_n)
    if study.trials and study.best_value <= -14.5:
        n_zero = sum(1 for t in study.trials if t.value is not None and t.value <= -14.5)
        print(
            f"      fold {tag} threshold WARNING: {n_zero}/{len(study.trials)} trials at "
            f"zero-trade floor (best={study.best_value:.3f}); edge floor may be too high "
            f"or CV windows too short",
            flush=True,
        )
    elif best is None and fallback is not None:
        print(
            f"      fold {tag} threshold: no trial passed constraints — using fallback "
            f"(objective={fallback.get('objective')}, signal_rows={fallback.get('signal_rows')})",
            flush=True,
        )

    if best is None and fallback is None and study.best_trial is not None:
        fallback = _evaluate_policy_cv(
            work,
            prices,
            _params_from_values(
                study.best_params,
                calibrated_edge=calibrated_edge,
                commission_bps=commission_bps,
                gate_mode=search_space.get("gate_mode"),
                search_space=search_space,
            ),
            folds,
            commission_bps=commission_bps,
        )

    winner = best or fallback or _default_policy_params(calibrated_edge=calibrated_edge)
    return _pack_threshold_result(
        optimizer="optuna",
        winner=winner,
        search_space=search_space,
        trial_results=trial_results,
        folds=folds,
        meta=meta,
        calibrated_edge=calibrated_edge,
        n_train=n_train,
        n_opt_rows=int(len(work)),
        seed=seed,
        optuna_study=study,
        had_constrained_best=best is not None,
    )


def _fallback_ticker_threshold_policy(sym: str) -> dict:
    from data_platform.universe import commission_bps_for_ticker
    from strategy.edge_gate import heuristic_gate_floor_bps

    comm = commission_bps_for_ticker(sym)
    edge = float(heuristic_gate_floor_bps(comm))
    return {
        **_default_policy_params(calibrated_edge=edge),
        "signal_quality_ok": False,
        "cv_objective": None,
        "cv_signal_rows": 0,
        "optimizer": "fallback",
    }


def _median_base_policy(by_ticker: dict[str, dict], *, calibrated_edge: float) -> dict:
    candidates = [
        v for v in by_ticker.values()
        if v.get("buy_threshold") is not None and v.get("hold_threshold") is not None
    ]
    if not candidates:
        return _default_policy_params(calibrated_edge=calibrated_edge)
    buy = int(round(float(np.median([float(v["buy_threshold"]) for v in candidates]))))
    hold = int(round(float(np.median([float(v["hold_threshold"]) for v in candidates]))))
    edge = float(np.median([float(v.get("min_expected_edge_bps", calibrated_edge)) for v in candidates]))
    impulse = float(np.median([float(v.get("impulse_min", 0.05)) for v in candidates]))
    gain = int(round(float(np.median([float(v.get("gain", 100)) for v in candidates]))))
    w_ml = float(np.median([float(v.get("w_ml", 0.45)) for v in candidates]))
    return _build_policy_params(
        buy_threshold=buy,
        min_expected_edge_bps=edge,
        impulse_min=impulse,
        hold_threshold=hold,
        gain=gain,
        w_ml=w_ml,
        calibrated_edge=calibrated_edge,
    )


def optimize_trading_policy_per_ticker_on_train(
    train: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    commission_bps: float,
    fold_meta: dict | None = None,
) -> dict:
    """Optuna threshold search per instrument; merged into ``by_ticker`` overrides."""
    from common.stage_log import stage_log
    from data_platform.universe import commission_bps_for_ticker
    from strategy.threshold_calibrator import merge_ticker_policies

    meta = fold_meta or {}
    fold_n = meta.get("fold", "?")

    if train.empty or "ml_proba" not in train.columns or "ticker" not in train.columns:
        fallback = _default_policy_params(calibrated_edge=10.0)
        return {
            "optimizer": "optuna_per_ticker",
            "best_params": fallback,
            "search_space": _search_space_spec(calibrated_edge=10.0),
            "cv": {"objective": None, "source": "fallback_empty_train"},
            "trial_results": [],
            "grid_results": [],
            "leaderboard": [],
            "param_stats": {},
            "n_trials": 0,
            "grid_size": 0,
            **meta,
        }

    min_rows = int(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_TRAIN_ROWS_PER_TICKER", 200))
    min_sessions = int(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_TRAIN_SESSIONS_PER_TICKER", 4))
    n_trials_pt = int(
        getattr(_cfg, "FUSION_THRESHOLD_OPTUNA_TRIALS_PER_TICKER", 0)
        or getattr(_cfg, "FUSION_THRESHOLD_OPTUNA_TRIALS", 25)
    )

    calibrated_edge = _calibrate_edge_on_train(train, commission_bps=commission_bps)
    by_ticker: dict[str, dict] = {}
    all_trials: list[dict] = []
    objectives: list[float] = []
    signal_rows: list[int] = []

    tickers = sorted(train["ticker"].astype(str).str.upper().unique())
    stage_log(
        "trading policy: per-instrument Optuna",
        fold=fold_n,
        detail=f"{len(tickers)} tickers x {n_trials_pt} trials",
    )

    for sym in tickers:
        sub = train[train["ticker"].astype(str).str.upper() == sym]
        if len(sub) < min_rows or sub["session"].nunique() < min_sessions:
            by_ticker[sym] = _fallback_ticker_threshold_policy(sym)
            by_ticker[sym]["source"] = "fallback_insufficient_rows"
            continue

        sym_comm = float(commission_bps_for_ticker(sym))
        sub_result = optimize_trading_policy_on_train(
            sub,
            prices,
            commission_bps=sym_comm,
            fold_meta={**meta, "ticker": sym},
            ticker=sym,
            n_trials=n_trials_pt,
        )
        bp = dict(sub_result.get("best_params") or {})
        cv = sub_result.get("cv") or {}
        sig_n = int(cv.get("signal_rows") or 0)
        obj = cv.get("objective")
        by_ticker[sym] = {
            **bp,
            "signal_quality_ok": bool(sig_n > 0 and not cv.get("trade_anomaly")),
            "cv_objective": obj,
            "cv_signal_rows": sig_n,
            "optimizer": "optuna",
            "commission_bps_per_side": sym_comm,
        }
        all_trials.extend(sub_result.get("trial_results") or [])
        if obj is not None and np.isfinite(float(obj)):
            objectives.append(float(obj))
        signal_rows.append(sig_n)
        stage_log(
            f"threshold [{sym}]",
            fold=fold_n,
            detail=(
                f"buy={bp.get('buy_threshold')} sell={bp.get('sell_threshold')} "
                f"edge={bp.get('min_expected_edge_bps')} signals={sig_n} obj={obj}"
            ),
        )

    base = _median_base_policy(by_ticker, calibrated_edge=calibrated_edge)
    merged = merge_ticker_policies(base, by_ticker)
    merged["objective"] = float(np.mean(objectives)) if objectives else None
    merged["signal_rows"] = int(round(np.mean(signal_rows))) if signal_rows else 0
    merged["constraints_ok"] = any(v.get("signal_quality_ok") for v in by_ticker.values())

    search_space = _search_space_spec(
        calibrated_edge=calibrated_edge,
        commission_bps=commission_bps,
    )
    result = _pack_threshold_result(
        optimizer="optuna_per_ticker",
        winner=merged,
        search_space=search_space,
        trial_results=all_trials,
        folds=[],
        meta=meta,
        calibrated_edge=calibrated_edge,
        n_train=int(len(train)),
        n_opt_rows=int(len(train)),
        had_constrained_best=merged.get("constraints_ok", False),
    )
    result.setdefault("cv", {}).update(
        {
            "by_ticker": by_ticker,
            "objective": merged.get("objective"),
            "signal_rows": merged.get("signal_rows"),
            "n_tickers_optimized": sum(1 for v in by_ticker.values() if v.get("optimizer") == "optuna"),
        }
    )
    return result
