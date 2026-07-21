"""Per-fold LightGBM hyperparameter search via Optuna (purged session CV)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any

import config as _cfg
from models.entry_model import (
    DEFAULT_LIGHTGBM_PARAMS,
    FUSION_ENTRY_MODEL_NAME,
    lightgbm_search_space_spec,
    params_from_trial,
    params_from_values,
)


def optimize_lightgbm_on_train(
    train: "Any",
    feat_cols: list[str],
    target_col: str,
    folds: list[tuple[set, set]],
    *,
    n_trials: int | None = None,
    seed: int | None = None,
    fold_meta: dict | None = None,
) -> dict:
    """Optuna TPE on train-only purged session CV (same composite as legacy grid)."""
    import optuna
    from optuna.samplers import TPESampler

    from strategy.pipeline import _score_entry_model_cv
    from models.model_selection import resolve_optuna_objective

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    n_trials = int(n_trials or getattr(_cfg, "FUSION_MODEL_OPTUNA_TRIALS", 20))
    seed = int(seed or getattr(_cfg, "FUSION_MODEL_OPTUNA_SEED", 42))
    progress_every = int(getattr(_cfg, "FUSION_MODEL_OPT_PROGRESS_EVERY", 5))
    search_space = lightgbm_search_space_spec()
    meta = fold_meta or {}
    fold_n = meta.get("fold", "?")

    from common.stage_log import stage_log

    stage_log(
        "entry model: Optuna hyperparameter search",
        fold=fold_n,
        detail=f"{n_trials} trials, {len(train):,} train rows, {len(folds)} CV folds",
    )

    trial_results: list[dict] = []
    best: dict | None = None
    fallback: dict | None = None
    soft_fallback: dict | None = None
    best_objective = float("-inf")
    fallback_objective = float("-inf")
    soft_fallback_net = float("-inf")

    def objective(trial: optuna.Trial) -> float:
        nonlocal best, fallback, soft_fallback, best_objective, fallback_objective, soft_fallback_net
        params = params_from_trial(trial)
        row = _score_entry_model_cv(train, feat_cols, target_col, folds, FUSION_ENTRY_MODEL_NAME, params)
        if row is None:
            raise optuna.TrialPruned()
        from models.profit_metrics import optuna_reject_reason, rejects_optuna_trial

        obj = resolve_optuna_objective(row)
        net_bps = row.get("top_decile_net_bps")
        trial.set_user_attr("top_decile_net_bps", net_bps)
        trial.set_user_attr("top_decile_gross_bps", row.get("top_decile_gross_bps"))
        trial.set_user_attr("train_oos_gap_bps", row.get("train_oos_gap_bps"))
        trial.set_user_attr("auc", row.get("auc"))
        trial.set_user_attr("accuracy", row.get("accuracy"))
        rejected = bool(rejects_optuna_trial(row))
        if rejected:
            trial.set_user_attr("reject_reason", optuna_reject_reason(row))
        slim = {
            "model_name": FUSION_ENTRY_MODEL_NAME,
            "params": dict(params),
            "composite": float(row["composite"]),
            "objective": obj,
            "accuracy": row.get("accuracy"),
            "top_decile_net_bps": net_bps,
            "top_decile_gross_bps": row.get("top_decile_gross_bps"),
            "top_decile_rt_bps": row.get("top_decile_rt_bps"),
            "train_oos_gap_bps": row.get("train_oos_gap_bps"),
            "auc": row.get("auc"),
            "pr_auc": row.get("pr_auc"),
            "log_loss": row.get("log_loss"),
            "brier": row.get("brier"),
            "n_folds": row.get("n_folds"),
            "rejected_gross_below_rt": rejected,
            "rejected_optuna": rejected,
        }
        trial_results.append(slim)
        if obj > fallback_objective:
            fallback_objective = obj
            fallback = slim
        # Prefer highest CV net when every trial is hard-rejected (obj=-1).
        try:
            net_f = float(net_bps) if net_bps is not None else float("-inf")
        except (TypeError, ValueError):
            net_f = float("-inf")
        if net_f > soft_fallback_net:
            soft_fallback_net = net_f
            soft_fallback = slim
        if not rejected and obj > best_objective:
            best_objective = obj
            best = slim
        return obj

    from common.optuna_progress import optuna_progress_callback
    from common.parallel import optuna_parallel_jobs

    objective_mode = str(getattr(_cfg, "FUSION_OPTUNA_OBJECTIVE", "composite")).lower()
    _on_trial = optuna_progress_callback(
        fold=meta.get("fold", "?"),
        label="model",
        n_trials=n_trials,
        progress_every=progress_every,
        metric_name=objective_mode,
        value_fmt=".4f",
        secondary_metric="accuracy" if objective_mode != "accuracy" else "top_decile_net_bps",
        secondary_fmt="+.1f" if objective_mode == "accuracy" else "+.3f",
    )

    n_jobs = optuna_parallel_jobs()
    sampler = TPESampler(seed=seed, multivariate=True, warn_independent_sampling=False)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
        callbacks=[_on_trial],
    )

    winner = best or soft_fallback or fallback
    if winner is not None and winner.get("rejected_optuna") and best is None and soft_fallback is not None:
        stage_log(
            "entry model: all trials hard-rejected — using best CV net fallback",
            fold=fold_n,
            detail=f"net_bps={soft_fallback.get('top_decile_net_bps')}",
        )

    if winner is None:
        try:
            if study.best_trial is not None and study.best_trial.value is not None:
                winner = {
                    "model_name": FUSION_ENTRY_MODEL_NAME,
                    "params": params_from_values(study.best_params),
                    "composite": float(study.best_value),
                }
        except ValueError:
            pass

    if winner is None:
        stage_log(
            "entry model: all Optuna trials pruned — default LightGBM params",
            fold=fold_n,
        )
        winner = {
            "model_name": FUSION_ENTRY_MODEL_NAME,
            "params": dict(DEFAULT_LIGHTGBM_PARAMS),
            "composite": None,
            "source": "fallback_all_pruned",
        }

    best_params = dict(winner.get("params") or DEFAULT_LIGHTGBM_PARAMS)
    best_cv = {k: v for k, v in winner.items() if k != "params"}
    save_all = bool(getattr(_cfg, "FUSION_OPTIMIZATION_SAVE_ALL_TRIALS", True))
    leaderboard = sorted(trial_results, key=lambda r: r["composite"], reverse=True)[:10]
    recorded = trial_results if save_all else leaderboard

    optuna_best: float | None = None
    try:
        if study.best_trial is not None and study.best_trial.value is not None:
            optuna_best = float(study.best_value)
    except ValueError:
        pass

    return {
        "optimizer": "optuna",
        "model_name": FUSION_ENTRY_MODEL_NAME,
        "model_params": best_params,
        "search_space": search_space,
        "cv": {k: v for k, v in best_cv.items() if k != "params"},
        "trial_results": recorded,
        "grid_results": recorded,
        "leaderboard": leaderboard,
        "grid_size": len(recorded),
        "n_trials": len(trial_results),
        "n_train_rows": int(len(train)),
        "n_opt_rows": int(len(train)),
        "cv_folds": len(folds),
        "optuna_best_value": optuna_best,
        "optuna_n_trials": len(study.trials),
        "optuna_seed": seed,
        **meta,
    }
