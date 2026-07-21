"""Aggregate per-fold optimization trials for audit and range narrowing."""

from __future__ import annotations

from typing import Any

import numpy as np


def _numeric_values(rows: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        val = row.get(key)
        if val is None:
            continue
        try:
            out.append(float(val))
        except (TypeError, ValueError):
            continue
    return out


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "min": round(float(np.min(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
    }


def summarize_numeric_params(
    rows: list[dict],
    param_keys: tuple[str, ...],
) -> dict[str, dict[str, float | int] | None]:
    return {key: _distribution(_numeric_values(rows, key)) for key in param_keys}


def top_fraction_rows(
    rows: list[dict],
    *,
    score_key: str = "objective",
    top_frac: float = 0.25,
    min_rows: int = 5,
) -> list[dict]:
    if not rows:
        return []
    scored = [r for r in rows if r.get(score_key) is not None]
    if not scored:
        return []
    scored.sort(key=lambda r: float(r[score_key]), reverse=True)
    n = max(min_rows, int(np.ceil(len(scored) * top_frac)))
    n = min(n, len(scored))
    return scored[:n]


def suggest_narrowed_range(
    dist: dict[str, float | int] | None,
    *,
    padding_frac: float = 0.15,
    step: float | None = None,
) -> dict[str, float] | None:
    """Suggest tighter bounds from distribution (typically top-quartile trials)."""
    if not dist:
        return None
    lo = float(dist["p25"])
    hi = float(dist["p75"])
    span = max(hi - lo, 1e-9)
    pad = span * padding_frac
    lo = lo - pad
    hi = hi + pad
    if step and step > 0:
        lo = np.floor(lo / step) * step
        hi = np.ceil(hi / step) * step
    return {"low": round(float(lo), 4), "high": round(float(hi), 4)}


def flatten_threshold_trials(folds: list[dict]) -> list[dict]:
    flat: list[dict] = []
    for fold in folds:
        th = fold.get("threshold_optimization") or {}
        for trial in th.get("trial_results") or th.get("grid_results") or []:
            flat.append(
                {
                    "optimizer": th.get("optimizer", "unknown"),
                    "fold": fold.get("fold"),
                    "train_start": fold.get("train_start"),
                    "train_end": fold.get("train_end"),
                    "test_start": fold.get("test_start"),
                    "test_end": fold.get("test_end"),
                    **trial,
                }
            )
    return flat


def flatten_ml_trials(folds: list[dict]) -> list[dict]:
    flat: list[dict] = []
    for fold in folds:
        mo = fold.get("model_optimization") or {}
        for trial in mo.get("grid_results") or mo.get("trial_results") or []:
            params = trial.get("params") or {}
            flat.append(
                {
                    "optimizer": mo.get("optimizer", "grid"),
                    "model_name": trial.get("model_name") or mo.get("model_name"),
                    "fold": fold.get("fold"),
                    "train_start": fold.get("train_start"),
                    "train_end": fold.get("train_end"),
                    "test_start": fold.get("test_start"),
                    "test_end": fold.get("test_end"),
                    "composite": trial.get("composite"),
                    "auc": trial.get("auc"),
                    "pr_auc": trial.get("pr_auc"),
                    "log_loss": trial.get("log_loss"),
                    "brier": trial.get("brier"),
                    "n_folds": trial.get("n_folds"),
                    **{f"param_{k}": v for k, v in params.items()},
                }
            )
    return flat


def summarize_fold_optimizations(
    folds: list[dict],
    *,
    top_frac: float = 0.25,
) -> dict[str, Any]:
    """Cross-fold stats + suggested narrowed ranges from strong trials."""
    policy_keys = (
        "buy_threshold",
        "min_expected_edge_bps",
        "impulse_min",
        "hold_threshold",
        "gain",
        "w_ml",
        "stop_loss_bps",
    )
    all_policy = flatten_threshold_trials(folds)
    top_policy = top_fraction_rows(all_policy, score_key="objective", top_frac=top_frac)
    policy_all = summarize_numeric_params(all_policy, policy_keys)
    policy_top = summarize_numeric_params(top_policy, policy_keys)

    ml_param_keys = (
        "param_num_leaves",
        "param_max_depth",
        "param_learning_rate",
        "param_n_estimators",
        "param_min_child_samples",
    )
    all_ml = flatten_ml_trials(folds)
    top_ml = top_fraction_rows(all_ml, score_key="composite", top_frac=top_frac)
    ml_all = summarize_numeric_params(all_ml, ml_param_keys)
    ml_top = summarize_numeric_params(top_ml, ml_param_keys)

    suggested_policy: dict[str, dict[str, float] | None] = {}
    for key in policy_keys:
        step = 0.5 if key == "min_expected_edge_bps" else (0.01 if key == "impulse_min" else (0.05 if key == "w_ml" else 1.0))
        suggested_policy[key] = suggest_narrowed_range(policy_top.get(key), step=step)

    suggested_ml: dict[str, dict[str, float] | None] = {}
    for key in ml_param_keys:
        step = 0.01 if key == "param_learning_rate" else 1.0
        suggested_ml[key] = suggest_narrowed_range(ml_top.get(key), step=step)

    search_spaces = [
        (f.get("threshold_optimization") or {}).get("search_space")
        for f in folds
        if (f.get("threshold_optimization") or {}).get("search_space")
    ]

    return {
        "n_folds": len(folds),
        "n_folds_with_threshold_opt": sum(1 for f in folds if f.get("threshold_optimization")),
        "n_folds_with_model_opt": sum(1 for f in folds if f.get("model_optimization")),
        "n_threshold_trials_total": len(all_policy),
        "n_ml_trials_total": len(all_ml),
        "top_fraction": top_frac,
        "threshold": {
            "all_trials": policy_all,
            "top_trials": policy_top,
            "suggested_ranges": suggested_policy,
            "last_search_space": search_spaces[-1] if search_spaces else None,
        },
        "ml": {
            "all_trials": ml_all,
            "top_trials": ml_top,
            "suggested_ranges": suggested_ml,
        },
    }
