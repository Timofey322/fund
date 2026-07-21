"""LightGBM entry classifier for fusion walk-forward OOS."""

from __future__ import annotations

from typing import Any

import config as _cfg

# Per-fold winner (13/16 folds): shallow, regularized — used as fallback + Optuna anchor.
DEFAULT_LIGHTGBM_PARAMS: dict[str, Any] = {
    "num_leaves": 15,
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 120,
    "min_child_samples": 96,
    "lambda_l1": 0.15,
    "lambda_l2": 1.5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.75,
    "bagging_freq": 1,
    "min_split_gain": 0.0,
    "max_bin": 255,
    "path_smooth": 0.0,
}

FUSION_ENTRY_MODEL_NAME = "lightgbm"


def lightgbm_search_space_spec() -> dict:
    """Optuna search bounds centered on DEFAULT_LIGHTGBM_PARAMS."""
    lo_leaves, hi_leaves = getattr(_cfg, "FUSION_MODEL_NUM_LEAVES_RANGE", (7, 31))
    lo_depth, hi_depth = getattr(_cfg, "FUSION_MODEL_MAX_DEPTH_RANGE", (3, 7))
    lo_lr, hi_lr = getattr(_cfg, "FUSION_MODEL_LEARNING_RATE_RANGE", (0.03, 0.12))
    lo_est, hi_est = getattr(_cfg, "FUSION_MODEL_N_ESTIMATORS_RANGE", (100, 280))
    lo_child, hi_child = getattr(_cfg, "FUSION_MODEL_MIN_CHILD_SAMPLES_RANGE", (40, 150))
    lo_l1, hi_l1 = getattr(_cfg, "FUSION_MODEL_LAMBDA_L1_RANGE", (0.0, 0.5))
    lo_l2, hi_l2 = getattr(_cfg, "FUSION_MODEL_LAMBDA_L2_RANGE", (0.5, 3.0))
    lo_ff, hi_ff = getattr(_cfg, "FUSION_MODEL_FEATURE_FRACTION_RANGE", (0.55, 0.95))
    lo_bf, hi_bf = getattr(_cfg, "FUSION_MODEL_BAGGING_FRACTION_RANGE", (0.55, 0.90))
    lo_bfreq, hi_bfreq = getattr(_cfg, "FUSION_MODEL_BAGGING_FREQ_RANGE", (1, 7))
    lo_msg, hi_msg = getattr(_cfg, "FUSION_MODEL_MIN_SPLIT_GAIN_RANGE", (0.0, 1.0))
    lo_bin, hi_bin = getattr(_cfg, "FUSION_MODEL_MAX_BIN_RANGE", (127, 511))
    lo_ps, hi_ps = getattr(_cfg, "FUSION_MODEL_PATH_SMOOTH_RANGE", (0.0, 1.0))
    return {
        "optimizer": "optuna",
        "sampler": "TPESampler",
        "model_name": FUSION_ENTRY_MODEL_NAME,
        "params": {
            "num_leaves": {"type": "int", "low": int(lo_leaves), "high": int(hi_leaves)},
            "max_depth": {"type": "int", "low": int(lo_depth), "high": int(hi_depth)},
            "learning_rate": {"type": "float", "low": float(lo_lr), "high": float(hi_lr), "log": True},
            "n_estimators": {"type": "int", "low": int(lo_est), "high": int(hi_est)},
            "min_child_samples": {"type": "int", "low": int(lo_child), "high": int(hi_child)},
            "lambda_l1": {"type": "float", "low": float(lo_l1), "high": float(hi_l1)},
            "lambda_l2": {"type": "float", "low": float(lo_l2), "high": float(hi_l2)},
            "feature_fraction": {"type": "float", "low": float(lo_ff), "high": float(hi_ff)},
            "bagging_fraction": {"type": "float", "low": float(lo_bf), "high": float(hi_bf)},
            "bagging_freq": {"type": "int", "low": int(lo_bfreq), "high": int(hi_bfreq)},
            "min_split_gain": {"type": "float", "low": float(lo_msg), "high": float(hi_msg)},
            "max_bin": {"type": "int", "low": int(lo_bin), "high": int(hi_bin)},
            "path_smooth": {"type": "float", "low": float(lo_ps), "high": float(hi_ps)},
        },
    }


def params_from_trial(trial) -> dict[str, Any]:
    spec = lightgbm_search_space_spec()["params"]
    return {
        "num_leaves": trial.suggest_int(
            "num_leaves", spec["num_leaves"]["low"], spec["num_leaves"]["high"]
        ),
        "max_depth": trial.suggest_int(
            "max_depth", spec["max_depth"]["low"], spec["max_depth"]["high"]
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            spec["learning_rate"]["low"],
            spec["learning_rate"]["high"],
            log=spec["learning_rate"].get("log", True),
        ),
        "n_estimators": trial.suggest_int(
            "n_estimators", spec["n_estimators"]["low"], spec["n_estimators"]["high"]
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            spec["min_child_samples"]["low"],
            spec["min_child_samples"]["high"],
        ),
        "lambda_l1": trial.suggest_float(
            "lambda_l1", spec["lambda_l1"]["low"], spec["lambda_l1"]["high"],
        ),
        "lambda_l2": trial.suggest_float(
            "lambda_l2", spec["lambda_l2"]["low"], spec["lambda_l2"]["high"],
        ),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", spec["feature_fraction"]["low"], spec["feature_fraction"]["high"],
        ),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", spec["bagging_fraction"]["low"], spec["bagging_fraction"]["high"],
        ),
        "bagging_freq": trial.suggest_int(
            "bagging_freq", spec["bagging_freq"]["low"], spec["bagging_freq"]["high"],
        ),
        "min_split_gain": trial.suggest_float(
            "min_split_gain", spec["min_split_gain"]["low"], spec["min_split_gain"]["high"],
        ),
        "max_bin": trial.suggest_int(
            "max_bin", spec["max_bin"]["low"], spec["max_bin"]["high"],
        ),
        "path_smooth": trial.suggest_float(
            "path_smooth", spec["path_smooth"]["low"], spec["path_smooth"]["high"],
        ),
    }


def params_from_values(values: dict[str, Any]) -> dict[str, Any]:
    return {k: values[k] for k in DEFAULT_LIGHTGBM_PARAMS}


def make_entry_classifier(model_name: str, params: dict[str, Any]):
    """Factory for binary LightGBM entry model (sklearn-like API)."""
    if model_name.lower() != FUSION_ENTRY_MODEL_NAME:
        raise ValueError(f"Only lightgbm is supported, got {model_name!r}")
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is not installed — pip install lightgbm") from exc
    from common.parallel import lightgbm_thread_count

    return lgb.LGBMClassifier(
        objective="binary",
        verbosity=-1,
        n_jobs=lightgbm_thread_count(),
        random_state=42,
        **params,
    )


def make_direction_classifier(
    model_name: str,
    params: dict[str, Any],
    *,
    num_classes: int = 3,
):
    """Factory for 3-class direction model: flat / long / short."""
    if model_name.lower() != FUSION_ENTRY_MODEL_NAME:
        raise ValueError(f"Only lightgbm is supported, got {model_name!r}")
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is not installed — pip install lightgbm") from exc
    from common.parallel import lightgbm_thread_count

    clean = {k: v for k, v in params.items() if k != "scale_pos_weight"}
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=int(num_classes),
        verbosity=-1,
        n_jobs=lightgbm_thread_count(),
        random_state=42,
        **clean,
    )
