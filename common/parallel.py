"""Shared worker-count helpers for process pools."""

from __future__ import annotations

import os

import config as _cfg


def resolve_worker_count(
    config_attr: str,
    *,
    cap: int | None = None,
    env_var: str | None = None,
    default: int | None = None,
) -> int:
    """Resolve pool size from env var, config attribute, or CPU count."""
    if env_var:
        raw = os.environ.get(env_var)
        if raw is not None and str(raw).strip():
            n = int(raw)
        else:
            n = _from_config(config_attr, default=default)
    else:
        n = _from_config(config_attr, default=default)
    if cap is not None:
        n = min(n, int(cap))
    return max(1, n)


def _from_config(config_attr: str, *, default: int | None) -> int:
    val = getattr(_cfg, config_attr, None)
    if val is not None:
        return max(1, int(val))
    if default is not None:
        return max(1, int(default))
    return max(1, (os.cpu_count() or 4) - 1)


def optuna_parallel_jobs() -> int:
    """Parallel Optuna trials per fold (None => all CPUs minus one)."""
    raw = os.environ.get("FUSION_OPTUNA_N_JOBS")
    if raw is not None and str(raw).strip():
        return max(1, int(raw))
    val = getattr(_cfg, "FUSION_OPTUNA_N_JOBS", None)
    if val is not None:
        return max(1, int(val))
    return max(1, (os.cpu_count() or 4) - 1)


def optuna_parallel_jobs_threshold() -> int:
    """Parallel trials for threshold CV (backtest-heavy; lower than model opt)."""
    raw = os.environ.get("FUSION_THRESHOLD_OPTUNA_N_JOBS")
    if raw is not None and str(raw).strip():
        return max(1, int(raw))
    val = getattr(_cfg, "FUSION_THRESHOLD_OPTUNA_N_JOBS", None)
    if val is not None:
        return max(1, int(val))
    cpus = os.cpu_count() or 4
    return max(1, min(8, cpus - 1))


def lightgbm_thread_count() -> int:
    """Threads per LightGBM fit (-1 = all cores)."""
    return int(getattr(_cfg, "FUSION_LGBM_N_JOBS", -1))
