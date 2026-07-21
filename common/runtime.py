"""Pipeline runtime: warning filters and fold-stage parallelism."""

from __future__ import annotations

import os
import warnings


def configure_pipeline_runtime() -> None:
    """Suppress noisy third-party warnings that pollute PowerShell stderr."""
    warnings.filterwarnings("ignore", message=".*multivariate.*", category=UserWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandas\..*")
    try:
        from optuna._experimental import ExperimentalWarning

        warnings.filterwarnings("ignore", category=ExperimentalWarning)
    except ImportError:
        pass


def fold_parallelism_summary() -> dict[str, int]:
    """Resolved worker counts for fold-stage Optuna + LightGBM."""
    from common.parallel import lightgbm_thread_count, optuna_parallel_jobs

    return {
        "optuna_n_jobs": optuna_parallel_jobs(),
        "lightgbm_n_jobs": lightgbm_thread_count(),
        "cpu_count": os.cpu_count() or 1,
    }


def log_fold_parallelism() -> None:
    info = fold_parallelism_summary()
    print(
        f"    fold opt parallelism: optuna_n_jobs={info['optuna_n_jobs']} "
        f"lightgbm_n_jobs={info['lightgbm_n_jobs']} "
        f"(cpus={info['cpu_count']})",
        flush=True,
    )
