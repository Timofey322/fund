"""Thread-safe Optuna progress logging for parallel fold optimization."""

from __future__ import annotations

import threading
from typing import Callable


def _progress_bar(completed: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    filled = int(round(width * completed / total))
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total}"


def optuna_progress_callback(
    *,
    fold: int | str,
    label: str,
    n_trials: int,
    progress_every: int = 5,
    metric_name: str = "composite",
    value_fmt: str = ".4f",
    secondary_metric: str | None = "top_decile_net_bps",
    secondary_fmt: str = "+.1f",
) -> Callable:
    """Log completed trials with progress bar and running best (parallel-safe).

    With ``n_jobs > 1`` trials finish out of order; lines show a bar, current
    score, best score, and optional secondary metric (e.g. profit net bps).
  """
    lock = threading.Lock()
    completed = 0
    last_best = float("-inf")

    def _secondary(trial) -> str:
        if not secondary_metric:
            return ""
        val = (trial.user_attrs or {}).get(secondary_metric)
        if val is None and trial.number > 0:
            return ""
        try:
            return f" {secondary_metric}={float(val):{secondary_fmt}}"
        except (TypeError, ValueError):
            return ""

    def _on_trial(study, trial) -> None:
        nonlocal completed, last_best
        with lock:
            completed += 1
            if trial.value is None:
                return
            cur = float(trial.value)
            best_val = float(study.best_value) if study.best_trial is not None else cur
            improved = best_val > last_best + 1e-12
            milestone = (
                completed == 1
                or completed % max(1, progress_every) == 0
                or completed == n_trials
            )
            if not (milestone or improved):
                return
            if improved:
                last_best = best_val
            star = " *" if improved else ""
            bar = _progress_bar(completed, n_trials)
            sec = _secondary(trial)
            best_sec = ""
            if secondary_metric and study.best_trial is not None:
                bv = (study.best_trial.user_attrs or {}).get(secondary_metric)
                if bv is not None:
                    try:
                        best_sec = f" best_{secondary_metric}={float(bv):{secondary_fmt}}"
                    except (TypeError, ValueError):
                        pass
            print(
                f"        fold {fold} {label} {bar} "
                f"trial#{trial.number + 1} {metric_name}={cur:{value_fmt}} "
                f"best={best_val:{value_fmt}}{best_sec}{sec}{star}",
                flush=True,
            )

    return _on_trial
