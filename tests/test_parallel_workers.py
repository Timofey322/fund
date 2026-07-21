"""Tests for parallel worker resolution."""

from __future__ import annotations

import config as cfg
from common.parallel import optuna_parallel_jobs, resolve_worker_count


def test_resolve_worker_count_uses_config(monkeypatch):
    monkeypatch.delenv("FUSION_TARGET_OPT_WORKERS", raising=False)
    old = getattr(cfg, "FUSION_TARGET_OPT_WORKERS", None)
    cfg.FUSION_TARGET_OPT_WORKERS = 3
    try:
        assert resolve_worker_count("FUSION_TARGET_OPT_WORKERS", cap=10) == 3
        assert resolve_worker_count("FUSION_TARGET_OPT_WORKERS", cap=2) == 2
    finally:
        cfg.FUSION_TARGET_OPT_WORKERS = old


def test_resolve_worker_count_env_overrides_config(monkeypatch):
    monkeypatch.setenv("FUSION_TARGET_OPT_WORKERS", "5")
    old = getattr(cfg, "FUSION_TARGET_OPT_WORKERS", None)
    cfg.FUSION_TARGET_OPT_WORKERS = 2
    try:
        assert resolve_worker_count("FUSION_TARGET_OPT_WORKERS", env_var="FUSION_TARGET_OPT_WORKERS") == 5
    finally:
        cfg.FUSION_TARGET_OPT_WORKERS = old


def test_optuna_parallel_jobs_at_least_one():
    assert optuna_parallel_jobs() >= 1


def test_optuna_parallel_jobs_defaults_to_cpu_minus_one(monkeypatch):
    monkeypatch.delenv("FUSION_OPTUNA_N_JOBS", raising=False)
    old = getattr(cfg, "FUSION_OPTUNA_N_JOBS", None)
    cfg.FUSION_OPTUNA_N_JOBS = None
    try:
        n = optuna_parallel_jobs()
        assert n >= 1
        assert n == max(1, (__import__("os").cpu_count() or 4) - 1)
    finally:
        cfg.FUSION_OPTUNA_N_JOBS = old
