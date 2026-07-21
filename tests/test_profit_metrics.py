"""Tests for profitability-first model metrics."""

from __future__ import annotations

import numpy as np

from models.profit_metrics import (
    optuna_reject_score,
    profitability_score,
    rejects_gross_below_rt,
    top_decile_net_bps,
    top_decile_stats,
)
from models.model_selection import compute_fold_metrics, entry_model_composite, resolve_optuna_objective


def test_top_decile_net_bps_positive_when_top_moves_cover_costs():
    n = 500
    proba = np.linspace(0.1, 0.9, n)
    fwd = np.where(proba > np.quantile(proba, 0.9), 0.005, -0.001)
    net = top_decile_net_bps(proba, fwd, commission_bps=1.1)
    assert net > 0.0


def test_top_decile_stats_exposes_gross_and_rt():
    n = 500
    proba = np.linspace(0.1, 0.9, n)
    fwd = np.where(proba > np.quantile(proba, 0.9), 0.005, -0.001)
    st = top_decile_stats(proba, fwd, commission_bps=1.1, slippage_bps=1.5)
    assert st["top_decile_gross_bps"] > st["top_decile_net_bps"]
    assert st["top_decile_rt_bps"] > 0
    assert abs(st["top_decile_net_bps"] - (st["top_decile_gross_bps"] - st["top_decile_rt_bps"])) < 1e-6


def test_rejects_gross_below_rt(monkeypatch):
    monkeypatch.setattr("config.FUSION_OPTUNA_REJECT_GROSS_BELOW_RT", True)
    monkeypatch.setattr("config.FUSION_OPTUNA_MIN_GROSS_OVER_RT_BPS", 0.0)
    monkeypatch.setattr("config.FUSION_OPTUNA_MIN_NET_BPS", None)
    monkeypatch.setattr("config.FUSION_OPTUNA_MAX_TRAIN_OOS_GAP_BPS", None)
    assert rejects_gross_below_rt({"top_decile_gross_bps": 3.0, "top_decile_rt_bps": 5.0}) is True
    assert rejects_gross_below_rt({"top_decile_gross_bps": 8.0, "top_decile_rt_bps": 5.0}) is False


def test_rejects_optuna_trial_on_net_and_gap(monkeypatch):
    from models.profit_metrics import rejects_optuna_trial

    monkeypatch.setattr("config.FUSION_OPTUNA_REJECT_GROSS_BELOW_RT", False)
    monkeypatch.setattr("config.FUSION_OPTUNA_MIN_NET_BPS", 5.0)
    monkeypatch.setattr("config.FUSION_OPTUNA_MAX_TRAIN_OOS_GAP_BPS", 25.0)
    assert rejects_optuna_trial({"top_decile_net_bps": 2.0}) is True
    assert rejects_optuna_trial({"top_decile_net_bps": 8.0, "train_oos_gap_bps": 40.0}) is True
    assert rejects_optuna_trial({"top_decile_net_bps": 8.0, "train_oos_gap_bps": 10.0}) is False


def test_rejects_optuna_disabled_net_floor_by_default(monkeypatch):
    """Search must not hard-kill trials solely for net < live gate (+5)."""
    from models.profit_metrics import rejects_optuna_trial

    monkeypatch.setattr("config.FUSION_OPTUNA_REJECT_GROSS_BELOW_RT", True)
    monkeypatch.setattr("config.FUSION_OPTUNA_MIN_GROSS_OVER_RT_BPS", 0.0)
    monkeypatch.setattr("config.FUSION_OPTUNA_MIN_NET_BPS", None)
    monkeypatch.setattr("config.FUSION_OPTUNA_MAX_TRAIN_OOS_GAP_BPS", None)
    # Net +3 with gross clearing RT → accepted for search ranking.
    assert rejects_optuna_trial({
        "top_decile_net_bps": 3.0,
        "top_decile_gross_bps": 8.0,
        "top_decile_rt_bps": 5.0,
        "train_oos_gap_bps": 10.0,
    }) is False
    # Gross below RT → still rejected.
    assert rejects_optuna_trial({
        "top_decile_net_bps": 3.0,
        "top_decile_gross_bps": 3.0,
        "top_decile_rt_bps": 5.0,
    }) is True


def test_resolve_optuna_objective_rejects_gross_below_rt(monkeypatch):
    monkeypatch.setattr("config.FUSION_OPTUNA_REJECT_GROSS_BELOW_RT", True)
    monkeypatch.setattr("config.FUSION_OPTUNA_MIN_GROSS_OVER_RT_BPS", 0.0)
    monkeypatch.setattr("config.FUSION_OPTUNA_MIN_NET_BPS", None)
    monkeypatch.setattr("config.FUSION_OPTUNA_MAX_TRAIN_OOS_GAP_BPS", None)
    monkeypatch.setattr("config.FUSION_OPTUNA_APPLY_GAP_PENALTY", False)
    monkeypatch.setattr("config.FUSION_OPTUNA_OBJECTIVE", "profit")
    bad = {
        "top_decile_net_bps": -2.0,
        "top_decile_gross_bps": 3.0,
        "top_decile_rt_bps": 5.0,
        "accuracy": 0.7,
        "composite": 0.9,
    }
    good = {
        "top_decile_net_bps": 10.0,
        "top_decile_gross_bps": 15.0,
        "top_decile_rt_bps": 5.0,
        "accuracy": 0.52,
        "composite": 0.5,
    }
    assert resolve_optuna_objective(bad) == optuna_reject_score()
    assert resolve_optuna_objective(good) > resolve_optuna_objective(bad)


def test_resolve_optuna_applies_gap_penalty(monkeypatch):
    monkeypatch.setattr("config.FUSION_OPTUNA_REJECT_GROSS_BELOW_RT", False)
    monkeypatch.setattr("config.FUSION_OPTUNA_MIN_NET_BPS", None)
    monkeypatch.setattr("config.FUSION_OPTUNA_MAX_TRAIN_OOS_GAP_BPS", None)
    monkeypatch.setattr("config.FUSION_OPTUNA_APPLY_GAP_PENALTY", True)
    monkeypatch.setattr("config.FUSION_TRAIN_OOS_GAP_WEIGHT", 0.45)
    monkeypatch.setattr("config.FUSION_TRAIN_OOS_GAP_SCALE_BPS", 8.0)
    monkeypatch.setattr("config.FUSION_OPTUNA_OBJECTIVE", "profit")
    base = {
        "top_decile_net_bps": 10.0,
        "top_decile_gross_bps": 15.0,
        "top_decile_rt_bps": 5.0,
        "train_oos_gap_bps": 0.0,
    }
    gappy = {**base, "train_oos_gap_bps": 16.0}
    assert resolve_optuna_objective(gappy) < resolve_optuna_objective(base)


def test_top_decile_rejects_small_samples():
    n = 40
    proba = np.linspace(0.1, 0.9, n)
    fwd = np.ones(n) * 0.01
    assert top_decile_net_bps(proba, fwd, commission_bps=1.1, min_rows=200) == float("-inf")


def test_profitability_score_penalizes_negative_edge():
    assert profitability_score(0.0) == 0.5
    assert profitability_score(25.0) == 1.0
    assert profitability_score(-25.0) == 0.0
    assert profitability_score(-5.0) < profitability_score(0.0)
    assert profitability_score(5.0) > profitability_score(0.0)


def test_composite_weights_profit_over_accuracy():
    profit_row = {
        "top_decile_net_bps": 25.0,
        "accuracy": 0.52,
        "auc": 0.52,
        "pr_auc": 0.2,
        "log_loss": 0.69,
        "brier": 0.24,
        "ic_spearman": 0.0,
        "edge_corr": 0.0,
        "calibration_mae": 0.2,
        "precision": 0.1,
        "recall": 0.1,
        "fold_stability": 0.0,
    }
    accurate_row = {
        "top_decile_net_bps": -5.0,
        "accuracy": 0.72,
        "auc": 0.75,
        "pr_auc": 0.4,
        "log_loss": 0.55,
        "brier": 0.15,
        "ic_spearman": 0.1,
        "edge_corr": 0.1,
        "calibration_mae": 0.1,
        "precision": 0.55,
        "recall": 0.50,
        "fold_stability": 0.0,
    }
    assert entry_model_composite(profit_row) > entry_model_composite(accurate_row)


def test_compute_fold_metrics_per_ticker_mean_profit():
    n = 120
    tickers = np.array(["BTC"] * 60 + ["ETH"] * 60)
    y = np.array([0, 1] * (n // 2))
    p = np.linspace(0.2, 0.95, n)
    fwd = np.where(p > np.quantile(p, 0.85), 0.006, -0.0005)
    m = compute_fold_metrics(y, p, fwd, commission_bps=1.1, tickers=tickers)
    assert "mean_ticker_top_decile_net_bps" in m
    assert m["top_decile_net_bps"] == m["mean_ticker_top_decile_net_bps"]
    assert "top_decile_gross_bps" in m
    assert "top_decile_rt_bps" in m
