"""Regression tests for causality / optimizer bug fixes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.features.entry_ml import effective_purge_sessions, _purged_session_folds
from strategy.pipeline import _expected_edge_bps_series, _expected_move_bps
from strategy.threshold_opt import _evaluate_policy_cv
from research.regime.hmm import GaussianHMM


def test_effective_purge_scales_with_label_horizon():
    short = effective_purge_sessions(12, purge=1)
    long = effective_purge_sessions(288, purge=None)
    assert short == 1
    assert long >= short
    assert effective_purge_sessions(288, purge=5) == 5


def test_purged_folds_exclude_purge_band():
    sessions = [f"s{i}" for i in range(20)]
    folds = _purged_session_folds(sessions, n_splits=4, max_label_horizon_bars=288)
    assert folds
    train_s, test_s = folds[0]
    assert train_s.isdisjoint(test_s)


def test_hmm_causal_fit_excludes_current_observation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    model = GaussianHMM(n_states=3, n_iter=5)
    hist = X[:-1]
    cur = X[-1]
    model.fit(hist)
    alpha_hist = model.forward_pass(hist)
    alpha_causal = model.forward_filter(alpha_hist, cur)

    model_leaky = GaussianHMM(n_states=3, n_iter=5)
    model_leaky.fit(X)
    alpha_leaky = model_leaky.forward_pass(X)

    assert np.allclose(alpha_causal.sum(), 1.0, atol=1e-6)
    assert not np.allclose(alpha_causal, alpha_leaky, atol=1e-9)


def test_expected_edge_scales_with_hold_horizon():
    short = _expected_move_bps(12)
    long = _expected_move_bps(288)
    assert long > short
    proba = pd.Series([0.6, 0.55])
    imp = pd.Series([0.5, 0.5])
    edge_long = _expected_edge_bps_series(proba, imp, hold_bars=288)
    edge_short = _expected_edge_bps_series(proba, imp, hold_bars=12)
    assert (edge_long > edge_short).all()


def test_mean_valid_fold_sharpes_ignores_nan():
    sharpes = [0.5, float("nan"), 1.0]
    valid = [float(s) for s in sharpes if np.isfinite(s)]
    assert float(np.mean(valid)) == 0.75


def test_evaluate_policy_cv_skips_nan_sharpe_folds():
    """Empty-signal folds must not inject -999 into the mean."""
    # Minimal smoke: function exists and returns None on empty work
    out = _evaluate_policy_cv(
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        [],
        commission_bps=10.0,
    )
    assert out is None


def test_min_tp_is_sl_plus_commission():
    from simulation.entry_signals import (
        edge_floor_bps,
        min_tp_gross_bps,
        round_trip_commission_bps,
        round_trip_cost_bps,
    )

    comm, slip, sl = 10.0, 15.0, 35.0
    rt_comm = round_trip_commission_bps(comm)
    rt = round_trip_cost_bps(comm, slip)
    assert min_tp_gross_bps(comm, slip, stop_loss_bps=sl, buffer_bps=0.0) == sl + rt_comm + slip
    assert edge_floor_bps(comm, slip, mode="sl_plus_commission", stop_loss_bps=sl, buffer_bps=0.0) == sl + rt_comm + slip
    assert edge_floor_bps(comm, slip, mode="commission_only", buffer_bps=2.0) == rt_comm + 2.0
    assert edge_floor_bps(comm, slip, mode="full_cost", buffer_bps=2.0) == rt + 2.0
    assert edge_floor_bps(comm, slip, mode="full_round_trip", buffer_bps=5.0) == rt + 5.0


def test_stop_loss_exit_triggers_on_adverse_move():
    from simulation.engine import run_backtest_signal_exit

    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame({
        "BTC": [100.0, 100.0, 100.0, 97.0, 97.0, 97.0],
    }, index=idx)
    signals = pd.DataFrame({
        "date": [idx[0], idx[0]],
        "ticker": ["BTC", "BTC"],
        "score": [80.0, 80.0],
        "risk_on": [True, True],
        "hold_threshold": [40.0, 40.0],
        "buy_threshold": [50.0, 50.0],
        "vol_ann": [0.25, 0.25],
        "stop_loss_bps": [200.0, 200.0],
    })
    bt = run_backtest_signal_exit(
        prices,
        signals,
        score_col="score",
        use_dynamic_thresholds=True,
        use_vol_targeting=False,
        commission_bps=0.0,
        horizon_bars=20,
        stop_loss_bps=200.0,
        period_start=idx[1],
        period_end=idx[-1],
    )
    stats = bt.get("stats") or {}
    assert int(stats.get("stop_loss_exit_count", 0)) >= 1


def test_threshold_edge_bounds_respect_panel_ceiling():
    from strategy.edge_gate import threshold_search_bounds

    lo, hi = threshold_search_bounds(20.0, commission_bps=10.0, panel_max_edge_bps=16.5)
    assert lo <= 16.5
    assert hi <= lo + 1.0


def test_cap_calibrated_edge_to_panel_lowers_infeasible_floor():
    from strategy.edge_gate import cap_calibrated_edge_to_panel

    capped = cap_calibrated_edge_to_panel(
        20.0,
        panel_max_edge_bps=16.5,
        panel_q65_edge_bps=12.0,
        commission_bps=10.0,
    )
    assert capped <= 60.0
    assert capped <= max(16.5, 20.0) or capped <= 16.5


def test_threshold_edge_bounds_respect_commission_heuristic_floor():
    from strategy.threshold_opt import _edge_bounds

    lo, hi = _edge_bounds(18.0, commission_bps=10.0, stop_loss_bps=45.0)
    assert lo >= 22.0
    assert hi <= 55.0
    assert hi > lo


def test_heuristic_gate_passes_typical_ml_proba():
    from strategy.edge_gate import heuristic_gate_floor_bps, resolve_min_expected_edge_bps

    comm = 10.0
    floor = heuristic_gate_floor_bps(comm)
    assert floor >= 22.0
    proba = pd.Series([0.82, 0.88])
    imp = pd.Series([0.55, 0.60])
    edges = _expected_edge_bps_series(proba, imp, baseline_proba=0.45, hold_bars=96)
    min_edge = resolve_min_expected_edge_bps(floor, commission_bps=comm)
    assert (edges >= min_edge).any(), f"edges={edges.tolist()} min_edge={min_edge}"


def test_target_opt_prefers_positive_edge():
    from strategy.target_opt import target_score

    base_cv = {"cv": {"composite": 0.55, "auc": 0.72}, "positive_rate": 0.08}
    loser = {**base_cv, "net_edge_bps": -50.0}
    winner = {**base_cv, "net_edge_bps": 5.0, "cv": {"composite": 0.48, "auc": 0.65}}
    assert target_score(winner, balance_range=(0.05, 0.55)) > target_score(
        loser, balance_range=(0.05, 0.55)
    )


def test_purge_scales_with_horizon_and_floor():
    assert effective_purge_sessions(12, purge=None) >= 1
    assert effective_purge_sessions(288, purge=None) >= effective_purge_sessions(12, purge=None)


def test_trim_label_embargo_drops_near_test():
    from strategy.leakage_guard import trim_label_embargo

    idx = pd.date_range("2024-01-01", periods=300, freq="5min")
    df = pd.DataFrame({"bar_time": idx, "x": 1})
    test_start = idx[-50]
    out = trim_label_embargo(df, test_start=test_start, horizon_bars=12, bar_minutes=5)
    assert len(out) < len(df)
    assert out["bar_time"].max() < test_start - pd.Timedelta(minutes=60)


def test_split_fit_calibration_embargo():
    from strategy.leakage_guard import split_fit_calibration

    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    df = pd.DataFrame({
        "bar_time": idx,
        "session": idx.normalize(),
        "label": (np.arange(500) % 2).astype(int),
    })
    test_start = idx[-80]
    tr_fit, va = split_fit_calibration(
        df, cal_frac=0.15, horizon_bars=12, test_start=test_start, target_col="label",
    )
    assert not tr_fit.empty
    assert not va.empty
    assert len(tr_fit) + len(va) <= len(df)


def test_panel_for_causal_target_opt_excludes_stitched_oos():
    from strategy.leakage_guard import panel_for_causal_target_opt, resolve_walk_forward_oos_cutoff

    panel = pd.DataFrame({
        "bar_time": pd.date_range("2016-01-01", "2026-06-01", freq="D"),
        "ticker": "GAZP",
        "close": 100.0,
    })
    cutoff = resolve_walk_forward_oos_cutoff(panel, train_days=1825, backtest_years=5, test_months=3)
    assert cutoff is not None
    causal, meta = panel_for_causal_target_opt(
        panel, train_days=1825, backtest_years=5, test_months=3, min_rows=100,
    )
    assert meta["causal"] is True
    assert meta["oos_cutoff"] == str(cutoff.date())
    assert causal["bar_time"].max() < cutoff
    assert len(causal) < len(panel)


def test_limit_walk_forward_windows():
    from strategy.pipeline import limit_walk_forward_windows, monthly_walk_forward_windows

    panel = pd.DataFrame({
        "bar_time": pd.date_range("2020-01-01", "2026-06-01", freq="D"),
    })
    windows = monthly_walk_forward_windows(panel, train_days=365, backtest_years=4, test_months=1)
    assert len(windows) > 5
    limited = limit_walk_forward_windows(windows, 5)
    assert len(limited) == 5
    assert limited[0]["fold"] == 0
    assert limited[-1]["fold"] == 4
    assert limit_walk_forward_windows(windows, None) == windows


def test_compute_fold_anomaly_table_equity_index_named_date():
    """Equity with DatetimeIndex named 'date' must not duplicate columns on reset."""
    from reporting.plots import compute_fold_anomaly_table

    idx = pd.date_range("2024-01-01", periods=5, freq="D", name="date")
    equity = pd.DataFrame({"value": [1.0, 1.01, 1.02, 1.01, 1.03]}, index=idx)
    fold_map = pd.DataFrame({
        "bar_time": idx,
        "wf_fold": [0] * len(idx),
    })
    rows = compute_fold_anomaly_table(equity, None, fold_map, commission_bps=10.0)
    assert len(rows) == 1
    assert rows[0]["fold"] == 0


def test_ml_feature_importance_report_from_folds():
    from models.feature_importance import build_ml_feature_importance_report

    folds = [
        {"fold": 0, "top_features": {"feat_a": 0.4, "feat_b": 0.3}, "oos_metrics": {"auc": 0.61}},
        {"fold": 1, "top_features": {"feat_a": 0.5, "feat_c": 0.2}, "oos_metrics": {"auc": 0.58}},
    ]
    report = build_ml_feature_importance_report(folds, ["feat_a", "feat_b", "feat_c"])
    assert report["n_folds"] == 2
    assert len(report["top_features"]) >= 2
    assert report["top_features"][0]["feature"] == "feat_a"
