"""Threshold optimization gate mode and fold anomaly flags."""

from __future__ import annotations

import pandas as pd

from strategy.threshold_opt import (
    _build_policy_params,
    _finalize_threshold_winner,
    build_calibrated_trading_policy,
)


def test_relaxed_gate_allows_lower_edge_than_production():
    prod = _build_policy_params(
        buy_threshold=40,
        min_expected_edge_bps=5.0,
        impulse_min=0.02,
        hold_threshold=30,
        gain=80,
        calibrated_edge=8.0,
        commission_bps=1.1,
        gate_mode="full_round_trip",
    )
    relaxed = _build_policy_params(
        buy_threshold=40,
        min_expected_edge_bps=5.0,
        impulse_min=0.02,
        hold_threshold=30,
        gain=80,
        calibrated_edge=8.0,
        commission_bps=1.1,
        gate_mode="commission_only",
    )
    assert relaxed["min_expected_edge_bps"] < prod["min_expected_edge_bps"]


def test_finalize_marks_zero_signals_as_anomaly_without_disabling_trading():
    out = _finalize_threshold_winner(
        {"objective": 1.0, "signal_rows": 0, "constraints_ok": False},
        best=None,
        calibrated_edge=10.0,
    )
    assert out["trade_anomaly"] is True
    assert out["cv_zero_signals"] is True
    assert out["disable_trading"] is False


def test_finalize_marks_negative_objective_as_anomaly():
    out = _finalize_threshold_winner(
        {"objective": -5.0, "signal_rows": 12, "constraints_ok": True},
        best={"objective": -5.0},
        calibrated_edge=10.0,
    )
    assert out["trade_anomaly"] is True
    assert out["disable_trading"] is False


def test_calibrated_policy_skips_optuna(monkeypatch):
    import strategy.threshold_opt as th

    def _fake_signal_count(train, prices, params, **kwargs):
        return 42

    monkeypatch.setattr(th, "_quick_signal_count", _fake_signal_count)
    monkeypatch.setattr(th, "_calibrate_edge_on_train", lambda train, **kw: 8.5)

    train = pd.DataFrame({"ml_proba": [0.1, 0.9, 0.5], "session": ["a", "a", "b"]})
    out = build_calibrated_trading_policy(
        train,
        pd.DataFrame(),
        commission_bps=1.1,
        fold_meta={"fold": 0},
    )
    assert out["optimizer"] == "calibrated"
    assert out["n_trials"] == 0
    assert out["best_params"]["min_expected_edge_bps"] == 8.5
    assert out["cv"]["signal_rows"] == 42


def test_evaluate_policy_cv_fails_constraints_on_negative_trade_net(monkeypatch):
    """Positive-trade-net requirement marks constraints_ok=False."""
    import numpy as np
    import strategy.threshold_opt as th

    monkeypatch.setattr("config.FUSION_THRESHOLD_OPT_REQUIRE_POSITIVE_TRADE_NET", True)
    monkeypatch.setattr(th, "_fusion_hold_default", lambda: 24)
    monkeypatch.setattr(th, "_passes_threshold_opt_constraints", lambda *a, **k: True)
    monkeypatch.setattr(th, "_threshold_opt_constraint_penalty", lambda *a, **k: 0.0)
    monkeypatch.setattr(th, "_threshold_opt_backtest_kwargs", lambda: {})
    monkeypatch.setattr(th, "_fusion_signal_frame", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(
        "simulation.engine.run_backtest_signal_exit",
        lambda *a, **k: {
            "stats": {"total_return_pct": 1.0, "sharpe": 0.5, "sharpe_bar_annualized": 0.5},
            "holdings": [],
        },
    )
    monkeypatch.setattr(th, "_trade_constraint_metrics", lambda *a, **k: {
        "signal_rows": 100, "active_rebalances": 5, "avg_exposure_pct": 10.0,
    })
    monkeypatch.setattr(
        "simulation.entry_signals.active_entry_signals",
        lambda sig: sig,
    )
    monkeypatch.setattr(
        "simulation.entry_signals.deoverlap_signals",
        lambda sig, prices, h: sig,
    )
    monkeypatch.setattr(
        "simulation.entry_signals.trade_returns_from_signals",
        lambda *a, **k: np.array([-0.001, -0.002]),
    )

    train = pd.DataFrame({
        "session": ["s1", "s1", "s2", "s2"],
        "bar_time": pd.date_range("2024-01-01", periods=4, freq="h"),
    })
    folds = [({"s1"}, {"s2"})]
    row = th._evaluate_policy_cv(train, pd.DataFrame(), {}, folds, commission_bps=1.1)
    assert row is not None
    assert row["mean_trade_net_bps"] < 0
    assert row["constraints_ok"] is False
