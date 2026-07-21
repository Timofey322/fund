"""Tests for decile audit, edge alignment, and objective kill-switch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.diagnostics.decile_audit import decile_monotonicity_check
from simulation.entry_signals import deoverlap_signals, deoverlap_signals_by_horizon, edge_floor_bps
from strategy.edge_gate import heuristic_gate_floor_bps
from strategy.objective import fusion_cv_objective, threshold_no_trade_objective
from strategy.target_opt import per_instrument_specs, ticker_min_expected_edge_bps


def test_decile_monotonicity_tradeable_when_top_decile_strong():
    n = 2000
    proba = np.linspace(0.01, 0.99, n)
    ret = proba * 0.01
    df = pd.DataFrame({"ml_proba": proba, "fwd_ret_entry": ret})
    out = decile_monotonicity_check(df, min_top_decile_net_bps=5.0)
    assert out["top_decile_net_bps"] is not None
    assert out["monotonic"] is True
    assert out["tradeable"] is True


def test_decile_monotonicity_not_tradeable_on_flat_signal():
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame({
        "ml_proba": rng.uniform(0, 1, n),
        "fwd_ret_entry": rng.normal(0, 0.0005, n),
    })
    out = decile_monotonicity_check(df, min_top_decile_net_bps=50.0)
    assert out["tradeable"] is False
    assert out["reasons"]


def test_gate_floor_at_least_round_trip_commission():
    floor = heuristic_gate_floor_bps(10.0, 15.0)
    assert floor >= 20.0


def test_edge_floor_full_round_trip_mode():
    floor = edge_floor_bps(10.0, 15.0, mode="full_round_trip", buffer_bps=5.0)
    assert floor >= 35.0


def test_threshold_no_trade_floor_from_config():
    assert threshold_no_trade_objective() == -2.0


def test_fusion_cv_objective_penalizes_negative_expectancy():
    good = fusion_cv_objective(
        sharpe=0.5, mean_return_pct=2.0, active_rebalances=10, mean_trade_net_bps=25.0,
    )
    bad = fusion_cv_objective(
        sharpe=0.5, mean_return_pct=2.0, active_rebalances=10, mean_trade_net_bps=-30.0,
    )
    assert good > bad


def test_deoverlap_respects_per_row_exit_horizon():
    idx = pd.date_range("2024-01-01", periods=20, freq="5min")
    prices = pd.DataFrame({"BTC": np.linspace(100, 101, 20)}, index=idx)
    signals = pd.DataFrame({
        "date": [idx[0], idx[2], idx[4]],
        "ticker": ["BTC", "BTC", "BTC"],
        "exit_horizon": [6, 6, 6],
        "fwd_ret_entry": [0.01, 0.01, 0.01],
    })
    out = deoverlap_signals_by_horizon(signals, prices)
    assert len(out) == 1


def test_deoverlap_signals_uses_exit_horizon_column():
    idx = pd.date_range("2024-01-01", periods=20, freq="5min")
    prices = pd.DataFrame({"BTC": np.linspace(100, 101, 20)}, index=idx)
    signals = pd.DataFrame({
        "date": [idx[0], idx[2]],
        "ticker": ["BTC", "BTC"],
        "exit_horizon": [8, 8],
        "fwd_ret_entry": [0.01, 0.01],
    })
    out = deoverlap_signals(signals, prices, horizon_bars=1)
    assert len(out) == 1


def test_per_instrument_specs_tradeable_only_excludes_negative_edge(monkeypatch):
    monkeypatch.setattr(
        "strategy.target_opt.load_target_optimization",
        lambda path=None: {
            "applied": True,
            "per_symbol": {
                "BTC": {"spec": {"horizon": 96, "threshold_bps": 50}, "tradeable": False},
                "SOL": {"spec": {"horizon": 96, "threshold_bps": 45}, "tradeable": True},
            },
        },
    )
    specs = per_instrument_specs(tradeable_only=True)
    assert "SOL" in specs
    assert "BTC" not in specs


def test_ticker_min_expected_edge_uses_threshold(monkeypatch):
    from data_platform.universe import commission_bps_for_ticker

    monkeypatch.setattr("config.FUSION_EDGE_GATE_FLOOR_MODE", "full_round_trip")
    comm = commission_bps_for_ticker("BTC")
    floor = ticker_min_expected_edge_bps("BTC", slippage_bps=2.0)
    rt_floor = heuristic_gate_floor_bps(comm, 2.0)
    assert floor >= rt_floor


def test_commission_bps_per_asset_class():
    from data_platform.universe import commission_bps_for_ticker

    assert commission_bps_for_ticker("BTC") == pytest.approx(1.1)
    assert commission_bps_for_ticker("SP500") == pytest.approx(0.5)


def test_desk_go_no_go_blocks_empty_tradeable():
    from reporting.desk_reports import desk_go_no_go

    report = {
        "impulse_optimization": {"best": {"disable_trading": True, "decile_gate_blocked": True}},
        "decile_audit": {"tradeable": False, "reasons": ["top_decile_net_bps=-3"], "top_decile_net_bps": -3},
        "backtest_walk_forward_oos": {"total_return_pct": -10},
        "ml_diagnostics": {},
        "walk_forward_folds": [],
    }
    out = desk_go_no_go(report)
    assert out["tradeable"] is False
    assert out["reasons"]


def test_desk_go_no_go_allows_partial_book():
    from reporting.desk_reports import desk_go_no_go

    report = {
        "impulse_optimization": {
            "best": {"disable_trading": False, "tradeable_tickers": ["SBER"]},
        },
        "decile_audit": {
            "tradeable": True,
            "tradeable_tickers": ["SBER"],
            "top_decile_net_bps": -1,
        },
        "backtest_walk_forward_oos": {"total_return_pct": 1.0},
        "ml_diagnostics": {},
        "walk_forward_folds": [],
    }
    out = desk_go_no_go(report)
    assert out["tradeable"] is True
    assert "SBER" in out["tradeable_tickers"]