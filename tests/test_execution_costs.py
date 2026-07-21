"""Execution cost model tests."""

from __future__ import annotations

import pytest

from simulation.execution_costs import (
    round_trip_cost_bps,
    round_trip_cost_bps_for_ticker,
    slippage_bps_per_side,
)
from simulation.entry_signals import round_trip_cost_bps as entry_rt


def test_crypto_round_trip_matches_backtest_legs():
    comm = 1.1
    slip = slippage_bps_per_side("BTC")
    assert slip == 1.5
    assert round_trip_cost_bps(comm, slip) == 2 * (comm + slip)
    assert round_trip_cost_bps_for_ticker("BTC") == pytest.approx(5.2)


def test_entry_signals_alias_consistent():
    assert entry_rt(1.1, 1.5) == round_trip_cost_bps(1.1, 1.5)


def test_tradfi_costs_use_liquid_etf_commission():
    # SPY: 0.5 comm + 2.0 slip → RT 5.0; BTC: 1.1 + 1.5 → RT 5.2
    assert slippage_bps_per_side("SPY") == 2.0
    assert round_trip_cost_bps_for_ticker("SPY") == pytest.approx(5.0)
    assert round_trip_cost_bps_for_ticker("BTC") == pytest.approx(5.2)
