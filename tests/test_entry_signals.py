"""Tests for symmetric long/short entry filtering."""

from __future__ import annotations

import pandas as pd

from simulation.entry_signals import active_entry_signals


def test_active_entry_signals_respects_position_side():
    df = pd.DataFrame(
        {
            "score": [60.0, 40.0, 55.0, 45.0],
            "position_side": [1, -1, 0, -1],
            "risk_on": [True, True, True, True],
            "buy_threshold": [55.0, 55.0, 55.0, 55.0],
            "sell_threshold": [45.0, 45.0, 45.0, 45.0],
        }
    )
    out = active_entry_signals(df)
    assert len(out) == 4
    assert set(out["position_side"].tolist()) == {1, -1, 0}


def test_active_entry_signals_ignores_zero_score_flat_side():
    df = pd.DataFrame(
        {
            "score": [0.0, 44.0],
            "position_side": [0, 0],
            "risk_on": [True, True],
            "buy_threshold": [55.0, 55.0],
            "sell_threshold": [45.0, 45.0],
        }
    )
    out = active_entry_signals(df)
    assert len(out) == 1
    assert float(out.iloc[0]["score"]) == 44.0
