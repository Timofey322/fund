"""Tests for tradfi timeframe selection (max bars)."""

from __future__ import annotations

from common.timeframe_policy import best_tradfi_timeframe_for_max_bars, estimate_tradfi_bars


def test_daily_has_most_tradfi_bars():
    tf, counts = best_tradfi_timeframe_for_max_bars(7300)
    assert tf == "1Day"
    assert counts["1Day"] > counts["1Hour"]
    assert counts["1Day"] > counts["5Min"]


def test_estimate_tradfi_bars_positive():
    assert estimate_tradfi_bars("1Day", 7300) > 1000
