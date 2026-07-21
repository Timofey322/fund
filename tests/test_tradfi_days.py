"""Tests for tradfi download day limits."""

from __future__ import annotations


def test_tradfi_max_days_by_timeframe():
    from config import tradfi_max_days

    assert tradfi_max_days("5Min") == 59
    assert tradfi_max_days("1Hour") == 730
    assert tradfi_max_days("1Day") == 5475
