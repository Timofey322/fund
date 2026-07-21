"""Tests for Finam Trade API bar parsing."""

from __future__ import annotations

from data_platform.finam_trade_api import _bars_to_frame, resolve_trade_symbol


def test_resolve_trade_symbols():
    assert resolve_trade_symbol("GAZP") == "GAZP@MISX"
    assert resolve_trade_symbol("IMOEX") == "IMOEX@MISX"
    assert resolve_trade_symbol("NASDAQ") == "NDX@_SCI"
    assert resolve_trade_symbol("SP500") == "SPY@ARCX"


def test_bars_to_frame():
    bars = [
        {
            "timestamp": "2025-06-02T10:00:00Z",
            "open": {"value": "100.1"},
            "high": {"value": "101.0"},
            "low": {"value": "99.5"},
            "close": {"value": "100.8"},
            "volume": {"value": "1200"},
        }
    ]
    frame = _bars_to_frame(bars)
    assert len(frame) == 1
    assert frame["close"].iloc[0] == 100.8
