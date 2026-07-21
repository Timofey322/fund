"""Per-ticker side allowlist (long_only / short_only)."""

from __future__ import annotations

import pandas as pd

from strategy.side_policy import (
    allowed_sides_for_ticker,
    side_allowed,
    side_policy_mask,
)


def test_allowed_sides_from_config(monkeypatch):
    monkeypatch.setattr(
        "config.FUSION_SIDE_POLICY",
        {"NASDAQ": "long_only", "IMOEX": "short_only", "GAZP": "both"},
    )
    assert allowed_sides_for_ticker("nasdaq") == "long_only"
    assert allowed_sides_for_ticker("IMOEX") == "short_only"
    assert allowed_sides_for_ticker("GAZP") == "both"
    assert allowed_sides_for_ticker("UNKNOWN") == "both"


def test_side_allowed_long_only(monkeypatch):
    monkeypatch.setattr("config.FUSION_SIDE_POLICY", {"NASDAQ": "long_only"})
    assert side_allowed("NASDAQ", 1) is True
    assert side_allowed("NASDAQ", -1) is False
    assert side_allowed("NASDAQ", 0) is False


def test_side_allowed_short_only(monkeypatch):
    monkeypatch.setattr("config.FUSION_SIDE_POLICY", {"IMOEX": "short_only"})
    assert side_allowed("IMOEX", -1) is True
    assert side_allowed("IMOEX", 1) is False


def test_side_policy_mask_filters_rows(monkeypatch):
    monkeypatch.setattr(
        "config.FUSION_SIDE_POLICY",
        {"NASDAQ": "long_only", "IMOEX": "short_only"},
    )
    tickers = pd.Series(["NASDAQ", "NASDAQ", "IMOEX", "IMOEX"])
    sides = pd.Series([1, -1, -1, 1])
    mask = side_policy_mask(tickers, sides)
    assert list(mask) == [True, False, True, False]
