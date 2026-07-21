"""Symmetric long/short threshold band tests."""

from __future__ import annotations

import pandas as pd

import config as cfg
from simulation.signal_frame import build_flow_signal_frame
from strategy.fusion_direction import (
    fusion_sell_threshold,
    normalize_buy_threshold,
    resolve_trading_thresholds,
)


def test_normalize_buy_threshold_clamps_below_midpoint():
    assert normalize_buy_threshold(42.0) >= 51.0
    assert normalize_buy_threshold(55.0) == 55.0


def test_resolve_trading_thresholds_symmetric():
    bands = resolve_trading_thresholds(55.0, 48.0)
    assert bands["buy_threshold"] == 55.0
    assert bands["sell_threshold"] == 45.0
    assert bands["hold_threshold"] == 48.0
    assert bands["sell_threshold"] < 50.0 < bands["buy_threshold"]


def test_fusion_sell_threshold_derived_when_explicit_none():
    prev = getattr(cfg, "FUSION_SELL_THRESHOLD", None)
    prev_sym = getattr(cfg, "FUSION_SYMMETRIC_THRESHOLDS", True)
    try:
        cfg.FUSION_SELL_THRESHOLD = None
        cfg.FUSION_SYMMETRIC_THRESHOLDS = True
        assert fusion_sell_threshold(55.0) == 45.0
    finally:
        cfg.FUSION_SELL_THRESHOLD = prev
        cfg.FUSION_SYMMETRIC_THRESHOLDS = prev_sym


def test_zero_score_does_not_assign_short_side():
    oos = pd.DataFrame(
        [
            {
                "ticker": "NASDAQ",
                "bar_time": "2024-01-02 10:00:00",
                "close": 100.0,
                "fusion_score": 0.0,
                "position_side": 0,
                "hmm_risk_on": True,
                "impulse_strength": 0.0,
            }
        ]
    )
    prices = pd.DataFrame(
        {"NASDAQ": [100.0]},
        index=pd.to_datetime(["2024-01-02 10:00:00"]),
    )
    sig = build_flow_signal_frame(
        oos,
        prices,
        buy_threshold=55.0,
        hold_threshold=48.0,
        score_col="fusion_score",
    )
    assert not sig.empty
    assert int(sig.iloc[0]["position_side"]) == 0
