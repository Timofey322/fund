"""Signed expected-edge gate: long/short must align with edge sign (universal rule)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.edge_gate import (
    passes_signed_edge_gate,
    resolve_ticker_min_edge_bps,
    signed_edge_active_mask,
)
from strategy.pipeline import _entry_active_mask, _gated_entries


def test_passes_signed_edge_gate_long_and_short():
    assert passes_signed_edge_gate(6.0, 5.0, 1) is True
    assert passes_signed_edge_gate(4.0, 5.0, 1) is False
    assert passes_signed_edge_gate(-6.0, 5.0, -1) is True
    assert passes_signed_edge_gate(-4.0, 5.0, -1) is False
    assert passes_signed_edge_gate(-9.0, 5.0, -1) is True
    assert passes_signed_edge_gate(9.0, 5.0, -1) is False
    assert passes_signed_edge_gate(9.0, 5.0, 0) is False


def test_signed_edge_active_mask_vectorized():
    edge = np.array([8.0, 3.0, -8.0, -3.0, 8.0])
    side = np.array([1, 1, -1, -1, 0])
    mask = signed_edge_active_mask(edge, side, 5.0)
    assert mask.tolist() == [True, False, True, False, False]


def test_entry_active_mask_allows_aligned_long_and_short(monkeypatch):
    monkeypatch.setattr(
        "strategy.edge_gate.resolve_ticker_min_edge_bps",
        lambda _t, _p: 5.0,
    )
    monkeypatch.setattr("config.FUSION_SIDE_POLICY", {"NASDAQ": "both"})
    df = pd.DataFrame({
        "ticker": ["NASDAQ"] * 2,
        "expected_edge_bps": [9.0, -9.0],
        "position_side": [1, -1],
        "impulse_strength": [0.5, 0.5],
    })
    params = {"impulse_min": 0.1, "min_expected_edge_bps": 5.0}
    mask = _entry_active_mask(df, params)
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[1]) is True


def test_entry_active_mask_blocks_misaligned_short(monkeypatch):
    monkeypatch.setattr(
        "strategy.edge_gate.resolve_ticker_min_edge_bps",
        lambda _t, _p: 5.0,
    )
    df = pd.DataFrame({
        "ticker": ["NASDAQ"] * 2,
        "expected_edge_bps": [9.0, -4.0],
        "position_side": [-1, -1],
        "impulse_strength": [0.5, 0.5],
    })
    params = {"impulse_min": 0.1, "min_expected_edge_bps": 5.0}
    mask = _entry_active_mask(df, params)
    assert bool(mask.iloc[0]) is False
    assert bool(mask.iloc[1]) is False


def test_gated_entries_uses_signed_edge_not_abs(monkeypatch):
    """Short rows with positive edge must not pass (regression: abs(edge) bug)."""
    idx = pd.date_range("2024-01-02 10:00", periods=2, freq="5min")
    oos = pd.DataFrame({
        "ticker": ["TEST"] * 2,
        "bar_time": idx,
        "close": [100.0, 100.0],
        "ml_proba": [0.3, 0.3],
        "ml_proba_short": [0.8, 0.55],
        "ml_base_rate": [0.5, 0.5],
        "impulse_strength": [0.5, 0.5],
        "prob_hmm_impulse": [0.3, 0.3],
        "prob_hmm_mean_revert": [0.2, 0.2],
        "prob_hmm_stress": [0.1, 0.1],
        "hmm_confidence": [0.5, 0.5],
        "hmm_prob_entropy": [0.5, 0.5],
    })
    params = {
        "w_ml": 0.45, "w_mom": 0.2, "w_nw": 0.15, "w_flow": 0.05, "w_vp": 0.15,
        "stress_max": 0.55, "hmm_impulse_min": 0.05, "hmm_confidence_min": 0.2,
        "hmm_entropy_max": 1.05, "allow_mean_revert": True,
        "impulse_min": 0.05, "min_expected_edge_bps": 2.0,
        "buy_threshold": 52, "hold_threshold": 49,
    }
    monkeypatch.setattr(
        "strategy.edge_gate.resolve_ticker_min_edge_bps",
        lambda _t, _p: 5.0,
    )
    out = _gated_entries(oos, params)
    if not out.empty:
        shorts = out[out["position_side"] < 0]
        if not shorts.empty:
            assert (shorts["expected_edge_bps"] <= -5.0).all()


def test_resolve_ticker_min_edge_uses_cost_floor_not_label_barrier(monkeypatch):
    """Label TP threshold must not become the ML edge gate floor."""
    monkeypatch.setattr(
        "strategy.target_opt.ticker_threshold_bps",
        lambda _t: 47.5,
    )
    from strategy.edge_gate import heuristic_gate_floor_bps, resolve_ticker_min_edge_bps
    from data_platform.universe import commission_bps_for_ticker

    pol = {"min_expected_edge_bps": 5.0}
    out = resolve_ticker_min_edge_bps("GAZP", pol)
    floor = heuristic_gate_floor_bps(commission_bps_for_ticker("GAZP"))
    assert out >= floor
    assert out < 20.0  # must not jump to ~47 label barrier
    assert abs(out - max(5.0, floor)) < 1.0


def test_threshold_calibrator_mask_respects_signed_short_edge():
    from strategy.threshold_calibrator import _mask_active

    score = np.array([40.0, 40.0, 60.0])
    edge = np.array([-8.0, 8.0, 8.0])
    impulse = np.array([0.5, 0.5, 0.5])
    side = np.array([-1, -1, 1])
    mask = _mask_active(
        score, edge, impulse,
        buy=55.0, min_edge=5.0, impulse_min=0.1,
        position_side=side, sell=45.0,
    )
    assert mask.tolist() == [True, False, True]
