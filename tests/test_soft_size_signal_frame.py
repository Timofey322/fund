"""Exposure soft-size reaches signal frame exposure_cap; hard-zero skips."""

from __future__ import annotations

import pandas as pd


def _oos_frame(side: int = 1) -> pd.DataFrame:
    prices_idx = pd.date_range("2024-01-02 10:00", periods=3, freq="5min")
    return pd.DataFrame({
        "ticker": ["NASDAQ"] * 3,
        "bar_time": prices_idx,
        "close": [100.0, 100.1, 100.2],
        "ml_proba": [0.7, 0.7, 0.7],
        "ml_proba_short": [0.3, 0.3, 0.3],
        "ml_base_rate": [0.5, 0.5, 0.5],
        "impulse_strength": [0.8, 0.8, 0.8],
        "expected_edge_bps": [12.0, 12.0, 12.0],
        "position_side": [side, side, side],
        "fusion_score": [70.0, 70.0, 70.0],
        "hmm_gate": [True, True, True],
        "prob_hmm_impulse": [0.4, 0.4, 0.4],
        "prob_hmm_mean_revert": [0.3, 0.3, 0.3],
        "prob_hmm_stress": [0.1, 0.1, 0.1],
        "hmm_confidence": [0.6, 0.6, 0.6],
        "hmm_prob_entropy": [0.5, 0.5, 0.5],
    })


def test_fusion_signal_frame_applies_soft_size_exposure(monkeypatch):
    monkeypatch.setattr("config.FUSION_QUALITY_SOFT_SIZE", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_HARD_ZERO", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_INVERTED_CAP", 0.25)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_MIN", 0.1)
    monkeypatch.setattr("config.FUSION_DISABLE_QUALITY_GATE", True)
    monkeypatch.setattr("config.FUSION_SIDE_POLICY", {"NASDAQ": "both"})

    prices = pd.DataFrame(
        {"NASDAQ": [100.0, 100.1, 100.2]},
        index=pd.date_range("2024-01-02 10:00", periods=3, freq="5min"),
    )
    oos = _oos_frame(side=1)
    params = {
        "w_ml": 0.45, "w_mom": 0.2, "w_nw": 0.15, "w_flow": 0.05, "w_vp": 0.15,
        "stress_max": 0.9, "hmm_impulse_min": 0.0, "hmm_confidence_min": 0.0,
        "hmm_entropy_max": 2.0, "allow_mean_revert": True,
        "impulse_min": 0.0, "min_expected_edge_bps": 1.0,
        "buy_threshold": 55, "hold_threshold": 50, "gain": 80,
        "signal_quality_ok": True,
        "cv_top_decile_net_bps": 6.0,
        "holdout_top_decile_net_bps": 5.0,
        "edge_alignment": -0.4,
    }
    monkeypatch.setattr(
        "strategy.edge_gate.resolve_ticker_min_edge_bps",
        lambda _t, _p: 1.0,
    )
    from strategy.pipeline import _fusion_signal_frame

    sig = _fusion_signal_frame(oos, prices, params)
    assert not sig.empty
    assert float(sig["exposure_cap"].iloc[0]) <= 0.25


def test_fusion_signal_frame_hard_zero_exposure(monkeypatch):
    monkeypatch.setattr("config.FUSION_QUALITY_SOFT_SIZE", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_HARD_ZERO", True)
    monkeypatch.setattr("config.FUSION_DISABLE_QUALITY_GATE", True)
    monkeypatch.setattr("config.FUSION_SIDE_POLICY", {"NASDAQ": "both"})

    prices = pd.DataFrame(
        {"NASDAQ": [100.0, 100.1, 100.2]},
        index=pd.date_range("2024-01-02 10:00", periods=3, freq="5min"),
    )
    oos = _oos_frame(side=1)
    params = {
        "w_ml": 0.45, "w_mom": 0.2, "w_nw": 0.15, "w_flow": 0.05, "w_vp": 0.15,
        "stress_max": 0.9, "hmm_impulse_min": 0.0, "hmm_confidence_min": 0.0,
        "hmm_entropy_max": 2.0, "allow_mean_revert": True,
        "impulse_min": 0.0, "min_expected_edge_bps": 1.0,
        "buy_threshold": 55, "hold_threshold": 50, "gain": 80,
        "signal_quality_ok": False,
        "cv_top_decile_net_bps": 6.0,
        "holdout_top_decile_net_bps": 5.0,
        "edge_alignment": 0.2,
    }
    monkeypatch.setattr(
        "strategy.edge_gate.resolve_ticker_min_edge_bps",
        lambda _t, _p: 1.0,
    )
    from strategy.pipeline import _fusion_signal_frame

    sig = _fusion_signal_frame(oos, prices, params)
    assert not sig.empty
    assert float(sig["exposure_cap"].iloc[0]) == 0.0


def test_entry_active_mask_respects_side_policy(monkeypatch):
    monkeypatch.setattr("config.FUSION_SIDE_POLICY", {"NASDAQ": "long_only"})
    monkeypatch.setattr(
        "strategy.edge_gate.resolve_ticker_min_edge_bps",
        lambda _t, _p: 1.0,
    )
    oos = _oos_frame(side=-1)
    oos["fusion_score"] = 30.0
    params = {
        "impulse_min": 0.0,
        "min_expected_edge_bps": 1.0,
    }
    from strategy.pipeline import _entry_active_mask

    mask = _entry_active_mask(oos, params)
    assert not bool(mask.any())
