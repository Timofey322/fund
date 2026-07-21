"""Edge gate in signal frame must use policy edge, not label barrier bps."""

from __future__ import annotations

import pandas as pd

from strategy.pipeline import _fusion_signal_frame
from strategy.target_opt import ticker_threshold_bps


def test_fusion_signal_frame_does_not_use_label_barrier_as_edge_gate(monkeypatch):
    """Label threshold_bps (~47.5) must not block entries when policy edge is low."""
    monkeypatch.setattr(
        "strategy.edge_gate.resolve_ticker_min_edge_bps",
        lambda _t, _p: 2.0,
    )
    label_thr = ticker_threshold_bps("GAZP")
    assert label_thr is not None and label_thr > 20.0

    idx = pd.date_range("2024-01-02 10:00", periods=4, freq="5min")
    prices = pd.DataFrame({"GAZP": [100.0, 101.0, 102.0, 103.0]}, index=idx)
    oos = pd.DataFrame(
        {
            "ticker": ["GAZP"] * 4,
            "bar_time": idx,
            "close": [100.0, 101.0, 102.0, 103.0],
            "ml_proba": [0.7, 0.3, 0.65, 0.35],
            "ml_proba_short": [0.3, 0.7, 0.35, 0.65],
            "ml_base_rate": [0.5] * 4,
            "impulse_strength": [0.2, 0.2, 0.2, 0.2],
            "prob_hmm_impulse": [0.3, 0.3, 0.3, 0.3],
            "prob_hmm_mean_revert": [0.2, 0.2, 0.2, 0.2],
            "prob_hmm_stress": [0.1, 0.1, 0.1, 0.1],
            "hmm_confidence": [0.5, 0.5, 0.5, 0.5],
            "hmm_prob_entropy": [0.5, 0.5, 0.5, 0.5],
        }
    )
    params = {
        "w_ml": 0.45,
        "w_mom": 0.2,
        "w_nw": 0.15,
        "w_flow": 0.05,
        "w_vp": 0.15,
        "stress_max": 0.55,
        "hmm_impulse_min": 0.05,
        "hmm_confidence_min": 0.2,
        "hmm_entropy_max": 1.05,
        "allow_mean_revert": True,
        "impulse_min": 0.05,
        "min_expected_edge_bps": 2.0,
        "gain": 100,
        "hold_threshold": 49,
        "buy_threshold": 52,
        "stop_loss_bps": 35.0,
        "edge_floor_mode": "commission_only",
        "disable_trading": False,
    }
    sig = _fusion_signal_frame(oos, prices, params)
    assert not sig.empty
    assert int((sig["score"] > 0).sum()) >= 1
