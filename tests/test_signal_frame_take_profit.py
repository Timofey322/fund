"""Take-profit fallback from label threshold / default config."""

from __future__ import annotations

import pandas as pd


def test_signal_frame_sets_take_profit_from_default(monkeypatch):
    monkeypatch.setattr("config.FUSION_DEFAULT_TAKE_PROFIT_BPS", 47.5)
    monkeypatch.setattr("config.FUSION_STOP_LOSS_BPS", 45.0)
    monkeypatch.setattr("config.FUSION_SIDE_POLICY", {"GAZP": "both"})
    monkeypatch.setattr(
        "strategy.target_opt.ticker_threshold_bps",
        lambda _t, default=None: None,
    )
    monkeypatch.setattr(
        "strategy.target_opt.ticker_hold_horizon_bars",
        lambda _t, d: int(d),
    )

    prices = pd.DataFrame(
        {"GAZP": [100.0, 100.5, 101.0]},
        index=pd.date_range("2024-01-02 10:00", periods=3, freq="5min"),
    )
    oos = pd.DataFrame({
        "ticker": ["GAZP"] * 3,
        "bar_time": prices.index,
        "close": [100.0, 100.5, 101.0],
        "ml_proba": [0.8, 0.8, 0.8],
        "impulse_strength": [0.9, 0.9, 0.9],
        "expected_edge_bps": [20.0, 20.0, 20.0],
        "position_side": [1, 1, 1],
        "fusion_score": [80.0, 80.0, 80.0],
        "prob_hmm_impulse": [0.5, 0.5, 0.5],
        "prob_hmm_mean_revert": [0.3, 0.3, 0.3],
        "prob_hmm_stress": [0.1, 0.1, 0.1],
        "hmm_confidence": [0.7, 0.7, 0.7],
        "hmm_prob_entropy": [0.4, 0.4, 0.4],
        "hmm_risk_on": [True, True, True],
        "risk_on": [True, True, True],
    })
    from simulation.signal_frame import build_flow_signal_frame

    sig = build_flow_signal_frame(
        oos, prices, buy_threshold=55, hold_threshold=50, gain=80, stop_loss_bps=45.0,
    )
    assert not sig.empty
    assert float(sig["take_profit_bps"].iloc[0]) > 0
