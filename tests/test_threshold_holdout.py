"""Holdout validation for threshold signal quality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.threshold_calibrator import fit_threshold_calibrator


def test_holdout_failure_blocks_signal_ok(monkeypatch):
    def _fake_fuse(df, params):
        out = df.copy()
        out["fusion_score"] = 30.0 + 50.0 * out["ml_proba"].astype(float)
        out["expected_edge_bps"] = 8.0
        out["impulse_strength"] = 0.1
        return out

    import strategy.pipeline as pipe

    monkeypatch.setattr(pipe, "apply_fusion_scores", _fake_fuse)

    n = 800
    proba = np.linspace(0.1, 0.9, n)
    fwd_train = np.where(proba > np.quantile(proba, 0.9), 0.008, -0.0003)
    train = pd.DataFrame({"ticker": "BTC", "ml_proba": proba, "fwd_ret": fwd_train})

    ho_proba = np.linspace(0.1, 0.9, n)
    fwd_hold = np.where(ho_proba > np.quantile(ho_proba, 0.9), 0.0002, -0.0005)
    holdout = pd.DataFrame({"ticker": "BTC", "ml_proba": ho_proba, "fwd_ret": fwd_hold})

    base = {"buy_threshold": 36, "min_expected_edge_bps": 4.0, "impulse_min": 0.05, "w_ml": 0.45}
    out = fit_threshold_calibrator(train, base, fold=0, holdout=holdout)
    assert out["BTC"]["signal_quality_ok"] is False
