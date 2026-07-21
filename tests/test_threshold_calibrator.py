"""Threshold model: gate-aligned signal quality + optional policy fit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.threshold_calibrator import (
    _chrono_splits,
    _cv_top_decile_net,
    _eval_policy_net,
    _threshold_from_isotonic,
    fit_threshold_calibrator,
    resolve_ticker_policy,
)
from sklearn.isotonic import IsotonicRegression


def test_chrono_splits_expanding():
    splits = _chrono_splits(400, 3)
    assert len(splits) >= 1
    for tr, va in splits:
        assert tr.max() < va.min()
        assert len(tr) > 0 and len(va) > 0


def test_threshold_from_isotonic_finds_positive_region():
    x = np.linspace(20, 80, 200)
    y = (x - 50.0) * 0.5
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(x, y)
    thr = _threshold_from_isotonic(
        model, x, target_net_bps=0.0, fallback_quantile=0.9, lo=20.0, hi=80.0
    )
    assert 45.0 <= thr <= 55.0


def test_eval_policy_requires_min_trades():
    n = 100
    score = np.linspace(30, 70, n)
    edge = np.full(n, 8.0)
    impulse = np.full(n, 0.1)
    net = np.ones(n)
    val, n_act = _eval_policy_net(
        score, edge, impulse, net, buy=60.0, min_edge=5.0, impulse_min=0.05, min_trades=50
    )
    assert val == float("-inf")
    assert n_act < 50


def test_cv_top_decile_reports_negative_honestly():
    n = 800
    proba = np.linspace(0.1, 0.9, n)
    # Top proba has only +2 bps gross — below RT cost (~5 bps).
    fwd = np.where(proba > np.quantile(proba, 0.9), 0.0002, -0.0005)
    grp = pd.DataFrame({"ml_proba": proba, "fwd_ret": fwd, "ticker": "BTC"})
    net, n_rows, ok = _cv_top_decile_net(grp, "BTC")
    assert net is not None
    assert net < 0.0
    assert ok is False
    assert n_rows >= 200


def test_fit_skips_fusion_when_signal_quality_fails(monkeypatch):
    calls = {"n": 0}

    def _fake_fuse(df, params):
        calls["n"] += 1
        out = df.copy()
        out["fusion_score"] = 50.0
        out["expected_edge_bps"] = 8.0
        out["impulse_strength"] = 0.1
        return out

    import strategy.pipeline as pipe

    monkeypatch.setattr(pipe, "apply_fusion_scores", _fake_fuse)

    n = 800
    proba = np.linspace(0.1, 0.9, n)
    fwd = np.where(proba > np.quantile(proba, 0.9), 0.0002, -0.0005)
    train = pd.DataFrame(
        {
            "ticker": ["BTC"] * (n // 2) + ["ETH"] * (n // 2),
            "ml_proba": proba,
            "fwd_ret": fwd,
        }
    )
    base = {
        "buy_threshold": 36,
        "min_expected_edge_bps": 4.0,
        "impulse_min": 0.05,
        "w_ml": 0.45,
    }
    out = fit_threshold_calibrator(train, base, fold=0)
    # Soft quality flag: still score once so defaults/fit can trade when gate is disabled.
    assert calls["n"] == 1
    assert "BTC" in out and "ETH" in out
    for pol in out.values():
        assert pol["signal_quality_ok"] is False
        assert pol["cv_top_decile_net_bps"] is not None
        assert pol["cv_top_decile_net_bps"] < 0.0
        assert pol["threshold_model"] != "signal_quality_skip"
        # No lottery: |cv_net| should be small (cost-scale), not hundreds of bps.
        assert abs(pol["cv_top_decile_net_bps"]) < 50.0


def test_fit_threshold_calibrator_score_once_on_positive_signal(monkeypatch):
    calls = {"n": 0}

    def _fake_fuse(df, params):
        calls["n"] += 1
        out = df.copy()
        out["fusion_score"] = 30.0 + 50.0 * out["ml_proba"].astype(float)
        out["expected_edge_bps"] = 4.0 + 20.0 * out["ml_proba"].astype(float)
        out["impulse_strength"] = 0.1
        return out

    import strategy.pipeline as pipe

    monkeypatch.setattr(pipe, "apply_fusion_scores", _fake_fuse)

    rng = np.random.default_rng(0)
    n = 1200
    proba = rng.uniform(0.2, 0.95, n)
    # Strong edge in top proba — clears costs.
    fwd = np.where(proba > np.quantile(proba, 0.9), 0.008, -0.0003)
    train = pd.DataFrame(
        {
            "ticker": ["BTC"] * (n // 2) + ["ETH"] * (n // 2),
            "ml_proba": proba,
            "fwd_ret": fwd,
        }
    )
    base = {
        "buy_threshold": 36,
        "min_expected_edge_bps": 4.0,
        "impulse_min": 0.05,
        "w_ml": 0.45,
    }
    out = fit_threshold_calibrator(train, base, fold=0)
    assert calls["n"] == 1
    assert "BTC" in out and "ETH" in out
    for pol in out.values():
        assert pol["signal_quality_ok"] is True
        assert pol["cv_top_decile_net_bps"] is not None
        assert pol["cv_top_decile_net_bps"] > 0.0


def test_resolve_ticker_policy_keeps_soft_size_diagnostics():
    """CV/quality/alignment must survive merge so soft_size_multiplier can act."""
    params = {
        "buy_threshold": 36,
        "min_expected_edge_bps": 5.0,
        "by_ticker": {
            "BTC": {
                "buy_threshold": 42,
                "min_expected_edge_bps": 7.0,
                "cv_net_bps": 1.2,
                "cv_top_decile_net_bps": 1.2,
                "signal_quality_ok": True,
                "edge_alignment": -0.2,
                "soft_size": 0.25,
                "train_top_net_bps": 1.2,
                "threshold_model": "isotonic_cv",
            }
        },
    }
    merged = resolve_ticker_policy(params, "BTC")
    assert merged["buy_threshold"] == 42
    assert merged["min_expected_edge_bps"] == 7.0
    assert merged["cv_top_decile_net_bps"] == 1.2
    assert merged["signal_quality_ok"] is True
    assert merged["edge_alignment"] == -0.2
    assert merged["soft_size"] == 0.25
    assert "threshold_model" not in merged
    assert "train_top_net_bps" not in merged
