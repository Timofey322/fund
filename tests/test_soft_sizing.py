"""Universal OOS soft-sizing from signal quality / edge alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.soft_sizing import (
    compute_edge_alignment,
    soft_size_block_reason,
    soft_size_multiplier,
)


def test_soft_size_full_when_quality_ok_and_positive_net(monkeypatch):
    monkeypatch.setattr("config.FUSION_QUALITY_SOFT_SIZE", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_HARD_ZERO", True)
    size = soft_size_multiplier(
        {
            "signal_quality_ok": True,
            "cv_top_decile_net_bps": 8.0,
            "holdout_top_decile_net_bps": 6.0,
            "edge_alignment": 0.4,
        }
    )
    assert size == 1.0


def test_soft_size_hard_zero_when_quality_fails(monkeypatch):
    monkeypatch.setattr("config.FUSION_QUALITY_SOFT_SIZE", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_HARD_ZERO", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_MIN", 0.1)
    size = soft_size_multiplier(
        {
            "signal_quality_ok": False,
            "cv_top_decile_net_bps": 8.0,
            "holdout_top_decile_net_bps": 6.0,
            "edge_alignment": 0.4,
        }
    )
    assert size == 0.0
    assert soft_size_block_reason(
        {"signal_quality_ok": False, "holdout_top_decile_net_bps": 6.0}
    ) == "signal_quality_ok=False"


def test_soft_size_hard_zero_on_negative_holdout(monkeypatch):
    monkeypatch.setattr("config.FUSION_QUALITY_SOFT_SIZE", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_HARD_ZERO", True)
    size = soft_size_multiplier(
        {
            "signal_quality_ok": True,
            "cv_top_decile_net_bps": 8.0,
            "holdout_top_decile_net_bps": -3.0,
            "edge_alignment": 0.2,
        }
    )
    assert size == 0.0
    reason = soft_size_block_reason(
        {"signal_quality_ok": True, "holdout_top_decile_net_bps": -3.0}
    )
    assert reason is not None and "holdout" in reason


def test_soft_size_legacy_fail_cap_when_hard_zero_off(monkeypatch):
    monkeypatch.setattr("config.FUSION_QUALITY_SOFT_SIZE", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_HARD_ZERO", False)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_QUALITY_FAIL", 0.35)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_MIN", 0.1)
    size = soft_size_multiplier(
        {
            "signal_quality_ok": False,
            "cv_top_decile_net_bps": 8.0,
            "holdout_top_decile_net_bps": 6.0,
            "edge_alignment": 0.4,
        }
    )
    assert 0.1 <= size <= 0.35


def test_soft_size_reduces_on_inverted_edge(monkeypatch):
    monkeypatch.setattr("config.FUSION_QUALITY_SOFT_SIZE", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_HARD_ZERO", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_INVERTED_CAP", 0.25)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_MIN", 0.1)
    size = soft_size_multiplier(
        {
            "signal_quality_ok": True,
            "cv_top_decile_net_bps": 5.0,
            "holdout_top_decile_net_bps": 5.0,
            "edge_alignment": -0.3,
        }
    )
    assert size <= 0.25


def test_soft_size_scales_with_negative_cv_net(monkeypatch):
    monkeypatch.setattr("config.FUSION_QUALITY_SOFT_SIZE", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_HARD_ZERO", True)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_CV_LO_BPS", -10.0)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_CV_HI_BPS", 5.0)
    monkeypatch.setattr("config.FUSION_SOFT_SIZE_MIN", 0.1)
    size = soft_size_multiplier(
        {
            "signal_quality_ok": True,
            "cv_top_decile_net_bps": -10.0,
            "holdout_top_decile_net_bps": 1.0,
            "edge_alignment": 0.2,
        }
    )
    assert abs(size - 0.1) < 1e-6


def test_soft_size_disabled_returns_one(monkeypatch):
    monkeypatch.setattr("config.FUSION_QUALITY_SOFT_SIZE", False)
    size = soft_size_multiplier({"signal_quality_ok": False, "edge_alignment": -1.0})
    assert size == 1.0


def test_compute_edge_alignment_positive_when_monotonic():
    n = 200
    abs_edge = np.linspace(1.0, 20.0, n)
    signed = abs_edge + np.random.default_rng(0).normal(0, 0.5, n)
    df = pd.DataFrame(
        {
            "expected_edge_bps": abs_edge,
            "position_side": np.ones(n, dtype=int),
            "fwd_ret_entry": signed / 10_000.0,
        }
    )
    align = compute_edge_alignment(df)
    assert align is not None and align > 0.5


def test_compute_edge_alignment_negative_when_inverted():
    n = 200
    abs_edge = np.linspace(1.0, 20.0, n)
    df = pd.DataFrame(
        {
            "expected_edge_bps": -abs_edge,
            "position_side": -np.ones(n, dtype=int),
            "fwd_ret_entry": abs_edge / 10_000.0,
        }
    )
    align = compute_edge_alignment(df)
    assert align is not None and align < 0.0
