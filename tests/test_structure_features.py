"""Tests for structure/path ML features and registry wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.features.registry import active_ml_feature_cols, feature_group, attach_fusion_derived_features
from research.features.structure import STRUCTURE_COLS, attach_structure_features


def test_structure_features_causal_finite():
    n = 200
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.001, n))
    df = pd.DataFrame(
        {
            "close": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "open": close,
            "volume": rng.uniform(1e3, 5e3, n),
            "nw_est": close * 0.999,
            "nw_band_width": 0.01,
        }
    )
    out = attach_structure_features(df)
    for c in STRUCTURE_COLS:
        assert c in out.columns
        # After warmup window, some finite values must exist
        assert np.isfinite(out[c].to_numpy(dtype=float)[80:]).any(), c


def test_structure_cols_in_active_ml_registry():
    active = set(active_ml_feature_cols())
    for c in STRUCTURE_COLS:
        assert c in active
        assert feature_group(c) == "structure_path"
    assert "spec_dominant_period" in active
    assert feature_group("spec_dominant_period") == "spectral"
    for c in (
        "hurst_rs_cs_rank",
        "spec_low_high_ratio_cs_rank",
        "vp_poc_dist_cs_rank",
        "garch_vol_ratio_cs_rank",
    ):
        assert c in active
        assert feature_group(c) == "cross_sectional"


def test_attach_fusion_derived_includes_structure():
    n = 120
    rng = np.random.default_rng(1)
    close = 50 + np.cumsum(rng.normal(0, 0.05, n))
    times = pd.date_range("2024-01-02 10:00", periods=n, freq="5min")
    df = pd.DataFrame(
        {
            "ticker": ["T"] * n,
            "bar_time": times,
            "close": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "volume": rng.uniform(100, 500, n),
            "ret_1": np.r_[np.nan, np.diff(close) / close[:-1]],
            "ret_12": 0.0,
            "vol_imbalance": 0.0,
            "nw_est": close,
            "nw_band_width": 0.02,
            "nw_env_pos": 0.0,
            "vp_poc_dist": 0.0,
        },
        index=times,
    )
    out = attach_fusion_derived_features(df)
    assert "trend_efficiency_24" in out.columns
    assert "vol_of_vol_24" in out.columns


def test_hl_range_z_handles_flat_ohlc():
    """When high==low, range z must not be all-NaN (would wipe WF dropna)."""
    n = 200
    close = np.linspace(100, 110, n) + np.sin(np.linspace(0, 8, n))
    df = pd.DataFrame(
        {
            "close": close,
            "high": close,
            "low": close,
            "open": close,
            "volume": np.full(n, 1000.0),
            "nw_est": close,
            "nw_band_width": 0.01,
        }
    )
    out = attach_structure_features(df)
    assert out["hl_range_z_24"].iloc[80:].notna().all()
    assert float(out["hl_range_z_24"].iloc[80:].std()) > 0.0
