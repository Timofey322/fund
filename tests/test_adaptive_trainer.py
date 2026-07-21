"""Tests for adaptive rolling LightGBM training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.adaptive_trainer import recency_sample_weights
from strategy.pipeline import is_adaptive_wf_mode, monthly_walk_forward_windows


def test_recency_weights_favor_recent_bars():
    ref = pd.Timestamp("2024-06-01")
    times = pd.Series(pd.date_range("2023-06-01", "2024-05-31", freq="7D"))
    w = recency_sample_weights(times, ref, halflife_days=90.0)
    assert len(w) == len(times)
    assert abs(float(np.mean(w)) - 1.0) < 1e-6
    assert float(w[-1]) > float(w[0])


def test_semiannual_windows_step_six_months():
    panel = pd.DataFrame({
        "bar_time": pd.date_range("2020-01-01", "2026-06-01", freq="D"),
    })
    windows = monthly_walk_forward_windows(panel, train_days=365, backtest_years=4, test_months=6)
    assert len(windows) >= 6
    w0 = windows[0]
    delta = (w0["test_end"] - w0["test_start"]).days
    assert 170 <= delta <= 190


def test_adaptive_wf_mode_default():
    assert is_adaptive_wf_mode("adaptive")
    assert is_adaptive_wf_mode("semiannual_adaptive")
    assert not is_adaptive_wf_mode("monthly4y")


def test_adaptive_entry_model_warm_start():
    pytest.importorskip("lightgbm")
    from models.adaptive_trainer import AdaptiveEntryModel

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((rng.random(n) > 0.55).astype(int))
    w = np.ones(n)
    params = {
        "num_leaves": 8,
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 20,
        "min_child_samples": 20,
    }
    model = AdaptiveEntryModel()
    model.fit_initial(X, y, w, params)
    state0 = model.training_state()
    model.fit_incremental(X, y, w, params)
    state1 = model.training_state()
    assert state1["n_fits"] == 2
    assert state1["total_trees"] > state0["total_trees"]
    proba = model.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 2)
