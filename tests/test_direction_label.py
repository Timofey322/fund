"""Tests for 3-class direction labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.direction_model import dual_binary_predict_direction
from research.labels.trade import (
    DIRECTION_FLAT,
    DIRECTION_LONG,
    DIRECTION_SHORT,
    TARGET_DIRECTION,
    attach_economic_entry_labels,
    build_direction_label,
    direction_class_rates,
    triple_barrier_direction_label,
)


def _synthetic_close(n: int = 500, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, n)
    return pd.Series(100.0 * np.cumprod(1.0 + rets))


def test_direction_label_mutually_exclusive():
    close = _synthetic_close()
    label = build_direction_label(close, horizon_bars=24, label_type="triple_barrier", threshold_bps=50.0)
    valid = label.dropna().astype(int)
    assert set(valid.unique()).issubset({0, 1, 2})
    rates = direction_class_rates(label)
    assert abs(sum(rates.values()) - 1.0) < 1e-6


def test_attach_economic_entry_labels_has_direction():
    close = _synthetic_close()
    out = attach_economic_entry_labels(pd.DataFrame({"close": close}), symbol="GAZP")
    assert TARGET_DIRECTION in out.columns
    assert out[TARGET_DIRECTION].dropna().astype(int).isin([0, 1, 2]).all()


def test_dual_binary_predict_direction_exclusive():
    pl = np.array([0.7, 0.4, 0.55, 0.5])
    ps = np.array([0.3, 0.7, 0.54, 0.5])
    pred = dual_binary_predict_direction(pl, ps, baseline=0.5, min_edge=0.05)
    assert pred[0] == DIRECTION_LONG
    assert pred[1] == DIRECTION_SHORT
    assert pred[2] in (DIRECTION_FLAT, DIRECTION_LONG)
    assert pred[3] == DIRECTION_FLAT


def test_triple_barrier_direction_no_nan_on_full_window():
    close = _synthetic_close(n=200)
    label = triple_barrier_direction_label(close, horizon_bars=12, take_profit_bps=40.0, stop_loss_bps=40.0)
    assert label.iloc[:-12].notna().all()
