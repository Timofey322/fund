"""Tests for TP/SL calibration and tail entry filters."""

from __future__ import annotations

import pandas as pd

from common.naming import COL_PROB_HMM_STRESS
from simulation.tp_sl_calibration import calibrate_tp_sl_bps, tighten_stop_loss_bps
from strategy.tail_filters import tail_entry_skip_mask


def test_calibrate_tp_sl_raises_tp_when_below_sl():
    tp, sl = calibrate_tp_sl_bps(25.0, 50.0, min_ratio=1.0, tp_floor_bps=40.0)
    assert tp >= sl
    assert tp >= 40.0


def test_tighten_stop_loss_in_stress():
    out = tighten_stop_loss_bps(50.0, stress_prob=0.5, vol_ann=0.10, vol_ratio=1.0)
    assert out < 50.0
    assert out >= 12.0


def test_tail_skip_requires_stress_and_vol():
    df = pd.DataFrame({
        COL_PROB_HMM_STRESS: [0.5, 0.5, 0.2],
        "vol_ann": [0.30, 0.10, 0.30],
        "vol_ratio": [1.0, 1.0, 1.5],
    })
    mask = tail_entry_skip_mask(df)
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[1]) is False
    assert bool(mask.iloc[2]) is False
