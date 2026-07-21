"""Tests for balanced per-instrument entry labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.labels.balanced import (
    TARGET_SL_BPS,
    TARGET_TP_BPS,
    attach_balanced_entry_label,
    build_balanced_entry_label,
    pick_balanced_horizon,
)
from research.labels.trade import FWD_RET_ENTRY, TARGET_ENTRY


def _synthetic_close(n: int = 400, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, n)
    prices = 100.0 * np.cumprod(1.0 + rets)
    return pd.Series(prices)


def test_pick_balanced_horizon_near_fifty_fifty():
    close = _synthetic_close()
    h, thr, rate = pick_balanced_horizon(close, horizons=(5, 10, 15, 20), target_rate=0.5)
    assert h in (5, 10, 15, 20)
    assert isinstance(thr, float)
    assert 0.25 <= rate <= 0.75


def test_build_balanced_entry_label_balance():
    close = _synthetic_close()
    label, fwd, meta = build_balanced_entry_label(close)
    valid = label.dropna()
    assert not valid.empty
    pos_rate = float(valid.mean())
    assert 0.2 <= pos_rate <= 0.8
    assert meta["label_type"] == "balanced"
    assert meta["horizon"] >= 1
    assert fwd.name == FWD_RET_ENTRY
    assert label.name == TARGET_ENTRY


def test_attach_balanced_entry_label_adds_tp_sl_columns():
    close = _synthetic_close()
    df = pd.DataFrame({"close": close})
    out = attach_balanced_entry_label(df, symbol="SPY")
    assert TARGET_ENTRY in out.columns
    assert TARGET_TP_BPS in out.columns
    assert TARGET_SL_BPS in out.columns
    assert out[TARGET_TP_BPS].notna().any()
    assert out[TARGET_SL_BPS].notna().any()
    assert out.attrs.get("balanced_spec", {}).get("symbol") == "SPY"
