"""Tests for fixed-threshold economic entry labels (no artificial 50/50 balance)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.labels.trade import (
    TARGET_ENTRY,
    TARGET_ENTRY_SHORT,
    attach_economic_entry_labels,
    build_entry_label,
    build_short_entry_label,
    default_entry_spec,
    resolve_entry_spec,
    triple_barrier_label,
    triple_barrier_label_short,
)


def _synthetic_close(n: int = 600, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.008, n)
    prices = 100.0 * np.cumprod(1.0 + rets)
    return pd.Series(prices)


def test_economic_long_label_not_forced_fifty_fifty():
    close = _synthetic_close()
    label, _ = build_entry_label(
        close,
        horizon_bars=48,
        label_type="triple_barrier",
        threshold_bps=50.0,
        commission_bps=1.1,
        stop_loss_bps=45.0,
    )
    valid = label.dropna()
    pos_rate = float(valid.mean())
    assert not valid.empty
    assert 0.02 <= pos_rate <= 0.98
    assert abs(pos_rate - 0.5) > 0.005


def test_economic_short_label_uses_same_economics():
    close = _synthetic_close(seed=13)
    long_label, _ = build_entry_label(
        close,
        horizon_bars=48,
        label_type="after_costs",
        threshold_bps=40.0,
        commission_bps=4.0,
        stop_loss_bps=45.0,
    )
    short_label = build_short_entry_label(
        close,
        horizon_bars=48,
        label_type="after_costs",
        threshold_bps=40.0,
        commission_bps=4.0,
        stop_loss_bps=45.0,
    )
    assert long_label.name == TARGET_ENTRY
    assert short_label.name == TARGET_ENTRY_SHORT
    assert long_label.dropna().mean() != short_label.dropna().mean()


def test_attach_economic_entry_labels_adds_metadata():
    close = _synthetic_close()
    df = pd.DataFrame({"close": close})
    out = attach_economic_entry_labels(df, symbol="GAZP")
    assert TARGET_ENTRY in out.columns
    assert TARGET_ENTRY_SHORT in out.columns
    assert "target_tp_bps" in out.columns
    assert "target_sl_bps" in out.columns
    assert "entry_label_horizon" in out.columns
    long_rate = float(out[TARGET_ENTRY].dropna().mean())
    assert abs(long_rate - 0.5) > 0.01 or long_rate < 0.45 or long_rate > 0.55


def test_default_entry_spec_is_fixed_not_data_dependent():
    a = default_entry_spec("GAZP")
    b = default_entry_spec("GAZP")
    assert a == b
    assert a["label_type"] in ("triple_barrier", "after_costs", "positive")
    assert int(a["horizon"]) >= 1
    assert float(a["threshold_bps"]) > 0.0


def test_triple_barrier_short_is_inverse_of_long_barrier():
    close = _synthetic_close(seed=21)
    long_tb = triple_barrier_label(
        close, horizon_bars=24, take_profit_bps=50.0, stop_loss_bps=45.0
    )
    short_tb = triple_barrier_label_short(
        close, horizon_bars=24, take_profit_bps=50.0, stop_loss_bps=45.0
    )
    valid = long_tb.notna() & short_tb.notna()
    if valid.sum() > 50:
        corr = float(long_tb[valid].corr(short_tb[valid]))
        assert corr < 0.85


def test_triple_barrier_timeout_is_zero_not_sign_of_path():
    """Full-window no-hit must be 0 so cost-floor TP is not diluted."""
    # Flat then tiny up-move: never hits ±50/45 bps barriers, but path max > 0.
    close = pd.Series([100.0] * 30 + [100.2] * 30)  # +20 bps < 50 TP
    lab = triple_barrier_label(
        close, horizon_bars=24, take_profit_bps=50.0, stop_loss_bps=45.0
    )
    # Rows with full future window and no barrier touch.
    valid = lab.iloc[:36].dropna()
    assert not valid.empty
    assert float(valid.max()) == 0.0

    short = triple_barrier_label_short(
        close, horizon_bars=24, take_profit_bps=50.0, stop_loss_bps=45.0
    )
    valid_s = short.iloc[:36].dropna()
    assert not valid_s.empty
    assert float(valid_s.max()) == 0.0


def test_resolve_entry_spec_returns_dict():
    spec = resolve_entry_spec("SP500")
    assert "horizon" in spec
    assert "label_type" in spec
    assert "threshold_bps" in spec
