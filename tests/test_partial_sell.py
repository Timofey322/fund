"""Partial sell (trim) on signal-dead exits."""

from __future__ import annotations

import pandas as pd

from simulation.engine import (
    _apply_signal_exits,
    _count_signal_exits,
    _deploy_cash_on_buy_picks,
    _merge_partial_sell_rebalance,
)


def test_apply_signal_exits_partial_trim_keeps_eighty_percent():
    weights = {"AAA": 1.0}
    entry_idx = {"AAA": 0}
    sub = pd.DataFrame({"ticker": ["AAA"], "score": [30.0], "risk_on": [True]})

    full = _apply_signal_exits(
        weights, entry_idx, 10, sub, "score", False, 100, "SPY", {}, min_hold_bars=0,
    )
    assert "AAA" not in full

    partial = _apply_signal_exits(
        weights, entry_idx, 10, sub, "score", False, 100, "SPY", {},
        min_hold_bars=0, partial_sell_frac=0.2,
    )
    assert abs(partial["AAA"] - 0.8) < 1e-9


def test_deploy_cash_on_buy_picks_new_entry_uses_all_cash():
    picks = [{"ticker": "AAA"}]
    out = _deploy_cash_on_buy_picks({}, picks)
    assert abs(out["AAA"] - 1.0) < 1e-9


def test_deploy_cash_on_buy_picks_reinvests_after_partial_trim():
    held = {"AAA": 0.9}
    picks = [{"ticker": "AAA"}]
    out = _deploy_cash_on_buy_picks(held, picks)
    assert abs(out["AAA"] - 1.0) < 1e-9


def test_deploy_cash_on_buy_picks_splits_cash_across_names():
    held = {"AAA": 0.9}
    picks = [{"ticker": "AAA"}, {"ticker": "BBB"}]
    out = _deploy_cash_on_buy_picks(held, picks)
    assert abs(out["AAA"] - 0.95) < 1e-9
    assert abs(out["BBB"] - 0.05) < 1e-9


def test_merge_partial_sell_rebalance_keeps_unpicked_holdings():
    held = {"AAA": 0.64}
    picks = {"AAA": 1.0, "CCC": 1.0}
    merged = _merge_partial_sell_rebalance(held, picks)
    assert merged["AAA"] == 1.0
    assert merged["CCC"] == 1.0
    assert merged.get("AAA", 0.0) == 1.0


def test_count_signal_exits_includes_partial_trims():
    before = {"AAA": 1.0}
    after = {"AAA": 0.8}
    assert _count_signal_exits(before, after, partial_sell_frac=0.2) == 1
    assert _count_signal_exits(before, {}, partial_sell_frac=0.2) == 1
    assert _count_signal_exits(before, after) == 0


def test_apply_signal_exits_partial_trim_respects_baseline_floor():
    weights = {"AAA": 0.20}
    entry_idx = {"AAA": 0}
    baseline = {"AAA": 0.10}
    sub = pd.DataFrame({"ticker": ["AAA"], "score": [30.0], "risk_on": [True]})

    partial = _apply_signal_exits(
        weights, entry_idx, 10, sub, "score", False, 100, "SPY", {},
        min_hold_bars=0, partial_sell_frac=0.1, baseline_weights=baseline,
    )
    assert abs(partial["AAA"] - 0.18) < 1e-9

    weights2 = {"AAA": 0.105}
    partial2 = _apply_signal_exits(
        weights2, entry_idx, 10, sub, "score", False, 100, "SPY", {},
        min_hold_bars=0, partial_sell_frac=0.1, baseline_weights=baseline,
    )
    assert abs(partial2["AAA"] - 0.10) < 1e-9
