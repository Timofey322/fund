"""Universe + LLN target-opt helpers."""
from __future__ import annotations

import pytest

from data_platform.universe import is_crypto_symbol, is_tradfi_symbol, parse_tickers, split_tickers
from strategy.target_opt import (
    _candidate_specs,
    _direction_balance_factor,
    direction_target_score,
    target_score,
)


def test_parse_tickers_multi_asset():
    tickers = parse_tickers("BTC,SPY")
    assert tickers == ["BTC", "SPY"]
    crypto, tradfi = split_tickers(tickers)
    assert crypto == ["BTC"]
    assert tradfi == ["SPY"]


def test_parse_tickers_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_tickers("BTC,FAKECOIN")


def test_candidate_specs_respects_max_horizon(monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "TARGET_OPT_MAX_HORIZON_BARS", 48)
    specs = _candidate_specs((12, 24, 72, 96), ("after_costs",), (1.0,), 35.0)
    horizons = {s["horizon"] for s in specs}
    assert horizons <= {12, 24}
    assert 72 not in horizons


def test_target_score_prefers_shorter_horizon_via_penalty(monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "TARGET_OPT_HORIZON_PENALTY_PER_BAR", 0.01)
    base = {
        "cv": {"composite": 0.5},
        "net_edge_bps": 15.0,
        "positive_rate": 0.2,
    }
    short = target_score({**base, "horizon": 24}, balance_range=(0.08, 0.55))
    long = target_score({**base, "horizon": 96}, balance_range=(0.08, 0.55))
    assert short > long


def test_direction_balance_factor_prefers_balanced_classes():
    balanced = _direction_balance_factor({"flat": 0.30, "long": 0.35, "short": 0.35})
    skewed = _direction_balance_factor({"flat": 0.05, "long": 0.90, "short": 0.05})
    assert balanced > skewed


def test_direction_target_score_prefers_positive_edges():
    good = direction_target_score(
        {
            "direction_accuracy": 0.42,
            "class_rates": {"flat": 0.30, "long": 0.35, "short": 0.35},
            "long_edge_bps": 90.0,
            "short_edge_bps": 80.0,
            "horizon": 48,
        },
        symbol="GAZP",
    )
    bad = direction_target_score(
        {
            "direction_accuracy": 0.42,
            "class_rates": {"flat": 0.30, "long": 0.35, "short": 0.35},
            "long_edge_bps": 90.0,
            "short_edge_bps": -10.0,
            "horizon": 48,
        },
        symbol="GAZP",
    )
    assert good > bad


def test_direction_target_score_penalizes_high_flat():
    balanced = direction_target_score(
        {
            "direction_accuracy": 0.42,
            "class_rates": {"flat": 0.35, "long": 0.33, "short": 0.32},
            "long_edge_bps": 90.0,
            "short_edge_bps": 80.0,
            "horizon": 48,
        },
        symbol="GAZP",
    )
    too_flat = direction_target_score(
        {
            "direction_accuracy": 0.42,
            "class_rates": {"flat": 0.80, "long": 0.10, "short": 0.10},
            "long_edge_bps": 90.0,
            "short_edge_bps": 80.0,
            "horizon": 48,
        },
        symbol="GAZP",
    )
    assert balanced > too_flat


def test_direction_label_feasible():
    from strategy.target_opt import _direction_label_feasible

    assert _direction_label_feasible({"class_rates": {"flat": 0.35, "long": 0.33, "short": 0.32}})
    assert not _direction_label_feasible({"class_rates": {"flat": 0.80, "long": 0.10, "short": 0.10}})


def test_tradfi_country_etfs_supported():
    tickers = parse_tickers("EWJ,EWG,INDA,EFA")
    assert tickers == ["EWJ", "EWG", "INDA", "EFA"]
    for sym in tickers:
        assert is_tradfi_symbol(sym)
        assert not is_crypto_symbol(sym)


def test_tradfi_not_crypto():
    assert is_tradfi_symbol("SPY")
    assert not is_crypto_symbol("SPY")
