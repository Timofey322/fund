"""Fold-history decile gate tests."""

from __future__ import annotations

from research.diagnostics.decile_audit import (
    resolve_tradeable_tickers,
    tradeable_tickers_from_wf_folds,
)


def _fold(ticker: str, oos_net: float, bottleneck: str) -> dict:
    return {
        "skipped": False,
        "fold_diagnostics": {
            "tickers": {
                ticker: {
                    "oos": {"net": oos_net},
                    "bottleneck": bottleneck,
                }
            }
        },
    }


def test_fold_history_gate_picks_ok_tickers():
    folds = [
        _fold("SOL", 7.0, "ok"),
        _fold("SOL", 6.5, "ok"),
        _fold("ETH", -2.0, "gross<5.2_cost"),
    ]
    syms = tradeable_tickers_from_wf_folds(folds, min_ok_folds=2, recent_folds=0)
    assert "SOL" in syms
    assert "ETH" not in syms


def test_fold_history_all_folds_not_just_recent():
    folds = [_fold("SOL", 7.0, "ok")] * 2 + [_fold("SOL", -1.0, "gross<5.2_cost")] * 10
    syms = tradeable_tickers_from_wf_folds(folds, min_ok_folds=2, recent_folds=0)
    assert "SOL" in syms
    syms_recent = tradeable_tickers_from_wf_folds(folds, min_ok_folds=2, recent_folds=8)
    assert "SOL" not in syms_recent


def test_resolve_tradeable_strict_blocks_negative_stitched():
    by_ticker = {
        "BTC": {"tradeable": False, "top_decile_net_bps": -3.0},
        "SOL": {"tradeable": False, "top_decile_net_bps": 2.0},
    }
    folds = [_fold("BTC", 7.0, "ok"), _fold("BTC", 8.0, "ok")]
    syms, source = resolve_tradeable_tickers(by_ticker, folds, mode="strict")
    assert syms == []
    assert source == "strict_blocked"


def test_resolve_tradeable_strict_allows_positive_stitched_fold():
    by_ticker = {
        "SOL": {"tradeable": False, "top_decile_net_bps": 1.5},
    }
    folds = [_fold("SOL", 7.0, "ok"), _fold("SOL", 8.0, "ok")]
    syms, source = resolve_tradeable_tickers(by_ticker, folds, mode="strict")
    assert "SOL" in syms
    assert source == "strict_fold_aligned"


def test_resolve_tradeable_union():
    by_ticker = {
        "BTC": {"tradeable": False, "top_decile_net_bps": -1.0},
        "SOL": {"tradeable": False, "top_decile_net_bps": 2.0},
    }
    folds = [_fold("SOL", 7.0, "ok"), _fold("SOL", 8.0, "ok")]
    syms, source = resolve_tradeable_tickers(by_ticker, folds, mode="union")
    assert "SOL" in syms
    assert source in ("union", "fold_history_only")
