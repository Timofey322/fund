"""Ticker research bundle tests."""

from __future__ import annotations

import json

from reporting.ticker_research import build_ticker_research, write_ticker_research


def test_build_ticker_research_minimal():
    fusion = {
        "tickers": ["BTC", "SPY"],
        "decile_audit": {
            "by_ticker": {
                "BTC": {"top_decile_net_bps": -5.0, "tradeable": False},
                "SPY": {"top_decile_net_bps": 6.0, "tradeable": True},
            }
        },
        "per_ticker_backtest": {
            "BTC": {"total_return_pct": -3.0, "sharpe": -0.5},
            "SPY": {"total_return_pct": 2.0, "sharpe": 0.4},
        },
        "go_no_go": {"tradeable_tickers": ["SPY"]},
        "walk_forward_folds": [],
    }
    btc = build_ticker_research("BTC", fusion)
    assert btc["ticker"] == "BTC"
    assert btc["asset_class"] == "crypto"
    assert btc["verdict"] in ("SKIP", "WATCH", "TRADE")

    spy = build_ticker_research("SPY", fusion)
    assert spy["tradeable"] is True


def test_write_ticker_research_files(tmp_path, monkeypatch):
    import reporting.ticker_research as tr

    monkeypatch.setattr(tr, "OUT_DIR", tmp_path)
    monkeypatch.setattr(tr, "RESEARCH_DIR", tmp_path / "research")
    fusion = {
        "tickers": ["ETH"],
        "decile_audit": {"by_ticker": {"ETH": {"top_decile_net_bps": 1.0}}},
        "per_ticker_backtest": {"ETH": {}},
        "go_no_go": {},
        "walk_forward_folds": [],
    }
    jp, hp = write_ticker_research("ETH", fusion)
    assert jp.is_file()
    assert hp.is_file()
    payload = json.loads(jp.read_text(encoding="utf-8"))
    assert payload["ticker"] == "ETH"
