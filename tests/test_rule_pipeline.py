"""Smoke tests for rule (non-ML) strategy pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def test_rule_config_paths():
    from rule.config import RULE_DEFAULT_TICKERS, RULE_NAME, RULE_REPORT_PATH

    assert RULE_NAME == "hmm_nw_buy"
    assert len(RULE_DEFAULT_TICKERS) >= 1
    assert "SPY" in RULE_DEFAULT_TICKERS
    assert RULE_REPORT_PATH.name == "rule_pipeline_report.json"


def test_rule_agents_chain():
    from rule.agents import DEFAULT_RULE_PIPELINE

    names = [a.name for a in DEFAULT_RULE_PIPELINE]
    assert names == ["data", "rule", "rule_plot"]


def test_build_signals_on_synthetic_prices(monkeypatch):
    """HMM exhaustion pipeline produces scored rows on synthetic OHLCV."""
    import numpy as np
    from rule.pipeline import _build_signals

    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        {"SPY": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx)))},
        index=idx,
    )
    monkeypatch.setattr("rule.pipeline.build_nw_hmm_buy_signal_frame", lambda p: _fake_signals(p))
    sig = _build_signals(prices)
    assert not sig.empty
    assert "score" in sig.columns
    assert sig["score"].between(0, 100).all()


def _fake_signals(prices: pd.DataFrame) -> pd.DataFrame:
  rows = []
  for col in prices.columns:
    for dt, close in prices[col].dropna().items():
      rows.append({"date": dt, "ticker": col, "close": close, "vol_ann": 0.15, "score": 55.0})
  return pd.DataFrame(rows)


def test_run_rule_pipeline_writes_report(tmp_path, monkeypatch):
    import numpy as np
    import rule.pipeline as rp

    report_path = tmp_path / "rule_pipeline_report.json"
    equity_path = tmp_path / "rule_bt_equity.parquet"
    summary_path = tmp_path / "rule_summary.md"
    mc_path = tmp_path / "rule_mc.json"

    monkeypatch.setattr(rp, "RULE_REPORT_PATH", report_path)
    monkeypatch.setattr(rp, "RULE_EQUITY_CACHE", equity_path)
    monkeypatch.setattr(rp, "RULE_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(rp, "RULE_MONTE_CARLO_PATH", mc_path)

    idx = pd.bdate_range("2019-01-01", periods=400)
    prices = pd.DataFrame({"SPY": 100.0 * (1.0002 ** np.arange(len(idx)))}, index=idx)

    monkeypatch.setattr(rp, "load_closes", lambda tickers, tf: prices)
    monkeypatch.setattr(
        rp,
        "run_portfolio_backtest",
        lambda prices, signals, tickers: {
            "stats": {
                "total_return_pct": 1.5,
                "sharpe": 0.8,
                "max_drawdown_pct": -3.0,
                "period_start": idx[0],
                "period_end": idx[-1],
            },
            "equity": pd.DataFrame({"bar_time": idx[:50], "value": np.linspace(1, 1.01, 50)}),
        },
    )
    monkeypatch.setattr(
        rp,
        "run_per_ticker_backtests",
        lambda prices, signals, tickers, include_equity=False: (
            (
                {
                    "SPY": {
                        "total_return_pct": 1.5,
                        "sharpe": 0.8,
                        "benchmark_return_pct": 1.0,
                        "excess_return_pct": 0.5,
                        "n_signals": 10,
                    }
                },
                {
                    "SPY": {
                        "equity": pd.DataFrame({"bar_time": idx[:50], "value": np.linspace(1, 1.01, 50)}),
                        "benchmark": pd.DataFrame({"bar_time": idx[:50], "value": np.linspace(1, 1.005, 50)}),
                    }
                },
            )
            if include_equity
            else {
                "SPY": {
                    "total_return_pct": 1.5,
                    "sharpe": 0.8,
                    "benchmark_return_pct": 1.0,
                    "excess_return_pct": 0.5,
                    "n_signals": 10,
                }
            }
        ),
    )
    monkeypatch.setattr(rp, "export_rule_web", lambda report: tmp_path / "rule_manifest.json")
    monkeypatch.setattr(
        rp,
        "survival_simulation",
        lambda eq, **kw: {"survival_rate": 0.9, "prob_terminal_loss": 0.1},
    )
    monkeypatch.setattr(rp, "_build_signals", lambda p: _fake_signals(p))

    from rule.pipeline import run_rule_pipeline

    report = run_rule_pipeline(["SPY"])
    assert report["strategy"] == "hmm_nw_buy"
    assert report_path.is_file()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["backtest"]["total_return_pct"] == 1.5
