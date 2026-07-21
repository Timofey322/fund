"""Tests for rule equity curve export."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def test_to_series_uses_datetime_index():
    from rule.equity_export import _to_series

    idx = pd.bdate_range("2024-06-01", periods=50)
    eq = pd.DataFrame({"value": np.linspace(1, 1.05, 50)}, index=idx)
    s = _to_series(eq)
    assert len(s) == 50
    assert isinstance(s.index, pd.DatetimeIndex)


def test_build_equity_chart_payload_beta():
    from rule.equity_export import build_equity_chart_payload

    idx = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(0)
    strat = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))), index=idx)
    bench = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.009, len(idx))), index=idx)
    eq = pd.DataFrame({"bar_time": strat.index, "value": strat.values})
    bq = pd.DataFrame({"bar_time": bench.index, "value": bench.values})

    payload = build_equity_chart_payload(eq, bq, ticker="EWJ", label="test")
    assert len(payload["series"]["strategy"]["v"]) >= 2
    assert payload["beta"] is not None
    assert payload["return_pct"]["strategy"] is not None


def test_build_equity_chart_payload_uses_prices_fallback():
    from rule.equity_export import build_equity_chart_payload, equal_weight_benchmark_series

    idx = pd.bdate_range("2024-01-01", periods=200)
    strat = pd.Series(np.linspace(1, 1.2, len(idx)), index=idx)
    eq = pd.DataFrame({"value": strat.values}, index=idx)
    px = pd.DataFrame(
        {"A": np.linspace(100, 120, len(idx)), "B": np.linspace(50, 55, len(idx))},
        index=idx,
    )
    payload = build_equity_chart_payload(eq, None, label="test", prices=px)
    assert len(payload["series"]["benchmark"]["v"]) >= 2
    assert payload["return_pct"]["benchmark"] is not None
    assert equal_weight_benchmark_series(px).iloc[-1] > 1.0


def test_export_rule_equity_curves(tmp_path, monkeypatch):
    from rule.config import RULE_INITIAL_CAPITAL_USD
    from rule import equity_export as ee

    monkeypatch.setattr(ee, "OUT_DIR", tmp_path)
    monkeypatch.setattr(ee, "RULE_EQUITY_DIR", tmp_path / "rule" / "equity")

    idx = pd.bdate_range("2024-01-01", periods=80)
    port = pd.DataFrame({"bar_time": idx, "value": np.linspace(1, 1.05, len(idx))})
    bench = pd.DataFrame({"bar_time": idx, "value": np.linspace(1, 1.02, len(idx))})
    per = {
        "SPY": {
            "equity": port.copy(),
            "benchmark": bench.copy(),
        }
    }
    paths = ee.export_rule_equity_curves(port, bench, per)
    assert paths["portfolio"] == "/output/rule/equity/portfolio.json"
    assert paths["SPY"] == "/output/rule/equity/SPY.json"
    loaded = json.loads((tmp_path / "rule" / "equity" / "portfolio.json").read_text(encoding="utf-8"))
    assert loaded["value_mode"] == "portfolio_dollars"
    assert loaded["series"]["strategy"]["v"][0] == RULE_INITIAL_CAPITAL_USD
