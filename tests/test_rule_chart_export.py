"""Tests for rule chart JSON export (client-side SVG)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def test_export_rule_charts_writes_json(tmp_path, monkeypatch):
    import rule.chart_export as ce

    monkeypatch.setattr(ce, "OUT_DIR", tmp_path)
    monkeypatch.setattr(ce, "RULE_CHARTS_PATH", tmp_path / "rule" / "charts.json")

    idx = pd.bdate_range("2024-01-01", periods=180)
    equity = pd.DataFrame({"bar_time": idx, "value": np.linspace(1.0, 1.12, len(idx))})
    report = {
        "generated_at": "2026-01-01T00:00:00Z",
        "backtest": {"total_return_pct": 12.0, "sharpe": 1.1},
        "per_ticker_backtest": {
            "SPY": {"total_return_pct": 2.0, "benchmark_return_pct": 1.0, "excess_return_pct": 1.0, "sharpe": 0.5},
            "EWJ": {"total_return_pct": 3.0, "benchmark_return_pct": 0.5, "excess_return_pct": 2.5, "sharpe": 1.1},
        },
        "per_ticker_signals": [
            {"ticker": "SPY", "pct_dump_buy": 20.0, "pct_rally_sell": 15.0},
            {"ticker": "EWJ", "pct_dump_buy": 22.0, "pct_rally_sell": 14.0},
        ],
    }

    charts = ce.export_rule_charts(report, equity)
    assert "rule_equity_curve" in charts
    assert charts["rule_equity_curve"]["type"] == "line"
    assert ce.RULE_CHARTS_PATH.is_file()

    payload = json.loads(ce.RULE_CHARTS_PATH.read_text(encoding="utf-8"))
    assert payload["charts"]["rule_per_ticker_returns"]["type"] == "grouped_bar"


def test_chart_manifest_plots():
    from rule.chart_export import chart_manifest_plots

    entries = chart_manifest_plots({"rule_equity_curve": {"type": "line", "title": "Equity"}})
    assert entries[0]["id"] == "rule_equity_curve"
    assert entries[0]["path"] == ""
