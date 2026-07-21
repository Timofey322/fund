"""Tests for quant desk reporting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reporting.desk_reports import (
    _trade_analytics,
    build_desk_risk_summary,
    build_walk_forward_attribution_table,
    plot_monthly_returns_heatmap,
    plot_rolling_sharpe,
    plot_trade_pnl_distribution,
    plot_underwater_drawdown,
)


def test_trade_analytics_expectancy():
    rets = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
    out = _trade_analytics(rets)
    assert out["n_trades"] == 5
    assert out["win_rate"] == pytest.approx(0.6)
    assert out["expectancy_bps"] is not None
    assert out["profit_factor"] is not None


def test_trade_analytics_empty():
    out = _trade_analytics(np.array([]))
    assert out["n_trades"] == 0
    assert out["win_rate"] is None


def test_build_walk_forward_attribution_merges_oos_metrics():
    report = {
        "tickers": ["BTC", "ETH"],
        "walk_forward_folds": [
            {
                "fold": 1,
                "test_start": "2024-01-01",
                "test_end": "2024-03-31",
                "skipped": False,
                "oos_metrics": {"auc": 0.56, "log_loss": 0.68},
                "trading_policy": {"buy_threshold": 0.55, "min_expected_edge_bps": 12},
                "threshold_optimization": {"cv": {"objective": -2.1}},
            },
        ],
    }
    anomaly = [{"fold": 1, "equity_return_pct": -3.2, "n_trades": 40, "avg_net_bps": -5.0}]
    rows = build_walk_forward_attribution_table(report, None, fold_anomaly_rows=anomaly)
    assert len(rows) == 1
    assert rows[0]["oos_auc"] == 0.56
    assert rows[0]["strategy_return_pct"] == -3.2
    assert rows[0]["policy_objective"] == -2.1


def test_build_desk_risk_summary_from_equity():
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    rets = np.random.default_rng(0).normal(0, 0.001, 100)
    vals = 10_000 * np.cumprod(1 + rets)
    eq = pd.DataFrame({"value": vals}, index=idx)
    trade_rets = np.array([0.01, -0.005, 0.008])
    risk = build_desk_risk_summary(eq, trade_rets, {"total_return_pct": 5.0, "sharpe": 1.2})
    assert risk["n_bars"] == 100
    assert risk["trade_analytics"]["n_trades"] == 3
    assert risk.get("realized_vol_annualized_pct") is not None


@pytest.mark.parametrize("plot_fn,kwargs", [
    (plot_rolling_sharpe, {}),
    (plot_underwater_drawdown, {}),
    (plot_trade_pnl_distribution, {"trade_returns": np.array([0.01, -0.005, 0.02])}),
    (plot_monthly_returns_heatmap, {}),
])
def test_desk_plots_write_file(plot_fn, kwargs, tmp_path):
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    vals = 10_000 * (1 + np.random.default_rng(1).normal(0, 0.0005, 500)).cumprod()
    eq = pd.DataFrame({"value": vals}, index=idx)
    out = tmp_path / f"{plot_fn.__name__}.png"
    if plot_fn is plot_trade_pnl_distribution:
        plot_fn(kwargs["trade_returns"], out)
    else:
        plot_fn(eq, out)
    assert out.is_file()
    assert out.stat().st_size > 100
