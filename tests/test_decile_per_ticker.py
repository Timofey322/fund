"""Tests for per-ticker decile audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.diagnostics.decile_audit import (
    decile_audit_by_ticker,
    tradeable_tickers_from_audit,
)


def _panel(ticker: str, edge_bps: float) -> pd.DataFrame:
    n = 200
    proba = np.linspace(0.1, 0.9, n)
    fwd = np.where(proba > np.quantile(proba, 0.9), edge_bps / 10_000.0, -0.0005)
    return pd.DataFrame({"ticker": ticker, "ml_proba": proba, "fwd_ret": fwd})


def test_tradeable_tickers_per_symbol():
    good = _panel("BTC", 40.0)
    bad = _panel("SPY", 2.0)
    df = pd.concat([good, bad], ignore_index=True)
    by_t = decile_audit_by_ticker(df, ret_col="fwd_ret", min_top_decile_net_bps=5.0)
    tradeable = tradeable_tickers_from_audit(by_t, min_top_decile_net_bps=5.0)
    assert "BTC" in tradeable
    assert "SPY" not in tradeable


def test_per_ticker_audit_uses_ticker_slippage():
    from simulation.execution_costs import slippage_bps_per_side

    df = _panel("SPY", 40.0)
    by_t = decile_audit_by_ticker(df, ret_col="fwd_ret", min_top_decile_net_bps=5.0)
    assert by_t["SPY"]["slippage_bps_per_side"] == slippage_bps_per_side("SPY")
    # Net must use tradfi slip (2.0), not crypto default (1.5).
    assert by_t["SPY"]["slippage_bps_per_side"] == 2.0
