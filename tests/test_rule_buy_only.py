"""Buy-only rule backtest: never sell after entry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rule.config import RULE_SCORE_ENTER
from simulation.engine import run_backtest_signal_exit


def _run(prices: pd.DataFrame, sig: pd.DataFrame, *, buy_only: bool) -> dict:
    return run_backtest_signal_exit(
        prices,
        sig,
        score_col="score",
        use_dynamic_thresholds=False,
        use_vol_targeting=False,
        equal_weight=True,
        commission_bps=0.0,
        horizon_bars=10_000,
        stop_loss_bps=0.0,
        min_hold_bars=0,
        rebalance_band=0.0,
        partial_sell_frac=0.0,
        keep_holdings_on_empty_picks=True,
        merge_new_picks_only=True,
        buy_only=buy_only,
    )


def test_buy_only_holds_while_legacy_sells_on_risk_off():
    idx = pd.bdate_range("2020-01-01", periods=120)
    px = 100.0 * (1 + np.linspace(-0.15, 0.05, len(idx)))
    prices = pd.DataFrame({"AAA": px}, index=idx)
    rows = []
    for i, (dt, close) in enumerate(prices["AAA"].items()):
        if i < 15:
            score, risk_on = 60.0, True
        else:
            score, risk_on = 45.0, False  # legacy: risk_off + score < buy → exit
        rows.append({
            "date": dt,
            "ticker": "AAA",
            "close": float(close),
            "vol_ann": 0.20,
            "score": score,
            "risk_on": risk_on,
        })
    sig = pd.DataFrame(rows)

    legacy = _run(prices, sig, buy_only=False)
    held = _run(prices, sig, buy_only=True)
    leg_exp = (legacy.get("stats") or {}).get("avg_exposure_pct", 0)
    buy_exp = (held.get("stats") or {}).get("avg_exposure_pct", 0)
    assert (held.get("stats") or {}).get("signal_exit_count", 0) == 0
    assert (held.get("stats") or {}).get("buy_only") is True
    assert buy_exp > leg_exp + 10.0


def test_buy_only_tracks_entry_discount_vs_twap():
    idx = pd.bdate_range("2020-01-01", periods=200)
    px = 100.0 + np.concatenate([
        np.linspace(0, 10, 80),
        np.linspace(10, -30, 40),
        np.linspace(-30, 15, 80),
    ])
    prices = pd.DataFrame({"AAA": px}, index=idx)
    s = prices["AAA"]
    floor = float(s.quantile(0.25))
    rows = []
    for dt, close in s.items():
        rows.append({
            "date": dt,
            "ticker": "AAA",
            "close": float(close),
            "vol_ann": 0.20,
            "score": float(RULE_SCORE_ENTER) + 5 if float(close) <= floor else 45.0,
            "risk_on": True,
        })
    sig = pd.DataFrame(rows)

    bt = _run(prices, sig, buy_only=True)
    discount = (bt.get("stats") or {}).get("avg_entry_discount_vs_twap_pct")
    assert discount is not None
    assert discount > 0.0
