"""Per-ticker Optuna threshold policy tests."""

from __future__ import annotations

import pandas as pd

from strategy.threshold_opt import (
    _policy_best_params,
    optimize_trading_policy_per_ticker_on_train,
)


def test_policy_best_params_preserves_by_ticker():
    finalized = {
        "buy_threshold": 52,
        "sell_threshold": 48,
        "hold_threshold": 49,
        "min_expected_edge_bps": 5.0,
        "by_ticker": {
            "GAZP": {"buy_threshold": 54, "sell_threshold": 46},
        },
        "threshold_calibrator": True,
    }
    bp = _policy_best_params(finalized)
    assert bp["buy_threshold"] == 52
    assert "GAZP" in bp["by_ticker"]
    assert bp["threshold_calibrator"] is True


def test_per_ticker_optuna_merges_instrument_policies(monkeypatch):
    import strategy.threshold_opt as th

    def _fake_single(train, prices, *, commission_bps, fold_meta=None, ticker=None, n_trials=None):
        sym = str(ticker or "X")
        buy = 52 if sym == "GAZP" else 55
        return {
            "optimizer": "optuna",
            "best_params": {
                "buy_threshold": buy,
                "sell_threshold": 100 - buy,
                "hold_threshold": 49,
                "min_expected_edge_bps": 4.0,
                "impulse_min": 0.05,
                "gain": 100,
                "w_ml": 0.45,
                "stop_loss_bps": 45.0,
                "edge_floor_mode": "commission_only",
                "disable_trading": False,
            },
            "cv": {
                "objective": 0.5,
                "signal_rows": 12,
                "trade_anomaly": False,
            },
            "trial_results": [{"objective": 0.5}],
        }

    monkeypatch.setattr(th, "optimize_trading_policy_on_train", _fake_single)
    monkeypatch.setattr(th, "_calibrate_edge_on_train", lambda train, **kw: 4.0)
    import config as cfg

    monkeypatch.setattr(cfg, "FUSION_THRESHOLD_OPT_MIN_TRAIN_ROWS_PER_TICKER", 2)
    monkeypatch.setattr(cfg, "FUSION_THRESHOLD_OPT_MIN_TRAIN_SESSIONS_PER_TICKER", 1)

    train = pd.DataFrame(
        {
            "ml_proba": [0.6, 0.4, 0.7, 0.3],
            "ml_proba_short": [0.3, 0.6, 0.2, 0.7],
            "ticker": ["GAZP", "GAZP", "NASDAQ", "NASDAQ"],
            "session": ["s1", "s1", "s2", "s2"],
            "bar_time": pd.date_range("2024-01-01", periods=4, freq="h"),
            "label": [1, 0, 1, 0],
        }
    )
    out = optimize_trading_policy_per_ticker_on_train(
        train,
        pd.DataFrame(),
        commission_bps=1.1,
        fold_meta={"fold": 0},
    )
    bp = out["best_params"]
    assert bp.get("by_ticker")
    assert bp["by_ticker"]["GAZP"]["buy_threshold"] == 52
    assert bp["by_ticker"]["NASDAQ"]["buy_threshold"] == 55
    assert out["optimizer"] == "optuna_per_ticker"
