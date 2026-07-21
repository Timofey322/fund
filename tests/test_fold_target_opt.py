"""Per-fold target optimization in walk-forward backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.labels.trade import ENTRY_LABEL_HORIZON, TARGET_ENTRY, attach_economic_entry_labels


def _mini_panel(n: int = 400, ticker: str = "GAZP") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0001, 0.002, n))
    base = pd.DataFrame(
        {
            "bar_time": idx,
            "session": idx.normalize(),
            "ticker": ticker,
            "close": close,
        }
    )
    return attach_economic_entry_labels(
        base,
        symbol=ticker,
        spec={"horizon": 12, "label_type": "triple_barrier", "threshold_bps": 40.0},
    )


def test_specs_dict_from_optimization_extracts_specs():
    from strategy.target_opt import specs_dict_from_optimization

    result = {
        "per_symbol": {
            "GAZP": {"spec": {"horizon": 48, "label_type": "triple_barrier", "threshold_bps": 50.0}},
            "IMOEX": {"spec": {"horizon": 24, "label_type": "after_costs", "threshold_bps": 30.0}},
        }
    }
    specs = specs_dict_from_optimization(result)
    assert specs["GAZP"]["horizon"] == 48
    assert specs["IMOEX"]["label_type"] == "after_costs"


def test_relabel_panel_entry_targets_updates_horizon():
    from strategy.target_opt import relabel_panel_entry_targets

    panel = _mini_panel()
    assert int(panel[ENTRY_LABEL_HORIZON].iloc[0]) == 12
    out = relabel_panel_entry_targets(
        panel,
        {"GAZP": {"horizon": 72, "label_type": "triple_barrier", "threshold_bps": 55.0}},
    )
    assert int(out[ENTRY_LABEL_HORIZON].iloc[0]) == 72
    assert TARGET_ENTRY in out.columns
    assert out[TARGET_ENTRY].notna().any()


def test_max_horizon_from_specs():
    from strategy.target_opt import max_horizon_from_specs

    specs = {
        "GAZP": {"horizon": 48},
        "NASDAQ": {"horizon": 96},
    }
    assert max_horizon_from_specs(specs, 12) == 96
    assert max_horizon_from_specs({}, 24) == 24


def test_optimize_targets_on_fold_train_skips_tiny_train(monkeypatch):
    from strategy.target_opt import optimize_targets_on_fold_train

    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        return {"per_symbol": {}}

    monkeypatch.setattr("strategy.target_opt.optimize_targets_per_instrument", _boom)
    panel = _mini_panel(n=50)
    out = optimize_targets_on_fold_train(panel, ["f1"], ["GAZP"], fold_meta={"fold": 0})
    assert out["skipped"] is True
    assert called["n"] == 0


def test_optimize_targets_on_fold_train_runs_on_train_only(monkeypatch):
    from strategy.target_opt import optimize_targets_on_fold_train

    seen: dict = {}

    def _fake_grid(train, symbols, **kwargs):
        seen["train_len"] = len(train)
        seen["symbols"] = list(symbols)
        seen["kwargs"] = kwargs
        return {
            "per_symbol": {
                "GAZP": {
                    "spec": {"horizon": 48, "label_type": "triple_barrier", "threshold_bps": 45.0},
                    "tradeable": True,
                }
            }
        }

    monkeypatch.setattr("strategy.target_opt.optimize_targets_per_instrument", _fake_grid)
    panel = _mini_panel(n=6000)
    out = optimize_targets_on_fold_train(
        panel,
        ["f0", "f1"],
        ["GAZP"],
        fold_meta={"fold": 1},
    )
    assert out["skipped"] is False
    assert out["optimizer"] == "fold_target_grid"
    assert seen["train_len"] == 6000
    assert seen["symbols"] == ["GAZP"]
    assert "horizons" in seen["kwargs"] or seen["kwargs"].get("horizons") is not None


def test_fold_target_opt_caps_horizon_grid(monkeypatch):
    import config as cfg
    from strategy.target_opt import optimize_targets_on_fold_train

    seen: dict = {}

    def _fake_grid(_train, _symbols, **kwargs):
        seen["horizons"] = kwargs.get("horizons")
        return {"per_symbol": {}}

    monkeypatch.setattr("strategy.target_opt.optimize_targets_per_instrument", _fake_grid)
    monkeypatch.setattr(cfg, "FUSION_FOLD_TARGET_OPT_MAX_HORIZON_BARS", 72)
    panel = _mini_panel(n=6000)
    optimize_targets_on_fold_train(panel, ["f0", "f1"], ["SP500"], fold_meta={"fold": 1})
    assert seen["horizons"] == (24, 48, 72)
