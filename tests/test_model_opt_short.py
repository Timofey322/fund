"""Per-ticker Optuna returns short_params when short optimization is on."""

from __future__ import annotations

import pandas as pd


def test_per_ticker_opt_includes_short_params(monkeypatch):
    import models.model_per_ticker_opt as mod

    calls: list[str] = []

    def fake_opt(train, feat_cols, target_col, **kwargs):
        calls.append(str(target_col))
        sym = (kwargs.get("fold_meta") or {}).get("ticker", "X")
        return {
            "model_name": "lightgbm",
            "model_params": {"num_leaves": 11 if "short" not in target_col else 7},
            "cv": {"composite": 0.5, "top_decile_net_bps": 6.0},
            "n_trials": 3,
            "ticker": sym,
        }

    monkeypatch.setattr(
        "strategy.pipeline.optimize_fusion_model_on_train_slice",
        fake_opt,
    )
    monkeypatch.setattr("config.FUSION_OPTUNA_SHORT_MODELS", True)
    monkeypatch.setattr("config.FUSION_ALLOW_SHORT", True)
    monkeypatch.setattr("config.FUSION_PER_TICKER_MIN_ROWS", 2)

    train = pd.DataFrame({
        "ticker": ["GAZP"] * 4 + ["NASDAQ"] * 4,
        "label_entry": [0, 1, 0, 1] * 2,
        "label_entry_short": [1, 0, 1, 0] * 2,
        "f1": [0.1, 0.2, 0.3, 0.4] * 2,
    })
    out = mod.optimize_fusion_model_per_ticker_on_train_slice(
        train, ["f1"], "label_entry", n_trials=3,
    )
    assert "label_entry" in calls
    assert "label_entry_short" in calls
    assert out["short_params_by_ticker"]["GAZP"]["num_leaves"] == 7
    assert out["per_ticker"]["GAZP"]["short_model_params"]["num_leaves"] == 7
