"""Per-ticker model bundle tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.per_ticker_models import PerTickerModelBundle, per_ticker_models_enabled


def test_per_ticker_models_enabled_default():
    assert per_ticker_models_enabled() is True


def test_bundle_fits_separate_tickers():
    rng = np.random.default_rng(1)
    n = 600
    tickers = ["BTC"] * (n // 2) + ["ETH"] * (n // 2)
    signal = np.concatenate([rng.uniform(0.2, 0.9, n // 2), rng.uniform(0.2, 0.9, n // 2)])
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "f1": signal,
            "f2": rng.normal(0, 1, n),
            "label": (signal > 0.55).astype(int),
        }
    )
    eth = df["ticker"] == "ETH"
    df.loc[eth, "label"] = (df.loc[eth, "f2"] > 0).astype(int)

    from models.entry_model import DEFAULT_LIGHTGBM_PARAMS

    bundle = PerTickerModelBundle("lightgbm")
    params = {**DEFAULT_LIGHTGBM_PARAMS, "n_estimators": 20, "min_child_samples": 15}
    bundle.fit(df, ["f1", "f2"], "label", sample_weight=None, params=params, min_rows=100)
    assert "BTC" in bundle.models
    assert "ETH" in bundle.models
    proba = bundle.predict_proba(df, ["f1", "f2"])
    assert proba.shape == (n, 2)


def test_gap_penalty_reduces_composite():
    from models.model_selection import aggregate_cv_metrics

    rows = [
        {"top_decile_net_bps": 10.0, "auc": 0.6, "train_oos_gap_bps": 20.0},
        {"top_decile_net_bps": 8.0, "auc": 0.58, "train_oos_gap_bps": 15.0},
    ]
    agg = aggregate_cv_metrics(rows)
    assert agg.get("train_oos_gap_bps") is not None
    assert agg.get("gap_penalty", 0) > 0
