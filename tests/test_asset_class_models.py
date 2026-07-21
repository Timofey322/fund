"""Asset-class model bundle tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.asset_class_models import (
    AssetClassModelBundle,
    asset_class_of,
    asset_class_series,
)


def test_asset_class_of():
    assert asset_class_of("BTC") == "crypto"
    assert asset_class_of("SPY") == "tradfi"


def test_bundle_fits_separate_classes():
    rng = np.random.default_rng(0)
    n = 800
    tickers = ["BTC"] * (n // 2) + ["SPY"] * (n // 2)
    # Different signal per class
    proba_signal = np.concatenate([
        rng.uniform(0.2, 0.9, n // 2),
        rng.uniform(0.2, 0.9, n // 2),
    ])
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "f1": proba_signal,
            "f2": rng.normal(0, 1, n),
            "label": (proba_signal > 0.55).astype(int),
        }
    )
    # Make SPY label depend on f2 instead of f1
    spy = df["ticker"] == "SPY"
    df.loc[spy, "label"] = (df.loc[spy, "f2"] > 0).astype(int)

    from models.entry_model import DEFAULT_LIGHTGBM_PARAMS

    bundle = AssetClassModelBundle("lightgbm")
    params = {**DEFAULT_LIGHTGBM_PARAMS, "n_estimators": 30, "min_child_samples": 20}
    bundle.fit(df, ["f1", "f2"], "label", sample_weight=None, params=params, min_rows=100)
    assert "crypto" in bundle.models
    assert "tradfi" in bundle.models
    proba = bundle.predict_proba(df, ["f1", "f2"])
    assert proba.shape == (n, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_asset_class_series():
    s = asset_class_series(pd.Series(["BTC", "ETH", "SPY"]))
    assert list(s) == ["crypto", "crypto", "tradfi"]
