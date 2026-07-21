"""Per-fold diagnostics tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.fold_diagnostics import _bottleneck, diagnose_fold_slice


def test_bottleneck_gross_below_cost():
    assert _bottleneck({"net": -1}, {"net": -3, "gross": 2.0}, rt_cost=5.2) == "gross<5.2_cost"


def test_bottleneck_train_ok_oos_fail():
    assert _bottleneck({"net": 2.0}, {"net": -1.0, "gross": 6.0}, rt_cost=5.2) == "train_ok_oos_fail"


def test_diagnose_fold_slice_per_ticker():
    n = 300
    proba = np.linspace(0.1, 0.9, n)
    fwd = np.where(proba > np.quantile(proba, 0.9), 0.0002, -0.0005)
    train = pd.DataFrame(
        {"ticker": ["BTC"] * n, "ml_proba": proba, "fwd_ret": fwd}
    )
    oos = train.copy()
    rep = diagnose_fold_slice(train, oos, fold=0, min_rows=50)
    assert "BTC" in rep["tickers"]
    assert rep["tickers"]["BTC"]["bottleneck"] in (
        "gross<5.2_cost",
        "no_edge_train_and_oos",
        "oos_negative",
        "oos_below_gate",
    )
