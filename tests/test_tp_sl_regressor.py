"""Tests for per-ticker TP/SL regressor bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.tp_sl_regressor import PerTickerTPSLRegressorBundle
from research.labels.balanced import TARGET_SL_BPS, TARGET_TP_BPS


def _panel(n: int = 300, tickers: tuple[str, ...] = ("SPY", "EFA")) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for sym in tickers:
        for i in range(n):
            rows.append(
                {
                    "ticker": sym,
                    "f1": float(rng.normal()),
                    "f2": float(rng.normal()),
                    TARGET_TP_BPS: float(rng.uniform(20, 80)),
                    TARGET_SL_BPS: float(rng.uniform(15, 60)),
                }
            )
    return pd.DataFrame(rows)


def test_per_ticker_tp_sl_regressor_fit_predict():
    frame = _panel()
    feat_cols = ["f1", "f2"]
    bundle = PerTickerTPSLRegressorBundle()
    bundle.fit(frame, feat_cols, min_rows=50)
    assert "SPY" in bundle.tp_models
    assert "EFA" in bundle.sl_models
    tp, sl = bundle.predict(frame, feat_cols)
    assert len(tp) == len(frame)
    assert len(sl) == len(frame)
    assert np.isfinite(tp).sum() > 0
    assert np.isfinite(sl).sum() > 0
    assert (tp[np.isfinite(tp)] >= 5).all()
    assert (sl[np.isfinite(sl)] >= 5).all()
