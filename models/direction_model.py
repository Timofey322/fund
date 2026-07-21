"""3-class direction models (flat / long / short) and dual-binary comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config as _cfg
from models.entry_model import DEFAULT_LIGHTGBM_PARAMS, make_direction_classifier, make_entry_classifier
from models.per_ticker_models import PerTickerModelBundle, _with_class_weight
from research.labels.trade import (
    DIRECTION_FLAT,
    DIRECTION_LONG,
    DIRECTION_NAMES,
    DIRECTION_SHORT,
    TARGET_DIRECTION,
    direction_class_rates,
)


def dual_binary_predict_direction(
    proba_long: np.ndarray,
    proba_short: np.ndarray,
    *,
    baseline: float = 0.5,
    min_edge: float = 0.05,
) -> np.ndarray:
    """
    Map separate long/short binary probabilities to direction {0,1,2}.

    Long if P(long) >= baseline + min_edge and P(long) > P(short).
    Short if P(short) >= baseline + min_edge and P(short) > P(long).
    Else flat.
    """
    pl = np.asarray(proba_long, dtype=float)
    ps = np.asarray(proba_short, dtype=float)
    base = float(np.clip(baseline, 1e-6, 1 - 1e-6))
    long_edge = pl - base
    short_edge = ps - base
    out = np.full(len(pl), DIRECTION_FLAT, dtype=int)
    long_ok = (long_edge >= min_edge) & (pl > ps)
    short_ok = (short_edge >= min_edge) & (ps > pl)
    out[long_ok] = DIRECTION_LONG
    out[short_ok] = DIRECTION_SHORT
    return out


def _multiclass_sample_weight(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights for flat/long/short."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts[y]
    weights *= len(y) / weights.sum()
    return weights


class PerTickerDirectionBundle:
    """One multiclass direction model per ticker."""

    def __init__(self, model_name: str = "lightgbm"):
        self.model_name = model_name
        self.models: dict[str, Any] = {}
        self.params_by_ticker: dict[str, dict[str, Any]] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        feat_cols: list[str],
        target_col: str = TARGET_DIRECTION,
        *,
        sample_weight: np.ndarray | None = None,
        params: dict[str, Any] | None = None,
        params_by_ticker: dict[str, dict[str, Any]] | None = None,
        min_rows: int = 200,
    ) -> "PerTickerDirectionBundle":
        pbt = params_by_ticker or {}
        default_params = dict(params or DEFAULT_LIGHTGBM_PARAMS)
        tickers = sorted(frame["ticker"].astype(str).str.upper().unique())
        for sym in tickers:
            mask = frame["ticker"].astype(str).str.upper().eq(sym).to_numpy()
            if int(mask.sum()) < min_rows:
                continue
            sub = frame.loc[mask]
            if sub[target_col].nunique() < 2:
                continue
            y = sub[target_col].astype(int).to_numpy()
            w_row = None if sample_weight is None else np.asarray(sample_weight)[mask]
            fold_params = dict(pbt.get(sym) or default_params)
            if bool(getattr(_cfg, "FUSION_CLASS_WEIGHT_AUTO", True)):
                cw = _multiclass_sample_weight(y)
                w_row = cw if w_row is None else cw * w_row
            model = make_direction_classifier(self.model_name, fold_params)
            model.fit(sub[feat_cols], y, sample_weight=w_row)
            self.models[sym] = model
            self.params_by_ticker[sym] = fold_params
        return self

    def predict_proba(self, frame: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
        """Return (n, 3) proba for [flat, long, short]."""
        n = len(frame)
        out = np.full((n, 3), 1.0 / 3.0, dtype=float)
        if not self.models or frame.empty or "ticker" not in frame.columns:
            return out
        tickers = frame["ticker"].astype(str).str.upper()
        for sym, model in self.models.items():
            mask = tickers.eq(sym).to_numpy()
            if not mask.any():
                continue
            out[mask] = model.predict_proba(frame.loc[mask, feat_cols])
        return out

    def predict_direction(self, frame: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
        proba = self.predict_proba(frame, feat_cols)
        return np.argmax(proba, axis=1).astype(int)


def direction_distribution(pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=int)
    if pred.size == 0:
        return {name: 0.0 for name in DIRECTION_NAMES}
    n = len(pred)
    return {
        name: float((pred == code).sum() / n)
        for code, name in zip((0, 1, 2), DIRECTION_NAMES)
    }


def compare_direction_predictions(
    y_true: np.ndarray,
    pred_multiclass: np.ndarray,
    pred_dual: np.ndarray,
) -> dict[str, Any]:
    """Compare true labels vs multiclass and dual-binary predicted directions."""
    y = np.asarray(y_true, dtype=int)
    pm = np.asarray(pred_multiclass, dtype=int)
    pd_ = np.asarray(pred_dual, dtype=int)
    mask = np.isfinite(y) & np.isin(y, (0, 1, 2))
    y, pm, pd_ = y[mask], pm[mask], pd_[mask]
    n = len(y)
    if n == 0:
        return {"n": 0}

    def _acc(a: np.ndarray, b: np.ndarray) -> float:
        return float((a == b).mean())

    confusion_mc = {
        f"true_{DIRECTION_NAMES[t]}": {
            DIRECTION_NAMES[p]: int(((y == t) & (pm == p)).sum())
            for p in range(3)
        }
        for t in range(3)
    }
    confusion_dual = {
        f"true_{DIRECTION_NAMES[t]}": {
            DIRECTION_NAMES[p]: int(((y == t) & (pd_ == p)).sum())
            for p in range(3)
        }
        for t in range(3)
    }
    agree = pm == pd_
    return {
        "n": int(n),
        "true_distribution": direction_class_rates(pd.Series(y)),
        "multiclass_pred_distribution": direction_distribution(pm),
        "dual_binary_pred_distribution": direction_distribution(pd_),
        "multiclass_accuracy": round(_acc(y, pm), 4),
        "dual_binary_accuracy": round(_acc(y, pd_), 4),
        "multiclass_vs_dual_agreement": round(float(agree.mean()), 4),
        "multiclass_confusion": confusion_mc,
        "dual_binary_confusion": confusion_dual,
        "multiclass_long_share": direction_distribution(pm)["long"],
        "dual_binary_long_share": direction_distribution(pd_)["long"],
        "multiclass_short_share": direction_distribution(pm)["short"],
        "dual_binary_short_share": direction_distribution(pd_)["short"],
    }
