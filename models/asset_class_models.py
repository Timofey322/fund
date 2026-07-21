"""Separate entry models per asset class (crypto vs tradfi)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config as _cfg
from data_platform.universe import is_crypto_symbol, is_tradfi_symbol
from models.adaptive_trainer import AdaptiveEntryModel
from models.entry_model import make_entry_classifier


def asset_class_of(ticker: str) -> str:
    sym = str(ticker).upper()
    if is_tradfi_symbol(sym):
        return "tradfi"
    if is_crypto_symbol(sym):
        return "crypto"
    return "other"


def asset_class_series(tickers: pd.Series) -> pd.Series:
    return tickers.astype(str).map(asset_class_of)


def asset_class_models_enabled() -> bool:
    from models.per_ticker_models import per_ticker_models_enabled

    if per_ticker_models_enabled():
        return False
    return bool(getattr(_cfg, "FUSION_ASSET_CLASS_MODELS", True))


class AssetClassModelBundle:
    """One classifier (or adaptive model) per asset class."""

    def __init__(self, model_name: str = "lightgbm"):
        self.model_name = model_name
        self.models: dict[str, Any] = {}
        self._adaptive = False

    def fit(
        self,
        frame: pd.DataFrame,
        feat_cols: list[str],
        target_col: str,
        *,
        sample_weight: np.ndarray | None,
        params: dict[str, Any],
        adaptive: bool = False,
        prev: "AssetClassModelBundle | None" = None,
        min_rows: int = 200,
    ) -> "AssetClassModelBundle":
        self._adaptive = adaptive
        classes = asset_class_series(frame["ticker"])
        for cls in sorted(classes.unique()):
            mask = (classes == cls).to_numpy()
            if int(mask.sum()) < min_rows:
                continue
            sub = frame.loc[mask]
            if sub[target_col].nunique() < 2:
                continue
            y = sub[target_col].astype(int)
            w = None if sample_weight is None else np.asarray(sample_weight)[mask]
            prev_model = prev.models.get(cls) if prev is not None else None
            if adaptive:
                model = prev_model if isinstance(prev_model, AdaptiveEntryModel) else AdaptiveEntryModel(self.model_name)
                if prev_model is None or not isinstance(prev_model, AdaptiveEntryModel) or prev_model.booster_ is None:
                    model.fit_initial(sub[feat_cols], y, w, params)
                else:
                    model.fit_incremental(sub[feat_cols], y, w, params)
            else:
                model = make_entry_classifier(self.model_name, params)
                model.fit(sub[feat_cols], y, sample_weight=w)
            self.models[cls] = model
        return self

    def predict_proba(self, frame: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
        """Return (n, 2) proba matrix; missing class → 0.5."""
        n = len(frame)
        out = np.full((n, 2), 0.5, dtype=float)
        if not self.models or frame.empty:
            return out
        classes = asset_class_series(frame["ticker"])
        for cls, model in self.models.items():
            mask = (classes == cls).to_numpy()
            if not mask.any():
                continue
            proba = model.predict_proba(frame.loc[mask, feat_cols])
            out[mask] = proba
        return out

    def training_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"classes": sorted(self.models), "adaptive": self._adaptive}
        for cls, model in self.models.items():
            if hasattr(model, "training_state"):
                state[cls] = model.training_state()
            else:
                state[cls] = {"type": type(model).__name__}
        return state

    @property
    def feature_importances_(self) -> np.ndarray:
        imps = [np.asarray(m.feature_importances_) for m in self.models.values() if hasattr(m, "feature_importances_")]
        if not imps:
            return np.array([])
        stacked = np.vstack(imps)
        return stacked.mean(axis=0)
