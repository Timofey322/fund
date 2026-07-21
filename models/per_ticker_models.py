"""Separate entry models per ticker (symbol)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config as _cfg
from models.adaptive_trainer import AdaptiveEntryModel
from models.entry_model import make_entry_classifier


def per_ticker_models_enabled() -> bool:
    return bool(getattr(_cfg, "FUSION_PER_TICKER_MODELS", True))


def multi_symbol_models_enabled(frame: pd.DataFrame) -> bool:
    """True when panel uses per-ticker or asset-class bundles."""
    if "ticker" not in frame.columns:
        return False
    from models.asset_class_models import asset_class_models_enabled

    return per_ticker_models_enabled() or asset_class_models_enabled()


def _with_class_weight(params: dict[str, Any], y: pd.Series | np.ndarray) -> dict[str, Any]:
    """Add LightGBM ``scale_pos_weight`` when train labels are naturally imbalanced."""
    out = dict(params)
    if not bool(getattr(_cfg, "FUSION_CLASS_WEIGHT_AUTO", True)):
        return out
    if "scale_pos_weight" in out:
        return out
    pos = float(np.asarray(y, dtype=float).mean())
    if pos <= 0.0 or pos >= 1.0:
        return out
    out["scale_pos_weight"] = (1.0 - pos) / max(pos, 1e-6)
    return out


class PerTickerModelBundle:
    """One classifier (or adaptive model) per ticker."""

    def __init__(self, model_name: str = "lightgbm"):
        self.model_name = model_name
        self.models: dict[str, Any] = {}
        self.short_models: dict[str, Any] = {}
        self.params_by_ticker: dict[str, dict[str, Any]] = {}
        self._adaptive = False

    def fit(
        self,
        frame: pd.DataFrame,
        feat_cols: list[str],
        target_col: str,
        *,
        sample_weight: np.ndarray | None,
        params: dict[str, Any] | None = None,
        params_by_ticker: dict[str, dict[str, Any]] | None = None,
        adaptive: bool = False,
        prev: "PerTickerModelBundle | None" = None,
        min_rows: int = 200,
    ) -> "PerTickerModelBundle":
        self._adaptive = adaptive
        pbt = params_by_ticker or {}
        default_params = dict(params or {})
        tickers = sorted(frame["ticker"].astype(str).str.upper().unique())
        for sym in tickers:
            mask = frame["ticker"].astype(str).str.upper().eq(sym).to_numpy()
            if int(mask.sum()) < min_rows:
                continue
            sub = frame.loc[mask]
            if sub[target_col].nunique() < 2:
                continue
            y = sub[target_col].astype(int)
            w = None if sample_weight is None else np.asarray(sample_weight)[mask]
            fold_params = _with_class_weight(dict(pbt.get(sym) or default_params), y)
            self.params_by_ticker[sym] = fold_params
            prev_model = prev.models.get(sym) if prev is not None else None
            if adaptive:
                model = (
                    prev_model
                    if isinstance(prev_model, AdaptiveEntryModel)
                    else AdaptiveEntryModel(self.model_name)
                )
                if (
                    prev_model is None
                    or not isinstance(prev_model, AdaptiveEntryModel)
                    or prev_model.booster_ is None
                ):
                    model.fit_initial(sub[feat_cols], y, w, fold_params)
                else:
                    model.fit_incremental(sub[feat_cols], y, w, fold_params)
            else:
                model = make_entry_classifier(self.model_name, fold_params)
                model.fit(sub[feat_cols], y, sample_weight=w)
            self.models[sym] = model
        return self

    def fit_short(
        self,
        frame: pd.DataFrame,
        feat_cols: list[str],
        target_col: str = "label_entry_short",
        *,
        sample_weight: np.ndarray | None,
        params: dict[str, Any] | None = None,
        params_by_ticker: dict[str, dict[str, Any]] | None = None,
        min_rows: int = 200,
    ) -> "PerTickerModelBundle":
        if target_col not in frame.columns:
            return self
        pbt = params_by_ticker or {}
        default_params = dict(params or {})
        tickers = sorted(frame["ticker"].astype(str).str.upper().unique())
        for sym in tickers:
            mask = frame["ticker"].astype(str).str.upper().eq(sym).to_numpy()
            if int(mask.sum()) < min_rows:
                continue
            sub = frame.loc[mask]
            if sub[target_col].nunique() < 2:
                continue
            y = sub[target_col].astype(int)
            w = None if sample_weight is None else np.asarray(sample_weight)[mask]
            fold_params = _with_class_weight(dict(pbt.get(sym) or default_params), y)
            model = make_entry_classifier(self.model_name, fold_params)
            model.fit(sub[feat_cols], y, sample_weight=w)
            self.short_models[sym] = model
        return self

    def predict_proba(self, frame: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
        """Return (n, 2) proba matrix; missing ticker → 0.5."""
        n = len(frame)
        out = np.full((n, 2), 0.5, dtype=float)
        if not self.models or frame.empty or "ticker" not in frame.columns:
            return out
        tickers = frame["ticker"].astype(str).str.upper()
        for sym, model in self.models.items():
            mask = tickers.eq(sym).to_numpy()
            if not mask.any():
                continue
            proba = model.predict_proba(frame.loc[mask, feat_cols])
            out[mask] = proba
        return out

    def predict_proba_short(self, frame: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
        """Return (n, 2) short-model proba; missing ticker → 0.5."""
        n = len(frame)
        out = np.full((n, 2), 0.5, dtype=float)
        if not self.short_models or frame.empty or "ticker" not in frame.columns:
            return out
        tickers = frame["ticker"].astype(str).str.upper()
        for sym, model in self.short_models.items():
            mask = tickers.eq(sym).to_numpy()
            if not mask.any():
                continue
            proba = model.predict_proba(frame.loc[mask, feat_cols])
            out[mask] = proba
        return out

    def training_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "tickers": sorted(self.models),
            "short_tickers": sorted(self.short_models),
            "adaptive": self._adaptive,
        }
        for sym, model in self.models.items():
            if hasattr(model, "training_state"):
                state[sym] = model.training_state()
            else:
                state[sym] = {"type": type(model).__name__}
        return state

    @property
    def feature_importances_(self) -> np.ndarray:
        imps = [
            np.asarray(m.feature_importances_)
            for m in self.models.values()
            if hasattr(m, "feature_importances_")
        ]
        if not imps:
            return np.array([])
        stacked = np.vstack(imps)
        return stacked.mean(axis=0)
