"""Rolling adaptive LightGBM: recency weights + warm-start incremental trees."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config as _cfg
from models.entry_model import FUSION_ENTRY_MODEL_NAME, make_entry_classifier


def adaptive_regularization_params() -> dict[str, Any]:
    """Extra LGBM regularization for stable incremental adaptation."""
    return {
        "lambda_l1": float(getattr(_cfg, "FUSION_ADAPTIVE_L1", 0.15)),
        "lambda_l2": float(getattr(_cfg, "FUSION_ADAPTIVE_L2", 1.5)),
        "feature_fraction": float(getattr(_cfg, "FUSION_ADAPTIVE_FEATURE_FRACTION", 0.8)),
        "bagging_fraction": float(getattr(_cfg, "FUSION_ADAPTIVE_BAGGING_FRACTION", 0.75)),
        "bagging_freq": int(getattr(_cfg, "FUSION_ADAPTIVE_BAGGING_FREQ", 1)),
    }


def recency_sample_weights(
    times: pd.Series,
    reference_time: pd.Timestamp,
    *,
    halflife_days: float | None = None,
) -> np.ndarray:
    """Exponential recency weights: newer bars in the rolling window weigh more."""
    hl = float(
        halflife_days
        if halflife_days is not None
        else getattr(_cfg, "FUSION_ADAPTIVE_RECENCY_HALFLIFE_DAYS", 120.0)
    )
    hl = max(hl, 1.0)
    ref = pd.Timestamp(reference_time)
    bt = pd.to_datetime(times)
    age_days = np.clip((ref - bt).dt.total_seconds().to_numpy() / 86400.0, 0.0, None)
    decay = np.log(2.0) / hl
    w = np.exp(-decay * age_days)
    w = np.clip(w, 1e-6, None)
    return w / float(np.mean(w))


def merge_model_params(params: dict[str, Any]) -> dict[str, Any]:
    """Base hyperparams + adaptive regularization."""
    return {**dict(params), **adaptive_regularization_params()}


class AdaptiveEntryModel:
    """LightGBM with rolling 12m window, recency weights, and warm-start updates."""

    def __init__(self, model_name: str = FUSION_ENTRY_MODEL_NAME):
        if model_name.lower() != FUSION_ENTRY_MODEL_NAME:
            raise ValueError(f"Adaptive training supports lightgbm only, got {model_name!r}")
        self.model_name = model_name
        self._clf: Any = None
        self._booster: Any = None
        self._params: dict[str, Any] = {}
        self._n_fits = 0
        self._total_trees = 0

    @property
    def booster_(self) -> Any:
        return self._booster

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._clf is None:
            return np.array([])
        return np.asarray(self._clf.feature_importances_)

    def fit_initial(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None,
        params: dict[str, Any],
    ) -> AdaptiveEntryModel:
        p = merge_model_params(params)
        clf = make_entry_classifier(self.model_name, p)
        clf.fit(X, y.astype(int), sample_weight=sample_weight)
        self._clf = clf
        self._booster = clf.booster_
        self._params = dict(p)
        self._n_fits = 1
        self._total_trees = int(clf.n_estimators_)
        return self

    def fit_incremental(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None,
        params: dict[str, Any],
    ) -> AdaptiveEntryModel:
        if self._booster is None:
            return self.fit_initial(X, y, sample_weight, params)
        inc_trees = int(getattr(_cfg, "FUSION_ADAPTIVE_INCREMENTAL_TREES", 72))
        inc_lr = float(getattr(_cfg, "FUSION_ADAPTIVE_INCREMENTAL_LR", 0.025))
        p = merge_model_params(params)
        p = {
            **p,
            "n_estimators": inc_trees,
            "learning_rate": inc_lr,
        }
        clf = make_entry_classifier(self.model_name, p)
        clf.fit(
            X,
            y.astype(int),
            sample_weight=sample_weight,
            init_model=self._booster,
        )
        self._clf = clf
        self._booster = clf.booster_
        self._params = dict(p)
        self._n_fits += 1
        self._total_trees += inc_trees
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("AdaptiveEntryModel is not fitted")
        return self._clf.predict_proba(X)

    def training_state(self) -> dict[str, Any]:
        return {
            "n_fits": self._n_fits,
            "total_trees": self._total_trees,
            "params": dict(self._params),
        }
