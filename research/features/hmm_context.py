"""HMM-derived ML features for XGBoost/trade models.

These features turn filtered HMM probabilities into model-ready context:
- current micro-regime probabilities
- entropy/confidence
- causal empirical next-state probabilities from prior observed transitions
- expected duration of the current dominant micro-regime
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import math

import numpy as np
import pandas as pd

from common.naming import (
    COL_PROB_HMM_IMPULSE,
    COL_PROB_HMM_MEAN_REVERT,
    COL_PROB_HMM_STRESS,
)

HMM_XGB_FEATURE_COLS = [
    "hmm_prob_entropy",
    "hmm_transition_entropy",
    "hmm_next_impulse",
    "hmm_next_mean_revert",
    "hmm_next_stress",
    "hmm_expected_duration_bars",
    "hmm_vol_ratio_1h_vs_1d",
]

_PROB_COLS = [COL_PROB_HMM_IMPULSE, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_STRESS]
_STATE_NAMES = ["impulse", "mean_revert", "stress"]
_NEXT_COLS = ["hmm_next_impulse", "hmm_next_mean_revert", "hmm_next_stress"]
_N_STATES = len(_STATE_NAMES)
_LOG_N = math.log(_N_STATES)


def _vector_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs.astype(float), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log(p)).sum(axis=1) / _LOG_N


def _transition_features(state_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(state_idx)
    next_probs = np.zeros((n, _N_STATES), dtype=float)
    next_entropy = np.zeros(n, dtype=float)
    durations = np.zeros(n, dtype=float)
    counts = np.ones((_N_STATES, _N_STATES), dtype=float)
    prev_state: int | None = None
    for i, s in enumerate(state_idx.astype(int)):
        row_counts = counts[s]
        total = float(row_counts.sum()) or 1.0
        p = row_counts / total
        next_probs[i] = p
        p_clip = np.clip(p, 1e-12, 1.0)
        next_entropy[i] = float(-(p_clip * np.log(p_clip)).sum() / _LOG_N)
        p_stay = row_counts[s] / total
        durations[i] = float(1.0 / max(1.0 - p_stay, 1e-6))
        if prev_state is not None:
            counts[prev_state, s] += 1.0
        prev_state = s
    return next_probs, next_entropy, durations


def _attach_transition_block(out: pd.DataFrame, idx: pd.Index, probs: np.ndarray) -> None:
    out.loc[idx, "hmm_prob_entropy"] = _vector_entropy(probs)
    tp, te, dur = _transition_features(np.argmax(probs, axis=1))
    for j, c in enumerate(_NEXT_COLS):
        out.loc[idx, c] = tp[:, j]
    out.loc[idx, "hmm_transition_entropy"] = te
    out.loc[idx, "hmm_expected_duration_bars"] = dur


def attach_hmm_xgb_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach causal HMM transition/duration features for ML models."""
    if panel.empty:
        return panel

    missing = [c for c in _PROB_COLS if c not in panel.columns]
    out = panel.copy()
    if missing:
        for c in HMM_XGB_FEATURE_COLS:
            out[c] = 0.0
        return out

    sort_cols = ["ticker", "bar_time"] if "ticker" in out.columns else ["bar_time"]
    out = out.sort_values(sort_cols).copy()
    for c in HMM_XGB_FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0.0

    if "ticker" in out.columns:
        for _, grp in out.groupby("ticker", sort=False):
            _attach_transition_block(out, grp.index, grp[_PROB_COLS].astype(float).to_numpy())
    else:
        _attach_transition_block(out, out.index, out[_PROB_COLS].astype(float).to_numpy())

    out["hmm_vol_ratio_1h_vs_1d"] = out.get("hmm_vol_ratio", 1.0)
    return out
