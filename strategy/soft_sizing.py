"""Universal soft sizing from CV quality and edge-alignment diagnostics.

Same rule for every ticker: reduce ``exposure_cap`` when fold OOS/CV net is
weak or when ``|expected_edge|`` is anti-correlated with signed outcomes.

When ``FUSION_SOFT_SIZE_HARD_ZERO`` is True (default), quality fail or negative
holdout net forces ``exposure_cap=0`` (fail-closed). Soft interpolation remains
for borderline names with holdout >= 0.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config as _cfg


def compute_edge_alignment(df: pd.DataFrame, *, min_rows: int = 80) -> float | None:
    """Spearman(|expected_edge|, signed_fwd_bps) on active sides.

    Positive ⇒ higher |edge| predicts better outcomes (healthy gate).
    Negative ⇒ inverted calibration (NASDAQ-like drag).
    """
    need = ("expected_edge_bps", "position_side")
    if any(c not in df.columns for c in need):
        return None
    ret_col = "fwd_ret_entry" if "fwd_ret_entry" in df.columns else "fwd_ret"
    if ret_col not in df.columns:
        return None

    side = df["position_side"].astype(float).to_numpy()
    edge = df["expected_edge_bps"].astype(float).to_numpy()
    fwd = df[ret_col].astype(float).to_numpy()
    active = (side != 0) & np.isfinite(edge) & np.isfinite(fwd)
    if int(active.sum()) < int(min_rows):
        return None

    abs_edge = np.abs(edge[active])
    signed_bps = fwd[active] * side[active] * 10_000.0
    if np.std(abs_edge) < 1e-12 or np.std(signed_bps) < 1e-12:
        return None
    # Rank correlation without scipy dependency.
    r_edge = pd.Series(abs_edge).rank().to_numpy(dtype=float)
    r_out = pd.Series(signed_bps).rank().to_numpy(dtype=float)
    corr = float(np.corrcoef(r_edge, r_out)[0, 1])
    if not math.isfinite(corr):
        return None
    return corr


def soft_size_block_reason(policy: dict | None) -> str | None:
    """Human-readable reason when soft-size hard-zeros exposure; else None."""
    if not bool(getattr(_cfg, "FUSION_QUALITY_SOFT_SIZE", True)):
        return None
    if not bool(getattr(_cfg, "FUSION_SOFT_SIZE_HARD_ZERO", True)):
        return None
    pol = policy or {}
    # Stitched-passers kept by SQ v2: soft-size only — never hard-zero on last-fold holdout.
    if pol.get("sq_soft_keep"):
        return None
    if pol.get("signal_quality_ok") is False:
        return "signal_quality_ok=False"
    ho = pol.get("holdout_top_decile_net_bps")
    if ho is not None and math.isfinite(float(ho)) and float(ho) < 0.0:
        return f"holdout_top_decile_net_bps={float(ho):.3f}"
    return None


def soft_size_multiplier(policy: dict | None) -> float:
    """Map per-ticker policy diagnostics → exposure_cap in [0, 1]."""
    if not bool(getattr(_cfg, "FUSION_QUALITY_SOFT_SIZE", True)):
        return 1.0
    pol = policy or {}
    hard_zero = bool(getattr(_cfg, "FUSION_SOFT_SIZE_HARD_ZERO", True))
    min_cap = float(getattr(_cfg, "FUSION_SOFT_SIZE_MIN", 0.1))
    fail_cap = float(getattr(_cfg, "FUSION_SOFT_SIZE_QUALITY_FAIL", 0.35))
    inv_cap = float(getattr(_cfg, "FUSION_SOFT_SIZE_INVERTED_CAP", 0.25))

    ticker = str(pol.get("ticker") or pol.get("symbol") or "").upper()
    cv_lo = float(getattr(_cfg, "FUSION_SOFT_SIZE_CV_LO_BPS", -10.0))
    cv_hi = float(getattr(_cfg, "FUSION_SOFT_SIZE_CV_HI_BPS", 5.0))
    if ticker:
        try:
            from strategy.instrument_economics import soft_size_cv_band_bps

            cv_lo, cv_hi = soft_size_cv_band_bps(ticker)
        except Exception:
            pass

    if hard_zero and soft_size_block_reason(pol) is not None:
        return 0.0

    size = 1.0
    if pol.get("sq_pass_rate") is not None and math.isfinite(float(pol["sq_pass_rate"])):
        size = min(size, float(np.clip(float(pol["sq_pass_rate"]), min_cap, 1.0)))
    elif pol.get("signal_quality_ok") is False:
        size = min(size, fail_cap)

    cv = pol.get("cv_top_decile_net_bps")
    if cv is None:
        cv = pol.get("cv_net_bps")
    if cv is not None and math.isfinite(float(cv)):
        t = (float(cv) - cv_lo) / max(cv_hi - cv_lo, 1e-9)
        size = min(size, float(np.clip(t, min_cap, 1.0)))

    ho = pol.get("holdout_top_decile_net_bps")
    if ho is not None and math.isfinite(float(ho)) and float(ho) < 0.0:
        size = min(size, fail_cap)

    align = pol.get("edge_alignment")
    if align is not None and math.isfinite(float(align)) and float(align) < 0.0:
        size = min(size, inv_cap)

    return float(max(min_cap, min(1.0, size)))
