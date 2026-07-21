"""
Canonical names — disambiguate HMM micro-regimes from per-stock factor scores.

HMM micro-regimes:         HMM_IMPULSE / HMM_MEAN_REVERT / HMM_STRESS
Stock factors (cross-sec): factor_trend_sub / factor_mean_rev_sub / factor_risk_sub
"""

from __future__ import annotations

# HMM hidden-state labels (market phase, not stock momentum)
# Assigned by emission means (ret_z, vol_z), NOT by forward returns.
# See output/hmm_state_selection_report.json for data-driven roles.
HMM_IMPULSE = "HMM_IMPULSE"          # highest ret_z emission → directional burst
HMM_MEAN_REVERT = "HMM_MEAN_REVERT"  # mid ret_z / quieter vol → range/chop
HMM_STRESS = "HMM_STRESS"            # low ret_z + elevated vol_z → risk-off context

HMM_REGIMES: tuple[str, ...] = (HMM_IMPULSE, HMM_MEAN_REVERT, HMM_STRESS)

# Human-readable aliases (economic reinterpretation; canonical keys unchanged for ML)
HMM_ECONOMIC_ALIASES: dict[str, str] = {
    HMM_IMPULSE: "directional_burst",
    HMM_MEAN_REVERT: "quiet_chop",
    HMM_STRESS: "elevated_vol_defensive",
}

# HMM regime frame columns
COL_HMM_DOMINANT = "hmm_dominant_regime"
COL_PROB_HMM_IMPULSE = "prob_hmm_impulse"
COL_PROB_HMM_MEAN_REVERT = "prob_hmm_mean_revert"
COL_PROB_HMM_STRESS = "prob_hmm_stress"

# Backward-compatible aliases for old notebooks/reports/imports.
# New code should use the micro-regime names above.
HMM_TREND = HMM_IMPULSE
HMM_RANGE = HMM_MEAN_REVERT
HMM_CRISIS = HMM_STRESS
COL_PROB_HMM_TREND = COL_PROB_HMM_IMPULSE
COL_PROB_HMM_RANGE = COL_PROB_HMM_MEAN_REVERT
COL_PROB_HMM_CRISIS = COL_PROB_HMM_STRESS

# Per-stock factor subscores (composite score inputs)
COL_FACTOR_TREND = "factor_trend_sub"
COL_FACTOR_MEAN_REV = "factor_mean_rev_sub"
COL_FACTOR_RISK = "factor_risk_sub"
COL_PRICE_TREND_12_1_PCT = "price_trend_12_1_pct"

SCORE_FACTOR_COLUMNS: tuple[str, ...] = (
    COL_FACTOR_TREND,
    COL_FACTOR_MEAN_REV,
    COL_FACTOR_RISK,
)

# Blended weights for factor subscores (HMM prior + walk-forward tilt)
COL_WEIGHT_TREND = "weight_factor_trend"
COL_WEIGHT_MEAN_REV = "weight_factor_mean_rev"
COL_WEIGHT_RISK = "weight_factor_risk"

COL_WEIGHT_TREND_PRIOR = "weight_factor_trend_prior"
COL_WEIGHT_MEAN_REV_PRIOR = "weight_factor_mean_rev_prior"
COL_WEIGHT_RISK_PRIOR = "weight_factor_risk_prior"

COL_TILT_TREND = "tilt_factor_trend"
COL_TILT_MEAN_REV = "tilt_factor_mean_rev"
COL_TILT_RISK = "tilt_factor_risk"

SIGNAL_PANEL_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    *SCORE_FACTOR_COLUMNS,
    COL_PRICE_TREND_12_1_PCT,
    "z_sma200",
    "risk_on",
    "vol_ratio",
    "close",
    "vol_ann",
    "score",
)
