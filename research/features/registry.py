"""ML feature registry: logical groups, active column set, derived features, importance rollups."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config as _cfg
from research.features.entry_ml import FLOW_FEATURE_COLS
from research.features.hmm_context import HMM_XGB_FEATURE_COLS
from research.features.impulse import IMPULSE_COLS
from research.features.advanced_ts import CROSS_SECTIONAL_COLS, ML_ADVANCED_TS_COLS
from research.features.structure import STRUCTURE_COLS, STRUCTURE_CS_COLS
from research.features.volume_profile import VP_COLS, VP_HMM_COLS
from common.naming import COL_PROB_HMM_IMPULSE, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_STRESS

# Columns computed for diagnostics / impulse logic but excluded from LightGBM inputs.
EXCLUDED_FROM_ML: frozenset[str] = frozenset(
    {
        "flow_source",  # constant in tick-only mode
        "session_progress",  # weak for 24/7 crypto; session boundary is arbitrary
        "hmm_risk_on",  # redundant with prob_hmm_* + hmm_vol_ratio
        "vp_in_value",
        "vp_above_poc",
        "vp_below_val",  # sparse binary flags; keep continuous VP geometry
        "nw_breakout_up",
        "nw_breakout_dn",  # sparse; nw_env_pos captures position
        "impulse_raw",  # collinear with side_shift / power_shift stack
        # flow_microstructure — feature eval: max IC ~0.009, redundant buy/imb
        "vol_imbalance",
        "imb_ma_6",
        "imb_ma_12",
        "imb_z",
        "buy_share",
        "tick_imbalance",
        # candle_geometry — keep wick ratios only in ML
        "body_pct",
        "clv",
        "body_ratio",
        "hammer_score",
    }
)

DERIVED_ML_COLS: tuple[str, ...] = (
    "ret_24",
    "vol_realized_12",
    "imb_z",
)

HMM_CONTEXT_ML_COLS: tuple[str, ...] = (
    COL_PROB_HMM_IMPULSE,
    COL_PROB_HMM_MEAN_REVERT,
    COL_PROB_HMM_STRESS,
    "hmm_confidence",
    "hmm_vol_ratio",
)

FEATURE_GROUPS: dict[str, dict[str, Any]] = {
    "price_momentum": {
        "label": "Price & momentum",
        "description": "Short-horizon returns and realized volatility on 5m bars.",
        "features": ("ret_1", "ret_6", "ret_12", "ret_24", "vol_realized_12", "vol_z"),
    },
    "flow_microstructure": {
        "label": "Flow & volume imbalance",
        "description": "Tick/candle hybrid flow: imbalance levels and z-scores.",
        "features": ("vol_imbalance", "imb_ma_6", "imb_ma_12", "imb_z", "buy_share", "tick_imbalance"),
    },
    "candle_geometry": {
        "label": "Candle geometry",
        "description": "Bar shape: body, wicks, close location, hammer pattern.",
        "features": (
            "body_pct",
            "clv",
            "lower_wick_ratio",
            "upper_wick_ratio",
            "body_ratio",
            "hammer_score",
        ),
    },
    "nw_envelope": {
        "label": "Nadaraya-Watson envelope",
        "description": "Smooth fair value, band position, slope; mean-reversion vs breakout context.",
        "features": ("nw_est", "nw_slope", "nw_env_pos", "nw_band_width"),
    },
    "classic_momentum": {
        "label": "Classic momentum stack",
        "description": "ROC, RSI, MACD and short/long momentum divergence.",
        "features": (
            "roc_6",
            "roc_12",
            "roc_24",
            "rsi_14",
            "macd_hist",
            "mom_short",
            "mom_long",
            "side_shift",
            "power_shift",
        ),
    },
    "volume_profile": {
        "label": "Volume profile (session)",
        "description": "Causal session VP: POC/VA geometry and distance to value.",
        "features": (
            "vp_poc",
            "vp_vah",
            "vp_val",
            "vp_poc_dist",
            "vp_va_width",
            "vp_near_poc",
        ),
    },
    "vp_hmm_blend": {
        "label": "VP x HMM interaction",
        "description": "Volume-profile signals modulated by HMM regime.",
        "features": VP_HMM_COLS,
    },
    "hmm_regime": {
        "label": "HMM regime levels",
        "description": "Filtered micro-regime probabilities and vol context (soft gate inputs).",
        "features": HMM_CONTEXT_ML_COLS,
    },
    "hmm_dynamics": {
        "label": "HMM dynamics",
        "description": "Transition entropy, next-state probs, duration, regime conviction.",
        "features": tuple(HMM_XGB_FEATURE_COLS) + ("hmm_impulse_edge", "hmm_regime_conviction"),
    },
    "long_memory": {
        "label": "Long memory / fractals",
        "description": "Rolling Hurst exponent (R/S) on 5m returns.",
        "features": ("hurst_rs",),
    },
    "spectral": {
        "label": "Spectral / cycles",
        "description": "FFT low/high band power ratio, entropy, dominant period.",
        "features": ("spec_low_high_ratio", "spec_entropy", "spec_dominant_period"),
    },
    "garch_vol": {
        "label": "GARCH stochastic vol",
        "description": "Conditional vol and vol ratio vs realized.",
        "features": ("garch_cond_vol", "garch_vol_ratio"),
    },
    "structure_path": {
        "label": "Structure / path",
        "description": "Trend efficiency, autocorrelation, range z, vol-of-vol, NW z, VWAP distance.",
        "features": STRUCTURE_COLS,
    },
    "cross_sectional": {
        "label": "Cross-sectional ranks",
        "description": "Per-bar percentile rank across universe tickers (momentum + importance leaders).",
        "features": tuple(CROSS_SECTIONAL_COLS) + tuple(STRUCTURE_CS_COLS),
    },
}


def _all_grouped_features() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for spec in FEATURE_GROUPS.values():
        for col in spec["features"]:
            if col not in seen:
                seen.add(col)
                ordered.append(col)
    return ordered


def active_ml_feature_cols(*, include_derived: bool = True) -> list[str]:
    """Ordered active LightGBM inputs: grouped catalog minus exclusions."""
    excluded_groups = set(getattr(_cfg, "ML_EXCLUDED_GROUPS", ()) or ())
    base = _all_grouped_features()
    if include_derived:
        for col in DERIVED_ML_COLS:
            if col not in base:
                base.append(col)
        for col in ML_ADVANCED_TS_COLS:
            if col not in base:
                base.append(col)
        for col in CROSS_SECTIONAL_COLS:
            if col not in base:
                base.append(col)
        for col in STRUCTURE_COLS:
            if col not in base:
                base.append(col)
        for col in STRUCTURE_CS_COLS:
            if col not in base:
                base.append(col)
    out: list[str] = []
    for col in base:
        if col in EXCLUDED_FROM_ML:
            continue
        gid = feature_group(col)
        if gid in excluded_groups:
            continue
        out.append(col)
    return out


def feature_group(name: str) -> str:
    """Return group id for a feature column."""
    for gid, spec in FEATURE_GROUPS.items():
        if name in spec["features"]:
            return gid
    if name in DERIVED_ML_COLS:
        if name.startswith("hmm_"):
            return "hmm_dynamics"
        if name in ("ret_24", "vol_realized_12"):
            return "price_momentum"
        if name == "imb_z":
            return "flow_microstructure"
    if name in ML_ADVANCED_TS_COLS:
        if name == "hurst_rs":
            return "long_memory"
        if name.startswith("garch_"):
            return "garch_vol"
        if name.startswith("spec_"):
            return "spectral"
    if name in STRUCTURE_COLS:
        return "structure_path"
    if name in CROSS_SECTIONAL_COLS or name in STRUCTURE_CS_COLS:
        return "cross_sectional"
    return "other"


def feature_group_catalog() -> list[dict]:
    """Human-readable catalog for reports and CLI."""
    rows: list[dict] = []
    active = set(active_ml_feature_cols())
    for gid, spec in FEATURE_GROUPS.items():
        feats = list(spec["features"])
        rows.append(
            {
                "id": gid,
                "label": spec["label"],
                "description": spec["description"],
                "n_features": len(feats),
                "active_in_ml": sum(1 for f in feats if f in active),
                "features": feats,
            }
        )
    return rows


def attach_price_flow_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Extra causal price/flow features (per-ticker when panel is stacked)."""
    if df.empty:
        return df
    out = df.copy()

    def _derive_block(block: pd.DataFrame) -> pd.DataFrame:
        g = block.copy()
        close = g["close"].astype(float)
        g["ret_24"] = close.pct_change(24, fill_method=None)
        if "ret_1" in g.columns:
            r1 = g["ret_1"]
            if isinstance(r1, pd.DataFrame):
                r1 = r1.iloc[:, 0]
        else:
            r1 = close.pct_change(1, fill_method=None)
        g["vol_realized_12"] = pd.to_numeric(r1, errors="coerce").rolling(12, min_periods=4).std()
        if "vol_imbalance" in g.columns:
            imb = g["vol_imbalance"].fillna(0.0)
            imb_mu = imb.rolling(48, min_periods=12).mean()
            imb_sd = imb.rolling(48, min_periods=12).std().replace(0, np.nan)
            g["imb_z"] = ((imb - imb_mu) / imb_sd).replace([np.inf, -np.inf], np.nan)
        return g

    if "ticker" in out.columns:
        parts = [
            _derive_block(g.sort_values("bar_time" if "bar_time" in g.columns else g.index))
            for _, g in out.groupby("ticker", sort=False)
        ]
        return pd.concat(parts).sort_index()
    return _derive_block(out)


def attach_hmm_derived_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Regime spread / conviction features for ML (after HMM columns exist)."""
    if df.empty:
        return df
    out = df.copy()
    cols = [COL_PROB_HMM_IMPULSE, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_STRESS]
    if not all(c in out.columns for c in cols):
        out["hmm_impulse_edge"] = 0.0
        out["hmm_regime_conviction"] = 0.0
        return out
    p = out[cols].astype(float).to_numpy()
    out["hmm_impulse_edge"] = (
        out[COL_PROB_HMM_IMPULSE].astype(float) - out[COL_PROB_HMM_STRESS].astype(float)
    )
    sorted_p = np.sort(p, axis=1)
    out["hmm_regime_conviction"] = sorted_p[:, -1] - sorted_p[:, -2]
    return out


def attach_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-bar percentile ranks across tickers (no future leak within bar)."""
    if panel.empty or "ticker" not in panel.columns or "bar_time" not in panel.columns:
        return panel
    out = panel.copy()
    rank_cols = {
        "ret_12_cs_rank": "ret_12",
        "vol_realized_12_cs_rank": "vol_realized_12",
        "nw_env_pos_cs_rank": "nw_env_pos",
        "hurst_rs_cs_rank": "hurst_rs",
        "spec_low_high_ratio_cs_rank": "spec_low_high_ratio",
        "vp_poc_dist_cs_rank": "vp_poc_dist",
        "garch_vol_ratio_cs_rank": "garch_vol_ratio",
    }
    for dst, src in rank_cols.items():
        if src not in out.columns:
            continue
        out[dst] = (
            out.groupby("bar_time", sort=False)[src]
            .rank(pct=True, method="average")
            .astype(float)
        )
    return out


def attach_fusion_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all derived ML feature columns to a panel."""
    from research.features.advanced_ts import attach_ml_advanced_ts_features
    from research.features.structure import attach_structure_features_by_ticker

    out = attach_price_flow_derived(df)
    out = attach_hmm_derived_ml(out)
    out = attach_ml_advanced_ts_features(out)
    out = attach_structure_features_by_ticker(out)
    return attach_cross_sectional_features(out)


def resolve_ml_feature_cols(panel: pd.DataFrame) -> list[str]:
    """Active registry columns present in panel."""
    return [c for c in active_ml_feature_cols() if c in panel.columns]


def aggregate_importance_by_group(per_feature: dict[str, float]) -> dict[str, Any]:
    """Roll up LightGBM importances into logical groups."""
    groups: dict[str, dict[str, Any]] = {}
    total = float(sum(abs(v) for v in per_feature.values())) or 1.0
    for name, raw in per_feature.items():
        gid = feature_group(name)
        spec = FEATURE_GROUPS.get(gid, {"label": gid})
        bucket = groups.setdefault(
            gid,
            {
                "id": gid,
                "label": spec.get("label", gid),
                "importance_sum": 0.0,
                "importance_pct": 0.0,
                "features": {},
            },
        )
        val = float(raw)
        bucket["importance_sum"] += val
        bucket["features"][name] = round(val, 6)
    for bucket in groups.values():
        bucket["importance_pct"] = round(100.0 * bucket["importance_sum"] / total, 2)
        bucket["importance_sum"] = round(bucket["importance_sum"], 6)
        top = sorted(bucket["features"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        bucket["top_features"] = {k: v for k, v in top}
    ranked = sorted(groups.values(), key=lambda x: x["importance_pct"], reverse=True)
    return {"groups": ranked, "n_features": len(per_feature)}


def merge_fold_importances(folds: list[dict]) -> dict[str, float]:
    """Average per-fold top_features dicts from monthly walk-forward."""
    acc: dict[str, list[float]] = {}
    for fold in folds:
        tops = fold.get("top_features") or {}
        for name, val in tops.items():
            acc.setdefault(name, []).append(float(val))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


# Backward-compatible alias used across the codebase.
FUSION_FEATURE_COLS: tuple[str, ...] = tuple(active_ml_feature_cols())

# Legacy superset (includes excluded columns still built in panel).
_LEGACY_PANEL_COLS = (
    tuple(FLOW_FEATURE_COLS)
    + tuple(IMPULSE_COLS)
    + tuple(VP_COLS)
    + tuple(VP_HMM_COLS)
    + HMM_CONTEXT_ML_COLS
    + ("hmm_risk_on",)
    + tuple(HMM_XGB_FEATURE_COLS)
    + DERIVED_ML_COLS
    + ML_ADVANCED_TS_COLS
    + STRUCTURE_COLS
    + STRUCTURE_CS_COLS
)

PANEL_FEATURE_COLS: tuple[str, ...] = _LEGACY_PANEL_COLS
