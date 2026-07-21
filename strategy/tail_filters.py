"""Entry filters for stress / high-vol regimes (left-tail control, keep volume)."""

from __future__ import annotations

import pandas as pd

import config as _cfg
from common.naming import COL_PROB_HMM_STRESS


def _col_num(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def tail_entry_skip_mask(df: pd.DataFrame) -> pd.Series:
    """True where new entries should be skipped (stress spike + elevated vol)."""
    if df.empty or not bool(getattr(_cfg, "FUSION_TAIL_ENTRY_FILTER", True)):
        return pd.Series(False, index=df.index)

    p_stress = _col_num(df, COL_PROB_HMM_STRESS, 1.0 / 3.0)
    stress_max = float(getattr(_cfg, "FUSION_TAIL_STRESS_ENTRY_MAX", 0.42))

    vol_ann = None
    for col in ("vol_ann", "garch_cond_vol", "vol_realized_12"):
        if col in df.columns:
            vol_ann = _col_num(df, col, 0.15)
            break
    vol_ann_thr = float(getattr(_cfg, "FUSION_TAIL_HIGH_VOL_ANN", 0.22))

    vol_ratio = None
    if "vol_ratio" in df.columns:
        vol_ratio = _col_num(df, "vol_ratio", 1.0)
    vol_ratio_thr = float(getattr(_cfg, "FUSION_TAIL_HIGH_VOL_RATIO", 1.35))

    stress_hit = p_stress >= stress_max
    vol_hit = pd.Series(False, index=df.index)
    if vol_ann is not None:
        vol_hit |= vol_ann >= vol_ann_thr
    if vol_ratio is not None:
        vol_hit |= vol_ratio >= vol_ratio_thr

    # Require both stress AND vol elevation — avoids zeroing entire book.
    return stress_hit & vol_hit


def apply_tail_entry_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Zero fusion entries on tail-skip rows."""
    if df.empty or not bool(getattr(_cfg, "FUSION_TAIL_ENTRY_FILTER", True)):
        return df
    out = df.copy()
    skip = tail_entry_skip_mask(out)
    if not skip.any():
        return out
    if "fusion_score" in out.columns:
        out.loc[skip, "fusion_score"] = 0.0
    if "position_side" in out.columns:
        out.loc[skip, "position_side"] = 0
    return out
