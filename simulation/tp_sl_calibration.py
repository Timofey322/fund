"""Enforce TP/SL economics: TP should cover SL + costs on average."""

from __future__ import annotations

import config as _cfg


def calibrate_tp_sl_bps(
    take_profit_bps: float,
    stop_loss_bps: float,
    *,
    min_ratio: float | None = None,
    tp_floor_bps: float | None = None,
    sl_cap_bps: float | None = None,
) -> tuple[float, float]:
    """Return (tp, sl) with TP >= SL * min_ratio and sane floors/caps."""
    tp = float(take_profit_bps)
    sl = float(stop_loss_bps)
    if not (tp > 0 and sl > 0):
        return tp, sl

    ratio = float(min_ratio if min_ratio is not None else getattr(_cfg, "FUSION_TP_SL_MIN_RATIO", 1.0))
    floor = float(tp_floor_bps if tp_floor_bps is not None else getattr(_cfg, "FUSION_TP_SL_TP_FLOOR_BPS", 35.0))
    cap = float(sl_cap_bps if sl_cap_bps is not None else getattr(_cfg, "FUSION_TP_SL_SL_CAP_BPS", 80.0))

    sl = min(max(sl, 5.0), cap)
    min_tp = max(floor, sl * max(ratio, 0.5))
    if tp < min_tp:
        tp = min_tp
    max_ratio = float(getattr(_cfg, "FUSION_TP_SL_MAX_RATIO", 0.0) or 0.0)
    if max_ratio > 0.0 and sl > 0.0:
        tp = min(tp, sl * max_ratio)
        tp = max(tp, min_tp)
    return float(tp), float(sl)


def tighten_stop_loss_bps(
    stop_loss_bps: float,
    *,
    stress_prob: float | None = None,
    vol_ann: float | None = None,
    vol_ratio: float | None = None,
) -> float:
    """Tighten SL in stress / high-vol regimes (left-tail control)."""
    if not bool(getattr(_cfg, "FUSION_TAIL_SL_TIGHTEN", True)):
        return float(stop_loss_bps)

    sl = float(stop_loss_bps)
    factor = 1.0
    stress_thr = float(getattr(_cfg, "FUSION_TAIL_STRESS_SL_MIN", 0.35))
    stress_mult = float(getattr(_cfg, "FUSION_TAIL_SL_TIGHTEN_IN_STRESS", 0.70))
    vol_ann_thr = float(getattr(_cfg, "FUSION_TAIL_HIGH_VOL_ANN", 0.22))
    vol_ann_mult = float(getattr(_cfg, "FUSION_TAIL_SL_TIGHTEN_HIGH_VOL", 0.80))
    vol_ratio_thr = float(getattr(_cfg, "FUSION_TAIL_HIGH_VOL_RATIO", 1.35))
    vol_ratio_mult = float(getattr(_cfg, "FUSION_TAIL_SL_TIGHTEN_HIGH_VOL_RATIO", 0.85))

    p_stress = float(stress_prob) if stress_prob is not None else 0.0
    if p_stress >= stress_thr:
        factor = min(factor, stress_mult)
    if vol_ann is not None and float(vol_ann) >= vol_ann_thr:
        factor = min(factor, vol_ann_mult)
    if vol_ratio is not None and float(vol_ratio) >= vol_ratio_thr:
        factor = min(factor, vol_ratio_mult)

    min_sl = float(getattr(_cfg, "FUSION_TAIL_SL_MIN_BPS", 12.0))
    return max(min_sl, sl * factor)
