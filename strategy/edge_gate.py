"""Edge gate: separate label economics (gross TP) from ML execution heuristic.

``expected_edge_bps`` estimates gross directional move from model probability.
The entry gate must compare it to **net round-trip costs**, not to SL+commission
(which is the minimum gross TP for *labels* in target-opt).
"""

from __future__ import annotations

import math

import config as _cfg
from simulation.entry_signals import edge_floor_bps
from research.labels.trade import DEFAULT_SLIPPAGE_BPS


def edge_gate_floor_mode() -> str:
    """Floor mode for comparing against ``expected_edge_bps`` heuristic."""
    return str(getattr(_cfg, "FUSION_EDGE_GATE_FLOOR_MODE", "commission_only"))


def label_tp_floor_mode() -> str:
    """Floor mode for target labels / take-profit economics."""
    return str(getattr(_cfg, "FUSION_EDGE_FLOOR_MODE", "sl_plus_commission"))


def heuristic_gate_floor_bps(
    commission_bps: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    *,
    buffer_bps: float | None = None,
) -> float:
    """Minimum edge (bps) for ML heuristic gate — net cost, not gross SL."""
    buf = buffer_bps
    if buf is None:
        buf = float(getattr(_cfg, "FUSION_EDGE_BUFFER_BPS", 2.0))
    return edge_floor_bps(
        commission_bps,
        slippage_bps,
        mode=edge_gate_floor_mode(),
        buffer_bps=buf,
    )


def label_tp_floor_bps(
    commission_bps: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    *,
    stop_loss_bps: float | None = None,
    buffer_bps: float | None = None,
) -> float:
    """Gross TP floor for per-instrument target optimization."""
    sl = stop_loss_bps
    if sl is None:
        sl = float(getattr(_cfg, "FUSION_STOP_LOSS_BPS", 25.0))
    return edge_floor_bps(
        commission_bps,
        slippage_bps,
        stop_loss_bps=sl,
        buffer_bps=buffer_bps,
        mode=label_tp_floor_mode(),
    )


def resolve_min_expected_edge_bps(
    requested: float,
    *,
    commission_bps: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    calibrated: float | None = None,
    instrument_floor: float | None = None,
) -> float:
    """Resolved signed-edge magnitude (bps) for one instrument or commission context."""
    floor = (
        float(instrument_floor)
        if instrument_floor is not None
        else heuristic_gate_floor_bps(commission_bps, slippage_bps)
    )
    cap = float(getattr(_cfg, "FUSION_EDGE_CALIBRATION_CAP_BPS", 50.0))
    fallback = float(getattr(_cfg, "FUSION_MIN_EXPECTED_EDGE_BPS", 0.0))
    resolved = max(float(requested), floor)
    if fallback > 0.0:
        resolved = max(resolved, min(fallback, cap))
    if calibrated is not None and math.isfinite(calibrated):
        resolved = max(resolved, min(float(calibrated), cap))
    return float(min(resolved, cap))


def resolve_ticker_min_edge_bps(
    ticker: str,
    impulse_params: dict,
    *,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> float:
    """Per-instrument gate floor: policy/calibration over **cost** floor (not label TP barrier).

    Label ``threshold_bps`` from target-opt is TP economics for training labels;
    using it as the ML edge gate zeroes volume. Same rule for all tickers:
    max(policy requested, commission/slippage heuristic, optional calibrated).
    """
    from data_platform.universe import commission_bps_for_ticker

    sym = str(ticker).upper()
    try:
        from strategy.threshold_calibrator import resolve_ticker_policy

        pol = resolve_ticker_policy(impulse_params, sym)
    except ImportError:
        pol = impulse_params

    comm = commission_bps_for_ticker(sym)
    instrument_floor = heuristic_gate_floor_bps(comm, slippage_bps)
    requested = float(
        pol.get("min_expected_edge_bps", impulse_params.get("min_expected_edge_bps", 0.0))
    )
    calibrated = pol.get("calibrated_min_expected_edge_bps")
    if calibrated is None:
        calibrated = impulse_params.get("calibrated_min_expected_edge_bps")
    return resolve_min_expected_edge_bps(
        requested,
        commission_bps=comm,
        slippage_bps=slippage_bps,
        calibrated=float(calibrated) if calibrated is not None else None,
        instrument_floor=instrument_floor,
    )


def passes_signed_edge_gate(
    edge_bps: float,
    min_edge_bps: float,
    position_side: int,
) -> bool:
    """Long: edge >= +min. Short: edge <= -min. Flat: never passes."""
    m = float(min_edge_bps)
    if m <= 0.0 or not math.isfinite(float(edge_bps)):
        return False
    side = int(position_side)
    e = float(edge_bps)
    if side > 0:
        return e >= m
    if side < 0:
        return e <= -m
    return False


def signed_edge_active_mask(
    edge_bps,
    position_side,
    min_edge_bps,
):
    """Vectorized signed edge gate aligned with ``position_side``."""
    import numpy as np
    import pandas as pd

    edge = np.asarray(edge_bps, dtype=float)
    side = np.asarray(position_side, dtype=int)
    if isinstance(min_edge_bps, pd.Series):
        min_e = np.asarray(min_edge_bps, dtype=float)
    else:
        min_e = np.full(len(edge), float(min_edge_bps), dtype=float)
    if len(min_e) != len(edge):
        min_e = np.broadcast_to(min_e, len(edge))

    out = np.zeros(len(edge), dtype=bool)
    long_m = side > 0
    short_m = side < 0
    if long_m.any():
        out[long_m] = np.isfinite(edge[long_m]) & (edge[long_m] >= min_e[long_m])
    if short_m.any():
        out[short_m] = np.isfinite(edge[short_m]) & (edge[short_m] <= -min_e[short_m])
    return out


def panel_positive_edge_stats(
    expected_edge_bps: "pd.Series | np.ndarray",
) -> tuple[float, float, int]:
    """Return (max, q65, count) for strictly positive expected-edge rows."""
    import numpy as np

    arr = np.asarray(expected_edge_bps, dtype=float)
    pos = arr[np.isfinite(arr) & (arr > 0.0)]
    if pos.size == 0:
        return 0.0, 0.0, 0
    return float(pos.max()), float(np.quantile(pos, 0.65)), int(pos.size)


def panel_abs_edge_stats(
    expected_edge_bps: "pd.Series | np.ndarray",
) -> tuple[float, float, int]:
    """Return (max, q65, count) for finite ``|expected_edge_bps|`` (long+short)."""
    import numpy as np

    arr = np.asarray(expected_edge_bps, dtype=float)
    mag = np.abs(arr[np.isfinite(arr)])
    if mag.size == 0:
        return 0.0, 0.0, 0
    return float(mag.max()), float(np.quantile(mag, 0.65)), int(mag.size)


def cap_calibrated_edge_to_panel(
    calibrated_edge: float,
    *,
    panel_max_edge_bps: float,
    panel_q65_edge_bps: float,
    commission_bps: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> float:
    """Lower calibrated edge when the policy panel cannot reach the heuristic floor."""
    floor = heuristic_gate_floor_bps(commission_bps, slippage_bps)
    cap = float(getattr(_cfg, "FUSION_EDGE_CALIBRATION_CAP_BPS", 35.0))
    out = float(calibrated_edge)
    if panel_max_edge_bps > 0.0 and panel_max_edge_bps < out:
        feasible = max(floor * 0.5, panel_q65_edge_bps if panel_q65_edge_bps > 0 else panel_max_edge_bps)
        out = min(out, feasible)
    return float(min(max(out, floor * 0.5), cap))


def threshold_search_bounds(
    calibrated_edge: float,
    *,
    commission_bps: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    panel_max_edge_bps: float | None = None,
) -> tuple[float, float]:
    """Optuna search range for ``min_expected_edge_bps`` aligned with heuristic."""
    floor = heuristic_gate_floor_bps(commission_bps, slippage_bps)
    explicit = getattr(_cfg, "FUSION_THRESHOLD_EDGE_RANGE", None)
    if explicit and len(explicit) >= 2:
        edge_lo = max(floor, float(explicit[0]))
        edge_hi = max(edge_lo + 1.0, float(explicit[1]))
    else:
        span = float(getattr(_cfg, "FUSION_THRESHOLD_EDGE_SPAN_BPS", 12.0))
        edge_lo = floor
        edge_hi = max(floor + span, min(float(calibrated_edge) + 5.0, floor + span * 2))
    if panel_max_edge_bps is not None and panel_max_edge_bps > 0.0:
        feasible_hi = float(panel_max_edge_bps)
        edge_hi = min(edge_hi, feasible_hi)
        edge_lo = min(edge_lo, feasible_hi)
    edge_lo = round(edge_lo * 2) / 2
    edge_hi = round(edge_hi * 2) / 2
    if edge_lo >= edge_hi:
        edge_hi = edge_lo + 0.5
    return edge_lo, edge_hi


def proportional_constraint_limits(
    val_rows: int,
    val_sessions: int,
) -> tuple[int, int, float]:
    """Scale threshold-opt constraints to CV window size."""
    base_rows = int(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_SIGNAL_ROWS", 5))
    base_reb = int(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_ACTIVE_REBALANCES", 1))
    base_exp = float(getattr(_cfg, "FUSION_THRESHOLD_OPT_MIN_AVG_EXPOSURE_PCT", 0.05))
    rows = max(3, min(base_rows, max(3, int(val_rows * 0.001))))
    reb = max(1, min(base_reb, max(1, val_sessions // 4)))
    exp = min(base_exp, max(0.02, base_exp * min(1.0, val_sessions / 20.0)))
    return rows, reb, exp
