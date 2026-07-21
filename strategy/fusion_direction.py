"""Signed long/short scoring for ML fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as _cfg


def fusion_allow_short() -> bool:
    return bool(getattr(_cfg, "FUSION_ALLOW_SHORT", False))


def fusion_sell_threshold(buy_threshold: float = 55.0) -> float:
    """Short entry ceiling; symmetric mode uses ``100 - buy`` around score=50."""
    explicit = getattr(_cfg, "FUSION_SELL_THRESHOLD", None)
    symmetric = bool(getattr(_cfg, "FUSION_SYMMETRIC_THRESHOLDS", True))
    buy = float(buy_threshold)
    if symmetric:
        buy = max(buy, 50.0 + 1e-6)
        return max(0.0, 100.0 - buy)
    if explicit is not None:
        return float(explicit)
    return max(0.0, 100.0 - buy)


def normalize_buy_threshold(buy_threshold: float) -> float:
    """Clamp buy threshold above the score midpoint for symmetric long/short bands."""
    buy = float(buy_threshold)
    if not bool(getattr(_cfg, "FUSION_SYMMETRIC_THRESHOLDS", True)):
        return buy
    lo, hi = getattr(_cfg, "FUSION_THRESHOLD_BUY_RANGE", (52, 58))
    buy = max(buy, float(lo))
    buy = min(buy, float(hi))
    if buy <= 50.0:
        buy = float(lo)
    return buy


def resolve_trading_thresholds(
    buy_threshold: float,
    hold_threshold: float | None = None,
) -> dict[str, float]:
    """Return symmetric buy/sell/hold band with sell < 50 < buy."""
    buy = normalize_buy_threshold(buy_threshold)
    sell = fusion_sell_threshold(buy)
    hold = float(
        hold_threshold
        if hold_threshold is not None
        else getattr(_cfg, "SCORE_EXIT", 40.0)
    )
    if hold >= buy:
        hold = sell + max(2.0, (buy - sell) * 0.35)
    if hold <= sell:
        hold = sell + 1.0
    return {
        "buy_threshold": buy,
        "sell_threshold": sell,
        "hold_threshold": hold,
    }


def signed_ml_edges(
    proba_long: np.ndarray,
    proba_short: np.ndarray,
    baseline: np.ndarray | float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative long and short conviction edges."""
    if isinstance(baseline, (int, float)):
        base = np.full(len(proba_long), np.clip(float(baseline), 1e-6, 1 - 1e-6))
    else:
        base = np.clip(np.asarray(baseline, dtype=float), 1e-6, 1 - 1e-6)
    pl = np.clip(np.asarray(proba_long, dtype=float), 1e-6, 1 - 1e-6)
    ps = np.clip(np.asarray(proba_short, dtype=float), 1e-6, 1 - 1e-6)
    long_edge = np.maximum(0.0, (pl - base) / np.maximum(1.0 - base, 1e-6))
    short_edge = np.maximum(0.0, (ps - base) / np.maximum(1.0 - base, 1e-6))
    return long_edge, short_edge


def fusion_signed_scores(
    proba_long: np.ndarray,
    proba_short: np.ndarray,
    impulse: np.ndarray,
    *,
    baseline: np.ndarray | float = 0.5,
    buy_threshold: float = 55.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map long/short probabilities + impulse to 0-100 score and position side.

    Score is symmetric around 50: >50 long-leaning, <50 short-leaning.
    ``position_side``: +1 long, -1 short, 0 flat.
    """
    bands = resolve_trading_thresholds(buy_threshold)
    buy_th = bands["buy_threshold"]
    sell_th = bands["sell_threshold"]
    long_edge, short_edge = signed_ml_edges(proba_long, proba_short, baseline)
    imp = np.clip(np.asarray(impulse, dtype=float), 0.0, 1.5)
    boost = 0.5 + 0.5 * np.minimum(imp, 1.0)
    min_edge = float(getattr(_cfg, "FUSION_MIN_DIRECTION_EDGE", 0.01))

    net = long_edge - short_edge
    denom = np.maximum(long_edge + short_edge, 1e-6)
    score = np.clip(50.0 + (net / denom) * 50.0 * boost, 0.0, 100.0)

    side = np.zeros(len(score), dtype=int)
    long_wins = long_edge >= short_edge + min_edge
    short_wins = short_edge >= long_edge + min_edge
    side[long_wins & (score >= buy_th)] = 1
    side[short_wins & (score <= sell_th)] = -1
    return score, side


def signed_expected_edge_bps(
    proba_long: np.ndarray,
    proba_short: np.ndarray,
    impulse: np.ndarray,
    *,
    baseline: np.ndarray | float = 0.5,
    move_bps: float = 20.0,
) -> np.ndarray:
    long_edge, short_edge = signed_ml_edges(proba_long, proba_short, baseline)
    imp = np.clip(np.asarray(impulse, dtype=float), 0.0, 1.5)
    net = long_edge - short_edge
    return net * float(move_bps) * (0.5 + imp)
