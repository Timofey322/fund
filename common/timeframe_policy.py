"""Choose tradfi bar timeframe that maximizes available history (bar count)."""

from __future__ import annotations

import config as _cfg
from config import RTH_MINUTES, _tf_minutes, tradfi_max_days


def estimate_tradfi_bars(timeframe: str, calendar_days: int | None = None) -> int:
    """Approximate bar count per symbol for yfinance tradfi limits."""
    days = int(calendar_days or getattr(_cfg, "FUSION_HISTORY_DAYS", 7300))
    cap = int(tradfi_max_days(timeframe))
    cal = min(days, cap)
    rth_bars_per_day = max(1, int(RTH_MINUTES) // max(1, _tf_minutes(timeframe)))

    if timeframe == "1Day":
        return int(cal * 252 / 365)  # ~trading days
    if timeframe.endswith("Hour"):
        return int(cal * 252 / 365 * rth_bars_per_day)
    # Intraday minute bars: yfinance caps calendar window tightly.
    return int(cal * 252 / 365 * rth_bars_per_day)


def best_tradfi_timeframe_for_max_bars(calendar_days: int | None = None) -> tuple[str, dict[str, int]]:
    """
    Pick timeframe with the largest expected bar count for US ETF history.

    For yfinance tradfi, daily bars dominate (~5k bars / 20y) vs 1Hour (~3.4k)
    or 5Min (~3.2k over only ~59 calendar days).
    """
    days = int(calendar_days or getattr(_cfg, "FUSION_HISTORY_DAYS", 7300))
    candidates = ("1Day", "1Hour", "30Min", "15Min", "5Min")
    counts = {tf: estimate_tradfi_bars(tf, days) for tf in candidates}
    best = max(counts, key=counts.get)
    return best, counts
