"""Per-ticker allowed trade sides (long_only / short_only / both / none)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as _cfg


def allowed_sides_for_ticker(ticker: str) -> str:
    """Return policy mode for ``ticker``: both|long_only|short_only|none."""
    policy = getattr(_cfg, "FUSION_SIDE_POLICY", None) or {}
    raw = policy.get(str(ticker).upper(), "both")
    mode = str(raw).strip().lower().replace("-", "_")
    if mode in ("long", "longs"):
        return "long_only"
    if mode in ("short", "shorts"):
        return "short_only"
    if mode in ("off", "flat", "block", "disabled"):
        return "none"
    if mode in ("long_only", "short_only", "both", "none"):
        return mode
    return "both"


def side_allowed(ticker: str, side: int) -> bool:
    """True if ``side`` (+1 long / -1 short) is permitted for ``ticker``."""
    mode = allowed_sides_for_ticker(ticker)
    s = int(side)
    if s == 0:
        return False
    if mode == "none":
        return False
    if mode == "both":
        return True
    if mode == "long_only":
        return s > 0
    if mode == "short_only":
        return s < 0
    return True


def side_policy_mask(tickers: pd.Series, sides: pd.Series) -> pd.Series:
    """Boolean mask: rows whose (ticker, side) pass ``FUSION_SIDE_POLICY``."""
    if tickers.empty:
        return pd.Series(dtype=bool)
    t = tickers.astype(str).str.upper()
    s = sides.fillna(0).astype(int)
    ok = np.array(
        [side_allowed(ti, si) for ti, si in zip(t.to_numpy(), s.to_numpy())],
        dtype=bool,
    )
    return pd.Series(ok, index=tickers.index)
