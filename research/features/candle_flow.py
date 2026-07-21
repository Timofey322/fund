"""
Impute buy/sell volume from OHLCV candles (no tick data).

Bullish body or long lower wick -> more volume attributed to buyers.
Long upper wick or bearish body -> more volume attributed to sellers.

Core metric: Close Location Value (CLV), blended with wick asymmetry and body
direction. Standard name in market microstructure: candle-based volume
classification / imputed order flow (related to Bulk Volume Classification).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd

CANDLE_FLOW_COLS = (
    "buy_vol_est",
    "sell_vol_est",
    "vol_imbalance",
    "buy_share",
    "clv",
    "lower_wick_ratio",
    "upper_wick_ratio",
    "body_ratio",
    "hammer_score",
)

# Weights for composite buy-share: CLV + lower-wick absorption + body direction
_W_CLV = 0.45
_W_WICK = 0.30
_W_BODY = 0.25


def _require_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV columns required, missing: {missing}")


def candle_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """Wick/body ratios and CLV per bar."""
    _require_ohlcv(df)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    hl = (h - l).replace(0, np.nan)
    lower_wick = (np.minimum(o, c) - l).clip(lower=0)
    upper_wick = (h - np.maximum(o, c)).clip(lower=0)
    body = (c - o).abs()

    clv = ((c - l) - (h - c)) / hl
    clv = clv.fillna(0.0).clip(-1.0, 1.0)

    lower_wick_ratio = (lower_wick / hl).fillna(0.0)
    upper_wick_ratio = (upper_wick / hl).fillna(0.0)
    body_ratio = (body / hl).fillna(0.0)

    # Hammer-like: long lower shadow relative to body (buyers absorbed selling)
    hammer_score = (lower_wick / body.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    hammer_score = hammer_score.fillna(lower_wick_ratio)

    return pd.DataFrame(
        {
            "clv": clv,
            "lower_wick_ratio": lower_wick_ratio,
            "upper_wick_ratio": upper_wick_ratio,
            "body_ratio": body_ratio,
            "hammer_score": hammer_score,
        },
        index=df.index,
    )


def decompose_candle_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split bar volume into estimated buy vs sell using candle shape.

    buy_share in [0, 1]:
      - close near high (CLV > 0) -> more buy volume
      - long lower wick -> buyers stepped in (bullish absorption)
      - bullish body (close > open) -> more buy volume
    """
    _require_ohlcv(df)
    out = df.copy()
    anatomy = candle_anatomy(df)

    o = out["open"].astype(float)
    c = out["close"].astype(float)
    v = out["volume"].astype(float).fillna(0.0)

    hl = (out["high"].astype(float) - out["low"].astype(float)).replace(0, np.nan)
    lower_wick = (np.minimum(o, c) - out["low"].astype(float)).clip(lower=0)
    upper_wick = (out["high"].astype(float) - np.maximum(o, c)).clip(lower=0)
    wick_bias = ((lower_wick - upper_wick) / hl).fillna(0.0).clip(-1.0, 1.0)

    body_sign = np.sign(c - o)
    body_sign = pd.Series(body_sign, index=df.index).replace(0, np.nan).ffill().fillna(0.0)

    clv_part = (anatomy["clv"] + 1.0) / 2.0
    wick_part = (wick_bias + 1.0) / 2.0
    body_part = (body_sign + 1.0) / 2.0

    buy_share = (_W_CLV * clv_part + _W_WICK * wick_part + _W_BODY * body_part).clip(0.0, 1.0)

    out["buy_share"] = buy_share
    out["buy_vol_est"] = v * buy_share
    out["sell_vol_est"] = v * (1.0 - buy_share)
    denom = v.replace(0, np.nan)
    out["vol_imbalance"] = ((out["buy_vol_est"] - out["sell_vol_est"]) / denom).fillna(0.0)

    for col in anatomy.columns:
        out[col] = anatomy[col]

    return out


def attach_candle_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Alias: full OHLCV panel with imputed flow + anatomy columns."""
    if df.empty:
        return df
    return decompose_candle_volume(df)
