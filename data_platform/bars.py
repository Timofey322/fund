"""Crypto bar cache helpers (Binance parquet layout)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from config import DATA_DIR

CACHE_DIR = DATA_DIR / "cache"
BARS_ROOT = CACHE_DIR / "bars"
OHLCV_COLS = ("open", "high", "low", "close", "volume")
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,12}$")


def safe_symbol(symbol: str) -> str:
    """Normalize and validate a cache filename symbol (no path traversal)."""
    raw = symbol.strip().upper()
    if not raw or ".." in raw or "/" in raw or "\\" in raw:
        raise ValueError(f"invalid symbol: {symbol!r}")
    cleaned = re.sub(r"[^A-Z0-9]", "", raw)
    if not _SYMBOL_RE.fullmatch(cleaned):
        raise ValueError(f"invalid symbol: {symbol!r}")
    return cleaned


def bars_cache_path(symbol: str, timeframe: str) -> Path:
    return BARS_ROOT / timeframe / f"{safe_symbol(symbol)}.parquet"


def load_closes(symbols: list[str], timeframe: str) -> pd.DataFrame:
    """Wide close matrix from cache."""
    series = {}
    for sym in symbols:
        path = bars_cache_path(sym, timeframe)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        series[sym] = df["close"].rename(sym)
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    from data_platform.binance import is_crypto_symbol, load_crypto_ohlcv

    if is_crypto_symbol(symbol):
        return load_crypto_ohlcv(symbol, timeframe)
    path = bars_cache_path(symbol, timeframe)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
