"""
Binance Spot data for crypto (public API, no keys).

Supported: BTCUSDT, ETHUSDT, SOLUSDT, ETHBTC (ETH priced in BTC).

OHLCV + taker buy/sell counts from klines (fields 8–9) — no aggTrades needed.
Cache layout matches stocks: cache/bars/<tf>/BTC.parquet, cache/bars_flow_5Min/.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from data_platform.bars import OHLCV_COLS, bars_cache_path
from config import BAR_TIMEFRAME, DATA_DIR

CACHE_DIR = DATA_DIR / "cache"
BARS_FLOW_SLIM = CACHE_DIR / "bars_flow_5Min"

BINANCE_API = "https://api.binance.com"
KLINES_URL = f"{BINANCE_API}/api/v3/klines"

BINANCE_PAIRS: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "ETHBTC": "ETHBTC",  # cross: ETH priced in BTC
}
CRYPTO_SYMBOLS = frozenset(BINANCE_PAIRS)

BINANCE_INTERVAL: dict[str, str] = {
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
    "30Min": "30m",
    "1Hour": "1h",
    "4Hour": "4h",
    "1Day": "1d",
}

KLINES_LIMIT = 1000
MAX_RETRIES = 8
SLIM_COLS = ("buy_count", "sell_count", "count_imbalance")


def is_crypto_symbol(symbol: str) -> bool:
    return symbol.upper() in CRYPTO_SYMBOLS


def binance_pair(symbol: str) -> str:
    sym = symbol.upper()
    if sym not in BINANCE_PAIRS:
        raise ValueError(
            f"Unsupported crypto symbol: {symbol} "
            f"(use {', '.join(sorted(BINANCE_PAIRS))})"
        )
    return BINANCE_PAIRS[sym]


def bars_flow_slim_path(symbol: str) -> Path:
    return BARS_FLOW_SLIM / f"{symbol.upper()}.parquet"


def _interval_ms(timeframe: str) -> int:
    mapping = {
        "1Min": 60_000,
        "5Min": 5 * 60_000,
        "15Min": 15 * 60_000,
        "30Min": 30 * 60_000,
        "1Hour": 60 * 60_000,
        "1Day": 24 * 60 * 60_000,
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported Binance timeframe: {timeframe}")
    return mapping[timeframe]


def _request_klines(
    pair: str,
    interval: str,
    start_ms: int,
    end_ms: int | None = None,
) -> list:
    params: dict = {
        "symbol": pair,
        "interval": interval,
        "limit": KLINES_LIMIT,
        "startTime": start_ms,
    }
    if end_ms is not None:
        params["endTime"] = end_ms

    backoff = 0.5
    for _ in range(MAX_RETRIES):
        r = requests.get(KLINES_URL, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Binance klines rate-limited: {pair}")


def _parse_klines(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        open_ms = int(row[0])
        vol = float(row[5])
        n_trades = int(row[8])
        taker_buy = float(row[9])
        taker_sell = max(vol - taker_buy, 0.0)
        if vol > 0 and n_trades > 0:
            buy_share = taker_buy / vol
            buy_count = n_trades * buy_share
            sell_count = n_trades - buy_count
        else:
            buy_count = sell_count = 0.0
        denom = buy_count + sell_count
        imb = (buy_count - sell_count) / denom if denom > 0 else 0.0
        records.append({
            "t": datetime.utcfromtimestamp(open_ms / 1000.0),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": vol,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "count_imbalance": imb,
        })
    df = pd.DataFrame(records).set_index("t").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def fetch_klines_history(
    symbol: str,
    timeframe: str = BAR_TIMEFRAME,
    *,
    days: int = 90,
    on_chunk: Callable[[int, int, int, object], None] | None = None,
) -> pd.DataFrame:
    """Download OHLCV + taker flow for last `days` calendar days."""
    pair = binance_pair(symbol)
    interval = BINANCE_INTERVAL.get(timeframe)
    if not interval:
        raise ValueError(f"No Binance interval for {timeframe}")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    step_ms = _interval_ms(timeframe)

    parts: list[pd.DataFrame] = []
    cursor = start_ms
    chunk_n = 0
    est_chunks = max(1, int((end_ms - start_ms) / (step_ms * KLINES_LIMIT)) + 1)
    while cursor < end_ms:
        rows = _request_klines(pair, interval, cursor, end_ms)
        if not rows:
            break
        piece = _parse_klines(rows)
        if piece.empty:
            break
        parts.append(piece)
        chunk_n += 1
        if on_chunk and chunk_n % 25 == 0:
            on_chunk(chunk_n, est_chunks, len(parts), piece.index[-1])
        last_ms = int(rows[-1][0])
        next_ms = last_ms + step_ms
        if next_ms <= cursor:
            break
        cursor = next_ms
        time.sleep(0.08)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def build_slim_panel(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + buy_count/sell_count from kline taker fields."""
    if df.empty:
        return df
    cols = list(OHLCV_COLS) + list(SLIM_COLS)
    have = [c for c in cols if c in df.columns]
    return df[have].copy()


def save_bars(symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
    path = bars_cache_path(symbol.upper(), timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    ohlcv = df[list(OHLCV_COLS)].copy()
    ohlcv.to_parquet(path)
    return path


def save_slim(symbol: str, df: pd.DataFrame) -> Path:
    path = bars_flow_slim_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = build_slim_panel(symbol, df)
    slim.to_parquet(path)
    return path


def download_crypto(
    symbols: list[str] | None = None,
    *,
    timeframe: str = BAR_TIMEFRAME,
    days: int = 90,
    force: bool = False,
) -> dict[str, int]:
    """Download crypto symbols from Binance into bar + slim flow caches."""
    syms = [s.upper() for s in (symbols or list(CRYPTO_SYMBOLS))]
    stats: dict[str, int] = {}
    for sym in syms:
        if not is_crypto_symbol(sym):
            print(f"  skip {sym}: not in {sorted(CRYPTO_SYMBOLS)}")
            continue
        bar_path = bars_cache_path(sym, timeframe)
        slim_path = bars_flow_slim_path(sym)
        if not force and bar_path.exists() and slim_path.exists():
            df = pd.read_parquet(slim_path)
            stats[sym] = len(df)
            print(f"  {sym}: cache {len(df):,} bars")
            continue
        print(f"  {sym}: Binance {BINANCE_PAIRS[sym]} | {days}d @ {timeframe}...")

        def _prog(n: int, est: int, _np: int, last_ts) -> None:
            print(f"    ... chunk {n}/{est} | through {last_ts}", flush=True)

        df = fetch_klines_history(sym, timeframe, days=days, on_chunk=_prog)
        if df.empty:
            stats[sym] = 0
            print(f"  {sym}: no data")
            continue
        save_bars(sym, timeframe, df)
        save_slim(sym, df)
        stats[sym] = len(df)
        print(f"  {sym}: {len(df):,} bars | {df.index.min()} .. {df.index.max()}")
    return stats


def load_crypto_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    path = bars_cache_path(symbol.upper(), timeframe)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_slim_panel(symbol: str) -> pd.DataFrame:
    path = bars_flow_slim_path(symbol)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
