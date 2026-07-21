"""Download US-listed ETF / index proxies into the shared bar cache."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import BAR_TIMEFRAME, tradfi_max_days
from data_platform.bars import OHLCV_COLS, bars_cache_path
from data_platform.universe import TRADFI_SYMBOLS, is_tradfi_symbol, yfinance_ticker

# yfinance per-request window (calendar days) — smaller for fine bars.
_CHUNK_DAYS_BY_TF: dict[str, int] = {
    "5Min": 59,
    "15Min": 59,
    "30Min": 59,
    "1Hour": 365,
    "1Day": 3650,
}


def _chunk_cal_days(timeframe: str) -> int:
    return int(_CHUNK_DAYS_BY_TF.get(timeframe, 59))
_INTERVAL_MAP = {
    "5Min": "5m",
    "15Min": "15m",
    "30Min": "30m",
    "1Hour": "1h",
    "1Day": "1d",
}


def _interval(timeframe: str) -> str:
    if timeframe not in _INTERVAL_MAP:
        raise ValueError(f"Unsupported yfinance timeframe: {timeframe}")
    return _INTERVAL_MAP[timeframe]


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower() for c in out.columns]
    else:
        out.columns = [str(c).lower() for c in out.columns]
    rename = {"adj close": "close"}
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    keep = [c for c in OHLCV_COLS if c in out.columns]
    if "close" not in keep:
        return pd.DataFrame()
    out = out[keep].astype(float)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[out["volume"] > 0]
    return out


def fetch_yfinance_history(
    symbol: str,
    *,
    timeframe: str = BAR_TIMEFRAME,
    days: int = 1826,
) -> pd.DataFrame:
    """Download OHLCV; uses ``period=`` for hourly/daily to respect Yahoo rolling windows."""
    sym = symbol.upper()
    if not is_tradfi_symbol(sym):
        raise ValueError(f"{sym} is not a tradfi symbol")
    yf_sym = yfinance_ticker(sym)
    interval = _interval(timeframe)
    max_days = tradfi_max_days(timeframe)
    req_days = min(int(days), max_days)

    if timeframe == "1Hour":
        period = "730d" if req_days >= 700 else f"{req_days}d"
        print(f"    ... yfinance {sym} period={period} @ {timeframe}", flush=True)
        raw = yf.download(
            yf_sym,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return _normalize_ohlcv(raw)

    if timeframe == "1Day":
        end = datetime.now(timezone.utc)
        if req_days >= 4000:
            start = end - timedelta(days=req_days)
            print(
                f"    ... yfinance {sym} {start.date()}..{end.date()} @ {timeframe}",
                flush=True,
            )
            raw = yf.download(
                yf_sym,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        else:
            period = "10y" if req_days >= 3000 else "5y" if req_days >= 1500 else f"{max(req_days, 30)}d"
            print(f"    ... yfinance {sym} period={period} @ {timeframe}", flush=True)
            raw = yf.download(
                yf_sym,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        return _normalize_ohlcv(raw)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=req_days)
    chunks: list[pd.DataFrame] = []
    chunk_start = start
    chunk_days = _chunk_cal_days(timeframe)
    est_chunks = max(1, int(req_days / chunk_days) + 1)
    n = 0
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        n += 1
        print(
            f"    ... yfinance chunk {n}/{est_chunks} {sym} "
            f"{chunk_start.date()} .. {chunk_end.date()}",
            flush=True,
        )
        raw = yf.download(
            yf_sym,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        part = _normalize_ohlcv(raw)
        if not part.empty:
            chunks.append(part)
        chunk_start = chunk_end
        time.sleep(0.35)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def download_tradfi(
    symbols: list[str] | None = None,
    *,
    timeframe: str = BAR_TIMEFRAME,
    days: int | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Download tradfi symbols into cache/bars/<tf>/<SYM>.parquet."""
    max_days = tradfi_max_days(timeframe)
    req_days = min(int(days or max_days), max_days)
    if days is not None and int(days) > max_days:
        print(
            f"  tradfi: capping request {days}d -> {req_days}d "
            f"(yfinance {timeframe} limit ~{max_days}d)",
            flush=True,
        )
    syms = [s.upper() for s in (symbols or list(TRADFI_SYMBOLS))]
    stats: dict[str, int] = {}
    for sym in syms:
        if not is_tradfi_symbol(sym):
            print(f"  skip {sym}: not in {sorted(TRADFI_SYMBOLS)}")
            continue
        bar_path = bars_cache_path(sym, timeframe)
        if not force and bar_path.is_file():
            df = pd.read_parquet(bar_path)
            stats[sym] = len(df)
            print(f"  {sym}: cache {len(df):,} bars")
            continue
        print(f"  {sym}: yfinance {yfinance_ticker(sym)} | {req_days}d @ {timeframe}...")
        df = fetch_yfinance_history(sym, timeframe=timeframe, days=req_days)
        if df.empty:
            stats[sym] = 0
            print(f"  {sym}: no data")
            continue
        bar_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(bar_path)
        stats[sym] = len(df)
        print(f"  {sym}: {len(df):,} bars | {df.index.min()} .. {df.index.max()}")
    return stats
