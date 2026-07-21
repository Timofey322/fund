"""MOEX ISS 1-minute candles resampled to 5-minute OHLCV (Finam export fallback)."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from data_platform.bars import OHLCV_COLS, bars_cache_path, safe_symbol

MOEX_ISS = "https://iss.moex.com/iss"
MOEX_PAGE_SIZE = 500
MOEX_DELAY_S = 0.25
MOEX_MAX_RETRIES = 2
MOEX_HTTP_TIMEOUT_S = 15

# Finam alias -> MOEX board path (engine, market, security)
MOEX_INSTRUMENTS: dict[str, dict[str, str]] = {
    "GAZP": {"engine": "stock", "market": "shares", "security": "GAZP"},
    "SBER": {"engine": "stock", "market": "shares", "security": "SBER"},
    "IMOEX": {"engine": "stock", "market": "index", "security": "IMOEX"},
}

# Finam alias -> yfinance ticker (5m fallback when MOEX ISS is unreachable)
YFINANCE_INDEX_TICKERS: dict[str, str] = {
    "NASDAQ": "^IXIC",
    "SP500": "^GSPC",
    "DAX": "^GDAXI",
    "GER40": "^GDAXI",
}
YFINANCE_SHARE_TICKERS: dict[str, str] = {
    "SBER": "SBER.ME",
    "GAZP": "GAZP.ME",
}

FINAM_MANAGED_SYMBOLS = frozenset(
    {**MOEX_INSTRUMENTS, **YFINANCE_INDEX_TICKERS, **YFINANCE_SHARE_TICKERS}.keys()
)


def is_finam_managed_symbol(symbol: str) -> bool:
    return safe_symbol(symbol) in FINAM_MANAGED_SYMBOLS


def _moex_json(url: str) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(MOEX_MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "fund-moex/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=MOEX_HTTP_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            time.sleep(MOEX_DELAY_S * (2**attempt))
    raise RuntimeError(f"MOEX ISS fetch failed after {MOEX_MAX_RETRIES} tries: {last_exc}") from last_exc


def _fetch_moex_1min_page(
    *,
    engine: str,
    market: str,
    security: str,
    from_d: date,
    till_d: date,
    start: int,
) -> pd.DataFrame:
    params = {
        "from": from_d.isoformat(),
        "till": till_d.isoformat(),
        "interval": 1,
        "start": start,
    }
    path = (
        f"/engines/{engine}/markets/{market}/securities/{security}/candles.json"
    )
    url = MOEX_ISS + path + "?" + urllib.parse.urlencode(params)
    payload = _moex_json(url)
    block = payload.get("candles") or {}
    cols = block.get("columns") or []
    rows = block.get("data") or []
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=cols)
    idx = pd.to_datetime(frame["begin"], errors="coerce")
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(frame["open"].values, errors="coerce"),
            "high": pd.to_numeric(frame["high"].values, errors="coerce"),
            "low": pd.to_numeric(frame["low"].values, errors="coerce"),
            "close": pd.to_numeric(frame["close"].values, errors="coerce"),
            "volume": pd.to_numeric(frame["volume"].values, errors="coerce"),
        },
        index=idx,
    )
    out = out[~out.index.isna()].sort_index()
    return out


def _fetch_moex_1min_range(
    *,
    engine: str,
    market: str,
    security: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    cur = start
    chunk_n = 0
    while cur <= end:
        chunk_end = min(cur + timedelta(days=7), end)
        chunk_n += 1
        print(f"      MOEX {security}: chunk {chunk_n} {cur}..{chunk_end}", flush=True)
        offset = 0
        while True:
            page = _fetch_moex_1min_page(
                engine=engine,
                market=market,
                security=security,
                from_d=cur,
                till_d=chunk_end,
                start=offset,
            )
            if page.empty:
                break
            parts.append(page)
            if len(page) < MOEX_PAGE_SIZE:
                break
            offset += MOEX_PAGE_SIZE
            time.sleep(MOEX_DELAY_S)
        cur = chunk_end + timedelta(days=1)
        time.sleep(MOEX_DELAY_S)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _resample_5min(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = frame.resample("5min", label="left", closed="left").agg(agg)
    out = out.dropna(subset=["close"])
    if (out["volume"] > 0).any():
        out = out[out["volume"] > 0]
    return out[list(OHLCV_COLS)]


def fetch_moex_5min(
    symbol: str,
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    sym = safe_symbol(symbol)
    spec = MOEX_INSTRUMENTS.get(sym)
    if spec is None:
        raise ValueError(f"No MOEX mapping for {sym!r}")
    raw = _fetch_moex_1min_range(
        engine=spec["engine"],
        market=spec["market"],
        security=spec["security"],
        start=start,
        end=end,
    )
    return _resample_5min(raw)


def _normalize_yfinance(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower() for c in out.columns]
    else:
        out.columns = [str(c).lower() for c in out.columns]
    keep = [c for c in OHLCV_COLS if c in out.columns]
    out = out[keep].astype(float)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if (out["volume"] > 0).any():
        out = out[out["volume"] > 0]
    return out


def fetch_yfinance_share_5min(
    symbol: str,
    *,
    days: int = 59,
) -> pd.DataFrame:
    import yfinance as yf

    sym = safe_symbol(symbol)
    yf_sym = YFINANCE_SHARE_TICKERS.get(sym)
    if yf_sym is None:
        raise ValueError(f"No yfinance share mapping for {sym!r}")
    req_days = max(1, min(int(days), 59))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=req_days)
    raw = yf.download(
        yf_sym,
        interval="5m",
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    return _normalize_yfinance(raw)


def fetch_yfinance_index_5min(
    symbol: str,
    *,
    days: int = 59,
) -> pd.DataFrame:
    import yfinance as yf

    sym = safe_symbol(symbol)
    yf_sym = YFINANCE_INDEX_TICKERS.get(sym)
    if yf_sym is None:
        raise ValueError(f"No yfinance index mapping for {sym!r}")
    req_days = max(1, min(int(days), 59))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=req_days)
    raw = yf.download(
        yf_sym,
        interval="5m",
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    return _normalize_yfinance(raw)


def download_moex_universe(
    symbols: list[str],
    *,
    timeframe: str = "5Min",
    years_back: int = 10,
    force: bool = False,
) -> dict[str, int]:
    if timeframe != "5Min":
        raise ValueError("MOEX downloader supports 5Min only")
    end_d = datetime.now(timezone.utc).date()
    start_d = end_d - timedelta(days=365 * years_back)
    counts: dict[str, int] = {}

    for raw in symbols:
        sym = safe_symbol(raw)
        out_path = bars_cache_path(sym, timeframe)
        if out_path.is_file() and not force:
            existing = pd.read_parquet(out_path)
            if not existing.empty:
                counts[sym] = len(existing)
                print(f"  {sym}: cache {len(existing):,} bars", flush=True)
                continue

        try:
            if sym in MOEX_INSTRUMENTS:
                print(f"  {sym}: MOEX ISS 1m -> 5m ({start_d}..{end_d})", flush=True)
                try:
                    frame = fetch_moex_5min(sym, start=start_d, end=end_d)
                except Exception as moex_exc:
                    if sym in YFINANCE_SHARE_TICKERS:
                        print(
                            f"  {sym}: MOEX failed ({moex_exc}) — yfinance {YFINANCE_SHARE_TICKERS[sym]} 5m (~59d)",
                            flush=True,
                        )
                        frame = fetch_yfinance_share_5min(sym, days=59)
                    else:
                        raise
            elif sym in YFINANCE_SHARE_TICKERS:
                print(f"  {sym}: yfinance share 5m (max ~59d)", flush=True)
                frame = fetch_yfinance_share_5min(sym, days=59)
            elif sym in YFINANCE_INDEX_TICKERS:
                print(f"  {sym}: yfinance index 5m (max ~59d)", flush=True)
                frame = fetch_yfinance_index_5min(sym, days=59)
            else:
                print(f"  {sym}: unsupported", flush=True)
                counts[sym] = 0
                continue
        except Exception as exc:
            print(f"  {sym}: download failed — {exc}", flush=True)
            counts[sym] = 0
            continue

        if frame.empty:
            print(f"  {sym}: no data returned", flush=True)
            counts[sym] = 0
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_path)
        counts[sym] = len(frame)
        print(f"  {sym}: saved {len(frame):,} bars -> {out_path}", flush=True)
    return counts
