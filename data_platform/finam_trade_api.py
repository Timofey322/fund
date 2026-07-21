"""Finam Trade API — 5-minute historical bars (REST)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any

from pathlib import Path

import pandas as pd

from data_platform.bars import OHLCV_COLS, bars_cache_path, safe_symbol

FINAM_API_BASE = os.environ.get("FINAM_API_BASE", "https://api.finam.ru")
FINAM_M5_CHUNK_DAYS = 14  # API max 30d; smaller chunks avoid IncompleteRead on liquid names
FINAM_REQUEST_DELAY_S = 0.35
FINAM_MAX_RETRIES = 2

# Desk alias -> Finam Trade API instrument symbol
FINAM_TRADE_SYMBOLS: dict[str, str] = {
    "GAZP": "GAZP@MISX",
    "SBER": "SBER@MISX",
    "IMOEX": "IMOEX@MISX",
    "NASDAQ": "NDX@_SCI",  # NASDAQ-100 index
    "SP500": "SPY@ARCX",  # S&P 500 ETF proxy (no cash SPX index in API)
    "DAX": "DAX@XNMS",  # Global X DAX Germany ETF
}

TRADE_API_ALIASES = frozenset(FINAM_TRADE_SYMBOLS.keys())

_ACCESS_TOKEN: str | None = None
_ACCESS_TOKEN_AT: float = 0.0
_TOKEN_TTL_S = 10 * 60.0


def _secret() -> str:
    secret = os.environ.get("FINAM_API_SECRET", "").strip()
    if not secret:
        raise ValueError(
            "FINAM_API_SECRET is not set. Add it to fund/.env (Finam portal -> Tokens)."
        )
    return secret


def _load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = val.strip().strip('"').strip("'")


def is_trade_api_configured() -> bool:
    _load_dotenv()
    return bool(os.environ.get("FINAM_API_SECRET", "").strip())


def is_trade_api_symbol(symbol: str) -> bool:
    return safe_symbol(symbol) in TRADE_API_ALIASES


def resolve_trade_symbol(alias: str) -> str:
    sym = safe_symbol(alias)
    if sym not in FINAM_TRADE_SYMBOLS:
        raise ValueError(f"No Finam Trade API mapping for {alias!r}")
    return FINAM_TRADE_SYMBOLS[sym]


class FinamAuthError(RuntimeError):
    """Access token expired or invalid."""


def _is_auth_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "401" in text or "unauthorized" in text or "invalid access key" in text


def _http_json(
    method: str,
    path: str,
    *,
    body: dict | None = None,
    token: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    url = FINAM_API_BASE.rstrip("/") + path
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = token

    last_exc: Exception | None = None
    for attempt in range(FINAM_MAX_RETRIES):
        try:
            import requests

            resp = requests.request(
                method,
                url,
                headers=headers,
                json=body if body is not None else None,
                timeout=(10, timeout),
            )
            if resp.status_code == 401:
                raise FinamAuthError(resp.text[:200])
            resp.raise_for_status()
            return resp.json()
        except FinamAuthError:
            raise
        except Exception as exc:
            last_exc = exc
            time.sleep(FINAM_REQUEST_DELAY_S * (2**attempt))
    raise RuntimeError(f"Finam API {method} {path} failed: {last_exc}") from last_exc


def get_access_token(*, force: bool = False) -> str:
    global _ACCESS_TOKEN, _ACCESS_TOKEN_AT
    _load_dotenv()
    now = time.time()
    if not force and _ACCESS_TOKEN and (now - _ACCESS_TOKEN_AT) < _TOKEN_TTL_S:
        return _ACCESS_TOKEN
    # Short connect timeout so unreachable API fails fast into MOEX/Yahoo fallback.
    payload = _http_json("POST", "/v1/sessions", body={"secret": _secret()}, timeout=20)
    token = str(payload.get("token") or "")
    if not token:
        raise RuntimeError(f"Finam session response missing token: {payload!r}")
    _ACCESS_TOKEN = token
    _ACCESS_TOKEN_AT = now
    return token


def _parse_bar_value(val: Any) -> float:
    if isinstance(val, dict):
        return float(val.get("value", 0) or 0)
    return float(val or 0)


def _bars_to_frame(bars: list[dict]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    rows = []
    for bar in bars:
        rows.append(
            {
                "open": _parse_bar_value(bar.get("open")),
                "high": _parse_bar_value(bar.get("high")),
                "low": _parse_bar_value(bar.get("low")),
                "close": _parse_bar_value(bar.get("close")),
                "volume": _parse_bar_value(bar.get("volume")),
                "ts": bar.get("timestamp"),
            }
        )
    frame = pd.DataFrame(rows)
    idx = pd.to_datetime(frame["ts"], utc=True, errors="coerce").dt.tz_convert(None)
    out = frame[["open", "high", "low", "close", "volume"]].copy()
    out.index = idx
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _split_ranges(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cur = start
    step = timedelta(days=chunk_days)
    while cur < end:
        nxt = min(cur + step, end)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def fetch_bars_5min(
    trade_symbol: str,
    *,
    start: datetime,
    end: datetime,
    token: str | None = None,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch 5m bars; splits into <=14-day API windows."""
    from pathlib import Path as _Path

    access = token or get_access_token()
    enc = urllib.parse.quote(trade_symbol, safe="")
    parts: list[pd.DataFrame] = []
    if checkpoint_path and _Path(checkpoint_path).is_file():
        cached = pd.read_parquet(checkpoint_path)
        if not cached.empty:
            parts.append(cached)
            start = max(start, cached.index.max().to_pydatetime().replace(tzinfo=timezone.utc))
            print(f"      resume {trade_symbol} from {start.date()} ({len(cached):,} bars)", flush=True)
    chunks = _split_ranges(start, end, FINAM_M5_CHUNK_DAYS)
    for i, (c0, c1) in enumerate(chunks, start=1):
        q = urllib.parse.urlencode(
            {
                "timeframe": "TIME_FRAME_M5",
                "interval.start_time": c0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "interval.end_time": c1.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        path = f"/v1/instruments/{enc}/bars?{q}"
        print(f"      Finam API {trade_symbol}: chunk {i}/{len(chunks)} {c0.date()}..{c1.date()}", flush=True)
        if i > 1 and (i - 1) % 15 == 0:
            access = get_access_token(force=True)
        for attempt in range(2):
            try:
                payload = _http_json("GET", path, token=access, timeout=180)
                break
            except FinamAuthError:
                access = get_access_token(force=True)
                if attempt == 1:
                    raise
            except RuntimeError as exc:
                if _is_auth_error(exc):
                    access = get_access_token(force=True)
                    if attempt == 1:
                        raise
                else:
                    raise
        chunk = _bars_to_frame(payload.get("bars") or [])
        if not chunk.empty:
            parts.append(chunk)
            if checkpoint_path is not None and i % 10 == 0:
                merged = pd.concat(parts).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                merged.to_parquet(checkpoint_path)
        time.sleep(FINAM_REQUEST_DELAY_S)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[list(OHLCV_COLS)]


def download_trade_api_universe(
    symbols: list[str],
    *,
    timeframe: str = "5Min",
    years_back: int = 10,
    force: bool = False,
) -> dict[str, int]:
    if timeframe != "5Min":
        raise ValueError("Finam Trade API downloader supports 5Min only")
    if not is_trade_api_configured():
        raise ValueError("FINAM_API_SECRET not configured")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years_back)
    token = get_access_token()
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

        trade_sym = resolve_trade_symbol(sym)
        print(f"  {sym}: Finam Trade API {trade_sym} ({years_back}y)", flush=True)
        ckpt = out_path.with_suffix(".download.parquet")
        try:
            frame = fetch_bars_5min(
                trade_sym, start=start, end=end, token=token, checkpoint_path=ckpt
            )
        except Exception as exc:
            if ckpt.is_file():
                partial = pd.read_parquet(ckpt)
                if not partial.empty:
                    partial.to_parquet(out_path)
                    counts[sym] = len(partial)
                    print(f"  {sym}: partial save {len(partial):,} bars (failed: {exc})", flush=True)
                    continue
            print(f"  {sym}: download failed — {exc}", flush=True)
            counts[sym] = 0
            continue
        if frame.empty:
            print(f"  {sym}: no data returned", flush=True)
            counts[sym] = 0
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_path)
        if ckpt.is_file():
            ckpt.unlink(missing_ok=True)
        counts[sym] = len(frame)
        print(
            f"  {sym}: saved {len(frame):,} bars | {frame.index.min()} .. {frame.index.max()}",
            flush=True,
        )
    return counts
