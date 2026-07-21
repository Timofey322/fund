"""Download OHLCV from Finam export API (5-minute bars, chunked requests).

Finam limits per request:
- trades: 1 day
- intraday candles: 4 months
- daily+: 5 years
"""

from __future__ import annotations

import http.cookiejar
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

from config import DATA_DIR
from data_platform.bars import bars_cache_path, safe_symbol

FINAM_EXPORT_HOST = "export.finam.ru"
FINAM_ICHARTS_URL = "https://www.finam.ru/cache/icharts/icharts.js"
FINAM_CACHE_DIR = DATA_DIR / "cache" / "finam"
FINAM_ICHARTS_CACHE = FINAM_CACHE_DIR / "icharts.js"
FINAM_META_CACHE = FINAM_CACHE_DIR / "instruments.parquet"

# Finam timeframe codes (p=): 3 = 5 minutes
FINAM_PERIOD_5MIN = 3

# User constraint: intraday candles <= 4 months per request
INTRADAY_CHUNK_DAYS = 120

# Market ids (finam.const.Market)
MARKET_SHARES = 1
MARKET_INDEXES = 6
MARKET_USA = 25

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.finam.ru/",
    "Accept": "*/*",
}

ERROR_TOO_LONG = "слишком большой временной период"
ERROR_IN_PROGRESS = "Система уже обрабатывает"
ERROR_FORBIDDEN = "Forbidden"

IMMUTABLE_PARAMS = {
    "d": "d",
    "f": "table",
    "e": ".csv",
    "dtf": "1",
    "tmf": "3",
    "MSOR": "0",
    "mstime": "on",
    "mstimever": "1",
    "sep": "3",
    "sep2": "1",
    "at": "1",
}


@dataclass(frozen=True)
class FinamInstrument:
    symbol: str
    em: int
    market: int
    code: str
    name: str


# Aliases requested by user -> lookup hints (resolved against icharts meta).
FINAM_ALIASES: dict[str, dict] = {
    "GAZP": {"code": "GAZP", "market": MARKET_SHARES},
    "SBER": {"code": "SBER", "market": MARKET_SHARES},
    "IMOEX": {"code": "IMOEX", "market": MARKET_INDEXES},
    "NASDAQ": {"code": "IXIC", "market": MARKET_INDEXES},
    "NDX": {"code": "NDX", "market": MARKET_INDEXES},
    "SP500": {"code": "INX", "market": MARKET_INDEXES},
    "SPX": {"code": "INX", "market": MARKET_INDEXES},
    "DAX": {"code": "ETF.DEDOW", "market": 28},
    "GER40": {"code": "ETF.DEDOW", "market": 28},
}

# Static fallback if meta lookup fails (em codes from Finam export pages).
FINAM_FALLBACK: dict[str, FinamInstrument] = {
    "GAZP": FinamInstrument("GAZP", 16842, MARKET_SHARES, "GAZP", "Gazprom"),
    "SBER": FinamInstrument("SBER", 3, MARKET_SHARES, "SBER", "Sberbank"),
    "IMOEX": FinamInstrument("IMOEX", 420450, MARKET_INDEXES, "IMOEX", "MOEX Index"),
    "NASDAQ": FinamInstrument("NASDAQ", 82075, MARKET_INDEXES, "IXIC", "NASDAQ Composite"),
    "SP500": FinamInstrument("SP500", 90, MARKET_INDEXES, "INX", "S&P 500"),
    "DAX": FinamInstrument("DAX", 19490, 28, "ETF.DEDOW", "Germany Index"),
    "GER40": FinamInstrument("GER40", 19490, 28, "ETF.DEDOW", "Germany Index"),
}


_COOKIE_JAR = http.cookiejar.CookieJar()
_URL_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_COOKIE_JAR))
_SESSION_WARMED = False


def _warm_export_session(code: str) -> None:
    global _SESSION_WARMED
    if _SESSION_WARMED:
        return
    warm_url = f"https://www.finam.ru/analysis/export/?code={code}"
    try:
        req = urllib.request.Request(warm_url, headers=DEFAULT_HEADERS)
        _URL_OPENER.open(req, timeout=30).read()
        _SESSION_WARMED = True
    except Exception:
        pass


def _fetch_text(url: str, *, retries: int = 4, sleep_s: float = 1.5, warm_code: str | None = None) -> str:
    if warm_code:
        _warm_export_session(warm_code)
    last_exc: Exception | None = None
    last_body = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
            raw = _URL_OPENER.open(req, timeout=90).read()
            for enc in ("cp1251", "utf-8"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            last_exc = exc
            try:
                last_body = exc.read().decode("cp1251", "replace")
            except Exception:
                last_body = ""
            time.sleep(sleep_s * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            time.sleep(sleep_s * (attempt + 1))
    hint = f" body={last_body[:200]!r}" if last_body else ""
    raise RuntimeError(f"Finam fetch failed for {url}: {last_exc}{hint}") from last_exc


def _load_icharts_js(*, force: bool = False, max_age_hours: int = 24) -> str:
    FINAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not force and FINAM_ICHARTS_CACHE.is_file():
        age_h = (time.time() - FINAM_ICHARTS_CACHE.stat().st_mtime) / 3600.0
        if age_h <= max_age_hours:
            return FINAM_ICHARTS_CACHE.read_text(encoding="cp1251", errors="replace")
    text = _fetch_text(FINAM_ICHARTS_URL)
    FINAM_ICHARTS_CACHE.write_text(text, encoding="cp1251", errors="replace")
    return text


def _parse_js_assignment(line: str) -> list[str]:
    start = line.find("[")
    end = line.find("]")
    if start < 0 or end < 0:
        raise ValueError(f"invalid js assignment: {line[:80]!r}")
    items = line[start + 1 : end]
    if items.startswith("'"):
        parts = items.split("','")
        parts[0] = parts[0].lstrip("'")
        parts[-1] = parts[-1].rstrip("'")
        return parts
    return [p.strip() for p in items.split(",") if p.strip()]


def _parse_instruments_meta(js_text: str) -> pd.DataFrame:
    lines = [ln for ln in js_text.splitlines() if ln.strip()]
    cols = ("id", "name", "code", "market")
    parsed: dict[str, list] = {c: [] for c in cols}
    for idx, col in enumerate(cols):
        parsed[col] = _parse_js_assignment(lines[idx])
    frame = pd.DataFrame(parsed)
    frame["market"] = pd.to_numeric(frame["market"], errors="coerce").astype("Int64")
    frame["id"] = pd.to_numeric(frame["id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["id", "market"])
    frame["id"] = frame["id"].astype(int)
    frame["market"] = frame["market"].astype(int)
    frame = frame[frame["market"] != -1]
    return frame.reset_index(drop=True)


def load_instruments_meta(*, force: bool = False) -> pd.DataFrame:
    FINAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not force and FINAM_META_CACHE.is_file():
        return pd.read_parquet(FINAM_META_CACHE)
    js = _load_icharts_js(force=force)
    meta = _parse_instruments_meta(js)
    meta.to_parquet(FINAM_META_CACHE, index=False)
    return meta


def resolve_instrument(symbol: str, meta: pd.DataFrame | None = None) -> FinamInstrument:
    sym = safe_symbol(symbol)
    if sym in FINAM_FALLBACK:
        return FINAM_FALLBACK[sym]
    hints = FINAM_ALIASES.get(sym)
    if hints is None:
        raise ValueError(f"Unknown Finam symbol {sym!r}; supported: {sorted(FINAM_ALIASES)}")

    meta = meta if meta is not None else load_instruments_meta()
    subset = meta
    if "market" in hints:
        subset = subset[subset["market"] == int(hints["market"])]
    if "code" in hints:
        subset = subset[subset["code"].astype(str).str.upper() == str(hints["code"]).upper()]
    if "name_contains" in hints:
        needle = str(hints["name_contains"]).upper()
        subset = subset[subset["name"].astype(str).str.upper().str.contains(needle, regex=False)]

    if subset.empty:
        if sym in FINAM_FALLBACK:
            return FINAM_FALLBACK[sym]
        raise ValueError(f"Finam instrument not found for alias {sym!r}")

    # Prefer exact code match when multiple rows remain.
    if "code" in hints:
        row = subset.iloc[0]
    else:
        row = subset.sort_values("name").iloc[0]

    return FinamInstrument(
        symbol=sym,
        em=int(row["id"]),
        market=int(row["market"]),
        code=str(row["code"]),
        name=str(row["name"]),
    )


def _split_date_range(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    if end < start:
        raise ValueError("end < start")
    chunks: list[tuple[date, date]] = []
    cur = start
    step = timedelta(days=chunk_days)
    while cur <= end:
        chunk_end = min(cur + step - timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _build_export_url(inst: FinamInstrument, start: date, end: date, *, period: int) -> str:
    params = {
        "p": period,
        "em": inst.em,
        "market": inst.market,
        "df": start.day,
        "mf": start.month - 1,
        "yf": start.year,
        "dt": end.day,
        "mt": end.month - 1,
        "yt": end.year,
        "cn": inst.code,
        "code": inst.code,
        "datf": 5,
        "fsp": 0,
    }
    q = urllib.parse.urlencode({**IMMUTABLE_PARAMS, **params})
    return f"http://{FINAM_EXPORT_HOST}/table.csv?{q}"


def _parse_export_csv(text: str) -> pd.DataFrame:
    if ERROR_TOO_LONG in text:
        raise ValueError("chunk exceeds Finam intraday window (reduce chunk_days)")
    if ERROR_FORBIDDEN in text:
        raise PermissionError("Finam export forbidden/throttled")
    if ERROR_IN_PROGRESS in text:
        raise RuntimeError("Finam export already in progress")
    if "<DATE>" not in text and "DATE" not in text:
        return pd.DataFrame()
    frame = pd.read_csv(StringIO(text), sep=";")
    if frame.empty:
        return frame
    rename = {
        "<DATE>": "date",
        "<TIME>": "time",
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<VOL>": "volume",
        "DATE": "date",
        "TIME": "time",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOL": "volume",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    if "date" not in frame.columns:
        return pd.DataFrame()
    if "time" in frame.columns:
        ts = frame["date"].astype(str).str.zfill(8) + frame["time"].astype(str).str.zfill(6)
        idx = pd.to_datetime(ts, format="%Y%m%d%H%M%S", errors="coerce")
    else:
        idx = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d", errors="coerce")
    out = frame[["open", "high", "low", "close", "volume"]].copy()
    out.index = idx
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"])
    return out


def fetch_finam_5min(
    inst: FinamInstrument,
    *,
    start: date,
    end: date,
    chunk_days: int = INTRADAY_CHUNK_DAYS,
    delay_s: float = 1.0,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    chunks = _split_date_range(start, end, chunk_days)
    for i, (c0, c1) in enumerate(chunks, start=1):
        url = _build_export_url(inst, c0, c1, period=FINAM_PERIOD_5MIN)
        print(f"    finam {inst.symbol}: chunk {i}/{len(chunks)} {c0}..{c1}", flush=True)
        retries = 0
        while True:
            try:
                text = _fetch_text(url, warm_code=inst.code)
                chunk = _parse_export_csv(text)
                break
            except RuntimeError:
                retries += 1
                if retries > 8:
                    raise
                time.sleep(delay_s * retries)
        if not chunk.empty:
            parts.append(chunk)
        time.sleep(delay_s)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def download_finam(
    symbols: list[str],
    *,
    timeframe: str = "5Min",
    start: date | None = None,
    end: date | None = None,
    years_back: int = 10,
    force: bool = False,
) -> dict[str, int]:
    """Download Finam intraday bars into standard parquet cache."""
    from data_platform.finam_trade_api import (
        download_trade_api_universe,
        is_trade_api_configured,
        is_trade_api_symbol,
    )
    from data_platform.moex_bars import download_moex_universe, is_finam_managed_symbol

    if timeframe != "5Min":
        raise ValueError("Finam downloader currently supports 5Min only")

    counts: dict[str, int] = {}
    need: list[str] = []
    for raw in symbols:
        sym = safe_symbol(raw)
        out_path = bars_cache_path(sym, timeframe)
        if out_path.is_file() and not force:
            existing = pd.read_parquet(out_path)
            if not existing.empty:
                counts[sym] = len(existing)
                print(f"  {sym}: cache {len(existing):,} bars", flush=True)
                continue
        need.append(sym)

    if not need:
        return counts

    trade_syms = [s for s in need if is_trade_api_symbol(s)]
    if is_trade_api_configured() and trade_syms:
        print("  source: Finam Trade API (FINAM_API_SECRET)", flush=True)
        try:
            api_counts = download_trade_api_universe(
                trade_syms,
                timeframe=timeframe,
                years_back=years_back,
                force=True,
            )
            counts.update({k: int(v) for k, v in api_counts.items()})
        except Exception as exc:
            print(f"  Finam Trade API unavailable ({exc}) — falling back to MOEX/Yahoo", flush=True)

    missing = [s for s in need if int(counts.get(s, 0)) <= 0]
    if not missing:
        return counts

    moex_syms = [s for s in missing if is_finam_managed_symbol(s)]
    if moex_syms:
        print("  source: MOEX ISS / yfinance fallback", flush=True)
        fb = download_moex_universe(
            moex_syms, timeframe=timeframe, years_back=years_back, force=True
        )
        counts.update({k: int(v) for k, v in fb.items()})
        missing = [s for s in missing if int(counts.get(s, 0)) <= 0]

    if not missing:
        return counts

    end_d = end or datetime.now(timezone.utc).date()
    start_d = start or (end_d - timedelta(days=365 * years_back))
    meta = load_instruments_meta()

    for raw in missing:
        sym = safe_symbol(raw)
        out_path = bars_cache_path(sym, timeframe)
        inst = resolve_instrument(sym, meta)
        print(
            f"  {sym}: Finam export em={inst.em} market={inst.market} code={inst.code} "
            f"({inst.name})",
            flush=True,
        )
        frame = pd.DataFrame()
        try:
            frame = fetch_finam_5min(inst, start=start_d, end=end_d)
        except Exception as exc:
            print(f"  {sym}: Finam export failed ({exc})", flush=True)
            counts[sym] = 0
            continue
        if frame.empty:
            print(f"  {sym}: no data returned", flush=True)
            counts[sym] = 0
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_path)
        counts[sym] = len(frame)
        print(
            f"  {sym}: {len(frame):,} bars | {frame.index.min()} .. {frame.index.max()}",
            flush=True,
        )
    return counts


DEFAULT_FINAM_UNIVERSE = ["GAZP", "SBER", "IMOEX", "NASDAQ", "SP500", "DAX"]
