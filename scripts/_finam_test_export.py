"""Test Finam export URL for known instruments."""
from __future__ import annotations

import urllib.parse
import urllib.request
from datetime import date, timedelta

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.finam.ru/",
}

CANDIDATES = [
    ("GAZP", 1, 16842, "GAZP"),
    ("IMOEX", 6, 420450, "IMOEX"),
    ("IMOEX2", 24, 420450, "IMOEX"),
    ("SP500", 25, 385008, "SPX"),
    ("SP500b", 24, 385008, "SPX"),
    ("NASDAQ", 25, 385001, "NDX"),
    ("NASDAQb", 24, 385001, "NDX"),
    ("DAX", 25, 385054, "DAX"),
    ("DAXb", 24, 385054, "DAX"),
    ("GER40", 24, 419520, "GER40"),
]

end = date.today()
start = end - timedelta(days=30)


def build_url(market: int, em: int, code: str) -> str:
    params = {
        "market": market,
        "em": em,
        "code": code,
        "apply": 0,
        "df": start.day,
        "mf": start.month - 1,
        "yf": start.year,
        "from": start.strftime("%d.%m.%Y"),
        "dt": end.day,
        "mt": end.month - 1,
        "yt": end.year,
        "to": end.strftime("%d.%m.%Y"),
        "p": 3,  # 5 min
        "f": f"{code}_test",
        "e": ".csv",
        "cn": code,
        "dtf": 1,
        "tmf": 1,
        "MSOR": 0,
        "mstime": "on",
        "mstimever": 1,
        "sep": 1,
        "sep2": 1,
        "datf": 1,
        "at": 1,
    }
    return "http://export.finam.ru/" + code + ".txt?" + urllib.parse.urlencode(params)


for label, market, em, code in CANDIDATES:
    url = build_url(market, em, code)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        raw = urllib.request.urlopen(req, timeout=30).read()
        text = raw.decode("cp1251", "replace")
        lines = [ln for ln in text.splitlines() if ln.strip()][:3]
        ok = "<DATE>" in text or "TICKER" in text or (lines and lines[0][0].isdigit())
        print(label, "OK" if ok else "BAD", "bytes", len(raw), "head", lines[:2])
        if not ok:
            print(" ", text[:200].replace("\n", " "))
    except Exception as exc:
        print(label, "ERR", exc)
