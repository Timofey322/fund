"""Debug Finam export 400 errors."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import urllib.parse
import urllib.request
from datetime import date

from data_platform.finam_bars import IMMUTABLE_PARAMS, FinamInstrument

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.finam.ru/",
}

inst = FinamInstrument("GAZP", 16842, 1, "GAZP", "Gazprom")
start = date(2024, 1, 1)
end = date(2024, 2, 1)

variants = [
    {"datf": 5, "fsp": 0, "https": False},
    {"datf": 1, "fsp": 0, "https": False, "from_to": True},
    {"datf": 5, "fsp": 0, "https": True},
]

for i, v in enumerate(variants, 1):
    params = {
        "p": 3,
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
        "datf": v["datf"],
        "fsp": v["fsp"],
    }
    if v.get("from_to"):
        params["from"] = start.strftime("%d.%m.%Y")
        params["to"] = end.strftime("%d.%m.%Y")
        params["apply"] = 0
    q = urllib.parse.urlencode({**IMMUTABLE_PARAMS, **params})
    scheme = "https" if v.get("https") else "http"
    url = f"{scheme}://export.finam.ru/table.csv?{q}"
    print(f"\n--- variant {i} ---")
    print(url[:180], "...")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        raw = urllib.request.urlopen(req, timeout=60).read()
        text = raw.decode("cp1251", "replace")
        print("OK bytes", len(raw), "head:", text[:200].replace("\n", " | "))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("cp1251", "replace") if exc.fp else ""
        print("HTTP", exc.code, body[:300])
