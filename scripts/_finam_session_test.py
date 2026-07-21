"""Test Finam export with session cookies."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import urllib.parse
import urllib.request
import http.cookiejar

from data_platform.finam_bars import IMMUTABLE_PARAMS, FinamInstrument

inst = FinamInstrument("GAZP", 16842, 1, "GAZP", "Gazprom")
start = date(2024, 1, 1)
end = date(2024, 1, 31)

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
    "datf": 5,
    "fsp": 0,
}
q = urllib.parse.urlencode({**IMMUTABLE_PARAMS, **params})
url = f"http://export.finam.ru/table.csv?{q}"

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.finam.ru/profile/moex-akcii/gazprom/export/",
    "Accept": "*/*",
}

print("warming session...")
warm = urllib.request.Request("https://www.finam.ru/profile/moex-akcii/gazprom/export/", headers=headers)
opener.open(warm, timeout=60).read()

print("fetch export", url[:120], "...")
req = urllib.request.Request(url, headers=headers)
try:
    raw = opener.open(req, timeout=90).read()
    text = raw.decode("cp1251", "replace")
    print("OK", len(raw), text[:300].replace("\n", " | "))
except Exception as exc:
    print("FAIL", exc)
