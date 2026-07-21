"""Probe Finam icharts.js for instrument IDs."""
from __future__ import annotations

import re
import urllib.request

URL = "https://www.finam.ru/cache/icharts/icharts.js"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.finam.ru/",
}
CODES = [
    "GAZP", "IMOEX", "MOEX", "NDX", "SPX", "SP500", "#SPX", "#NQ", "DAX", "GER40",
    "NDX100", "NASDAQ", "ES", "RTS", "SBER", "NDXm", "NQ", "SPY", "QQQ",
]

req = urllib.request.Request(URL, headers=HEADERS)
txt = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
print(f"icharts.js bytes={len(txt)}")
for code in CODES:
    pat = re.escape(code) + r"':(\d+)"
    m = re.search(pat, txt)
    print(f"{code:8} -> {m.group(1) if m else 'NA'}")

for s in ["GAZP", "gazprom", "16842", "IMOEX", "SPX", "NDX", "DAX"]:
    i = txt.find(s)
    print(f"\nfind {s!r} @ {i}")
    if i >= 0:
        print(txt[max(0, i - 100) : i + 150])

# fuzzy search
for needle in ["IMOEX", "NASDAQ", "SP500", "DAX", "GER"]:
    hits = re.findall(r"'([^']*" + needle + r"[^']*)':(\d+)", txt, flags=re.I)
    print(f"\n-- contains {needle!r} ({len(hits)}) --")
    for name, em in hits[:15]:
        print(f"  {name}: {em}")
