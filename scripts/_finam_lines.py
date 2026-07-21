"""Inspect icharts.js first lines."""
from __future__ import annotations

import urllib.request

URL = "https://www.finam.ru/cache/icharts/icharts.js"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.finam.ru/",
}
req = urllib.request.Request(URL, headers=HEADERS)
lines = urllib.request.urlopen(req, timeout=60).read().decode("cp1251", "replace").splitlines()
for i, ln in enumerate(lines[:8]):
    print(i, ln[:200])
