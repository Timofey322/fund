"""Scrape Finam export pages for em/market/code."""
from __future__ import annotations

import re
import urllib.request

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,*/*",
    "Referer": "https://www.finam.ru/",
}

PAGES = {
    "GAZP": "https://www.finam.ru/profile/moex-akcii/gazprom/export/",
    "IMOEX": "https://www.finam.ru/profile/moex-indeksy/mosbirzhi/export/",
    "NASDAQ": "https://www.finam.ru/profile/indeksy-ssha/nasdaq-composite/export/",
    "SP500": "https://www.finam.ru/profile/indeksy-ssha/s-p-500/export/",
    "DAX": "https://www.finam.ru/profile/indeksy-evropy/dax/export/",
    "GER40": "https://www.finam.ru/profile/indeksy-evropy/dax/export/",
}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")


for name, url in PAGES.items():
    try:
        html = fetch(url)
    except Exception as exc:
        print(name, "ERR", exc)
        continue
    em = re.search(r"em=(\d+)", html)
    market = re.search(r"market=(\d+)", html)
    code = re.search(r"code=([A-Za-z0-9#._-]+)", html)
    print(
        name,
        f"em={em.group(1) if em else 'NA'}",
        f"market={market.group(1) if market else 'NA'}",
        f"code={code.group(1) if code else 'NA'}",
    )
