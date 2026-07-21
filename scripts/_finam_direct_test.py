"""Test finam-export with direct icharts.js URL."""
from __future__ import annotations

import datetime as dt

from finam.const import Market, Timeframe
from finam.export import Exporter, ExporterMetaFile, LookupComparator

ICHARTS = "https://www.finam.ru/cache/icharts/icharts.js"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://www.finam.ru/",
}


def fetch_url(url, lines=False):
    import urllib.request

    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=60).read()
    if lines:
        return raw.decode("cp1251", "replace").splitlines(True)
    return raw.decode("cp1251", "replace")


meta = ExporterMetaFile(ICHARTS, fetcher=fetch_url).parse_df()
print("meta rows", len(meta))

exporter = Exporter(fetcher=fetch_url)
exporter._meta._meta = meta

queries = [
    ("GAZP", dict(code="GAZP", market=Market.SHARES)),
    ("IMOEX", dict(code="IMOEX", market=Market.INDEXES)),
    ("NASDAQ", dict(name="NASDAQ", market=Market.USA, name_comparator=LookupComparator.CONTAINS)),
    ("SP500", dict(name="S&P 500", market=Market.USA, name_comparator=LookupComparator.CONTAINS)),
    ("DAX", dict(name="DAX", market=Market.USA, name_comparator=LookupComparator.CONTAINS)),
]

start = dt.date.today() - dt.timedelta(days=60)
end = dt.date.today()

for label, kwargs in queries:
    try:
        found = exporter.lookup(**kwargs)
        print(f"\n{label}: matches={len(found)}")
        print(found[["code", "name", "market"]].head(8).to_string())
        row = found.iloc[0]
        data = exporter.download(
            row.name,
            Market(int(row["market"])),
            start_date=start,
            end_date=end,
            timeframe=Timeframe.MINUTES5,
            delay=1.0,
        )
        print(f"rows={len(data)}")
        print(data.head(2).to_string())
    except Exception as exc:
        print(f"\n{label}: {type(exc).__name__}: {exc}")
