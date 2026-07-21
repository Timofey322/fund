"""Test finam-export library lookups and 5m download."""
from __future__ import annotations

import datetime as dt

from finam.const import Market, Timeframe
from finam.export import Exporter, LookupComparator

QUERIES = [
    ("GAZP", dict(code="GAZP", market=Market.SHARES)),
    ("IMOEX", dict(name="MOEX", market=Market.INDEXES, name_comparator=LookupComparator.CONTAINS)),
    ("NASDAQ", dict(name="NASDAQ", market=Market.USA, name_comparator=LookupComparator.CONTAINS)),
    ("SP500", dict(name="S&P", market=Market.USA, name_comparator=LookupComparator.CONTAINS)),
    ("DAX", dict(name="DAX", market=Market.USA, name_comparator=LookupComparator.CONTAINS)),
    ("DAX_EU", dict(name="DAX", market=Market.INDEXES, name_comparator=LookupComparator.CONTAINS)),
]

exporter = Exporter()
start = dt.date.today() - dt.timedelta(days=30)
end = dt.date.today()

for label, kwargs in QUERIES:
    try:
        found = exporter.lookup(**kwargs)
        print(f"\n{label}: {len(found)} matches")
        print(found[["id", "code", "name", "market"]].head(5).to_string())
        row = found.iloc[0]
        data = exporter.download(
            row["id"],
            Market(row["market"]),
            start_date=start,
            end_date=end,
            timeframe=Timeframe.MINUTES5,
            delay=1.2,
        )
        print(f"  downloaded rows={len(data)} cols={list(data.columns)}")
        print(data.tail(2).to_string())
    except Exception as exc:
        print(f"\n{label}: FAIL {type(exc).__name__}: {exc}")
