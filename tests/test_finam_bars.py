"""Tests for Finam 5-minute bar downloader."""

from __future__ import annotations

from datetime import date

import pandas as pd

from data_platform.finam_bars import (
    INTRADAY_CHUNK_DAYS,
    _parse_export_csv,
    _split_date_range,
    resolve_instrument,
)


def test_split_date_range_four_month_chunks():
    start = date(2024, 1, 1)
    end = date(2024, 12, 31)
    chunks = _split_date_range(start, end, INTRADAY_CHUNK_DAYS)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    assert all((b - a).days + 1 <= INTRADAY_CHUNK_DAYS for a, b in chunks)


def test_parse_export_csv_sample():
    raw = """<TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>
GAZP;5;20240110;100000;165.1;165.4;164.9;165.2;1200
GAZP;5;20240110;100500;165.2;165.5;165.0;165.4;900
"""
    frame = _parse_export_csv(raw)
    assert len(frame) == 2
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame["close"].iloc[-1] == 165.4


def test_resolve_gazp_fallback():
    inst = resolve_instrument("GAZP", pd.DataFrame())
    assert inst.em == 16842
    assert inst.code == "GAZP"
