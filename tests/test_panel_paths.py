"""Tests for per-instrument panel cache paths."""

from __future__ import annotations

import pandas as pd

from strategy.panel_paths import load_panel, panel_cache_path, save_panel


def test_panel_cache_path_per_symbol(tmp_path, monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cfg, "PANEL_CACHE_VERSION", 99)
    path = panel_cache_path("spy")
    assert path.name == "fusion_panel_SPY_v99.parquet"
    assert "panels" in str(path)


def test_save_and_load_panel(tmp_path, monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cfg, "PANEL_CACHE_VERSION", 1)
    df = pd.DataFrame({"ticker": ["SPY"], "close": [100.0]})
    save_panel("SPY", df)
    loaded = load_panel("SPY")
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded.iloc[0]["ticker"] == "SPY"
