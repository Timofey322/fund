"""Per-instrument fusion panel cache paths (no shared cross-ticker cache)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config as _cfg
from config import OUT_DIR


def panel_cache_version() -> int:
    return int(getattr(_cfg, "PANEL_CACHE_VERSION", 1))


def panel_cache_path(symbol: str) -> Path:
    sym = str(symbol).upper()
    ver = panel_cache_version()
    return OUT_DIR / "cache" / "panels" / f"fusion_panel_{sym}_v{ver}.parquet"


def panel_isolated_enabled() -> bool:
    return bool(getattr(_cfg, "FUSION_PANEL_ISOLATED", True))


def save_panel(symbol: str, panel: pd.DataFrame) -> Path:
    path = panel_cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return path


def load_panel(symbol: str) -> pd.DataFrame | None:
    path = panel_cache_path(symbol)
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def load_panels(symbols: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for sym in symbols:
        sub = load_panel(sym)
        if sub is not None and not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("bar_time").reset_index(drop=True)
