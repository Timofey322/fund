"""Data role: download / load bars."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from operations.pipeline.agents.base import Agent
from operations.checkpoint import PipelineCheckpoint, PipelineContext
from data_platform.binance import download_crypto, load_slim_panel
from data_platform.universe import is_crypto_symbol, is_tradfi_symbol, split_tickers
from data_platform.yfinance_bars import download_tradfi
from data_platform.bars import bars_cache_path, load_ohlcv
from data_platform.moex_bars import is_finam_managed_symbol
from data_platform.finam_bars import download_finam
from config import BAR_TIMEFRAME, FUSION_HISTORY_DAYS, tradfi_max_days


class DataAgent(Agent):
    name = "data"

    def __init__(self, *, days: int | None = None, skip_download: bool = False, force: bool = False, timeframe: str | None = None):
        from config import FUSION_HISTORY_DAYS
        self.days = int(days if days is not None else FUSION_HISTORY_DAYS)
        self.skip_download = skip_download
        self.force = force
        self.timeframe = timeframe or BAR_TIMEFRAME

    def run(self, ctx: PipelineContext, ckpt: PipelineCheckpoint | None = None) -> PipelineContext:
        if ckpt:
            cached = ckpt.load(ctx.run_id, self.name)
            if cached and cached.get("tickers") == ctx.tickers:
                ctx.artifacts["data"] = cached
                return ctx

        crypto, tradfi = split_tickers(ctx.tickers)
        if not self.skip_download:
            if crypto:
                download_crypto(crypto, timeframe=self.timeframe, days=self.days, force=self.force)
            if tradfi:
                finam_syms = [t for t in tradfi if is_finam_managed_symbol(t)]
                yf_syms = [t for t in tradfi if t not in finam_syms]
                if finam_syms and self.timeframe == "5Min":
                    years_back = max(1, min(self.days, 3650) // 365)
                    download_finam(
                        finam_syms,
                        timeframe=self.timeframe,
                        years_back=years_back,
                        force=self.force,
                    )
                if yf_syms:
                    tradfi_days = tradfi_max_days(self.timeframe)
                    download_tradfi(
                        yf_syms,
                        timeframe=self.timeframe,
                        days=min(self.days, tradfi_days),
                        force=self.force,
                    )
        else:
            # Cache-only: report source without hitting Finam/Yahoo.
            finam_syms = [t for t in tradfi if is_finam_managed_symbol(t)]
            if finam_syms:
                print("  source: Finam Trade API (FINAM_API_SECRET)", flush=True)
            for t in ctx.tickers:
                path = bars_cache_path(t, self.timeframe)
                if path.is_file():
                    try:
                        n = len(load_ohlcv(t, self.timeframe))
                        print(f"  {t}: cache {n:,} bars", flush=True)
                    except Exception:
                        print(f"  {t}: cache present", flush=True)
                else:
                    print(f"  {t}: MISSING cache", flush=True)

        loaded = {}
        for t in ctx.tickers:
            if is_crypto_symbol(t):
                loaded[t] = not load_slim_panel(t).empty
            elif is_tradfi_symbol(t):
                loaded[t] = (
                    bars_cache_path(t, self.timeframe).is_file()
                    and not load_ohlcv(t, self.timeframe).empty
                )
            else:
                loaded[t] = False
        payload = {
            "tickers": ctx.tickers,
            "loaded": loaded,
            "days": self.days,
            "crypto": crypto,
            "tradfi": tradfi,
        }
        ctx.artifacts["data"] = payload
        if ckpt:
            ckpt.save(ctx.run_id, self.name, payload)
        return ctx
