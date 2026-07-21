"""HMM role: bar-level Markov regime."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd

from operations.pipeline.agents.base import Agent
from operations.checkpoint import PipelineCheckpoint, PipelineContext
from data_platform.bars import load_closes
from data_platform.binance import load_crypto_ohlcv
from config import HMM_FREQUENCY, HMM_BAR_MAX_ROWS, HMM_BAR_PLOT_ROWS, BAR_TIMEFRAME, HMM_REGIME_CACHE_PATH
from research.regime.hmm import LAST_HMM_META, build_hmm_regime_frame


class HmmAgent(Agent):
    name = "hmm"

    def __init__(self, *, max_bars: int | None = None):
        self.max_bars = max_bars

    def run(self, ctx: PipelineContext, ckpt: PipelineCheckpoint | None = None) -> PipelineContext:
        if ckpt:
            cached = ckpt.load(ctx.run_id, self.name)
            if cached:
                ctx.artifacts["hmm"] = cached
                return ctx

        if self._regime_cache_fresh(ctx.tickers):
            regime = pd.read_parquet(HMM_REGIME_CACHE_PATH)
            print(f"    HMM regime from cache ({len(regime):,} rows)", flush=True)
            payload = {
                "rows": len(regime),
                "hmm_frequency": HMM_FREQUENCY,
                "regime_tickers": ctx.tickers,
                "tail_distribution": self._tail_dist(regime),
                "regime_cache": str(HMM_REGIME_CACHE_PATH),
                "source": "cache",
            }
            ctx.artifacts["hmm"] = payload
            ctx.artifacts["hmm_frame"] = regime
            if ckpt:
                ckpt.save(ctx.run_id, self.name, payload)
            return ctx

        print(f"    HMM: fitting regime ({len(ctx.tickers)} tickers)...", flush=True)
        prices = self._load_prices(ctx.tickers)
        n_bars = int(len(prices)) if not prices.empty else 0
        print(f"    HMM: {n_bars:,} bars x {len(prices.columns)} symbols", flush=True)
        regime = build_hmm_regime_frame(prices)
        HMM_REGIME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        regime.to_parquet(HMM_REGIME_CACHE_PATH, index=False)
        payload = {
            "rows": len(regime),
            "hmm_frequency": LAST_HMM_META.get("hmm_frequency", HMM_FREQUENCY),
            "regime_tickers": LAST_HMM_META.get("regime_tickers", ctx.tickers),
            "tail_distribution": self._tail_dist(regime),
            "regime_cache": str(HMM_REGIME_CACHE_PATH),
        }
        ctx.artifacts["hmm"] = payload
        ctx.artifacts["hmm_frame"] = regime
        if ckpt:
            ckpt.save(ctx.run_id, self.name, payload)
        return ctx

    def _load_prices(self, tickers: list[str]) -> pd.DataFrame:
        from config import REGIME_TICKERS

        regime_syms = [t for t in REGIME_TICKERS if t in tickers]
        load_syms = regime_syms or list(tickers)
        cap = self.max_bars if self.max_bars is not None else HMM_BAR_MAX_ROWS
        cols: dict[str, pd.Series] = {}
        failed: list[str] = []
        for t in load_syms:
            try:
                ohlcv = load_crypto_ohlcv(t, BAR_TIMEFRAME)
                if not ohlcv.empty:
                    s = ohlcv["close"].astype(float)
                    if cap:
                        s = s.iloc[-int(cap):]
                    cols[t] = s
                    continue
            except Exception as exc:
                failed.append(f"{t}:{exc}")
        if failed:
            print(f"    HMM price load warnings: {', '.join(failed[:4])}", flush=True)
        if not cols:
            closes = load_closes(load_syms, BAR_TIMEFRAME)
            for t in load_syms:
                if t in closes.columns:
                    s = closes[t].dropna()
                    if cap:
                        s = s.iloc[-int(cap):]
                    cols[t] = s
        return pd.DataFrame(cols)

    @staticmethod
    def _regime_cache_fresh(tickers: list[str]) -> bool:
        if not HMM_REGIME_CACHE_PATH.is_file():
            return False
        cache_mtime = HMM_REGIME_CACHE_PATH.stat().st_mtime
        from config import REGIME_TICKERS
        from data_platform.bars import bars_cache_path

        watch = [t for t in REGIME_TICKERS if t in tickers] or list(tickers)
        for sym in watch:
            path = bars_cache_path(sym, BAR_TIMEFRAME)
            if path.is_file() and path.stat().st_mtime > cache_mtime:
                return False
        return True

    @staticmethod
    def _tail_dist(regime: pd.DataFrame) -> dict[str, float]:
        if regime.empty:
            return {}
        from common.naming import COL_PROB_HMM_STRESS, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_IMPULSE

        tail = regime.tail(5000)
        dom = tail[[COL_PROB_HMM_IMPULSE, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_STRESS]].idxmax(axis=1)
        counts = dom.value_counts(normalize=True)
        return {str(k): round(float(v), 4) for k, v in counts.items()}
