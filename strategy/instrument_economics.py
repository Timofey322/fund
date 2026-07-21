"""Instrument-relative economics — floors, sizing, SQ from vol & RT cost.

No absolute per-ticker bps hardcodes. All thresholds derive from:
  - round-trip cost (commission + vol-scaled slippage)
  - realized annualized volatility of the instrument
  - configurable *multipliers* (dimensionless), not ticker-specific bps
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

import config as _cfg
from data_platform.universe import is_crypto_symbol, is_tradfi_symbol
from simulation.execution_costs import (
    round_trip_cost_bps_for_ticker,
    slippage_bps_per_side,
)


def _bars_per_year() -> float:
    return float(getattr(_cfg, "BARS_PER_YEAR", 252.0))


def _vol_ref_for_ticker(ticker: str) -> float:
    sym = str(ticker).upper()
    if is_tradfi_symbol(sym):
        return float(getattr(_cfg, "SLIPPAGE_VOL_REF_TRADFI", 0.18))
    if is_crypto_symbol(sym):
        return float(getattr(_cfg, "SLIPPAGE_VOL_REF_CRYPTO", 0.80))
    return float(getattr(_cfg, "SLIPPAGE_VOL_REF_DEFAULT", 0.50))


def realized_vol_ann(
    closes: pd.Series,
    *,
    bars_per_year: float | None = None,
    min_bars: int = 50,
) -> float | None:
    """Annualized realized vol from close series (log returns)."""
    c = pd.to_numeric(closes, errors="coerce").dropna()
    if len(c) < int(min_bars):
        return None
    rets = np.log(c).diff().dropna()
    if len(rets) < max(20, min_bars // 2):
        return None
    bpy = float(bars_per_year if bars_per_year is not None else _bars_per_year())
    vol = float(rets.std(ddof=1) * math.sqrt(bpy))
    if not math.isfinite(vol) or vol <= 0:
        return None
    return vol


@lru_cache(maxsize=64)
def instrument_vol_ann(ticker: str) -> float:
    """Best-effort realized vol for ``ticker`` from bar cache; else asset-class ref."""
    sym = str(ticker).upper()
    try:
        from config import BAR_TIMEFRAME
        from data_platform.bars import bars_cache_path, load_ohlcv

        path = bars_cache_path(sym, BAR_TIMEFRAME)
        if path.is_file():
            df = load_ohlcv(sym, BAR_TIMEFRAME)
            if df is not None and not df.empty and "close" in df.columns:
                # Trailing ~1y of bars for stable estimate.
                bpy = _bars_per_year()
                tail = df["close"].iloc[-int(min(len(df), max(bpy, 500))) :]
                vol = realized_vol_ann(tail, bars_per_year=bpy)
                if vol is not None:
                    return float(vol)
    except Exception:
        pass
    return float(_vol_ref_for_ticker(sym))


def vol_scale(ticker: str, *, vol_ann: float | None = None) -> float:
    """``vol / vol_ref`` clipped — how noisy the instrument is vs its class baseline."""
    vol = float(vol_ann if vol_ann is not None else instrument_vol_ann(ticker))
    ref = max(_vol_ref_for_ticker(ticker), 1e-6)
    lo = float(getattr(_cfg, "FUSION_VOL_SCALE_CLIP_LO", 0.5))
    hi = float(getattr(_cfg, "FUSION_VOL_SCALE_CLIP_HI", 2.5))
    return float(np.clip(vol / ref, lo, hi))


def round_trip_bps(ticker: str, *, vol_ann: float | None = None) -> float:
    vol = float(vol_ann if vol_ann is not None else instrument_vol_ann(ticker))
    return float(round_trip_cost_bps_for_ticker(ticker, vol_ann=vol))


def economics_floor_bps(ticker: str, *, vol_ann: float | None = None) -> float:
    """Minimum top-decile *net* bps to clear gate / SQ for this instrument.

    ``floor = RT(vol) × over_rt_mult × vol_scale^vol_exp``

    - RT grows with vol via slippage scaling
    - ``over_rt_mult`` (>1) requires edge above pure cost
    - ``vol_exp`` (>0) raises the bar for noisier names
    """
    vol = float(vol_ann if vol_ann is not None else instrument_vol_ann(ticker))
    rt = round_trip_bps(ticker, vol_ann=vol)
    over = float(getattr(_cfg, "FUSION_ECONOMICS_OVER_RT_MULT", 1.25))
    exp = float(getattr(_cfg, "FUSION_ECONOMICS_VOL_EXP", 0.5))
    scale = vol_scale(ticker, vol_ann=vol)
    floor = rt * over * (scale ** exp)
    # Never below RT itself (must clear costs).
    return float(max(rt, floor))


def soft_size_cv_band_bps(
    ticker: str,
    *,
    vol_ann: float | None = None,
) -> tuple[float, float]:
    """CV net band for soft-size interpolation: ``[−k×floor, +floor]``."""
    floor = economics_floor_bps(ticker, vol_ann=vol_ann)
    lo_mult = float(getattr(_cfg, "FUSION_SOFT_SIZE_CV_LO_OVER_FLOOR", 2.0))
    return (-lo_mult * floor, floor)


def preferred_trade_side(
    side_audits: dict[str, dict],
    *,
    ticker: str,
) -> str:
    """Choose long/short from side audits by vol-adjusted net (no ticker hardcode)."""
    if not side_audits:
        return "long"
    floor = economics_floor_bps(ticker)
    scored: list[tuple[float, float, str]] = []
    for side, audit in side_audits.items():
        net = audit.get("top_decile_net_bps")
        if net is None or not math.isfinite(float(net)):
            continue
        # Prefer sides that clear floor; else least-negative / best net.
        cleared = 1.0 if float(net) > floor and audit.get("monotonic") else 0.0
        scored.append((cleared, float(net), str(side)))
    if not scored:
        return next(iter(side_audits.keys()))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]


def inverse_vol_exposure_budget(
    tradeable_tickers: list[str],
    *,
    vol_by_ticker: dict[str, float] | None = None,
) -> dict[str, float]:
    """Exposure caps from inverse-vol weights, clipped by max weight."""
    syms = [str(t).upper() for t in tradeable_tickers]
    if not syms:
        return {}
    if not bool(getattr(_cfg, "FUSION_PER_TICKER_EXPOSURE_BUDGET", True)):
        return {s: 1.0 for s in syms}

    max_w = float(getattr(_cfg, "FUSION_PER_TICKER_MAX_WEIGHT", 0.35))
    vols: dict[str, float] = {}
    for s in syms:
        if vol_by_ticker and s in vol_by_ticker and vol_by_ticker[s] > 0:
            vols[s] = float(vol_by_ticker[s])
        else:
            vols[s] = max(instrument_vol_ann(s), 1e-4)

    inv = {s: 1.0 / vols[s] for s in syms}
    total = sum(inv.values())
    raw = {s: inv[s] / total for s in syms}
    # Iterative cap + redistribute so no weight exceeds max_w after renorm.
    weights = dict(raw)
    for _ in range(len(syms) + 3):
        over = [s for s, w in weights.items() if w > max_w + 1e-12]
        if not over:
            break
        free = [s for s in syms if s not in over]
        capped_mass = max_w * len(over)
        for s in over:
            weights[s] = max_w
        rem = 1.0 - capped_mass
        if not free or rem <= 0:
            # All capped / residual nowhere to go — equalize within max_w.
            eq = min(max_w, 1.0 / len(syms))
            return {s: eq for s in syms}
        free_raw = sum(raw[s] for s in free)
        if free_raw <= 0:
            eq = rem / len(free)
            for s in free:
                weights[s] = eq
        else:
            for s in free:
                weights[s] = rem * (raw[s] / free_raw)
    return {s: float(weights[s]) for s in syms}


def _fold_ticker_policies(wf_folds: list[dict]) -> dict[str, list[dict]]:
    """Collect per-fold threshold policies keyed by ticker (chronological)."""
    by_sym: dict[str, list[dict]] = {}
    for fold in wf_folds:
        if fold.get("skipped"):
            continue
        th = fold.get("threshold_optimization") or {}
        by_t = (th.get("cv") or {}).get("by_ticker") or (th.get("best_params") or {}).get("by_ticker") or {}
        for sym, pol in by_t.items():
            by_sym.setdefault(str(sym).upper(), []).append(dict(pol or {}))
    return by_sym


def signal_quality_pass_stats(
    policies: list[dict],
    *,
    ticker: str,
) -> dict[str, Any]:
    """Pass rate / mean holdout net from fold policies — instrument-relative."""
    floor = economics_floor_bps(ticker)
    if not policies:
        return {
            "n": 0,
            "pass_rate": None,
            "mean_holdout_net_bps": None,
            "economics_floor_bps": floor,
            "ok": True,
        }
    oks = [bool(p.get("signal_quality_ok")) for p in policies]
    holdouts: list[float] = []
    for p in policies:
        ho = p.get("holdout_top_decile_net_bps")
        if ho is not None and math.isfinite(float(ho)):
            holdouts.append(float(ho))
    pass_rate = float(sum(oks) / len(oks))
    mean_ho = float(np.mean(holdouts)) if holdouts else None
    min_rate = float(getattr(_cfg, "FUSION_SQ_MIN_PASS_RATE", 0.40))
    ok = pass_rate >= min_rate
    if mean_ho is not None and mean_ho >= floor:
        ok = True
    return {
        "n": len(policies),
        "pass_rate": round(pass_rate, 4),
        "mean_holdout_net_bps": None if mean_ho is None else round(mean_ho, 4),
        "economics_floor_bps": round(floor, 4),
        "min_pass_rate": min_rate,
        "ok": ok,
        "soft_size": float(np.clip(pass_rate, float(getattr(_cfg, "FUSION_SOFT_SIZE_MIN", 0.1)), 1.0)),
    }


def filter_tradeable_by_signal_quality_v2(
    wf_folds: list[dict],
    tradeable_syms: list[str],
) -> tuple[list[str], list[str], dict[str, dict]]:
    """Keep stitched-passers unless fold SQ pass-rate is structurally weak.

    Unlike last-fold veto: uses majority / mean-holdout vs economics_floor.
    Soft-size multipliers returned in ``detail[sym]['soft_size']`` for live book.
    """
    require = bool(getattr(_cfg, "FUSION_GATE_REQUIRE_SIGNAL_QUALITY", True))
    detail: dict[str, dict] = {}
    if not require or not wf_folds or not tradeable_syms:
        for s in tradeable_syms:
            detail[str(s).upper()] = {"ok": True, "soft_size": 1.0, "reason": "sq_filter_off"}
        return list(tradeable_syms), [], detail

    hist = _fold_ticker_policies(wf_folds)
    kept: list[str] = []
    removed: list[str] = []
    for sym in tradeable_syms:
        sym_u = str(sym).upper()
        stats = signal_quality_pass_stats(hist.get(sym_u, []), ticker=sym_u)
        detail[sym_u] = stats
        if stats.get("ok"):
            kept.append(sym_u)
        else:
            removed.append(sym_u)
            detail[sym_u]["reason"] = (
                f"pass_rate={stats.get('pass_rate')} < {stats.get('min_pass_rate')} "
                f"and mean_holdout < floor={stats.get('economics_floor_bps')}"
            )
    return kept, removed, detail


def instrument_economics_snapshot(ticker: str) -> dict[str, Any]:
    """Debug/report dict for one symbol."""
    vol = instrument_vol_ann(ticker)
    return {
        "symbol": str(ticker).upper(),
        "vol_ann": round(vol, 6),
        "vol_ref": round(_vol_ref_for_ticker(ticker), 6),
        "vol_scale": round(vol_scale(ticker, vol_ann=vol), 4),
        "rt_cost_bps": round(round_trip_bps(ticker, vol_ann=vol), 4),
        "economics_floor_bps": round(economics_floor_bps(ticker, vol_ann=vol), 4),
        "soft_size_cv_band_bps": soft_size_cv_band_bps(ticker, vol_ann=vol),
        "slippage_bps_per_side": round(float(slippage_bps_per_side(ticker, vol_ann=vol)), 4),
    }
