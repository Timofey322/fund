"""Decile monotonicity and tradeability checks for ML probability ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as _cfg
from simulation.entry_signals import round_trip_cost_bps
from research.labels.trade import DEFAULT_SLIPPAGE_BPS


def decile_side_candidates(
    ticker: str,
    df: pd.DataFrame,
    *,
    ret_col: str = "fwd_ret_entry",
) -> list[dict]:
    """Side-specific proba / signed-return specs for decile gate (matches live book)."""
    from strategy.side_policy import allowed_sides_for_ticker

    mode = allowed_sides_for_ticker(ticker)
    if ret_col not in df.columns:
        return []
    candidates: list[dict] = []
    if mode in ("long_only", "both") and "ml_proba" in df.columns:
        candidates.append({
            "side": "long",
            "proba_col": "ml_proba",
            "invert_ret": False,
        })
    if mode in ("short_only", "both"):
        if "ml_proba_short" in df.columns:
            candidates.append({
                "side": "short",
                "proba_col": "ml_proba_short",
                "invert_ret": True,
            })
        elif mode == "short_only" and "ml_proba" in df.columns:
            candidates.append({
                "side": "short",
                "proba_col": "ml_proba",
                "invert_ret": True,
            })
    return candidates


def _prepare_side_frame(
    grp: pd.DataFrame,
    *,
    proba_col: str,
    ret_col: str,
    invert_ret: bool,
) -> pd.DataFrame:
    """Working frame with finite proba and signed forward return for one side."""
    out = grp[[proba_col, ret_col]].copy()
    out = out.rename(columns={proba_col: "_gate_proba", ret_col: "_gate_ret"})
    out["_gate_ret"] = out["_gate_ret"].astype(float)
    if invert_ret:
        out["_gate_ret"] = -out["_gate_ret"]
    return out.dropna(subset=["_gate_proba", "_gate_ret"])


def decile_monotonicity_check(
    df: pd.DataFrame,
    *,
    proba_col: str = "ml_proba",
    ret_col: str = "fwd_ret_entry",
    commission_bps: float | None = None,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    min_top_decile_net_bps: float | None = None,
) -> dict:
    """Net bps by probability decile; tradeable when top decile clears threshold and ranks well."""
    min_top = float(
        min_top_decile_net_bps
        if min_top_decile_net_bps is not None
        else getattr(_cfg, "FUSION_MIN_TOP_DECILE_NET_BPS", 15.0)
    )
    default_comm = float(getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 1.1))
    comm = float(commission_bps if commission_bps is not None else default_comm)

    out: dict = {
        "tradeable": False,
        "monotonic": False,
        "top_decile_gross_bps": None,
        "top_decile_net_bps": None,
        "min_top_decile_net_bps": min_top,
        "deciles": [],
        "reasons": [],
    }
    if df.empty or proba_col not in df.columns or ret_col not in df.columns:
        out["reasons"].append("missing_oos_or_columns")
        return out

    use_per_ticker = commission_bps is None and "ticker" in df.columns
    cols = [proba_col, ret_col] + (["ticker"] if use_per_ticker else [])
    q = df[cols].dropna()
    if len(q) < 100:
        out["reasons"].append("insufficient_rows")
        return out

    try:
        q = q.copy()
        q["decile"] = pd.qcut(q[proba_col], 10, labels=False, duplicates="drop")
    except ValueError:
        out["reasons"].append("decile_binning_failed")
        return out

    if use_per_ticker:
        from simulation.execution_costs import round_trip_cost_series

        q["net_ret"] = q[ret_col].astype(float) - round_trip_cost_series(q["ticker"]) / 10_000.0

    deciles: list[dict] = []
    for d, part in q.groupby("decile"):
        gross = float(part[ret_col].mean()) * 10_000.0
        if use_per_ticker:
            net = float(part["net_ret"].mean()) * 10_000.0
        else:
            net = gross - round_trip_cost_bps(comm, slippage_bps)
        deciles.append({
            "decile": int(d),
            "n": int(len(part)),
            "mean_proba": round(float(part[proba_col].mean()), 4),
            "gross_bps": round(gross, 3),
            "net_bps": round(net, 3),
        })
    deciles.sort(key=lambda r: r["decile"])
    out["deciles"] = deciles

    if not deciles:
        out["reasons"].append("empty_deciles")
        return out

    top = deciles[-1]
    out["top_decile_gross_bps"] = top["gross_bps"]
    out["top_decile_net_bps"] = top["net_bps"]

    net_vals = [d["net_bps"] for d in deciles]
    bottom_net = net_vals[0]
    top_net = net_vals[-1]
    out["monotonic"] = bool(top_net >= max(bottom_net, net_vals[-2] if len(net_vals) > 1 else bottom_net))

    if top_net <= min_top:
        out["reasons"].append(f"top_decile_net_bps={top_net} <= min={min_top}")
    if not out["monotonic"]:
        out["reasons"].append("deciles_not_monotonic")

    out["tradeable"] = top_net > min_top and out["monotonic"]
    return out


def _audit_side_frame(
    frame: pd.DataFrame,
    *,
    side: str,
    proba_col: str,
    commission_bps: float,
    slippage_bps: float,
    min_top_decile_net_bps: float | None,
) -> dict:
    audit = decile_monotonicity_check(
        frame,
        proba_col="_gate_proba",
        ret_col="_gate_ret",
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        min_top_decile_net_bps=min_top_decile_net_bps,
    )
    audit["side"] = side
    audit["proba_col"] = proba_col
    return audit


def decile_audit_for_ticker(
    grp: pd.DataFrame,
    ticker: str,
    *,
    ret_col: str = "fwd_ret_entry",
    min_top_decile_net_bps: float | None = None,
    slippage_bps: float | None = None,
) -> dict:
    """Side-aware decile audit for one symbol (long / short / both)."""
    from data_platform.universe import commission_bps_for_ticker
    from simulation.execution_costs import slippage_bps_per_side

    sym = str(ticker).upper()
    comm = commission_bps_for_ticker(sym)
    slip = float(slippage_bps) if slippage_bps is not None else slippage_bps_per_side(sym)
    from strategy.instrument_economics import economics_floor_bps, preferred_trade_side

    floor = (
        float(min_top_decile_net_bps)
        if min_top_decile_net_bps is not None
        else economics_floor_bps(sym)
    )
    candidates = decile_side_candidates(sym, grp, ret_col=ret_col)
    if not candidates:
        audit = decile_monotonicity_check(
            grp,
            ret_col=ret_col,
            commission_bps=comm,
            slippage_bps=slip,
            min_top_decile_net_bps=floor,
        )
        audit["ticker"] = sym
        audit["active_side"] = "long"
        audit["commission_bps_per_side"] = comm
        audit["slippage_bps_per_side"] = slip
        audit["economics_floor_bps"] = floor
        return audit

    side_audits: dict[str, dict] = {}
    for spec in candidates:
        frame = _prepare_side_frame(
            grp,
            proba_col=spec["proba_col"],
            ret_col=ret_col,
            invert_ret=bool(spec["invert_ret"]),
        )
        if len(frame) < 100:
            continue
        side_audits[spec["side"]] = _audit_side_frame(
            frame,
            side=spec["side"],
            proba_col=spec["proba_col"],
            commission_bps=comm,
            slippage_bps=slip,
            min_top_decile_net_bps=floor,
        )

    if not side_audits:
        audit = decile_monotonicity_check(
            grp,
            ret_col=ret_col,
            commission_bps=comm,
            slippage_bps=slip,
            min_top_decile_net_bps=floor,
        )
        audit["ticker"] = sym
        audit["active_side"] = "long"
        audit["commission_bps_per_side"] = comm
        audit["slippage_bps_per_side"] = slip
        audit["economics_floor_bps"] = floor
        audit["reasons"] = list(audit.get("reasons") or []) + ["side_frame_insufficient_rows"]
        return audit

    tradeable_sides = [s for s, a in side_audits.items() if a.get("tradeable")]
    if tradeable_sides:
        active = preferred_trade_side(
            {s: side_audits[s] for s in tradeable_sides},
            ticker=sym,
        )
    else:
        active = preferred_trade_side(side_audits, ticker=sym)

    best = dict(side_audits[active])
    best["ticker"] = sym
    best["active_side"] = active
    best["sides"] = side_audits
    best["commission_bps_per_side"] = comm
    best["slippage_bps_per_side"] = slip
    best["economics_floor_bps"] = floor
    if len(side_audits) > 1:
        best["tradeable"] = bool(tradeable_sides)
        if tradeable_sides:
            best["reasons"] = [
                r for r in (best.get("reasons") or [])
                if r not in ("deciles_not_monotonic",) or active in tradeable_sides
            ]
            best["reasons"] = [r for r in best["reasons"] if not r.startswith("top_decile_net_bps=")]
        else:
            best["reasons"] = list(best.get("reasons") or [])
            if not any("side" in r for r in best["reasons"]):
                best["reasons"].append(
                    f"no_side_passed; best={active} net={best.get('top_decile_net_bps')}"
                )
    return best


def attach_side_aware_gate_columns(
    df: pd.DataFrame,
    *,
    ret_col: str = "fwd_ret_entry",
) -> pd.DataFrame:
    """Add ``gate_proba`` / ``gate_ret`` per row using ticker side policy."""
    if df.empty or "ticker" not in df.columns or ret_col not in df.columns:
        return df
    out = df.copy()
    gate_proba = np.full(len(out), np.nan, dtype=float)
    gate_ret = np.full(len(out), np.nan, dtype=float)
    gate_side = np.full(len(out), "", dtype=object)

    for ticker, idx in out.groupby("ticker").groups.items():
        grp = out.loc[idx]
        loc = out.index.get_indexer(idx)
        sym = str(ticker).upper()
        candidates = decile_side_candidates(sym, grp, ret_col=ret_col)
        if not candidates:
            continue
        spec = candidates[0]
        if len(candidates) > 1 and "position_side" in grp.columns:
            side_vals = grp["position_side"].fillna(0).astype(int)
            short_mask = side_vals < 0
            long_mask = side_vals > 0
            if short_mask.any():
                short_spec = next((c for c in candidates if c["side"] == "short"), spec)
                pcol = short_spec["proba_col"]
                gate_proba[loc[short_mask.to_numpy()]] = grp.loc[short_mask, pcol].astype(float).to_numpy()
                ret_v = grp.loc[short_mask, ret_col].astype(float).to_numpy()
                gate_ret[loc[short_mask.to_numpy()]] = -ret_v if short_spec["invert_ret"] else ret_v
                gate_side[loc[short_mask.to_numpy()]] = "short"
            if long_mask.any():
                long_spec = next((c for c in candidates if c["side"] == "long"), spec)
                pcol = long_spec["proba_col"]
                gate_proba[loc[long_mask.to_numpy()]] = grp.loc[long_mask, pcol].astype(float).to_numpy()
                ret_v = grp.loc[long_mask, ret_col].astype(float).to_numpy()
                gate_ret[loc[long_mask.to_numpy()]] = -ret_v if long_spec["invert_ret"] else ret_v
                gate_side[loc[long_mask.to_numpy()]] = "long"
            neutral = ~(short_mask | long_mask)
            if neutral.any():
                pcol = spec["proba_col"]
                gate_proba[loc[neutral.to_numpy()]] = grp.loc[neutral, pcol].astype(float).to_numpy()
                ret_v = grp.loc[neutral, ret_col].astype(float).to_numpy()
                gate_ret[loc[neutral.to_numpy()]] = -ret_v if spec["invert_ret"] else ret_v
                gate_side[loc[neutral.to_numpy()]] = spec["side"]
        else:
            pcol = spec["proba_col"]
            gate_proba[loc] = grp[pcol].astype(float).to_numpy()
            ret_v = grp[ret_col].astype(float).to_numpy()
            gate_ret[loc] = (-ret_v) if spec["invert_ret"] else ret_v
            gate_side[loc] = spec["side"]

    out["gate_proba"] = gate_proba
    out["gate_ret"] = gate_ret
    out["gate_side"] = gate_side
    return out


def decile_monotonicity_check_side_aware(
    df: pd.DataFrame,
    *,
    ret_col: str = "fwd_ret_entry",
    commission_bps: float | None = None,
    min_top_decile_net_bps: float | None = None,
) -> dict:
    """Portfolio decile audit on side-normalized proba/return columns."""
    if df.empty:
        return decile_monotonicity_check(df, ret_col=ret_col, commission_bps=commission_bps)
    gated = attach_side_aware_gate_columns(df, ret_col=ret_col)
    ok = np.isfinite(gated["gate_proba"].to_numpy()) & np.isfinite(gated["gate_ret"].to_numpy())
    work = gated.loc[ok, ["gate_proba", "gate_ret"] + (["ticker"] if "ticker" in gated.columns else [])]
    if len(work) < 100:
        return decile_monotonicity_check(df, ret_col=ret_col, commission_bps=commission_bps)
    audit = decile_monotonicity_check(
        work,
        proba_col="gate_proba",
        ret_col="gate_ret",
        commission_bps=commission_bps,
        min_top_decile_net_bps=min_top_decile_net_bps,
    )
    audit["side_aware"] = True
    return audit


def decile_audit_by_ticker(
    df: pd.DataFrame,
    *,
    proba_col: str = "ml_proba",
    ret_col: str = "fwd_ret_entry",
    min_top_decile_net_bps: float | None = None,
    slippage_bps: float | None = None,
    side_aware: bool = True,
) -> dict[str, dict]:
    """Per-symbol decile audit with per-ticker commission and slippage."""
    del proba_col  # legacy arg; side-aware path picks columns per ticker
    if df.empty or "ticker" not in df.columns:
        return {}

    out: dict[str, dict] = {}
    for ticker, grp in df.groupby("ticker", sort=False):
        sym = str(ticker).upper()
        if side_aware:
            audit = decile_audit_for_ticker(
                grp,
                sym,
                ret_col=ret_col,
                min_top_decile_net_bps=min_top_decile_net_bps,
                slippage_bps=slippage_bps,
            )
        else:
            from data_platform.universe import commission_bps_for_ticker
            from simulation.execution_costs import slippage_bps_per_side

            comm = commission_bps_for_ticker(sym)
            slip = float(slippage_bps) if slippage_bps is not None else slippage_bps_per_side(sym)
            audit = decile_monotonicity_check(
                grp,
                proba_col="ml_proba",
                ret_col=ret_col,
                commission_bps=comm,
                slippage_bps=slip,
                min_top_decile_net_bps=min_top_decile_net_bps,
            )
            audit["ticker"] = sym
            audit["active_side"] = "long"
            audit["commission_bps_per_side"] = comm
            audit["slippage_bps_per_side"] = slip
        out[sym] = audit
    return out


def tradeable_tickers_from_audit(
    by_ticker: dict[str, dict],
    *,
    min_top_decile_net_bps: float | None = None,
) -> list[str]:
    """Symbols that pass per-ticker decile gate (instrument-relative floor)."""
    from strategy.instrument_economics import economics_floor_bps

    tradeable: list[str] = []
    for sym, audit in sorted(by_ticker.items()):
        top_net = audit.get("top_decile_net_bps")
        floor = (
            float(min_top_decile_net_bps)
            if min_top_decile_net_bps is not None
            else float(audit.get("economics_floor_bps") or economics_floor_bps(str(sym)))
        )
        if audit.get("tradeable") and top_net is not None and float(top_net) > floor:
            tradeable.append(str(sym).upper())
    return tradeable


def tradeable_tickers_from_wf_folds(
    wf_folds: list[dict],
    *,
    recent_folds: int | None = None,
    min_ok_folds: int | None = None,
    min_median_oos_net_bps: float | None = None,
) -> list[str]:
    """Symbols with repeated fold-level OOS edge (avoids stitched-only gate blind spot).

    Median floors are instrument-relative (``economics_floor_bps`` / fraction of it),
    not absolute bps hardcodes.
    """
    from strategy.instrument_economics import economics_floor_bps

    recent = int(recent_folds if recent_folds is not None else getattr(_cfg, "FUSION_GATE_RECENT_FOLDS", 8))
    min_ok = int(min_ok_folds if min_ok_folds is not None else getattr(_cfg, "FUSION_GATE_MIN_OK_FOLDS", 2))
    # Soft fold median: fraction of economics floor (configurable multiplier, not absolute bps).
    median_floor_frac = float(getattr(_cfg, "FUSION_GATE_FOLD_MEDIAN_FLOOR_FRAC", 0.6))

    active = [f for f in wf_folds if not f.get("skipped")]
    if recent > 0:
        active = active[-recent:]
    ok_counts: dict[str, int] = {}
    oos_nets: dict[str, list[float]] = {}
    for fold in active:
        fd = fold.get("fold_diagnostics") or {}
        for sym, entry in (fd.get("tickers") or {}).items():
            sym_u = str(sym).upper()
            if entry.get("bottleneck") == "ok":
                ok_counts[sym_u] = ok_counts.get(sym_u, 0) + 1
            o_net = (entry.get("oos") or {}).get("net")
            if o_net is not None:
                oos_nets.setdefault(sym_u, []).append(float(o_net))

    tradeable: list[str] = []
    for sym, nets in sorted(oos_nets.items()):
        floor = economics_floor_bps(sym)
        min_median = (
            float(min_median_oos_net_bps)
            if min_median_oos_net_bps is not None
            else floor * median_floor_frac
        )
        if ok_counts.get(sym, 0) >= min_ok:
            tradeable.append(sym)
            continue
        if len(nets) >= max(2, min_ok) and float(np.median(nets)) >= min_median:
            tradeable.append(sym)
            continue
        if len(nets) >= 1 and float(np.median(nets)) > floor:
            tradeable.append(sym)
    return tradeable


def resolve_tradeable_tickers(
    decile_by_ticker: dict[str, dict],
    wf_folds: list[dict] | None,
    *,
    mode: str | None = None,
) -> tuple[list[str], str]:
    """Merge stitched decile audit with fold-history gate."""
    gate_mode = str(mode or getattr(_cfg, "FUSION_DECILE_GATE_MODE", "union")).lower()
    stitched = tradeable_tickers_from_audit(decile_by_ticker)
    fold_syms = tradeable_tickers_from_wf_folds(wf_folds or []) if wf_folds else []

    if gate_mode == "stitched":
        return stitched, "stitched"
    if gate_mode == "fold_history":
        chosen = fold_syms if fold_syms else stitched
        return chosen, "fold_history" if fold_syms else "stitched_fallback"
    if gate_mode == "strict":
        min_stitched = float(
            getattr(_cfg, "FUSION_GATE_STRICT_MIN_STITCHED_NET_BPS", 0.0)
        )
        positive_stitched = {
            str(sym).upper()
            for sym, audit in decile_by_ticker.items()
            if audit.get("top_decile_net_bps") is not None
            and float(audit["top_decile_net_bps"]) > min_stitched
        }
        if stitched:
            return stitched, "stitched"
        strict_fold = sorted(set(fold_syms) & positive_stitched)
        if strict_fold:
            return strict_fold, "strict_fold_aligned"
        return [], "strict_blocked"
    merged = sorted(set(stitched) | set(fold_syms))
    source = "union"
    if merged and not stitched and fold_syms:
        source = "fold_history_only"
    elif merged and stitched and not fold_syms:
        source = "stitched_only"
    return merged, source


def filter_tradeable_by_signal_quality(
    wf_folds: list[dict],
    tradeable_syms: list[str],
) -> tuple[list[str], list[str]]:
    """Keep tradeable names unless fold SQ pass-rate is structurally weak.

    Uses instrument-relative economics floor + pass-rate (not last-fold veto).
    """
    from strategy.instrument_economics import filter_tradeable_by_signal_quality_v2

    kept, removed, _detail = filter_tradeable_by_signal_quality_v2(wf_folds, tradeable_syms)
    return kept, removed


def filter_tradeable_by_signal_quality_detailed(
    wf_folds: list[dict],
    tradeable_syms: list[str],
) -> tuple[list[str], list[str], dict[str, dict]]:
    """Same as ``filter_tradeable_by_signal_quality`` plus per-ticker SQ detail."""
    from strategy.instrument_economics import filter_tradeable_by_signal_quality_v2

    return filter_tradeable_by_signal_quality_v2(wf_folds, tradeable_syms)
