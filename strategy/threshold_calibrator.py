"""Per-instrument threshold model aligned with the OOS decile gate.

Primary metric (same as gate / model CV):
    ``top_decile_net_bps(ml_proba, fwd_ret, per-ticker costs)``

Flow:
1. Chronological CV signal-quality per ticker (min rows enforced).
2. If signal quality is not positive → keep base thresholds, report honest CV net.
3. If positive → score fusion panel once, fit buy/edge with min active rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import config as _cfg
from common.stage_log import stage_log
from data_platform.universe import commission_bps_for_ticker
from models.profit_metrics import top_decile_net_bps
from simulation.execution_costs import round_trip_cost_bps_for_ticker, slippage_bps_per_side


def _ret_col(df: pd.DataFrame) -> str:
    if "fwd_ret_entry" in df.columns:
        return "fwd_ret_entry"
    return "fwd_ret"


def _side_gate_arrays(grp: pd.DataFrame, ticker: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Proba and signed fwd arrays aligned with ticker side policy."""
    from research.diagnostics.decile_audit import (
        _prepare_side_frame,
        decile_audit_for_ticker,
        decile_side_candidates,
    )

    ret = _ret_col(grp)
    sym = str(ticker).upper()
    specs = decile_side_candidates(sym, grp, ret_col=ret)
    if not specs:
        proba = grp["ml_proba"].to_numpy(dtype=float)
        fwd = grp[ret].to_numpy(dtype=float)
        return proba, fwd, "long"

    if len(specs) == 1:
        spec = specs[0]
    else:
        audit = decile_audit_for_ticker(grp, sym, ret_col=ret)
        active = str(audit.get("active_side") or "long")
        spec = next((s for s in specs if s["side"] == active), specs[0])

    frame = _prepare_side_frame(
        grp,
        proba_col=spec["proba_col"],
        ret_col=ret,
        invert_ret=bool(spec["invert_ret"]),
    )
    return (
        frame["_gate_proba"].to_numpy(dtype=float),
        frame["_gate_ret"].to_numpy(dtype=float),
        str(spec["side"]),
    )


def _row_net_bps(grp: pd.DataFrame, ticker: str) -> pd.Series:
    ret = _ret_col(grp)
    if ret not in grp.columns:
        return pd.Series(np.nan, index=grp.index, dtype=float)
    rt = round_trip_cost_bps_for_ticker(ticker) / 10_000.0
    return (grp[ret].astype(float) - rt) * 10_000.0


def _chrono_splits(n: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological train/val index splits (no shuffle)."""
    if n < 60 or n_folds < 2:
        idx = np.arange(n)
        cut = max(1, int(n * 0.7))
        return [(idx[:cut], idx[cut:])]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    block = n // (n_folds + 1)
    if block < 20:
        idx = np.arange(n)
        cut = max(1, int(n * 0.7))
        return [(idx[:cut], idx[cut:])]
    for i in range(n_folds):
        val_start = block * (i + 1)
        val_end = block * (i + 2) if i < n_folds - 1 else n
        if val_end - val_start < 15 or val_start < 30:
            continue
        folds.append((np.arange(0, val_start), np.arange(val_start, val_end)))
    if not folds:
        idx = np.arange(n)
        cut = max(1, int(n * 0.7))
        return [(idx[:cut], idx[cut:])]
    return folds


def _min_rows() -> int:
    return int(getattr(_cfg, "FUSION_THRESHOLD_CAL_MIN_ROWS", 200))


def _cv_top_decile_net(grp: pd.DataFrame, ticker: str) -> tuple[float | None, int, bool]:
    """Chronological CV of gate-aligned top-decile net bps.

    Returns (mean_cv_net, total_val_rows, signal_quality_ok).
    """
    ret = _ret_col(grp)
    if ret not in grp.columns:
        return None, 0, False

    proba, fwd, _side = _side_gate_arrays(grp, ticker)
    ok = np.isfinite(proba) & np.isfinite(fwd)
    if ok.sum() < _min_rows():
        return None, int(ok.sum()), False

    proba, fwd = proba[ok], fwd[ok]
    n = len(proba)
    tickers = np.full(n, ticker, dtype=object)
    n_folds = int(getattr(_cfg, "FUSION_THRESHOLD_CAL_CV_FOLDS", 3))
    min_rows = _min_rows()
    splits = _chrono_splits(n, n_folds)

    fold_nets: list[float] = []
    total_val = 0
    for _, val_idx in splits:
        if len(val_idx) < min_rows:
            continue
        net = top_decile_net_bps(
            proba[val_idx],
            fwd[val_idx],
            commission_bps=commission_bps_for_ticker(ticker),
            tickers=tickers[val_idx],
            min_rows=min_rows,
        )
        if np.isfinite(net):
            fold_nets.append(float(net))
            total_val += int(len(val_idx))

    if not fold_nets:
        # Fallback: full-sample top decile (still honest, not gated lottery).
        net = top_decile_net_bps(
            proba,
            fwd,
            commission_bps=commission_bps_for_ticker(ticker),
            tickers=tickers,
            min_rows=min_rows,
        )
        if not np.isfinite(net):
            return None, n, False
        require_pos = bool(getattr(_cfg, "FUSION_THRESHOLD_CAL_REQUIRE_POSITIVE_SIGNAL", True))
        from strategy.instrument_economics import economics_floor_bps

        floor = economics_floor_bps(ticker) if require_pos else -np.inf
        return float(net), n, float(net) >= floor

    mean_net = float(np.mean(fold_nets))
    require_pos = bool(getattr(_cfg, "FUSION_THRESHOLD_CAL_REQUIRE_POSITIVE_SIGNAL", True))
    from strategy.instrument_economics import economics_floor_bps

    floor = economics_floor_bps(ticker) if require_pos else -np.inf
    ok_signal = mean_net >= floor
    return mean_net, total_val, ok_signal


def _holdout_top_decile_net(grp: pd.DataFrame, ticker: str) -> tuple[float | None, int]:
    """Top-decile net on a chronological holdout slice (post-train, pre-OOS test)."""
    ret = _ret_col(grp)
    if ret not in grp.columns:
        return None, 0
    proba, fwd, _side = _side_gate_arrays(grp, ticker)
    ok = np.isfinite(proba) & np.isfinite(fwd)
    min_rows = _min_rows()
    if ok.sum() < min_rows:
        return None, int(ok.sum())
    net = top_decile_net_bps(
        proba[ok],
        fwd[ok],
        commission_bps=commission_bps_for_ticker(ticker),
        tickers=np.full(int(ok.sum()), ticker, dtype=object),
        min_rows=min_rows,
    )
    if not np.isfinite(net):
        return None, int(ok.sum())
    return float(net), int(ok.sum())


def _holdout_signal_ok(holdout_net: float | None, *, n_rows: int, ticker: str) -> bool:
    """Require holdout top-decile net ≥ instrument economics floor."""
    from strategy.instrument_economics import economics_floor_bps

    min_rows = _min_rows()
    min_net = economics_floor_bps(ticker)
    if n_rows < min_rows or holdout_net is None:
        return False
    return float(holdout_net) >= min_net


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 50:
        return None
    xs, ys = x[mask], y[mask]
    if xs.size > 40_000:
        rng = np.random.default_rng(42)
        take = rng.choice(xs.size, size=40_000, replace=False)
        xs, ys = xs[take], ys[take]
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(xs, ys)
    return model


def _threshold_from_isotonic(
    model: IsotonicRegression | None,
    x: np.ndarray,
    *,
    target_net_bps: float,
    fallback_quantile: float,
    lo: float,
    hi: float,
) -> float:
    x_ok = x[np.isfinite(x)]
    if x_ok.size == 0:
        return float(lo)
    if model is None:
        return float(np.clip(np.quantile(x_ok, fallback_quantile), lo, hi))
    grid = np.unique(np.quantile(x_ok, np.linspace(0.05, 0.99, 40)))
    pred = model.predict(grid)
    hits = grid[pred >= target_net_bps]
    if hits.size == 0:
        return float(np.clip(np.quantile(x_ok, fallback_quantile), lo, hi))
    return float(np.clip(hits.min(), lo, hi))


def _mask_active(
    score: np.ndarray,
    edge: np.ndarray,
    impulse: np.ndarray,
    *,
    buy: float,
    min_edge: float,
    impulse_min: float,
    position_side: np.ndarray | None = None,
    sell: float | None = None,
) -> np.ndarray:
    imp_ok = np.isfinite(impulse) & (impulse >= impulse_min)
    if position_side is not None:
        from strategy.edge_gate import signed_edge_active_mask

        side = np.asarray(position_side, dtype=int)
        sell_th = float(sell) if sell is not None else max(0.0, 100.0 - float(buy))
        edge_ok = signed_edge_active_mask(edge, side, float(min_edge))
        score_ok = np.zeros(len(score), dtype=bool)
        long_m = side > 0
        short_m = side < 0
        if long_m.any():
            score_ok[long_m] = np.isfinite(score[long_m]) & (score[long_m] >= float(buy))
        if short_m.any():
            score_ok[short_m] = np.isfinite(score[short_m]) & (score[short_m] <= sell_th)
        return (
            np.isfinite(score)
            & np.isfinite(edge)
            & imp_ok
            & edge_ok
            & score_ok
        )
    return (
        np.isfinite(score)
        & np.isfinite(edge)
        & (score >= buy)
        & (edge >= min_edge)
        & imp_ok
    )


def _eval_policy_net(
    score: np.ndarray,
    edge: np.ndarray,
    impulse: np.ndarray,
    net: np.ndarray,
    *,
    buy: float,
    min_edge: float,
    impulse_min: float,
    min_trades: int,
    position_side: np.ndarray | None = None,
    sell: float | None = None,
) -> tuple[float, int]:
    active = _mask_active(
        score, edge, impulse,
        buy=buy, min_edge=min_edge, impulse_min=impulse_min,
        position_side=position_side, sell=sell,
    )
    n = int(active.sum())
    if n < min_trades:
        return float("-inf"), n
    vals = net[active]
    vals = vals[np.isfinite(vals)]
    if vals.size < min_trades:
        return float("-inf"), int(vals.size)
    return float(np.mean(vals)), int(vals.size)


def _cv_policy_net(
    score: np.ndarray,
    edge: np.ndarray,
    impulse: np.ndarray,
    net: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    buy: float,
    min_edge: float,
    impulse_min: float,
    min_trades: int,
    position_side: np.ndarray | None = None,
    sell: float | None = None,
) -> tuple[float, int]:
    fold_nets: list[float] = []
    total_n = 0
    # Do not dilute min_trades across folds — each val fold must be large enough.
    min_val = max(min_trades, _min_rows())
    for _, val_idx in splits:
        side_val = position_side[val_idx] if position_side is not None else None
        v_net, n = _eval_policy_net(
            score[val_idx],
            edge[val_idx],
            impulse[val_idx],
            net[val_idx],
            buy=buy,
            min_edge=min_edge,
            impulse_min=impulse_min,
            min_trades=min_val,
            position_side=side_val,
            sell=sell,
        )
        if np.isfinite(v_net):
            fold_nets.append(v_net)
            total_n += n
    if not fold_nets or total_n < min_trades:
        return float("-inf"), total_n
    return float(np.mean(fold_nets)), total_n


def _candidate_grid(
    buy_center: float,
    edge_center: float,
    *,
    buy_lo: float,
    buy_hi: float,
    edge_lo: float,
    edge_hi: float,
) -> list[tuple[float, float]]:
    buy_steps = np.unique(
        np.clip(
            np.array(
                [buy_center - 6, buy_center - 3, buy_center, buy_center + 3, buy_center + 6, buy_lo, buy_hi],
                dtype=float,
            ),
            buy_lo,
            buy_hi,
        )
    )
    edge_steps = np.unique(
        np.clip(
            np.array(
                [
                    edge_center - 2.0,
                    edge_center - 1.0,
                    edge_center,
                    edge_center + 1.0,
                    edge_center + 2.0,
                    edge_lo,
                    edge_hi,
                ],
                dtype=float,
            ),
            edge_lo,
            edge_hi,
        )
    )
    return [(float(b), float(e)) for b in buy_steps for e in edge_steps]


def _default_ticker_policy(
    base_params: dict,
    *,
    ticker: str,
    cv_top_decile_net: float | None,
    signal_quality_ok: bool,
    n_rows: int,
) -> dict:
    comm = commission_bps_for_ticker(ticker)
    slip = slippage_bps_per_side(ticker)
    net_round = round(float(cv_top_decile_net), 3) if cv_top_decile_net is not None else None
    out = {
        "buy_threshold": int(base_params.get("buy_threshold", 36)),
        "min_expected_edge_bps": float(base_params.get("min_expected_edge_bps", 4.0)),
        "train_top_net_bps": net_round,
        "cv_net_bps": net_round,
        "cv_top_decile_net_bps": net_round,
        "train_net_bps": net_round,
        "train_active_rows": int(n_rows),
        "cv_active_rows": int(n_rows),
        "signal_quality_ok": bool(signal_quality_ok),
        "model_buy_proposal": None,
        "model_edge_proposal": None,
        "commission_bps_per_side": comm,
        "slippage_bps_per_side": round(slip, 2),
        "round_trip_cost_bps": round(round_trip_cost_bps_for_ticker(ticker), 2),
        "threshold_model": "signal_quality_skip" if not signal_quality_ok else "defaults",
    }
    from strategy.soft_sizing import soft_size_multiplier

    out["soft_size"] = round(soft_size_multiplier(out), 4)
    return out


def _fit_ticker_thresholds(
    grp: pd.DataFrame,
    base_params: dict,
    *,
    impulse_min: float,
    edge_floor: float,
    cv_top_decile_net: float,
) -> dict:
    sym = str(grp["ticker"].iloc[0]).upper()
    score = grp["fusion_score"].to_numpy(dtype=float)
    edge = grp["expected_edge_bps"].to_numpy(dtype=float)
    impulse = grp["impulse_strength"].to_numpy(dtype=float)
    position_side = (
        grp["position_side"].to_numpy(dtype=int)
        if "position_side" in grp.columns
        else None
    )
    net = _row_net_bps(grp, sym).to_numpy(dtype=float)

    ok = np.isfinite(score) & np.isfinite(edge) & np.isfinite(net)
    min_trades = int(getattr(_cfg, "FUSION_THRESHOLD_CAL_MIN_TRADES", 200))
    if ok.sum() < max(_min_rows(), min_trades):
        return _default_ticker_policy(
            base_params,
            ticker=sym,
            cv_top_decile_net=cv_top_decile_net,
            signal_quality_ok=True,
            n_rows=int(ok.sum()),
        )

    score, edge, impulse, net = score[ok], edge[ok], impulse[ok], net[ok]
    if position_side is not None:
        position_side = position_side[ok]
    n = len(score)

    target = float(getattr(_cfg, "FUSION_THRESHOLD_CAL_TARGET_NET_BPS", 0.0))
    buy_lo = float(getattr(_cfg, "FUSION_THRESHOLD_CAL_BUY_LO", 30))
    buy_hi = float(getattr(_cfg, "FUSION_THRESHOLD_CAL_BUY_HI", 55))
    edge_lo = float(edge_floor)
    edge_hi = float(getattr(_cfg, "FUSION_THRESHOLD_CAL_EDGE_HI_BPS", 12.0))
    edge_hi = max(edge_hi, edge_lo + 1.0)
    n_folds = int(getattr(_cfg, "FUSION_THRESHOLD_CAL_CV_FOLDS", 3))
    fallback_q = float(getattr(_cfg, "FUSION_THRESHOLD_CAL_FALLBACK_QUANTILE", 0.85))

    score_model = _fit_isotonic(score, net)
    edge_model = _fit_isotonic(edge, net)
    buy_prop = _threshold_from_isotonic(
        score_model, score, target_net_bps=target, fallback_quantile=fallback_q, lo=buy_lo, hi=buy_hi,
    )
    edge_prop = _threshold_from_isotonic(
        edge_model, edge, target_net_bps=target, fallback_quantile=fallback_q, lo=edge_lo, hi=edge_hi,
    )

    splits = _chrono_splits(n, n_folds)
    candidates = _candidate_grid(
        buy_prop, edge_prop, buy_lo=buy_lo, buy_hi=buy_hi, edge_lo=edge_lo, edge_hi=edge_hi,
    )

    best_cv = float("-inf")
    best_buy = float(base_params.get("buy_threshold", buy_prop))
    best_edge = float(base_params.get("min_expected_edge_bps", edge_prop))
    best_n = 0
    from strategy.fusion_direction import fusion_sell_threshold

    for buy, min_edge in candidates:
        sell_th = fusion_sell_threshold(buy)
        cv_net, n_act = _cv_policy_net(
            score, edge, impulse, net, splits,
            buy=buy, min_edge=min_edge, impulse_min=impulse_min, min_trades=min_trades,
            position_side=position_side, sell=sell_th,
        )
        better = cv_net > best_cv + 1e-9
        tie = abs(cv_net - best_cv) <= 1e-9 and np.isfinite(cv_net)
        if better or (tie and (n_act > best_n or (n_act == best_n and buy > best_buy))):
            best_cv = cv_net
            best_buy = buy
            best_edge = min_edge
            best_n = n_act

    # If no candidate met min_n, keep defaults — never report lottery nets.
    if not np.isfinite(best_cv) or best_n < min_trades:
        out = _default_ticker_policy(
            base_params,
            ticker=sym,
            cv_top_decile_net=cv_top_decile_net,
            signal_quality_ok=True,
            n_rows=n,
        )
        out["threshold_model"] = "top_decile_defaults"
        out["model_buy_proposal"] = round(float(buy_prop), 2)
        out["model_edge_proposal"] = round(float(edge_prop), 2)
        return out

    train_net, train_n = _eval_policy_net(
        score, edge, impulse, net,
        buy=best_buy, min_edge=best_edge, impulse_min=impulse_min,
        min_trades=min_trades,
        position_side=position_side,
        sell=fusion_sell_threshold(best_buy),
    )
    comm = commission_bps_for_ticker(sym)
    slip = slippage_bps_per_side(sym)
    # Primary reported metric stays gate-aligned top-decile; policy net is secondary.
    return {
        "buy_threshold": int(round(best_buy)),
        "min_expected_edge_bps": round(float(best_edge), 2),
        "train_top_net_bps": round(float(cv_top_decile_net), 3),
        "cv_net_bps": round(float(cv_top_decile_net), 3),
        "cv_top_decile_net_bps": round(float(cv_top_decile_net), 3),
        "policy_cv_net_bps": round(float(best_cv), 3),
        "train_net_bps": round(float(train_net), 3) if np.isfinite(train_net) else None,
        "train_active_rows": int(train_n),
        "cv_active_rows": int(best_n),
        "signal_quality_ok": True,
        "model_buy_proposal": round(float(buy_prop), 2),
        "model_edge_proposal": round(float(edge_prop), 2),
        "commission_bps_per_side": comm,
        "slippage_bps_per_side": round(slip, 2),
        "round_trip_cost_bps": round(round_trip_cost_bps_for_ticker(sym), 2),
        "threshold_model": "isotonic_cv",
    }


def fit_threshold_calibrator(
    train: pd.DataFrame,
    base_params: dict,
    *,
    fold: int | str | None = None,
    holdout: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Per-ticker thresholds; signal quality uses train CV + chronological holdout."""
    if train.empty or "ticker" not in train.columns or "ml_proba" not in train.columns:
        return {}

    impulse_min = float(base_params.get("impulse_min", 0.05))
    global_edge_floor = float(
        base_params.get("min_expected_edge_bps")
        or getattr(_cfg, "FUSION_MIN_EXPECTED_EDGE_BPS", 0.0)
    )

    qualities: dict[str, tuple[float | None, int, bool]] = {}
    holdout_stats: dict[str, tuple[float | None, int, bool]] = {}
    for ticker, grp in train.groupby("ticker", sort=True):
        sym = str(ticker).upper()
        qualities[sym] = _cv_top_decile_net(grp, sym)
        net, n_rows, ok = qualities[sym]
        ho_net, ho_rows, ho_ok = None, 0, True
        if holdout is not None and not holdout.empty and "ticker" in holdout.columns:
            hgrp = holdout[holdout["ticker"].astype(str).str.upper() == sym]
            if not hgrp.empty:
                ho_net, ho_rows = _holdout_top_decile_net(hgrp, sym)
                ho_ok = _holdout_signal_ok(ho_net, n_rows=ho_rows, ticker=sym)
        holdout_stats[sym] = (ho_net, ho_rows, ho_ok)
        signal_ok = bool(ok and ho_ok)
        stage_log(
            f"signal quality [{sym}]",
            fold=fold,
            detail=(
                f"cv_top_decile_net={None if net is None else round(net, 3)} bps "
                f"holdout_net={None if ho_net is None else round(ho_net, 3)} bps "
                f"n_val={n_rows} n_holdout={ho_rows} ok={signal_ok}"
            ),
        )
        qualities[sym] = (net, n_rows, signal_ok)

    any_ok = any(ok for _, _, ok in qualities.values())
    fused: pd.DataFrame | None = None
    # Always score for per-ticker threshold fit; signal_quality_ok is a soft flag
    # (hard skip only when FUSION_DISABLE_QUALITY_GATE is False — see signal frame).
    from strategy.pipeline import apply_fusion_scores

    stage_log(
        "threshold model: scoring panel once",
        fold=fold,
        detail=(
            f"{len(train):,} rows "
            f"(signal_ok={sum(1 for _, _, ok in qualities.values() if ok)}/{len(qualities)}"
            f"{'' if any_ok else '; fitting defaults for all'})"
        ),
    )
    scored = apply_fusion_scores(train, base_params)
    if not scored.empty and "fusion_score" in scored.columns:
        fused = scored

    out: dict[str, dict] = {}
    for ticker, grp in train.groupby("ticker", sort=True):
        sym = str(ticker).upper()
        cv_net, n_rows, ok = qualities[sym]
        ho_net = holdout_stats.get(sym, (None, 0, True))[0]
        if fused is None:
            fitted = _default_ticker_policy(
                base_params,
                ticker=sym,
                cv_top_decile_net=cv_net,
                signal_quality_ok=bool(ok),
                n_rows=n_rows,
            )
            if ho_net is not None:
                fitted["holdout_top_decile_net_bps"] = round(float(ho_net), 3)
        else:
            fgrp = fused.loc[fused["ticker"].astype(str).str.upper() == sym]
            if fgrp.empty:
                fitted = _default_ticker_policy(
                    base_params,
                    ticker=sym,
                    cv_top_decile_net=cv_net,
                    signal_quality_ok=bool(ok),
                    n_rows=n_rows,
                )
            else:
                from strategy.edge_gate import resolve_ticker_min_edge_bps
                from strategy.soft_sizing import compute_edge_alignment, soft_size_multiplier

                ticker_edge_floor = resolve_ticker_min_edge_bps(sym, base_params)
                fitted = _fit_ticker_thresholds(
                    fgrp,
                    base_params,
                    impulse_min=impulse_min,
                    edge_floor=max(global_edge_floor, ticker_edge_floor),
                    cv_top_decile_net=float(cv_net) if cv_net is not None else 0.0,
                )
                fitted["signal_quality_ok"] = bool(ok)
                align = compute_edge_alignment(fgrp)
                if align is not None:
                    fitted["edge_alignment"] = round(float(align), 4)
                fitted["soft_size"] = round(soft_size_multiplier(fitted), 4)
            if ho_net is not None:
                fitted["holdout_top_decile_net_bps"] = round(float(ho_net), 3)
        out[sym] = fitted
        stage_log(
            f"threshold model [{sym}]",
            fold=fold,
            detail=(
                f"buy={fitted['buy_threshold']} edge={fitted['min_expected_edge_bps']} "
                f"cv_top_decile={fitted['cv_top_decile_net_bps']} bps "
                f"signal_ok={fitted['signal_quality_ok']} "
                f"mode={fitted['threshold_model']}"
            ),
        )
    return out


def merge_ticker_policies(base_params: dict, by_ticker: dict[str, dict]) -> dict:
    """Attach per-ticker overrides to a fold trading policy."""
    return {**base_params, "by_ticker": by_ticker, "threshold_calibrator": True}


def resolve_ticker_policy(params: dict, ticker: str) -> dict:
    """Merge base policy with per-ticker calibrated overrides."""
    sym = str(ticker).upper()
    overrides = (params.get("by_ticker") or {}).get(sym) or {}
    if not overrides:
        return params
    # Diagnostics kept for soft-sizing / reporting; omit only train-fit proposals.
    skip = {
        "train_top_net_bps",
        "train_active_rows",
        "train_net_bps",
        "cv_active_rows",
        "model_buy_proposal",
        "model_edge_proposal",
        "threshold_model",
        "commission_bps_per_side",
        "slippage_bps_per_side",
        "round_trip_cost_bps",
    }
    return {**params, **{k: v for k, v in overrides.items() if k not in skip}}
