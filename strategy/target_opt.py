"""Per-instrument entry-target optimization.

Each crypto has a different volatility profile, so a single global label
(``label_12_after_costs``) clears the cost floor too rarely for low-vol symbols
(BTC, ETHBTC) and too easily for high-vol ones (SOL). This module sweeps, per
symbol, the target's ``(horizon, label_type, threshold)`` and scores each
candidate on three axes:

    - learnability: purged-session CV composite of a fixed LightGBM,
    - economics:    net edge (bps) of the top-decile predictions, after costs,
    - balance:      positive-class share inside an acceptable range.

The winner per symbol is persisted to ``cache/target_optimization.json`` and,
when applied, consumed at runtime to attach a unified ``label_entry`` column.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import config as _cfg
from common.parallel import resolve_worker_count
from config import OUT_DIR
from research.features.entry_ml import _purged_session_folds
from models.entry_model import make_entry_classifier
from models.model_selection import aggregate_cv_metrics, compute_fold_metrics
from models.entry_model import make_direction_classifier
from research.labels.trade import (
    DEFAULT_SLIPPAGE_BPS,
    build_direction_label,
    build_entry_label,
    direction_class_rates,
)
from simulation.entry_signals import min_tp_gross_bps

TARGET_OPT_PATH = OUT_DIR / "cache" / "target_optimization.json"

_FALLBACK_LGBM_PARAMS = {
    "num_leaves": 15,
    "max_depth": 5,
    "learning_rate": 0.08,
    "n_estimators": 150,
    "min_child_samples": 80,
}


def _eval_params() -> dict:
    """Fixed LightGBM params for fair target comparison across candidates."""
    return dict(_FALLBACK_LGBM_PARAMS)


def _subsample(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= int(max_rows):
        return df
    return df.sample(n=int(max_rows), random_state=seed)


def _multiclass_sample_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts[y]
    weights *= len(y) / weights.sum()
    return weights


def _label_direction_edges_bps(label: pd.Series, fwd: pd.Series) -> dict[str, float]:
    """Mean forward return (bps) by direction class on the label itself."""
    sub = pd.DataFrame({"y": label, "fwd": fwd}).dropna()
    if sub.empty:
        return {}
    out: dict[str, float] = {}
    for code, name in ((0, "flat"), (1, "long"), (2, "short")):
        mask = sub["y"].astype(int) == code
        if mask.any():
            out[name] = float(sub.loc[mask, "fwd"].mean() * 10_000.0)
    if "short" in out:
        out["short_edge_bps"] = float(-out["short"])
    if "long" in out:
        out["long_edge_bps"] = float(out["long"])
    return out


def _direction_rate_factor(rate: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    pos = float(rate)
    if lo <= pos <= hi:
        return 1.0
    if pos < lo:
        return max(0.0, pos / lo) if lo > 0 else 0.0
    return max(0.0, 1.0 - (pos - hi) / max(1.0 - hi, 1e-6))


def _direction_balance_factor(
    class_rates: dict[str, float],
    ranges: dict[str, tuple[float, float]] | None = None,
) -> float:
    bounds = ranges or getattr(
        _cfg,
        "TARGET_OPT_DIRECTION_CLASS_RANGES",
        {"flat": (0.12, 0.45), "long": (0.20, 0.45), "short": (0.20, 0.45)},
    )
    factors = [
        _direction_rate_factor(float(class_rates.get(name, 0.0)), tuple(bounds[name]))
        for name in ("flat", "long", "short")
        if name in bounds
    ]
    return float(np.mean(factors)) if factors else 0.0


def evaluate_direction_target(
    work_sym: pd.DataFrame,
    feat_cols: list[str],
    folds: list[tuple[set, set]],
    label: pd.Series,
    fwd: pd.Series,
    params: dict,
    *,
    max_train_rows: int | None,
) -> dict | None:
    """Purged-session CV for 3-class direction labels + label economics."""
    from sklearn.metrics import accuracy_score

    df = work_sym.copy()
    df["_y"] = np.asarray(label, dtype=float)
    df["_fwd"] = np.asarray(fwd, dtype=float)
    df = df.dropna(subset=feat_cols + ["_y"])
    if df.empty or df["_y"].nunique() < 2:
        return None

    fold_accs: list[float] = []
    class_rates_list: list[dict[str, float]] = []
    for f_i, (train_s, val_s) in enumerate(folds):
        tr = df[df["session"].isin(train_s)]
        va = df[df["session"].isin(val_s)]
        if len(tr) < 200 or len(va) < 50:
            continue
        tr = _subsample(tr, max_train_rows, seed=31 + f_i)
        y_tr = tr["_y"].astype(int).values
        if len(np.unique(y_tr)) < 2:
            continue
        clf = make_direction_classifier("lightgbm", params)
        clf.fit(tr[feat_cols], y_tr, sample_weight=_multiclass_sample_weight(y_tr))
        pred = clf.predict(va[feat_cols]).astype(int)
        y_va = va["_y"].astype(int).values
        fold_accs.append(float(accuracy_score(y_va, pred)))
        class_rates_list.append(direction_class_rates(tr["_y"]))

    if not fold_accs:
        return None

    edges = _label_direction_edges_bps(df["_y"], df["_fwd"])
    rates = direction_class_rates(df["_y"])
    mean_rates = {
        name: float(np.mean([r[name] for r in class_rates_list])) if class_rates_list else rates[name]
        for name in ("flat", "long", "short")
    }
    return {
        "direction_accuracy": round(float(np.mean(fold_accs)), 4),
        "direction_accuracy_std": round(float(np.std(fold_accs)) if len(fold_accs) > 1 else 0.0, 4),
        "class_rates": {k: round(v, 4) for k, v in rates.items()},
        "train_class_rates": {k: round(v, 4) for k, v in mean_rates.items()},
        "long_edge_bps": round(float(edges.get("long_edge_bps", edges.get("long", 0.0))), 3),
        "short_edge_bps": round(float(edges.get("short_edge_bps", -edges.get("short", 0.0))), 3),
        "flat_edge_bps": round(float(edges.get("flat", 0.0)), 3),
        "n_rows": int(len(df)),
        "n_folds": len(fold_accs),
    }


def _direction_label_feasible(metrics: dict) -> bool:
    """Direction targets must have enough L/S signal and not be mostly flat."""
    rates = metrics.get("class_rates") or {}
    flat = float(rates.get("flat", 1.0))
    directional = float(rates.get("long", 0.0)) + float(rates.get("short", 0.0))
    max_flat = float(getattr(_cfg, "TARGET_OPT_DIRECTION_MAX_FLAT_RATE", 0.50))
    min_dir = float(getattr(_cfg, "TARGET_OPT_DIRECTION_MIN_DIRECTIONAL_RATE", 0.35))
    return flat <= max_flat and directional >= min_dir


def _direction_trade_penalties(metrics: dict) -> tuple[float, float]:
    """Return (flat_penalty, directional_penalty) in [0, 1] for direction scoring."""
    rates = metrics.get("class_rates") or {}
    flat = float(rates.get("flat", 1.0))
    directional = float(rates.get("long", 0.0)) + float(rates.get("short", 0.0))
    max_flat = float(getattr(_cfg, "TARGET_OPT_DIRECTION_MAX_FLAT_RATE", 0.50))
    min_dir = float(getattr(_cfg, "TARGET_OPT_DIRECTION_MIN_DIRECTIONAL_RATE", 0.35))
    flat_penalty = 1.0
    if flat > max_flat:
        flat_penalty = max(0.0, 1.0 - (flat - max_flat) / max(1.0 - max_flat, 1e-6))
    dir_penalty = 1.0
    if directional < min_dir:
        dir_penalty = max(0.0, directional / min_dir) if min_dir > 0 else 0.0
    return flat_penalty, dir_penalty


def direction_target_score(
    metrics: dict,
    *,
    class_ranges: dict[str, tuple[float, float]] | None = None,
    weights: dict[str, float] | None = None,
    symbol: str | None = None,
) -> float:
    """Composite score for 3-class direction targets."""
    w = weights or getattr(
        _cfg,
        "TARGET_OPT_DIRECTION_SCORE_WEIGHTS",
        {"accuracy": 0.35, "economics": 0.35, "balance": 0.20, "frequency": 0.10},
    )
    acc = float(metrics.get("direction_accuracy", 0.0))
    acc_score = max(0.0, min(1.0, (acc - 0.33) / 0.20))
    long_edge = float(metrics.get("long_edge_bps", 0.0))
    short_edge = float(metrics.get("short_edge_bps", 0.0))
    econ_raw = min(long_edge, short_edge)
    econ_score = max(0.0, min(1.0, econ_raw / 80.0))
    balance = _direction_balance_factor(metrics.get("class_rates", {}), class_ranges)

    rates = metrics.get("class_rates", {})
    directional_rate = float(rates.get("long", 0.0)) + float(rates.get("short", 0.0))
    horizon = max(1.0, float(metrics.get("horizon", 48.0)))
    bars_per_year = float(metrics.get("bars_per_year") or _bars_per_year_for_symbol(symbol or ""))
    est_trades_per_year = directional_rate * bars_per_year / horizon
    target_tpy = float(getattr(_cfg, "TARGET_OPT_TRADES_PER_YEAR_TARGET", 30.0))
    freq_score = max(0.0, min(1.0, est_trades_per_year / max(target_tpy, 1.0)))

    total_w = sum(w.values()) or 1.0
    blended = (
        w.get("accuracy", 0.0) * acc_score
        + w.get("economics", 0.0) * econ_score
        + w.get("balance", 0.0) * balance
        + w.get("frequency", 0.0) * freq_score
    ) / total_w
    flat_penalty, dir_penalty = _direction_trade_penalties(metrics)
    blended *= flat_penalty * dir_penalty
    horizon_pen = float(getattr(_cfg, "TARGET_OPT_HORIZON_PENALTY_PER_BAR", 0.0)) * horizon

    if long_edge > 0 and short_edge > 0:
        return round(max(0.0, 0.5 + 0.5 * blended - horizon_pen), 4)
    losers = (
        w.get("accuracy", 0.0) * acc_score
        + w.get("balance", 0.0) * balance
        + w.get("frequency", 0.0) * freq_score
    ) / max(w.get("accuracy", 0.0) + w.get("balance", 0.0) + w.get("frequency", 0.0), 1e-6)
    return round(max(0.0, 0.25 * losers - horizon_pen), 4)


def _top_decile_net_edge_bps(
    prob: np.ndarray,
    fwd_ret: np.ndarray,
    *,
    commission_bps: float,
    slippage_bps: float | None = None,
) -> float:
    """Top-proba decile net bps using round-trip costs (aligned with decile gate).

    Historically this subtracted ``cost_floor_bps`` (TP min ≈ SL+RT), which made
    almost every candidate look untradeable. Gate / Optuna / soft-size all use RT.
    """
    from models.profit_metrics import top_decile_net_bps

    return float(
        top_decile_net_bps(
            prob,
            fwd_ret,
            commission_bps=float(commission_bps),
            slippage_bps=slippage_bps,
        )
    )


def evaluate_target(
    work_sym: pd.DataFrame,
    feat_cols: list[str],
    folds: list[tuple[set, set]],
    label: pd.Series,
    fwd: pd.Series,
    params: dict,
    *,
    cost_floor_bps: float | None = None,
    max_train_rows: int | None,
    symbol: str | None = None,
    commission_bps: float | None = None,
) -> dict | None:
    """Purged-session CV of one candidate target → CV metrics + economics + balance.

    ``cost_floor_bps`` is unused for edge scoring (kept for call-site compat);
    economics use per-ticker round-trip costs like the live decile gate.
    """
    del cost_floor_bps  # TP floor is for label thresholds, not net-edge scoring
    from data_platform.universe import commission_bps_for_ticker

    df = work_sym.copy()
    df["_y"] = np.asarray(label, dtype=float)
    df["_fwd"] = np.asarray(fwd, dtype=float)
    df = df.dropna(subset=feat_cols + ["_y"])
    if df.empty or df["_y"].nunique() < 2:
        return None

    if commission_bps is not None:
        comm = float(commission_bps)
    elif symbol:
        comm = float(commission_bps_for_ticker(str(symbol)))
    elif "ticker" in df.columns and len(df):
        comm = float(commission_bps_for_ticker(str(df["ticker"].iloc[0])))
    else:
        comm = float(getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 1.1))

    fold_rows: list[dict] = []
    edges: list[float] = []
    pos_rates: list[float] = []
    for f_i, (train_s, val_s) in enumerate(folds):
        tr = df[df["session"].isin(train_s)]
        va = df[df["session"].isin(val_s)]
        if len(tr) < 200 or len(va) < 50 or tr["_y"].nunique() < 2:
            continue
        tr = _subsample(tr, max_train_rows, seed=17 + f_i)
        clf = make_entry_classifier("lightgbm", params)
        clf.fit(tr[feat_cols], tr["_y"].astype(int))
        pr = clf.predict_proba(va[feat_cols])[:, 1]
        y = va["_y"].astype(int).values
        fr = va["_fwd"].values
        fold_rows.append(compute_fold_metrics(y, pr, fr, commission_bps=comm))
        edges.append(
            _top_decile_net_edge_bps(pr, fr, commission_bps=comm)
        )
        pos_rates.append(float(tr["_y"].mean()))

    if not fold_rows:
        return None
    agg = aggregate_cv_metrics(fold_rows)
    # Prefer aggregated CV top-decile (same formula as Optuna) when finite.
    cv_top = agg.get("top_decile_net_bps")
    finite_edges = [float(e) for e in edges if e is not None and np.isfinite(float(e))]
    mean_edge = float(np.mean(finite_edges)) if finite_edges else 0.0
    if cv_top is not None and np.isfinite(float(cv_top)):
        mean_edge = float(cv_top)
    return {
        "cv": {k: v for k, v in agg.items()},
        "positive_rate": round(float(np.mean(pos_rates)) if pos_rates else float(df["_y"].mean()), 4),
        "net_edge_bps": round(mean_edge, 3),
        "net_edge_bps_std": round(float(np.std(finite_edges)) if len(finite_edges) > 1 else 0.0, 3),
        "n_rows": int(len(df)),
        "n_folds": len(fold_rows),
    }


def _balance_factor(positive_rate: float, balance_range: tuple[float, float]) -> float:
    lo, hi = balance_range
    pos = float(positive_rate)
    if lo <= pos <= hi:
        return 1.0
    if pos < lo:
        return max(0.0, pos / lo) if lo > 0 else 0.0
    return max(0.0, 1.0 - (pos - hi) / max(1.0 - hi, 1e-6))


def _bars_per_year_for_symbol(symbol: str) -> float:
    from data_platform.universe import is_crypto_symbol
    import config as cfg

    if is_crypto_symbol(symbol):
        return float(getattr(cfg, "CRYPTO_BARS_PER_DAY", 288)) * float(
            getattr(cfg, "CRYPTO_TRADING_DAYS_PER_YEAR", 365)
        )
    return float(getattr(cfg, "BARS_PER_YEAR", 78 * 252))


def target_score(
    metrics: dict,
    *,
    balance_range: tuple[float, float],
    weights: dict[str, float] | None = None,
    symbol: str | None = None,
) -> float:
    """Economics-gated composite score with trade-frequency term for LLN."""
    w = weights or getattr(
        _cfg,
        "TARGET_OPT_SCORE_WEIGHTS",
        {"cv": 0.30, "economics": 0.35, "balance": 0.15, "frequency": 0.20},
    )
    cv_composite = float(metrics.get("cv", {}).get("composite", 0.0))
    cv_score = max(0.0, min(1.0, cv_composite / 0.6))
    net_edge = float(metrics.get("net_edge_bps", 0.0))
    econ_score = max(0.0, min(1.0, net_edge / 20.0))
    balance = _balance_factor(float(metrics.get("positive_rate", 0.0)), balance_range)

    pos_rate = float(metrics.get("positive_rate", 0.0))
    horizon = max(1.0, float(metrics.get("horizon", 48.0)))
    bars_per_year = float(metrics.get("bars_per_year") or _bars_per_year_for_symbol(symbol or ""))
    est_trades_per_year = pos_rate * bars_per_year / horizon
    target_tpy = float(getattr(_cfg, "TARGET_OPT_TRADES_PER_YEAR_TARGET", 80.0))
    freq_score = max(0.0, min(1.0, est_trades_per_year / max(target_tpy, 1.0)))

    total_w = sum(w.values()) or 1.0
    blended = (
        w.get("cv", 0.0) * cv_score
        + w.get("economics", 0.0) * econ_score
        + w.get("balance", 0.0) * balance
        + w.get("frequency", 0.0) * freq_score
    ) / total_w

    horizon_pen = float(getattr(_cfg, "TARGET_OPT_HORIZON_PENALTY_PER_BAR", 0.0)) * horizon
    if net_edge > 0:
        return round(max(0.0, 0.5 + 0.5 * blended - horizon_pen), 4)
    losers = (
        w.get("cv", 0.0) * cv_score
        + w.get("balance", 0.0) * balance
        + w.get("frequency", 0.0) * freq_score
    ) / max(w.get("cv", 0.0) + w.get("balance", 0.0) + w.get("frequency", 0.0), 1e-6)
    return round(max(0.0, 0.25 * losers - horizon_pen), 4)


def _candidate_specs(
    horizons: tuple[int, ...],
    label_types: tuple[str, ...],
    thr_mults: tuple[float, ...],
    cost_floor_bps: float,
) -> list[dict]:
    max_h = int(getattr(_cfg, "TARGET_OPT_MAX_HORIZON_BARS", 0) or 0)
    specs: list[dict] = []
    for h in horizons:
        if max_h > 0 and int(h) > max_h:
            continue
        for lt in label_types:
            mults = (1.0,) if lt == "positive" else thr_mults
            for m in mults:
                specs.append({
                    "horizon": int(h),
                    "label_type": str(lt),
                    "threshold_bps": round(float(cost_floor_bps) * float(m), 2),
                    "threshold_mult": float(m),
                })
    return specs


def _pick_best_symbol_result(
    sym: str,
    results: list[dict],
    *,
    scoring_mode: str = "entry",
) -> tuple[dict | None, list[str]]:
    """Select winner and human-readable log lines for one symbol."""
    logs: list[str] = []
    if not results:
        logs.append(f"  [{sym}] no valid candidates")
        return None, logs

    score_key = "direction_score" if scoring_mode == "direction" else "score"
    if scoring_mode == "direction":
        require_edge = bool(getattr(_cfg, "TARGET_OPT_DIRECTION_REQUIRE_EDGE", True))
        feasible = [r for r in results if _direction_label_feasible(r)]
        tradeable = [
            r for r in feasible
            if float(r.get("long_edge_bps", 0.0)) > 0.0 and float(r.get("short_edge_bps", 0.0)) > 0.0
        ]
        edge_note = lambda r: (
            f"L/S edge={r.get('long_edge_bps', 0):+.1f}/{r.get('short_edge_bps', 0):+.1f}bps "
            f"acc={r.get('direction_accuracy', 0):.3f} "
            f"F/L/S={r.get('class_rates', {}).get('flat', 0):.0%}/"
            f"{r.get('class_rates', {}).get('long', 0):.0%}/"
            f"{r.get('class_rates', {}).get('short', 0):.0%}"
        )
        if require_edge and tradeable:
            tradeable.sort(key=lambda r: r[score_key], reverse=True)
            best = tradeable[0]
        elif feasible:
            feasible.sort(key=lambda r: r[score_key], reverse=True)
            best = feasible[0]
            if require_edge:
                logs.append(
                    f"  [{sym}] WARNING: no tradeable feasible candidate — "
                    f"using best feasible {edge_note(best)} H={best['horizon']}",
                )
        else:
            results.sort(
                key=lambda r: (
                    min(float(r.get("long_edge_bps", -999.0)), float(r.get("short_edge_bps", -999.0))),
                    r[score_key],
                ),
                reverse=True,
            )
            best = results[0]
            logs.append(
                f"  [{sym}] WARNING: no feasible direction label (flat>{getattr(_cfg, 'TARGET_OPT_DIRECTION_MAX_FLAT_RATE', 0.5):.0%} "
                f"or L+S<{getattr(_cfg, 'TARGET_OPT_DIRECTION_MIN_DIRECTIONAL_RATE', 0.35):.0%}) — "
                f"using fallback {edge_note(best)} H={best['horizon']}",
            )
    else:
        require_edge = bool(getattr(_cfg, "TARGET_OPT_REQUIRE_POSITIVE_EDGE", True))
        tradeable = [r for r in results if float(r.get("net_edge_bps", 0.0)) > 0.0]
        edge_note = lambda r: f"edge={r.get('net_edge_bps', 0):+.1f}bps"
        if require_edge and tradeable:
            tradeable.sort(key=lambda r: r[score_key], reverse=True)
            best = tradeable[0]
        elif require_edge:
            results.sort(
                key=lambda r: (float(r.get("net_edge_bps", -999.0)), r[score_key]),
                reverse=True,
            )
            best = results[0]
            logs.append(
                f"  [{sym}] WARNING: no candidate with positive net edge — "
                f"using least-negative {edge_note(best)} H={best['horizon']}",
            )
        else:
            results.sort(key=lambda r: r[score_key], reverse=True)
            best = results[0]

    candidate_keys = (
        "horizon", "label_type", "threshold_bps", score_key,
        "direction_accuracy", "long_edge_bps", "short_edge_bps",
        "positive_rate", "net_edge_bps",
    )
    per_symbol = {
        "spec": {
            "horizon": best["horizon"],
            "label_type": best["label_type"],
            "threshold_bps": best["threshold_bps"],
        },
        "scoring_mode": scoring_mode,
        "score": best.get("direction_score") if scoring_mode == "direction" else best.get("score"),
        "tradeable": bool(
            float(best.get("long_edge_bps", 0.0)) > 0 and float(best.get("short_edge_bps", 0.0)) > 0
            if scoring_mode == "direction"
            else float(best.get("net_edge_bps", 0.0)) > 0
        ),
        "cv": best.get("cv"),
        "positive_rate": best.get("positive_rate"),
        "net_edge_bps": best.get("net_edge_bps"),
        "direction": {
            "accuracy": best.get("direction_accuracy"),
            "class_rates": best.get("class_rates"),
            "long_edge_bps": best.get("long_edge_bps"),
            "short_edge_bps": best.get("short_edge_bps"),
            "score": best.get("direction_score"),
        },
        "candidates": [
            {k: r[k] for k in candidate_keys if k in r}
            for r in sorted(results, key=lambda r: r.get(score_key, 0.0), reverse=True)[:8]
        ],
    }
    if scoring_mode == "direction":
        logs.append(
            f"  -> [{sym}] BEST H={best['horizon']} {best['label_type']} "
            f"thr={best['threshold_bps']}bps dir_score={best.get('direction_score', 0):.4f} "
            f"acc={best.get('direction_accuracy', 0):.3f} "
            f"flat/long/short={best.get('class_rates', {}).get('flat', 0):.2%}/"
            f"{best.get('class_rates', {}).get('long', 0):.2%}/"
            f"{best.get('class_rates', {}).get('short', 0):.2%}",
        )
    else:
        logs.append(
            f"  -> [{sym}] BEST H={best['horizon']} {best['label_type']} "
            f"thr={best['threshold_bps']}bps score={best.get('score', 0):.4f}",
        )
    return per_symbol, logs


def _optimize_single_symbol(
    sym: str,
    work_sym: pd.DataFrame,
    cols: list[str],
    specs: list[dict],
    params: dict,
    *,
    cost_floor: float,
    max_rows: int | None,
    brange: tuple[float, float],
    scoring_mode: str = "entry",
) -> dict:
    """Sweep all target candidates for one symbol (process-pool safe)."""
    from data_platform.universe import commission_bps_for_ticker
    from simulation.entry_signals import min_tp_gross_bps

    logs: list[str] = []
    if work_sym.empty:
        logs.append(f"  [{sym}] no rows in panel — skipped")
        return {"symbol": sym, "per_symbol": None, "leaderboard": [], "logs": logs}

    sym_comm = commission_bps_for_ticker(sym)
    sym_floor = min_tp_gross_bps(sym_comm, DEFAULT_SLIPPAGE_BPS, buffer_bps=0.0)
    if specs:
        horizons = tuple(sorted({int(s["horizon"]) for s in specs}))
        label_types = tuple(dict.fromkeys(str(s["label_type"]) for s in specs))
        thr_mults = tuple(sorted({float(s.get("threshold_mult", 1.0)) for s in specs}))
        specs = _candidate_specs(horizons, label_types, thr_mults, sym_floor)

    close = work_sym.set_index("bar_time")["close"].astype(float)
    sessions = sorted(work_sym["session"].unique())
    results: list[dict] = []
    leaderboard: list[dict] = []

    for c_i, spec in enumerate(specs, start=1):
        horizon = int(spec["horizon"])
        folds = _purged_session_folds(sessions, max_label_horizon_bars=horizon)
        label_full, fwd_full = build_entry_label(
            close,
            horizon_bars=spec["horizon"],
            label_type=spec["label_type"],
            threshold_bps=spec["threshold_bps"],
            commission_bps=sym_comm,
        )
        dir_full = build_direction_label(
            close,
            horizon_bars=spec["horizon"],
            label_type=spec["label_type"],
            threshold_bps=spec["threshold_bps"],
            commission_bps=sym_comm,
        )
        bar_index = pd.DatetimeIndex(pd.to_datetime(work_sym["bar_time"]))
        label = label_full.reindex(bar_index)
        fwd = fwd_full.reindex(bar_index)
        direction = dir_full.reindex(bar_index)
        metrics = evaluate_target(
            work_sym, cols, folds, label, fwd, params,
            cost_floor_bps=cost_floor,
            max_train_rows=max_rows,
            symbol=sym,
            commission_bps=sym_comm,
        )
        dir_metrics = evaluate_direction_target(
            work_sym, cols, folds, direction, fwd, params, max_train_rows=max_rows,
        )
        if metrics is None and dir_metrics is None:
            continue
        if metrics is None:
            metrics = {"positive_rate": 0.0, "net_edge_bps": 0.0, "cv": {}}
        if dir_metrics is None:
            dir_metrics = {
                "direction_accuracy": 0.0,
                "class_rates": {"flat": 0.0, "long": 0.0, "short": 0.0},
                "long_edge_bps": 0.0,
                "short_edge_bps": 0.0,
            }
        merged = {
            **metrics,
            **dir_metrics,
            "horizon": int(spec["horizon"]),
            "bars_per_year": _bars_per_year_for_symbol(sym),
        }
        entry_score = target_score(merged, balance_range=brange, symbol=sym)
        direction_score = direction_target_score(merged, symbol=sym)
        pick_score = direction_score if scoring_mode == "direction" else entry_score
        row = {
            "symbol": sym,
            "score": entry_score,
            "direction_score": direction_score,
            **spec,
            **merged,
        }
        results.append(row)
        leaderboard.append({
            "symbol": sym,
            "score": pick_score,
            "entry_score": entry_score,
            "direction_score": direction_score,
            "horizon": spec["horizon"],
            "label_type": spec["label_type"],
            "threshold_bps": spec["threshold_bps"],
            "auc": metrics.get("cv", {}).get("auc"),
            "cv_composite": metrics.get("cv", {}).get("composite"),
            "positive_rate": metrics.get("positive_rate"),
            "net_edge_bps": metrics.get("net_edge_bps"),
            "direction_accuracy": dir_metrics.get("direction_accuracy"),
            "long_edge_bps": dir_metrics.get("long_edge_bps"),
            "short_edge_bps": dir_metrics.get("short_edge_bps"),
        })
        if scoring_mode == "direction":
            cr = dir_metrics.get("class_rates", {})
            logs.append(
                f"  [{sym} {c_i}/{len(specs)}] H={spec['horizon']:>3} {spec['label_type']:<14} "
                f"thr={spec['threshold_bps']:>5}bps | dir={direction_score:.4f} "
                f"acc={dir_metrics.get('direction_accuracy', 0):.3f} "
                f"F/L/S={cr.get('flat', 0):.2%}/{cr.get('long', 0):.2%}/{cr.get('short', 0):.2%} "
                f"edge={dir_metrics.get('long_edge_bps', 0):+.0f}/{dir_metrics.get('short_edge_bps', 0):+.0f}bps",
            )
        else:
            logs.append(
                f"  [{sym} {c_i}/{len(specs)}] H={spec['horizon']:>3} {spec['label_type']:<14} "
                f"thr={spec['threshold_bps']:>5}bps | score={entry_score:.4f} "
                f"auc={metrics.get('cv', {}).get('auc', 0):.4f} pos={metrics.get('positive_rate', 0):.3f} "
                f"edge={metrics.get('net_edge_bps', 0):+.1f}bps",
            )

    per_symbol, pick_logs = _pick_best_symbol_result(sym, results, scoring_mode=scoring_mode)
    logs.extend(pick_logs)
    return {
        "symbol": sym,
        "per_symbol": per_symbol,
        "leaderboard": leaderboard,
        "logs": logs,
    }


def _optimize_symbol_worker(payload: dict) -> dict:
    """Unpack dict payload for Windows process spawn."""
    return _optimize_single_symbol(
        payload["sym"],
        payload["work_sym"],
        payload["cols"],
        payload["specs"],
        payload["params"],
        cost_floor=payload["cost_floor"],
        max_rows=payload["max_rows"],
        brange=tuple(payload["brange"]),
        scoring_mode=str(payload.get("scoring_mode", "entry")),
    )


def optimize_targets_per_instrument(
    panel: pd.DataFrame,
    symbols: list[str],
    prices: pd.DataFrame | None = None,
    *,
    horizons: tuple[int, ...] | None = None,
    label_types: tuple[str, ...] | None = None,
    thr_mults: tuple[float, ...] | None = None,
    cost_floor_bps: float | None = None,
    max_train_rows: int | None = None,
    balance_range: tuple[float, float] | None = None,
    feat_cols: list[str] | None = None,
    scoring_mode: str | None = None,
) -> dict:
    """Sweep (horizon, label_type, threshold) per symbol; return best spec + leaderboard.

    Callers must pass a panel that excludes stitched walk-forward OOS dates
    (see ``leakage_guard.panel_for_causal_target_opt``).
    """
    from research.features.registry import FUSION_FEATURE_COLS, resolve_ml_feature_cols  # local import avoids cycle

    mode = str(
        scoring_mode
        or getattr(_cfg, "TARGET_OPT_SCORING_MODE", "entry")
    ).lower()
    horizons = horizons or tuple(getattr(_cfg, "TARGET_OPT_HORIZON_GRID", (12, 24, 48, 96, 144)))
    if mode == "direction":
        min_h = int(getattr(_cfg, "TARGET_OPT_DIRECTION_MIN_HORIZON_BARS", 24) or 24)
        horizons = tuple(h for h in horizons if int(h) >= min_h)
        if not horizons:
            raise ValueError(f"No horizons >= {min_h} for direction target-opt")
    label_types = label_types or tuple(getattr(_cfg, "TARGET_OPT_LABEL_TYPES", ("after_costs", "triple_barrier")))
    thr_mults = thr_mults or tuple(getattr(_cfg, "TARGET_OPT_THRESHOLD_MULTS", (1.0, 1.5, 2.0)))
    cfg_floor = getattr(_cfg, "TARGET_OPT_COST_FLOOR_BPS", None)
    cost_floor = float(
        cost_floor_bps
        if cost_floor_bps is not None
        else (
            cfg_floor
            if cfg_floor is not None
            else min_tp_gross_bps(
                float(getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 10.0)),
                DEFAULT_SLIPPAGE_BPS,
                buffer_bps=0.0,
            )
        )
    )
    max_rows = max_train_rows if max_train_rows is not None else getattr(_cfg, "TARGET_OPT_MAX_TRAIN_ROWS", 60_000)
    brange = balance_range or tuple(getattr(_cfg, "TARGET_OPT_BALANCE_RANGE", (0.05, 0.55)))

    cols = feat_cols or resolve_ml_feature_cols(panel)
    if len(cols) < 10:
        raise ValueError(f"Too few fusion features in panel: {len(cols)}")

    specs = _candidate_specs(horizons, label_types, thr_mults, cost_floor)
    params = _eval_params()
    per_symbol: dict[str, dict] = {}
    leaderboard: list[dict] = []

    print(
        f"target-opt: {len(symbols)} symbols x {len(specs)} candidates | "
        f"mode={mode} | cost_floor={cost_floor}bps | feats={len(cols)} | max_train_rows={max_rows}",
        flush=True,
    )
    workers = resolve_worker_count(
        "FUSION_TARGET_OPT_WORKERS",
        cap=len(symbols),
        env_var="FUSION_TARGET_OPT_WORKERS",
    )
    symbol_frames = {
        sym: panel[panel["ticker"] == sym]
        for sym in symbols
        if not panel[panel["ticker"] == sym].empty
    }
    for sym in symbols:
        if sym not in symbol_frames:
            print(f"  [{sym}] no rows in panel — skipped", flush=True)

    if workers <= 1 or len(symbol_frames) <= 1:
        for sym, work_sym in symbol_frames.items():
            out = _optimize_single_symbol(
                sym, work_sym, cols, specs, params,
                cost_floor=cost_floor, max_rows=max_rows, brange=brange,
                scoring_mode=mode,
            )
            for line in out["logs"]:
                print(line, flush=True)
            if out["per_symbol"] is not None:
                per_symbol[sym] = out["per_symbol"]
            leaderboard.extend(out["leaderboard"])
    else:
        print(f"target-opt: parallel workers={workers}", flush=True)
        payloads = [
            {
                "sym": sym,
                "work_sym": work_sym,
                "cols": cols,
                "specs": specs,
                "params": params,
                "cost_floor": cost_floor,
                "max_rows": max_rows,
                "brange": brange,
                "scoring_mode": mode,
            }
            for sym, work_sym in symbol_frames.items()
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_optimize_symbol_worker, p) for p in payloads]
            for fut in as_completed(futures):
                out = fut.result()
                for line in out["logs"]:
                    print(line, flush=True)
                if out["per_symbol"] is not None:
                    per_symbol[out["symbol"]] = out["per_symbol"]
                leaderboard.extend(out["leaderboard"])

    leaderboard.sort(key=lambda r: r["score"], reverse=True)
    return {
        "per_symbol": per_symbol,
        "leaderboard": leaderboard[:40],
        "scoring_mode": mode,
        "cost_floor_bps": cost_floor,
        "eval_params": params,
        "horizons": list(horizons),
        "label_types": list(label_types),
        "threshold_mults": list(thr_mults),
        "n_features": len(cols),
    }


def specs_dict_from_optimization(result: dict) -> dict[str, dict]:
    """Extract ``{SYMBOL: spec}`` from ``optimize_targets_per_instrument`` output."""
    out: dict[str, dict] = {}
    for sym, entry in (result.get("per_symbol") or {}).items():
        if not entry or not entry.get("spec"):
            continue
        out[str(sym).upper()] = dict(entry["spec"])
    return out


def max_horizon_from_specs(specs: dict[str, dict], fallback: int) -> int:
    """Largest label horizon across per-fold specs (for embargo trimming)."""
    horizons = [int(s.get("horizon", 0)) for s in specs.values() if int(s.get("horizon", 0)) > 0]
    return max(horizons) if horizons else int(fallback)


def relabel_panel_entry_targets(
    panel: pd.DataFrame,
    specs_by_ticker: dict[str, dict],
) -> pd.DataFrame:
    """Re-attach unified entry/direction labels using per-fold target specs."""
    from research.labels.trade import attach_economic_entry_labels

    if panel.empty or "ticker" not in panel.columns or "close" not in panel.columns:
        return panel
    if not specs_by_ticker:
        return panel

    parts: list[pd.DataFrame] = []
    for ticker, grp in panel.groupby("ticker", sort=False):
        sym = str(ticker).upper()
        spec = specs_by_ticker.get(sym)
        if spec is None:
            parts.append(grp)
            continue
        g = grp.sort_values("bar_time").copy()
        parts.append(attach_economic_entry_labels(g, close_col="close", symbol=sym, spec=spec))
    if not parts:
        return panel
    out = pd.concat(parts, ignore_index=True)
    out["bar_time"] = pd.to_datetime(out["bar_time"])
    return out.sort_values(["bar_time", "ticker"]).reset_index(drop=True)


def optimize_targets_on_fold_train(
    train: pd.DataFrame,
    feat_cols: list[str],
    symbols: list[str],
    *,
    fold_meta: dict | None = None,
    scoring_mode: str | None = None,
) -> dict:
    """Per-fold target grid on train-only rows (does not write global cache)."""
    from common.stage_log import stage_log

    meta = dict(fold_meta or {})
    fold_n = meta.get("fold", "?")
    min_rows = int(getattr(_cfg, "FUSION_FOLD_TARGET_OPT_MIN_TRAIN_ROWS", 5_000))
    if train.empty or len(train) < min_rows:
        return {
            "optimizer": "fold_target_grid",
            "skipped": True,
            "reason": "insufficient_train_rows",
            "per_symbol": {},
            **meta,
        }

    stage_log("target optimization: grid search starting", fold=fold_n)

    max_rows = getattr(_cfg, "FUSION_FOLD_TARGET_OPT_MAX_TRAIN_ROWS", None)
    if max_rows is None:
        max_rows = getattr(_cfg, "TARGET_OPT_MAX_TRAIN_ROWS", 60_000)

    mode = str(
        scoring_mode or getattr(_cfg, "TARGET_OPT_SCORING_MODE", "entry")
    ).lower()

    horizons: tuple[int, ...] | None = None
    label_types: tuple[str, ...] | None = None
    thr_mults: tuple[float, ...] | None = None
    if bool(getattr(_cfg, "FUSION_FOLD_TARGET_OPT_COMPACT_GRID", True)):
        horizons = tuple(getattr(_cfg, "FUSION_FOLD_TARGET_OPT_HORIZON_GRID", (24, 48, 72, 96)))
        max_h = int(getattr(_cfg, "FUSION_FOLD_TARGET_OPT_MAX_HORIZON_BARS", 0) or 0)
        if max_h > 0:
            horizons = tuple(h for h in horizons if int(h) <= max_h)
        label_types = tuple(getattr(_cfg, "TARGET_OPT_LABEL_TYPES", ("triple_barrier", "after_costs")))
        thr_mults = tuple(
            getattr(_cfg, "FUSION_FOLD_TARGET_OPT_THRESHOLD_MULTS", (1.0, 1.5))
        )

    tickers = sorted({str(s).upper() for s in symbols if s})
    if "ticker" in train.columns:
        present = set(train["ticker"].astype(str).str.upper().unique())
        tickers = [t for t in tickers if t in present] or sorted(present)

    result = optimize_targets_per_instrument(
        train,
        tickers,
        feat_cols=feat_cols,
        max_train_rows=int(max_rows) if max_rows is not None else None,
        scoring_mode=mode,
        horizons=horizons,
        label_types=label_types,
        thr_mults=thr_mults,
    )

    specs = specs_dict_from_optimization(result)
    if specs:
        summary = ", ".join(f"{sym}:H{specs[sym].get('horizon')}" for sym in sorted(specs))
        print(f"      fold {fold_n} target-opt: {summary}", flush=True)

    return {
        "optimizer": "fold_target_grid",
        "skipped": False,
        **result,
        **meta,
    }


def save_target_optimization(result: dict, *, applied: bool = False, path: Path | None = None) -> Path:
    """Persist experiment result; ``applied=True`` activates it at runtime."""
    out = path or TARGET_OPT_PATH
    payload = dict(result)
    payload["applied"] = bool(applied)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def load_target_optimization(path: Path | None = None) -> dict | None:
    src = path or TARGET_OPT_PATH
    if not src.is_file():
        return None
    try:
        return json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def ticker_hold_horizon_bars(ticker: str, default: int) -> int:
    """Per-symbol hold horizon from applied target specs, else ``default``."""
    sym = str(ticker).upper()
    specs = per_instrument_specs()
    if sym in specs:
        h = int(specs[sym].get("horizon", 0) or 0)
        if h > 0:
            return h
    return int(default)


def applied_hold_default(fallback: int) -> int:
    """Median holding horizon implied by the active per-instrument target specs.

    Keeps the backtest's default/fallback holding horizon consistent with where the
    optimized targets actually measure edge. Returns ``fallback`` when no specs are active.
    """
    tradeable_only = not bool(getattr(_cfg, "FUSION_APPLY_TARGET_CACHE", True))
    specs = per_instrument_specs(tradeable_only=tradeable_only)
    horizons = [int(s.get("horizon", 0)) for s in specs.values() if int(s.get("horizon", 0)) > 0]
    if not horizons:
        return int(fallback)
    return int(np.median(horizons))


def per_instrument_specs(*, tradeable_only: bool = True) -> dict[str, dict]:
    """Resolve active per-symbol target specs (inline config > applied cache).

    When ``tradeable_only`` is True, skip symbols whose cached optimization had
    non-positive net edge (falls back to global labels for those symbols).
    """
    inline = getattr(_cfg, "PER_INSTRUMENT_TARGETS", {}) or {}
    if getattr(_cfg, "USE_PER_INSTRUMENT_TARGETS", False) and inline:
        return {str(k).upper(): dict(v) for k, v in inline.items()}

    cached = load_target_optimization()
    apply_cache = bool(getattr(_cfg, "FUSION_APPLY_TARGET_CACHE", False))
    cache_active = (
        cached
        and not getattr(_cfg, "FUSION_IGNORE_APPLIED_TARGETS", False)
        and (bool(cached.get("applied")) or apply_cache)
    )
    if cache_active:
        specs: dict[str, dict] = {}
        for sym, entry in (cached.get("per_symbol") or {}).items():
            if not entry.get("spec"):
                continue
            if tradeable_only and not bool(entry.get("tradeable", False)):
                continue
            specs[str(sym).upper()] = dict(entry["spec"])
        if specs:
            return specs
    if getattr(_cfg, "USE_PER_INSTRUMENT_TARGETS", False) and inline:
        return {str(k).upper(): dict(v) for k, v in inline.items()}
    return {}


def ticker_threshold_bps(ticker: str, *, default: float | None = None) -> float | None:
    """Per-symbol label TP threshold from applied target spec."""
    spec = per_instrument_specs().get(str(ticker).upper())
    if spec and spec.get("threshold_bps") is not None:
        return float(spec["threshold_bps"])
    return default


def ticker_min_expected_edge_bps(
    ticker: str,
    *,
    commission_bps: float | None = None,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> float:
    """Resolved minimum expected edge for one symbol (TP floor vs round-trip cost)."""
    from data_platform.universe import commission_bps_for_ticker
    from strategy.edge_gate import heuristic_gate_floor_bps

    comm = float(
        commission_bps
        if commission_bps is not None
        else commission_bps_for_ticker(ticker)
    )
    floor = heuristic_gate_floor_bps(comm, slippage_bps)
    thr = ticker_threshold_bps(ticker)
    if thr is not None:
        return max(floor, float(thr))
    return floor
