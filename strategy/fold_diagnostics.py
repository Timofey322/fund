"""Per-fold diagnostics: what fails on each walk-forward iteration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config as _cfg
from common.stage_log import stage_log
from models.asset_class_models import asset_class_of, asset_class_series
from models.profit_metrics import profitability_score, top_decile_net_bps
from simulation.execution_costs import round_trip_cost_bps_for_ticker


def _ret_col(df: pd.DataFrame) -> str:
    if "fwd_ret_entry" in df.columns:
        return "fwd_ret_entry"
    return "fwd_ret"


def _slice_net(
    df: pd.DataFrame,
    *,
    min_rows: int = 100,
) -> dict:
    """Top-decile net/gross for a panel slice."""
    ret = _ret_col(df)
    if df.empty or "ml_proba" not in df.columns or ret not in df.columns:
        return {"n": int(len(df)), "net": None, "gross": None, "ok": False}
    p = df["ml_proba"].to_numpy(dtype=float)
    r = df[ret].to_numpy(dtype=float)
    tickers = df["ticker"].to_numpy() if "ticker" in df.columns else None
    net = top_decile_net_bps(
        p,
        r,
        commission_bps=float(getattr(_cfg, "COMMISSION_BPS_PER_SIDE", 1.1)),
        tickers=tickers,
        min_rows=min_rows,
    )
    mask = np.isfinite(p) & np.isfinite(r)
    if mask.sum() < min_rows:
        return {"n": int(mask.sum()), "net": None, "gross": None, "ok": False}
    try:
        edges = np.quantile(p[mask], np.linspace(0.0, 1.0, 11))
        top = p[mask] >= edges[-2]
    except Exception:
        return {"n": int(mask.sum()), "net": None, "gross": None, "ok": False}
    gross = float(np.mean(r[mask][top])) * 10_000.0 if top.any() else None
    net_f = float(net) if np.isfinite(net) else None
    return {
        "n": int(mask.sum()),
        "net": None if net_f is None else round(net_f, 3),
        "gross": None if gross is None else round(gross, 3),
        "ok": bool(net_f is not None and net_f >= 0.0),
        "score": None if net_f is None else round(profitability_score(net_f), 3),
    }


def _bottleneck(
    train: dict,
    oos: dict,
    *,
    rt_cost: float | None = None,
    ok_floor: float | None = None,
) -> str:
    """One-line reason what is failing for this slice."""
    if oos.get("net") is None and train.get("net") is None:
        return "insufficient_rows"
    t_net = train.get("net")
    o_net = oos.get("net")
    o_gross = oos.get("gross")
    # Instrument-relative ok floor (economics); fallback to RT cost, never absolute 5bps.
    if ok_floor is not None and math.isfinite(float(ok_floor)):
        floor = float(ok_floor)
    elif rt_cost is not None and math.isfinite(float(rt_cost)):
        over = float(getattr(_cfg, "FUSION_ECONOMICS_OVER_RT_MULT", 1.25))
        floor = float(rt_cost) * over
    else:
        floor = float(getattr(_cfg, "FUSION_ECONOMICS_OVER_RT_MULT", 1.25)) * 2.0
    if o_net is not None and o_net >= floor:
        return "ok"
    if o_gross is not None and rt_cost is not None and o_gross < rt_cost:
        return f"gross<{rt_cost:.1f}_cost"
    if t_net is not None and t_net < 0 and (o_net is None or o_net < 0):
        return "no_edge_train_and_oos"
    if t_net is not None and t_net >= 0 and o_net is not None and o_net < 0:
        return "train_ok_oos_fail"
    if o_net is not None and o_net < 0:
        return "oos_negative"
    if o_net is not None and 0 <= o_net < floor:
        return "oos_below_gate"
    return "unknown"


def diagnose_fold_slice(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    *,
    fold: int | str,
    min_rows: int = 100,
) -> dict:
    """Per-ticker and per-asset-class train vs OOS top-decile diagnostics."""
    report: dict = {
        "fold": fold,
        "portfolio": {},
        "groups": {},
        "tickers": {},
        "worst": None,
    }

    tr_port = _slice_net(train, min_rows=min_rows)
    oo_port = _slice_net(oos, min_rows=min_rows)
    report["portfolio"] = {
        "train": tr_port,
        "oos": oo_port,
        "bottleneck": _bottleneck(tr_port, oo_port, rt_cost=5.2),
    }

    if "ticker" not in oos.columns:
        return report

    # Asset-class groups
    for label, mask_fn in (
        ("crypto", lambda s: asset_class_of(s) == "crypto"),
        ("tradfi", lambda s: asset_class_of(s) == "tradfi"),
    ):
        tr = train[train["ticker"].astype(str).map(mask_fn)] if not train.empty else train
        oo = oos[oos["ticker"].astype(str).map(mask_fn)]
        tr_m = _slice_net(tr, min_rows=min_rows)
        oo_m = _slice_net(oo, min_rows=min_rows)
        report["groups"][label] = {
            "train": tr_m,
            "oos": oo_m,
            "bottleneck": _bottleneck(tr_m, oo_m, rt_cost=5.2 if label == "crypto" else 5.0),
        }

    # Per ticker
    tickers = sorted(set(oos["ticker"].astype(str).str.upper()) | set(train["ticker"].astype(str).str.upper()))
    worst_net = float("inf")
    worst_sym = None
    for sym in tickers:
        tr = train[train["ticker"].astype(str).str.upper() == sym] if not train.empty else train
        oo = oos[oos["ticker"].astype(str).str.upper() == sym]
        tr_m = _slice_net(tr, min_rows=min(min_rows, 50))
        oo_m = _slice_net(oo, min_rows=min(min_rows, 50))
        rt = round_trip_cost_bps_for_ticker(sym)
        try:
            from strategy.instrument_economics import economics_floor_bps

            ok_floor = economics_floor_bps(sym)
        except Exception:
            ok_floor = float(rt) * float(getattr(_cfg, "FUSION_ECONOMICS_OVER_RT_MULT", 1.25))
        bn = _bottleneck(tr_m, oo_m, rt_cost=rt, ok_floor=ok_floor)
        report["tickers"][sym] = {
            "asset_class": asset_class_of(sym),
            "rt_cost_bps": rt,
            "train": tr_m,
            "oos": oo_m,
            "bottleneck": bn,
        }
        o_net = oo_m.get("net")
        if o_net is not None and o_net < worst_net:
            worst_net = o_net
            worst_sym = sym

    report["worst"] = worst_sym
    return report


def log_fold_diagnostics(report: dict) -> None:
    """Print compact per-fold diagnosis."""
    fold = report.get("fold", "?")
    port = report.get("portfolio", {})
    stage_log(
        "diag portfolio",
        fold=fold,
        detail=(
            f"train_net={port.get('train', {}).get('net')} "
            f"oos_net={port.get('oos', {}).get('net')} "
            f"oos_gross={port.get('oos', {}).get('gross')} "
            f"fail={port.get('bottleneck')}"
        ),
    )
    for gname, g in (report.get("groups") or {}).items():
        stage_log(
            f"diag [{gname}]",
            fold=fold,
            detail=(
                f"train={g.get('train', {}).get('net')} "
                f"oos={g.get('oos', {}).get('net')} "
                f"gross={g.get('oos', {}).get('gross')} "
                f"fail={g.get('bottleneck')}"
            ),
        )
    parts = []
    for sym, t in sorted((report.get("tickers") or {}).items()):
        o = t.get("oos") or {}
        tr = t.get("train") or {}
        parts.append(
            f"{sym}:tr={tr.get('net')}/oos={o.get('net')}/{t.get('bottleneck')}"
        )
    if parts:
        stage_log("diag tickers", fold=fold, detail=" | ".join(parts))
    if report.get("worst"):
        w = report["tickers"][report["worst"]]
        stage_log(
            "diag worst",
            fold=fold,
            detail=(
                f"{report['worst']} oos_net={w.get('oos', {}).get('net')} "
                f"fail={w.get('bottleneck')}"
            ),
        )


def append_fold_diagnostics(report: dict, path: Path | None = None) -> Path:
    """Append fold report to JSONL for post-run analysis."""
    out = path or (Path(getattr(_cfg, "OUT_DIR", Path("output"))) / "fold_diagnostics.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report, default=str) + "\n")
    return out


def summarize_fold_diagnostics(path: Path | None = None) -> dict:
    """Aggregate JSONL fold diagnostics into bottleneck counts."""
    out = path or (Path(getattr(_cfg, "OUT_DIR", Path("output"))) / "fold_diagnostics.jsonl")
    if not out.exists():
        return {"folds": 0}
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    bn_port: dict[str, int] = {}
    bn_ticker: dict[str, dict[str, int]] = {}
    oos_nets: list[float] = []
    for r in rows:
        b = (r.get("portfolio") or {}).get("bottleneck") or "unknown"
        bn_port[b] = bn_port.get(b, 0) + 1
        o = (r.get("portfolio") or {}).get("oos") or {}
        if o.get("net") is not None:
            oos_nets.append(float(o["net"]))
        for sym, t in (r.get("tickers") or {}).items():
            bn_ticker.setdefault(sym, {})
            tb = t.get("bottleneck") or "unknown"
            bn_ticker[sym][tb] = bn_ticker[sym].get(tb, 0) + 1
    return {
        "folds": len(rows),
        "portfolio_bottlenecks": bn_port,
        "ticker_bottlenecks": bn_ticker,
        "mean_oos_net": round(float(np.mean(oos_nets)), 3) if oos_nets else None,
        "median_oos_net": round(float(np.median(oos_nets)), 3) if oos_nets else None,
    }
