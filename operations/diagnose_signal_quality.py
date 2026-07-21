"""Diagnose OOS signal quality by portfolio, asset class, and ticker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from models.asset_class_models import asset_class_of, asset_class_series
from models.profit_metrics import profitability_score, top_decile_net_bps
from research.diagnostics.decile_audit import (
    decile_audit_by_ticker,
    decile_monotonicity_check,
    tradeable_tickers_from_audit,
)
from simulation.execution_costs import (
    round_trip_cost_bps_for_ticker,
    slippage_bps_per_side,
)
from data_platform.universe import commission_bps_for_ticker
from strategy.threshold_calibrator import _cv_top_decile_net


def _ret_col(df: pd.DataFrame) -> str:
    return "fwd_ret_entry" if "fwd_ret_entry" in df.columns else "fwd_ret"


def _group_audit(df: pd.DataFrame, ret: str, label: str) -> dict:
    if df.empty or "ml_proba" not in df.columns:
        return {"group": label, "n": 0, "error": "empty"}
    port = decile_monotonicity_check(df, ret_col=ret, commission_bps=None)
    tickers = sorted(df["ticker"].astype(str).str.upper().unique()) if "ticker" in df.columns else []
    costs = {t: round_trip_cost_bps_for_ticker(t) for t in tickers}
    return {
        "group": label,
        "n": int(len(df)),
        "tickers": tickers,
        "top_decile_gross_bps": port.get("top_decile_gross_bps"),
        "top_decile_net_bps": port.get("top_decile_net_bps"),
        "monotonic": port.get("monotonic"),
        "tradeable": port.get("tradeable"),
        "reasons": port.get("reasons"),
        "rt_costs": costs,
        "profitability_score": (
            profitability_score(float(port["top_decile_net_bps"]))
            if port.get("top_decile_net_bps") is not None
            else None
        ),
    }


def _ticker_signal_quality(df: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if df.empty or "ticker" not in df.columns:
        return out
    for ticker, grp in df.groupby("ticker", sort=True):
        sym = str(ticker).upper()
        net, n_rows, ok = _cv_top_decile_net(grp, sym)
        out[sym] = {
            "cv_top_decile_net_bps": None if net is None else round(float(net), 3),
            "n_val": int(n_rows),
            "signal_quality_ok": bool(ok),
            "commission_bps": commission_bps_for_ticker(sym),
            "slippage_bps": slippage_bps_per_side(sym),
            "rt_cost_bps": round_trip_cost_bps_for_ticker(sym),
            "asset_class": asset_class_of(sym),
        }
    return out


def diagnose(oos: pd.DataFrame) -> dict:
    ret = _ret_col(oos)
    classes = asset_class_series(oos["ticker"]) if "ticker" in oos.columns else pd.Series(dtype=str)

    groups = [
        _group_audit(oos, ret, "portfolio"),
    ]
    if not classes.empty:
        for cls in sorted(classes.unique()):
            groups.append(_group_audit(oos.loc[classes == cls], ret, cls))

    by_ticker = decile_audit_by_ticker(oos, ret_col=ret)
    tradeable = tradeable_tickers_from_audit(by_ticker)
    signal_quality = _ticker_signal_quality(oos)

    # Align profit_metrics vs audit
    alignment: dict[str, dict] = {}
    for sym, grp in oos.groupby("ticker"):
        s = str(sym).upper()
        metrics_net = top_decile_net_bps(
            grp["ml_proba"].to_numpy(),
            grp[ret].to_numpy(),
            commission_bps=1.1,
            tickers=grp["ticker"].to_numpy(),
            min_rows=200,
        )
        audit_net = by_ticker.get(s, {}).get("top_decile_net_bps")
        alignment[s] = {
            "profit_metrics_net": None if not np.isfinite(metrics_net) else round(float(metrics_net), 3),
            "audit_net": audit_net,
            "delta": (
                None
                if audit_net is None or not np.isfinite(metrics_net)
                else round(abs(float(metrics_net) - float(audit_net)), 3)
            ),
        }

    return {
        "n_rows": int(len(oos)),
        "ret_col": ret,
        "groups": groups,
        "by_ticker": {
            k: {
                "gross": v.get("top_decile_gross_bps"),
                "net": v.get("top_decile_net_bps"),
                "tradeable": v.get("tradeable"),
                "comm": v.get("commission_bps_per_side"),
                "slip": v.get("slippage_bps_per_side"),
                "monotonic": v.get("monotonic"),
                "reasons": v.get("reasons"),
            }
            for k, v in sorted(by_ticker.items())
        },
        "tradeable_tickers": tradeable,
        "signal_quality": signal_quality,
        "metric_alignment": alignment,
        "config": {
            "min_top_decile_net_bps": float(__import__("config").FUSION_MIN_TOP_DECILE_NET_BPS),
            "asset_class_models": bool(__import__("config").FUSION_ASSET_CLASS_MODELS),
            "commission_tradfi": float(__import__("config").COMMISSION_BPS_TRADFI),
            "commission_crypto": float(__import__("config").COMMISSION_BPS_CRYPTO),
        },
    }


def _print_report(report: dict) -> None:
    print(f"OOS rows={report['n_rows']:,} ret_col={report['ret_col']}")
    print(f"config={report['config']}")
    print()
    print("=== GROUPS ===")
    for g in report["groups"]:
        print(
            f"  [{g['group']}] n={g['n']:,} "
            f"gross={g.get('top_decile_gross_bps')} net={g.get('top_decile_net_bps')} "
            f"tradeable={g.get('tradeable')} score={g.get('profitability_score')} "
            f"tickers={g.get('tickers')}"
        )
        if g.get("reasons"):
            print(f"    reasons={g['reasons']}")
        if g.get("rt_costs"):
            print(f"    rt_costs={g['rt_costs']}")

    print()
    print("=== PER TICKER (decile audit) ===")
    for sym, a in report["by_ticker"].items():
        print(
            f"  {sym}: gross={a['gross']} net={a['net']} "
            f"comm={a['comm']} slip={a['slip']} ok={a['tradeable']}"
        )

    print()
    print("=== SIGNAL QUALITY (threshold gate metric, chrono CV) ===")
    for sym, sq in report["signal_quality"].items():
        print(
            f"  {sym} ({sq['asset_class']}): cv_top_decile={sq['cv_top_decile_net_bps']} "
            f"ok={sq['signal_quality_ok']} n={sq['n_val']} rt={sq['rt_cost_bps']}"
        )

    print()
    print("=== METRIC ALIGNMENT (audit vs profit_metrics) ===")
    for sym, a in report["metric_alignment"].items():
        print(f"  {sym}: metrics={a['profit_metrics_net']} audit={a['audit_net']} |delta|={a['delta']}")

    print()
    print(f"TRADEABLE={report['tradeable_tickers']}")


def main() -> int:
    oos_path = Path(ROOT).parent / "output" / "cache" / "fusion_oos_adaptive.parquet"
    if not oos_path.exists():
        print(f"missing {oos_path}")
        return 1

    oos = pd.read_parquet(oos_path)
    report = diagnose(oos)
    _print_report(report)

    out_path = Path(ROOT).parent / "output" / "signal_quality_diagnosis.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
