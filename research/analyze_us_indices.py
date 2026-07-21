"""Summarize NASDAQ/SP500 OOS weakness from fold diagnostics and pipeline report."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = ROOT.parent / "output"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fold_stats(fd: list[dict], ticker: str) -> dict:
    rows = []
    for fold in fd:
        t = (fold.get("tickers") or {}).get(ticker) or {}
        oos = t.get("oos") or {}
        rows.append(
            {
                "fold": fold["fold"],
                "oos_gross": oos.get("gross"),
                "oos_net": oos.get("net"),
                "oos_ok": oos.get("ok"),
                "bottleneck": t.get("bottleneck"),
                "worst_fold": fold.get("worst") == ticker,
            }
        )
    gross = [r["oos_gross"] for r in rows if r["oos_gross"] is not None]
    bottlenecks: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("bottleneck"):
            bottlenecks[str(r["bottleneck"])] += 1
    return {
        "folds": len(rows),
        "oos_ok_folds": sum(1 for r in rows if r["oos_ok"]),
        "oos_gross_mean": round(statistics.mean(gross), 3) if gross else None,
        "oos_gross_median": round(statistics.median(gross), 3) if gross else None,
        "oos_gross_min": round(min(gross), 3) if gross else None,
        "oos_gross_max": round(max(gross), 3) if gross else None,
        "positive_oos_folds": sum(1 for g in gross if g > 0),
        "bottlenecks": dict(sorted(bottlenecks.items(), key=lambda x: -x[1])),
        "worst_ticker_count": sum(1 for r in rows if r["worst_fold"]),
    }


def main() -> None:
    fd = _load_jsonl(OUTPUT / "fold_diagnostics.jsonl")
    report = json.loads((OUTPUT / "fusion_pipeline_report.json").read_text(encoding="utf-8"))
    target = json.loads((OUTPUT / "cache" / "target_optimization.json").read_text(encoding="utf-8"))

    tickers = ["NASDAQ", "SP500", "GAZP", "IMOEX"]
    fold_stats = {tk: _fold_stats(fd, tk) for tk in tickers}
    pt = report.get("per_ticker_backtest") or {}
    target_specs = {k: v.get("spec") for k, v in (target.get("per_symbol") or {}).items()}
    target_dir = {k: v.get("direction") for k, v in (target.get("per_symbol") or {}).items()}

    analysis = {
        "summary": {
            "portfolio_oos_gross_pct": (report.get("oos_backtest") or {}).get("gross_return_pct"),
            "portfolio_oos_net_pct": (report.get("oos_backtest") or {}).get("net_return_pct"),
            "threshold_bug": "buy=42 < sell=45 caused zero-score rows to enter short",
            "symmetric_fix": "FUSION_SELL_THRESHOLD=None, buy>=52, inactive rows zero position_side",
        },
        "fold_diagnostics": fold_stats,
        "per_ticker_isolated_backtest": {tk: pt.get(tk) for tk in tickers if pt.get(tk)},
        "target_specs": target_specs,
        "direction_label_stats": target_dir,
        "us_indices_diagnosis": {
            "NASDAQ": {
                "issue": "OOS gross below 5% rt_cost floor in most folds",
                "fold_ok_rate": f"{fold_stats['NASDAQ']['oos_ok_folds']}/{fold_stats['NASDAQ']['folds']}",
                "label_horizon_bars": (target_specs.get("NASDAQ") or {}).get("horizon"),
                "direction_accuracy": (target_dir.get("NASDAQ") or {}).get("accuracy"),
                "class_balance": (target_dir.get("NASDAQ") or {}).get("class_rates"),
                "isolated_gross_pct": (pt.get("NASDAQ") or {}).get("gross_return_pct"),
            },
            "SP500": {
                "issue": "Same cost-floor bottleneck; H=36 labels vs NASDAQ H=24",
                "fold_ok_rate": f"{fold_stats['SP500']['oos_ok_folds']}/{fold_stats['SP500']['folds']}",
                "label_horizon_bars": (target_specs.get("SP500") or {}).get("horizon"),
                "direction_accuracy": (target_dir.get("SP500") or {}).get("accuracy"),
                "class_balance": (target_dir.get("SP500") or {}).get("class_rates"),
                "isolated_gross_pct": (pt.get("SP500") or {}).get("gross_return_pct"),
            },
        },
    }

    out = OUTPUT / "research" / "us_indices_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(fold_stats, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
