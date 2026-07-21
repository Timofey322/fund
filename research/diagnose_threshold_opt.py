"""Diagnose per-ticker threshold Optuna zero-signal issue."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
import pandas as pd
from research.features.entry_ml import _purged_session_folds
from simulation.entry_signals import active_entry_signals
from strategy.leakage_guard import resolve_label_horizon_bars
from strategy.panel_paths import load_panel
from strategy.pipeline import _fusion_signal_frame, apply_fusion_scores
from strategy.threshold_opt import (
    _calibrate_edge_on_train,
    _default_policy_params,
    _evaluate_policy_cv,
)


def _load_prices(symbols: list[str]) -> pd.DataFrame:
    from config import BAR_TIMEFRAME
    from data_platform.bars import load_closes

    return load_closes(symbols, BAR_TIMEFRAME)


def diagnose_symbol(sym: str, prices: pd.DataFrame) -> dict:
    train = load_panel(sym)
    train = train.copy()
    train["bar_time"] = pd.to_datetime(train["bar_time"])
    train = train[train["bar_time"] < "2021-08-01"]
    if "label_entry" in train.columns:
        train["ml_proba"] = train["label_entry"].astype(float) * 0.25 + 0.4
    else:
        train["ml_proba"] = 0.5
    train["ml_proba_short"] = 1.0 - train["ml_proba"].clip(0.05, 0.95)
    train["ml_base_rate"] = 0.5

    comm = 1.1
    cal = _calibrate_edge_on_train(train, commission_bps=comm)
    params = _default_policy_params(calibrated_edge=cal)
    sub = train.head(12_000)

    fused = apply_fusion_scores(sub, params)
    sig = _fusion_signal_frame(sub, prices, params)
    act = active_entry_signals(sig) if not sig.empty else sig

    buy_stats = {}
    for buy in (52, 53, 55):
        p = dict(params)
        from strategy.fusion_direction import resolve_trading_thresholds

        p.update(resolve_trading_thresholds(buy, p["hold_threshold"]))
        s2 = _fusion_signal_frame(sub, prices, p)
        a2 = active_entry_signals(s2) if not s2.empty else s2
        buy_stats[buy] = {
            "sig_rows": len(s2),
            "active_rows": len(a2),
            "score_ge_buy": int((s2["score"] >= s2["buy_threshold"]).sum()) if not s2.empty else 0,
        }

    horizon = resolve_label_horizon_bars()
    folds = _purged_session_folds(sorted(train["session"].unique()), max_label_horizon_bars=horizon)
    cv = _evaluate_policy_cv(train, prices, params, folds, commission_bps=comm)

    return {
        "train_rows": len(train),
        "params": {
            k: params.get(k)
            for k in ("buy_threshold", "sell_threshold", "min_expected_edge_bps", "impulse_min")
        },
        "fusion_score_mean": float(fused["fusion_score"].mean()),
        "fusion_score_p90": float(fused["fusion_score"].quantile(0.9)),
        "position_side": fused["position_side"].value_counts().to_dict(),
        "sig_rows": len(sig),
        "active_rows": len(act),
        "score_ge_buy": int((sig["score"] >= sig["buy_threshold"]).sum()) if not sig.empty else 0,
        "buy_sweep": buy_stats,
        "cv": {k: cv.get(k) for k in ("objective", "signal_rows", "active_rebalances", "constraints_ok")},
    }


def main() -> None:
    symbols = ["GAZP", "IMOEX", "NASDAQ", "SP500"]
    prices = _load_prices(symbols)
    for sym in symbols:
        out = diagnose_symbol(sym, prices)
        print(sym, out)


if __name__ == "__main__":
    main()
