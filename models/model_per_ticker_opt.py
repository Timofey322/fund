"""Per-ticker Optuna on a single walk-forward train slice."""

from __future__ import annotations

from typing import Any

import config as _cfg


def optimize_fusion_model_per_ticker_on_train_slice(
    train: "Any",
    feat_cols: list[str],
    target_col: str,
    *,
    model_name: str | None = None,
    fold_meta: dict | None = None,
    label_horizon_bars: int | None = None,
    n_trials: int | None = None,
) -> dict:
    """Run Optuna separately for each ticker in the train panel.

    When ``FUSION_OPTUNA_SHORT_MODELS`` and ``label_entry_short`` exist, also
    tunes short-side HPs independently (does not reuse long winners).
    """
    from strategy.pipeline import optimize_fusion_model_on_train_slice

    if "ticker" not in train.columns:
        opt = optimize_fusion_model_on_train_slice(
            train,
            feat_cols,
            target_col,
            model_name=model_name,
            fold_meta=fold_meta,
            label_horizon_bars=label_horizon_bars,
            n_trials=n_trials,
        )
        return {
            **opt,
            "optimizer": "optuna",
            "per_ticker": {},
            "short_params_by_ticker": {},
        }

    min_rows = int(getattr(_cfg, "FUSION_PER_TICKER_MIN_ROWS", 200))
    tickers = sorted(train["ticker"].astype(str).str.upper().unique())
    per_ticker: dict[str, dict] = {}
    short_params_by_ticker: dict[str, dict] = {}
    composites: list[float] = []
    profit_nets: list[float] = []

    for sym in tickers:
        sub = train[train["ticker"].astype(str).str.upper() == sym]
        if len(sub) < min_rows or sub[target_col].nunique() < 2:
            continue
        meta = {**(fold_meta or {}), "ticker": sym}
        opt = optimize_fusion_model_on_train_slice(
            sub,
            feat_cols,
            target_col,
            model_name=model_name,
            fold_meta=meta,
            label_horizon_bars=label_horizon_bars,
            n_trials=n_trials,
        )
        per_ticker[sym] = opt
        cv = opt.get("cv") or {}
        comp = cv.get("composite")
        net = cv.get("top_decile_net_bps")
        if comp is not None:
            composites.append(float(comp))
        if net is not None:
            profit_nets.append(float(net))

    short_target = "label_entry_short"
    run_short = (
        bool(getattr(_cfg, "FUSION_OPTUNA_SHORT_MODELS", True))
        and bool(getattr(_cfg, "FUSION_ALLOW_SHORT", False))
        and short_target in train.columns
    )
    if run_short:
        short_trials = int(
            n_trials
            if n_trials is not None
            else getattr(_cfg, "FUSION_MODEL_OPTUNA_SHORT_TRIALS", 60)
        )
        for sym in tickers:
            sub = train[train["ticker"].astype(str).str.upper() == sym]
            if len(sub) < min_rows or sub[short_target].nunique() < 2:
                continue
            meta = {**(fold_meta or {}), "ticker": sym, "side": "short"}
            opt_s = optimize_fusion_model_on_train_slice(
                sub,
                feat_cols,
                short_target,
                model_name=model_name,
                fold_meta=meta,
                label_horizon_bars=label_horizon_bars,
                n_trials=short_trials,
            )
            params = dict(opt_s.get("model_params") or {})
            if params:
                short_params_by_ticker[sym] = params
                if sym in per_ticker:
                    per_ticker[sym]["short_model_params"] = params
                    per_ticker[sym]["short_cv"] = opt_s.get("cv")

    if not per_ticker:
        return optimize_fusion_model_on_train_slice(
            train,
            feat_cols,
            target_col,
            model_name=model_name,
            fold_meta=fold_meta,
            label_horizon_bars=label_horizon_bars,
            n_trials=n_trials,
        )

    agg_cv: dict[str, Any] = {
        "composite": round(float(sum(composites) / len(composites)), 4) if composites else None,
        "top_decile_net_bps": round(float(sum(profit_nets) / len(profit_nets)), 3)
        if profit_nets
        else None,
        "n_tickers_optimized": len(per_ticker),
        "n_short_tickers_optimized": len(short_params_by_ticker),
        "source": "per_ticker_mean",
    }
    first = next(iter(per_ticker.values()))
    return {
        "optimizer": "optuna_per_ticker",
        "model_name": first.get("model_name"),
        "model_params": {},
        "per_ticker": per_ticker,
        "short_params_by_ticker": short_params_by_ticker,
        "cv": agg_cv,
        "trial_results": [],
        "grid_results": [],
        "leaderboard": [],
        "grid_size": 0,
        "n_trials": sum(int(v.get("n_trials") or 0) for v in per_ticker.values()),
        **(fold_meta or {}),
    }
