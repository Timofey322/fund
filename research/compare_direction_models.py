"""Train multiclass vs dual-binary direction models and compare predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as _cfg
from config import FINAM_UNIVERSE, OUT_DIR
from models.direction_model import (
    PerTickerDirectionBundle,
    compare_direction_predictions,
    dual_binary_predict_direction,
)
from models.per_ticker_models import PerTickerModelBundle
from research.features.registry import resolve_ml_feature_cols
from research.labels.trade import (
    TARGET_DIRECTION,
    TARGET_ENTRY,
    TARGET_ENTRY_SHORT,
    attach_economic_entry_labels,
    direction_class_rates,
)
from strategy.panel_paths import load_panel


def _ensure_direction_labels(panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if TARGET_DIRECTION in panel.columns and panel[TARGET_DIRECTION].notna().any():
        return panel
    if "close" not in panel.columns:
        raise ValueError(f"{symbol}: panel missing close column")
    return attach_economic_entry_labels(panel, close_col="close", symbol=symbol)


def _time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.sort_values("bar_time")
    cut = int(len(work) * (1.0 - test_frac))
    return work.iloc[:cut], work.iloc[cut:]


def compare_ticker(
    symbol: str,
    *,
    test_frac: float = 0.2,
    min_rows: int = 5000,
) -> dict:
    panel = load_panel(symbol)
    if panel is None or panel.empty:
        raise ValueError(f"No panel cache for {symbol}")
    panel = _ensure_direction_labels(panel, symbol)
    feat_cols = resolve_ml_feature_cols(panel)
    need = feat_cols + [TARGET_DIRECTION, TARGET_ENTRY, TARGET_ENTRY_SHORT]
    work = panel.dropna(subset=need).copy()
    if len(work) < min_rows:
        raise ValueError(f"{symbol}: only {len(work)} labeled rows")

    tr, te = _time_split(work, test_frac=test_frac)
    tr = tr.copy()
    te = te.copy()
    tr["ticker"] = symbol
    te["ticker"] = symbol

    dual = PerTickerModelBundle("lightgbm")
    dual.fit(tr, feat_cols, TARGET_ENTRY, sample_weight=None, min_rows=min_rows)
    dual.fit_short(tr, feat_cols, sample_weight=None, min_rows=min_rows)

    multi = PerTickerDirectionBundle("lightgbm")
    multi.fit(tr, feat_cols, TARGET_DIRECTION, sample_weight=None, min_rows=min_rows)

    pl = dual.predict_proba(te, feat_cols)[:, 1]
    ps = dual.predict_proba_short(te, feat_cols)[:, 1]
    pred_dual = dual_binary_predict_direction(pl, ps, baseline=0.5, min_edge=0.05)
    pred_multi = multi.predict_direction(te, feat_cols)
    proba_multi = multi.predict_proba(te, feat_cols)

    y = te[TARGET_DIRECTION].astype(int).to_numpy()
    cmp = compare_direction_predictions(y, pred_multi, pred_dual)
    cmp["symbol"] = symbol
    cmp["train_rows"] = int(len(tr))
    cmp["test_rows"] = int(len(te))
    cmp["label_distribution_train"] = direction_class_rates(tr[TARGET_DIRECTION])
    cmp["multiclass_proba_mean"] = {
        name: float(proba_multi[:, i].mean())
        for i, name in enumerate(("flat", "long", "short"))
    }
    return cmp


def run_comparison(
    symbols: list[str] | None = None,
    *,
    out_path: Path | None = None,
) -> dict:
    syms = [str(s).upper() for s in (symbols or FINAM_UNIVERSE)]
    per_symbol: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for sym in syms:
        try:
            per_symbol[sym] = compare_ticker(sym)
            print(
                f"{sym}: mc acc={per_symbol[sym]['multiclass_accuracy']:.3f} "
                f"dual acc={per_symbol[sym]['dual_binary_accuracy']:.3f} "
                f"agree={per_symbol[sym]['multiclass_vs_dual_agreement']:.3f} "
                f"mc L/S={per_symbol[sym]['multiclass_long_share']:.2%}/"
                f"{per_symbol[sym]['multiclass_short_share']:.2%} "
                f"dual L/S={per_symbol[sym]['dual_binary_long_share']:.2%}/"
                f"{per_symbol[sym]['dual_binary_short_share']:.2%}",
                flush=True,
            )
        except Exception as exc:
            errors[sym] = str(exc)
            print(f"{sym}: ERROR {exc}", flush=True)

    payload = {
        "symbols": syms,
        "per_symbol": per_symbol,
        "errors": errors,
    }
    if per_symbol:
        keys = (
            "multiclass_accuracy",
            "dual_binary_accuracy",
            "multiclass_vs_dual_agreement",
            "multiclass_long_share",
            "multiclass_short_share",
            "dual_binary_long_share",
            "dual_binary_short_share",
        )
        payload["aggregate"] = {
            k: round(float(np.mean([v[k] for v in per_symbol.values()])), 4) for k in keys
        }
        payload["aggregate"]["true_long_share"] = round(
            float(np.mean([v["true_distribution"]["long"] for v in per_symbol.values()])), 4
        )
        payload["aggregate"]["true_short_share"] = round(
            float(np.mean([v["true_distribution"]["short"] for v in per_symbol.values()])), 4
        )
        payload["aggregate"]["true_flat_share"] = round(
            float(np.mean([v["true_distribution"]["flat"] for v in per_symbol.values()])), 4
        )

    dest = out_path or (OUT_DIR / "research" / "direction_model_comparison.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {dest}", flush=True)
    return payload


if __name__ == "__main__":
    run_comparison()
