"""Per-ticker LightGBM regressors for take-profit and stop-loss (bps)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config as _cfg
from research.labels.balanced import TARGET_SL_BPS, TARGET_TP_BPS


def make_entry_regressor(params: dict[str, Any] | None = None):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is not installed — pip install lightgbm") from exc
    from common.parallel import lightgbm_thread_count
    from models.entry_model import DEFAULT_LIGHTGBM_PARAMS

    p = dict(DEFAULT_LIGHTGBM_PARAMS)
    if params:
        p.update(params)
    return lgb.LGBMRegressor(
        objective="regression",
        verbosity=-1,
        n_jobs=lightgbm_thread_count(),
        random_state=42,
        **p,
    )


class PerTickerTPSLRegressorBundle:
    """Separate TP and SL regressors per ticker — no cross-instrument sharing."""

    def __init__(self):
        self.tp_models: dict[str, Any] = {}
        self.sl_models: dict[str, Any] = {}
        self.params_by_ticker: dict[str, dict[str, Any]] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        feat_cols: list[str],
        *,
        sample_weight: np.ndarray | None = None,
        params: dict[str, Any] | None = None,
        params_by_ticker: dict[str, dict[str, Any]] | None = None,
        min_rows: int | None = None,
        fwd_ret_col: str | None = None,
    ) -> "PerTickerTPSLRegressorBundle":
        min_rows = int(min_rows or getattr(_cfg, "FUSION_PER_TICKER_MIN_ROWS", 200))
        pbt = params_by_ticker or {}
        default_params = dict(params or {})
        profit_weights = bool(getattr(_cfg, "FUSION_TP_SL_PROFIT_WEIGHTS", False))
        if "ticker" not in frame.columns:
            return self
        for sym in sorted(frame["ticker"].astype(str).str.upper().unique()):
            mask = frame["ticker"].astype(str).str.upper().eq(sym).to_numpy()
            sub = frame.loc[mask]
            if len(sub) < min_rows:
                continue
            if TARGET_TP_BPS not in sub.columns or TARGET_SL_BPS not in sub.columns:
                continue
            fold_params = dict(pbt.get(sym) or default_params)
            self.params_by_ticker[sym] = fold_params
            w = None if sample_weight is None else np.asarray(sample_weight)[mask]
            if profit_weights and fwd_ret_col and fwd_ret_col in sub.columns:
                fr = sub[fwd_ret_col].astype(float).to_numpy()
                pw = np.abs(fr)
                pw = np.where(np.isfinite(pw), pw, 0.0)
                if pw.sum() > 0:
                    pw = pw / (pw.mean() + 1e-9)
                    w = pw if w is None else w * pw
            x = sub[feat_cols]
            y_tp = sub[TARGET_TP_BPS].astype(float)
            y_sl = sub[TARGET_SL_BPS].astype(float)
            valid = y_tp.notna() & y_sl.notna()
            if int(valid.sum()) < min_rows:
                continue
            tp_model = make_entry_regressor(fold_params)
            sl_model = make_entry_regressor(fold_params)
            tp_model.fit(x.loc[valid], y_tp.loc[valid], sample_weight=None if w is None else w[valid.to_numpy()])
            sl_model.fit(x.loc[valid], y_sl.loc[valid], sample_weight=None if w is None else w[valid.to_numpy()])
            self.tp_models[sym] = tp_model
            self.sl_models[sym] = sl_model
        return self

    def predict(self, frame: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (pred_tp_bps, pred_sl_bps) aligned to frame rows."""
        n = len(frame)
        tp = np.full(n, np.nan, dtype=float)
        sl = np.full(n, np.nan, dtype=float)
        if "ticker" not in frame.columns:
            return tp, sl
        tickers = frame["ticker"].astype(str).str.upper()
        for sym, tp_model in self.tp_models.items():
            sl_model = self.sl_models.get(sym)
            if sl_model is None:
                continue
            mask = tickers.eq(sym).to_numpy()
            if not mask.any():
                continue
            x = frame.loc[mask, feat_cols]
            from simulation.tp_sl_calibration import calibrate_tp_sl_bps

            raw_tp = np.clip(tp_model.predict(x), 5.0, 500.0)
            raw_sl = np.clip(sl_model.predict(x), 5.0, 500.0)
            for i, (t_bps, s_bps) in enumerate(zip(raw_tp, raw_sl)):
                ct, cs = calibrate_tp_sl_bps(float(t_bps), float(s_bps))
                raw_tp[i] = ct
                raw_sl[i] = cs
            tp[mask] = raw_tp
            sl[mask] = raw_sl
        return tp, sl

    def training_state(self) -> dict:
        return {
            "tp_models": sorted(self.tp_models),
            "sl_models": sorted(self.sl_models),
        }


def tp_sl_regressor_enabled() -> bool:
    return bool(getattr(_cfg, "FUSION_USE_TP_SL_REGRESSOR", True))
