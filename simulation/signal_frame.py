"""Build bar-level trade signal frames from OOS ML rows."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from simulation.engine import _daily_closes_from_intraday, _intraday_regime_series
from config import BARS_PER_YEAR, BENCHMARK, REGIME_TICKER
from research.features.entry_ml import FWD_HORIZON_BARS
from strategy.fusion_direction import resolve_trading_thresholds

_MARKET_CTX_CACHE: dict[int, tuple] = {}


def _get_market_context(prices: pd.DataFrame) -> tuple:
    key = id(prices)
    cached = _MARKET_CTX_CACHE.get(key)
    if cached is not None:
        return cached
    daily = _daily_closes_from_intraday(prices)
    risk_on_s, vol_ratio_s = _intraday_regime_series(daily, REGIME_TICKER)
    bar_rets = prices.pct_change(fill_method=None)
    vol_wide = bar_rets.rolling(20, min_periods=5).std() * np.sqrt(BARS_PER_YEAR)
    ctx = (risk_on_s, vol_ratio_s, vol_wide)
    _MARKET_CTX_CACHE.clear()
    _MARKET_CTX_CACHE[key] = ctx
    return ctx


def _proba_to_score(proba: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(50.0 + (proba - 0.5) * gain, 0.0, 100.0)


def build_flow_signal_frame(
    oos: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    gain: float = 100.0,
    hold_threshold: float = 50.0,
    buy_threshold: float = 55.0,
    score_col: str = "ml_proba",
    regime_horizons: dict[str, int] | None = None,
    default_horizon: int = FWD_HORIZON_BARS,
    stop_loss_bps: float | None = None,
    exposure_cap: float = 1.0,
) -> pd.DataFrame:
    """Bar-level signals compatible with backtest.py."""
    if oos.empty:
        return pd.DataFrame()

    risk_on_s, vol_ratio_s, vol_wide = _get_market_context(prices)

    import config as _cfg

    sl_default = float(
        stop_loss_bps
        if stop_loss_bps is not None
        else getattr(_cfg, "FUSION_STOP_LOSS_BPS", 35.0)
    )
    bands = resolve_trading_thresholds(buy_threshold, hold_threshold)
    buy_th = bands["buy_threshold"]
    sell_th = bands["sell_threshold"]
    hold_th = bands["hold_threshold"]

    rows: list[dict] = []
    for _, row in oos.iterrows():
        t = row["ticker"]
        bt = pd.Timestamp(row["bar_time"])
        if t not in prices.columns:
            continue
        close = row.get("close")
        if pd.isna(close) and bt in prices.index:
            close = prices.loc[bt, t]
        if pd.isna(close):
            continue

        vol = 0.25
        if t in vol_wide.columns and bt in vol_wide.index:
            v = vol_wide.loc[bt, t]
            if v == v and v > 0.01:
                vol = float(v)

        risk_on = True
        if not risk_on_s.empty:
            idx = risk_on_s.index.searchsorted(bt, side="right") - 1
            if idx >= 0:
                risk_on = bool(risk_on_s.iloc[idx])

        bench_vr = 1.0
        if BENCHMARK in prices.columns and not vol_ratio_s.empty:
            idx = vol_ratio_s.index.searchsorted(bt, side="right") - 1
            if idx >= 0:
                bench_vr = float(vol_ratio_s.iloc[idx])

        if score_col == "ml_proba":
            sc = float(_proba_to_score(np.array([row["ml_proba"]]), gain)[0])
        elif score_col in row.index:
            sc = float(row[score_col])
        else:
            sc = 50.0

        if "hmm_gate" in row.index and not bool(row["hmm_gate"]):
            sc = 0.0
        elif "hmm_risk_on" in row.index:
            risk_on = bool(row["hmm_risk_on"])

        from simulation.tp_sl_calibration import calibrate_tp_sl_bps, tighten_stop_loss_bps
        from common.naming import COL_PROB_HMM_STRESS

        sl_val = sl_default
        if "pred_sl_bps" in row.index:
            raw_sl = row.get("pred_sl_bps")
            if raw_sl == raw_sl and float(raw_sl) > 0:
                sl_val = float(raw_sl)
        tp_val = None
        if "pred_tp_bps" in row.index:
            raw_tp = row.get("pred_tp_bps")
            if raw_tp == raw_tp and float(raw_tp) > 0:
                tp_val = float(raw_tp)
        if tp_val is None or tp_val <= 0:
            try:
                from strategy.target_opt import ticker_threshold_bps

                thr = ticker_threshold_bps(str(t))
                if thr is not None and float(thr) > 0:
                    tp_val = float(thr)
            except Exception:
                tp_val = None
        if tp_val is None or tp_val <= 0:
            default_tp = float(getattr(_cfg, "FUSION_DEFAULT_TAKE_PROFIT_BPS", 0.0) or 0.0)
            if default_tp > 0:
                tp_val = default_tp
        if tp_val is not None and sl_val > 0:
            tp_val, sl_val = calibrate_tp_sl_bps(tp_val, sl_val)
        elif sl_val > 0 and tp_val is None:
            tp_val, sl_val = calibrate_tp_sl_bps(sl_val * 1.0, sl_val)

        p_stress = float(row.get(COL_PROB_HMM_STRESS, 1.0 / 3.0) or (1.0 / 3.0))
        sl_val = tighten_stop_loss_bps(
            sl_val,
            stress_prob=p_stress,
            vol_ann=vol,
            vol_ratio=bench_vr,
        )

        side = int(row.get("position_side", 0) or 0)
        if sc <= 0.0:
            side = 0
        elif side == 0 and sc >= buy_th:
            side = 1
        elif side == 0 and sc <= sell_th:
            side = -1

        rec = {
            "date": bt,
            "ticker": t,
            "close": float(close),
            "vol_ann": vol,
            "score": sc,
            "risk_on": risk_on,
            "hold_threshold": hold_th,
            "buy_threshold": buy_th,
            "sell_threshold": sell_th,
            "stop_loss_bps": sl_val,
            "take_profit_bps": tp_val,
            "exposure_cap": float(max(0.0, min(1.0, exposure_cap))),
            "vol_ratio": bench_vr,
            "impulse_strength": float(row.get("impulse_strength", 0.0)),
            "position_side": side,
        }
        for c in (
            "fwd_ret",
            "fwd_ret_entry",
            "label_entry",
            "ml_proba",
            "ml_proba_raw",
            "ml_proba_short",
            "position_side",
            "ml_base_rate",
            "expected_edge_bps",
            "prob_hmm_impulse",
            "prob_hmm_mean_revert",
            "prob_hmm_stress",
            "hmm_confidence",
            "hmm_prob_entropy",
        ):
            if c in row.index:
                rec[c] = row.get(c)
        rows.append(rec)

    frame = pd.DataFrame(rows).sort_values(["date", "ticker"])
    if frame.empty:
        return frame

    try:
        from strategy.target_opt import ticker_hold_horizon_bars
    except ImportError:
        ticker_hold_horizon_bars = None  # type: ignore[assignment]

    if ticker_hold_horizon_bars is not None:
        ticker_default = frame["ticker"].map(
            lambda t: int(ticker_hold_horizon_bars(str(t), default_horizon))
        )
    else:
        ticker_default = pd.Series(int(default_horizon), index=frame.index)

    if regime_horizons:
        from simulation.adaptive_horizon import dominant_regime_series

        regimes = dominant_regime_series(frame)
        frame["exit_horizon"] = [
            int(regime_horizons.get(r, int(ticker_default.iloc[i])))
            for i, r in enumerate(regimes)
        ]
    else:
        frame["exit_horizon"] = ticker_default.astype(int)
    return frame
