"""Advanced time-series features: long memory, stochastic vol, spectral cycles."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ADVANCED_TS_COLS: tuple[str, ...] = (
    "hurst_rs",
    "garch_cond_vol",
    "garch_vol_ratio",
    "spec_dominant_period",
    "spec_entropy",
    "spec_low_high_ratio",
)

CROSS_SECTIONAL_COLS: tuple[str, ...] = (
    "ret_12_cs_rank",
    "vol_realized_12_cs_rank",
    "nw_env_pos_cs_rank",
)

# Subset promoted to production ML (feature evaluation).
# spec_dominant_period kept: spectral family leads WF importance.
ML_ADVANCED_TS_COLS: tuple[str, ...] = (
    "hurst_rs",
    "garch_cond_vol",
    "garch_vol_ratio",
    "spec_entropy",
    "spec_low_high_ratio",
    "spec_dominant_period",
)

DEFAULT_HURST_WINDOW = 192
DEFAULT_GARCH_WINDOW = 256
DEFAULT_GARCH_REFIT = 48
DEFAULT_SPEC_WINDOW = 96


def _rolling_hurst_rs(x: np.ndarray, window: int, *, step: int = 1) -> np.ndarray:
    """Rolling Hurst exponent via rescaled range (R/S) on log returns."""
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if window < 32 or n < window:
        return out
    step = max(1, int(step))
    last_val = np.nan
    for end in range(window, n + 1, step):
        seg = x[end - window : end]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 32 or np.std(seg) < 1e-12:
            continue
        y = np.cumsum(seg - np.mean(seg))
        r = float(np.max(y) - np.min(y))
        s = float(np.std(seg, ddof=1))
        if s < 1e-12 or r <= 0:
            continue
        rs = r / s
        last_val = math.log(rs) / math.log(float(len(seg)))
        out[end - 1 : min(n, end - 1 + step)] = last_val
    if step > 1 and np.isfinite(last_val):
        mask = np.isnan(out) & (np.arange(n) >= window - 1)
        idx = np.where(mask)[0]
        if len(idx):
            out[idx] = last_val
    return out


def _garch_cond_vol_series(returns: np.ndarray, *, window: int, refit: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling GARCH(1,1) conditional vol; refit every `refit` bars."""
    n = len(returns)
    cond = np.full(n, np.nan, dtype=float)
    ratio = np.full(n, np.nan, dtype=float)
    try:
        from arch import arch_model
    except ImportError:
        return cond, ratio

    r = np.asarray(returns, dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    last_fit = -10**9
    last_sigma: float | None = None
    for i in range(window, n):
        if i - last_fit < refit and last_sigma is not None:
            cond[i] = last_sigma
            rv = float(np.std(r[i - 12 : i], ddof=1)) if i >= 12 else np.nan
            ratio[i] = last_sigma / rv if rv and rv > 1e-12 else np.nan
            continue
        sample = r[i - window : i] * 100.0
        if np.std(sample) < 1e-10:
            continue
        try:
            am = arch_model(sample, vol="Garch", p=1, q=1, mean="Zero")
            res = am.fit(disp="off", show_warning=False)
            sigma = float(res.conditional_volatility.iloc[-1]) / 100.0
            last_sigma = sigma
            last_fit = i
            cond[i] = sigma
            rv = float(np.std(r[i - 12 : i], ddof=1))
            ratio[i] = sigma / rv if rv > 1e-12 else np.nan
        except Exception:
            continue
    return cond, ratio


def _rolling_spectral_features(
    returns: np.ndarray, window: int, *, step: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dominant FFT period, normalized spectral entropy, low/high band power ratio."""
    n = len(returns)
    period = np.full(n, np.nan, dtype=float)
    entropy = np.full(n, np.nan, dtype=float)
    lh_ratio = np.full(n, np.nan, dtype=float)
    if window < 16 or n < window:
        return period, entropy, lh_ratio

    step = max(1, int(step))
    last_p = last_e = last_l = np.nan
    for end in range(window, n + 1, step):
        seg = returns[end - window : end]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 16 or np.std(seg) < 1e-12:
            continue
        seg = seg - np.mean(seg)
        spec = np.abs(np.fft.rfft(seg)) ** 2
        if len(spec) < 3:
            continue
        power = spec[1:]
        total = float(power.sum())
        if total < 1e-18:
            continue
        p_norm = power / total
        p_clip = np.clip(p_norm, 1e-12, 1.0)
        last_e = float(-(p_clip * np.log(p_clip)).sum() / math.log(len(p_clip)))
        peak = int(np.argmax(power)) + 1
        last_p = float(len(seg) / max(peak, 1))
        mid = max(1, len(power) // 3)
        low = float(power[:mid].sum())
        high = float(power[mid:].sum())
        last_l = low / high if high > 1e-18 else np.nan
        sl = slice(end - 1, min(n, end - 1 + step))
        period[sl] = last_p
        entropy[sl] = last_e
        lh_ratio[sl] = last_l
    return period, entropy, lh_ratio


def attach_advanced_ts_features(
    df: pd.DataFrame,
    *,
    hurst_window: int = DEFAULT_HURST_WINDOW,
    garch_window: int = DEFAULT_GARCH_WINDOW,
    garch_refit: int = DEFAULT_GARCH_REFIT,
    spec_window: int = DEFAULT_SPEC_WINDOW,
    include_garch: bool = True,
    compute_step: int = 12,
) -> pd.DataFrame:
    """Attach Hurst, GARCH vol, and spectral features from close returns."""
    if df.empty or "close" not in df.columns:
        return df
    out = df.copy()
    close = out["close"].astype(float)
    ret = close.pct_change(1, fill_method=None).to_numpy(dtype=float)

    out["hurst_rs"] = _rolling_hurst_rs(ret, hurst_window, step=compute_step)
    if include_garch:
        g_vol, g_ratio = _garch_cond_vol_series(ret, window=garch_window, refit=max(garch_refit, compute_step))
        out["garch_cond_vol"] = g_vol
        out["garch_vol_ratio"] = g_ratio
    else:
        out["garch_cond_vol"] = np.nan
        out["garch_vol_ratio"] = np.nan

    p, ent, lh = _rolling_spectral_features(ret, spec_window, step=compute_step)
    out["spec_dominant_period"] = p
    out["spec_entropy"] = ent
    out["spec_low_high_ratio"] = lh
    return out


def attach_advanced_ts_by_ticker(
    panel: pd.DataFrame,
    *,
    max_rows_per_ticker: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Compute advanced TS features per ticker (causal, no cross-leakage)."""
    if panel.empty or "ticker" not in panel.columns:
        return attach_advanced_ts_features(panel, **kwargs)
    parts: list[pd.DataFrame] = []
    for ticker, grp in panel.groupby("ticker", sort=False):
        g = grp.sort_values("bar_time").copy()
        if max_rows_per_ticker and len(g) > max_rows_per_ticker:
            g = g.iloc[-int(max_rows_per_ticker) :].copy()
        parts.append(attach_advanced_ts_features(g, **kwargs))
    if not parts:
        return panel.copy()
    return pd.concat(parts).sort_index()


def attach_ml_advanced_ts_features(
    df: pd.DataFrame,
    *,
    hurst_window: int = DEFAULT_HURST_WINDOW,
    garch_window: int = DEFAULT_GARCH_WINDOW,
    garch_refit: int = DEFAULT_GARCH_REFIT,
    spec_window: int = DEFAULT_SPEC_WINDOW,
    compute_step: int = 12,
    include_garch: bool = True,
) -> pd.DataFrame:
    """Production ML subset: Hurst, GARCH vol, spectral features."""
    if df.empty or "close" not in df.columns:
        return df
    if "ticker" in df.columns:
        parts: list[pd.DataFrame] = []
        for _, grp in df.groupby("ticker", sort=False):
            sort_col = "bar_time" if "bar_time" in grp.columns else grp.index
            parts.append(
                _attach_ml_advanced_ts_block(
                    grp.sort_values(sort_col),
                    hurst_window=hurst_window,
                    garch_window=garch_window,
                    garch_refit=garch_refit,
                    spec_window=spec_window,
                    compute_step=compute_step,
                    include_garch=include_garch,
                )
            )
        if not parts:
            return df
        return pd.concat(parts).sort_index()
    return _attach_ml_advanced_ts_block(
        df,
        hurst_window=hurst_window,
        garch_window=garch_window,
        garch_refit=garch_refit,
        spec_window=spec_window,
        compute_step=compute_step,
        include_garch=include_garch,
    )


def _attach_ml_advanced_ts_block(
    df: pd.DataFrame,
    *,
    hurst_window: int,
    garch_window: int,
    garch_refit: int,
    spec_window: int,
    compute_step: int,
    include_garch: bool,
) -> pd.DataFrame:
    out = df.copy()
    ret = out["close"].astype(float).pct_change(1, fill_method=None).to_numpy(dtype=float)
    out["hurst_rs"] = _rolling_hurst_rs(ret, hurst_window, step=compute_step)
    if include_garch:
        g_vol, g_ratio = _garch_cond_vol_series(ret, window=garch_window, refit=max(garch_refit, compute_step))
        if not np.isfinite(g_vol).any():
            rv = pd.Series(ret).rolling(12, min_periods=4).std().to_numpy(dtype=float)
            med = pd.Series(rv).rolling(48, min_periods=12).median().replace(0, np.nan).to_numpy(dtype=float)
            g_vol = rv
            g_ratio = np.where(np.isfinite(rv) & np.isfinite(med) & (med > 1e-12), rv / med, 1.0)
        out["garch_cond_vol"] = g_vol
        out["garch_vol_ratio"] = g_ratio
    else:
        out["garch_cond_vol"] = np.nan
        out["garch_vol_ratio"] = np.nan
    period, ent, lh = _rolling_spectral_features(ret, spec_window, step=compute_step)
    out["spec_dominant_period"] = period
    out["spec_entropy"] = ent
    out["spec_low_high_ratio"] = lh
    return out
