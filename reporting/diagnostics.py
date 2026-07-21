"""HMM / ret_z diagnostic plots for the crypto fusion pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BAR_TIMEFRAME,
    CRYPTO_BARS_PER_HOUR,
    HMM_BAR_PLOT_ROWS,
    HMM_RET_Z_LOOKBACK_BARS,
    HMM_STATUS_MAX_ROWS,
    OUT_DIR,
)
from research.features.hmm_observations import compute_bar_features, ret_z_step_table
from data_platform.binance import load_crypto_ohlcv
from common.naming import (
    COL_HMM_DOMINANT,
    COL_PROB_HMM_IMPULSE,
    COL_PROB_HMM_MEAN_REVERT,
    COL_PROB_HMM_STRESS,
    HMM_IMPULSE,
    HMM_MEAN_REVERT,
    HMM_STRESS,
)
from research.regime.hmm import build_hmm_regime_frame

PLOTS_DIR = OUT_DIR / "plots"


def _plots_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def _load_close(symbol: str) -> pd.Series:
    from data_platform.bars import load_ohlcv
    from data_platform.universe import is_crypto_symbol

    sym = symbol.upper()
    ohlcv = load_crypto_ohlcv(sym, BAR_TIMEFRAME) if is_crypto_symbol(sym) else load_ohlcv(sym, BAR_TIMEFRAME)
    if ohlcv.empty:
        raise FileNotFoundError(f"No cached OHLCV for {symbol}")
    close = ohlcv["close"].astype(float)
    if HMM_STATUS_MAX_ROWS and len(close) > HMM_STATUS_MAX_ROWS:
        return close.iloc[-int(HMM_STATUS_MAX_ROWS):]
    return close


def plot_ret_z_explainer(symbol: str, t_idx: int | None = None) -> Path:
    """Step table + formula for ret_z at the latest bar (or t_idx)."""
    close = _load_close(symbol)
    if t_idx is None:
        t_idx = len(close) - 1
    step = ret_z_step_table(close, t_idx=t_idx, lookback=HMM_RET_Z_LOOKBACK_BARS)
    mu = step.attrs.get("mu")
    sigma = step.attrs.get("sigma")
    ret_z = step.attrs.get("ret_z")
    hours = step.attrs.get("lookback_hours", HMM_RET_Z_LOOKBACK_BARS / CRYPTO_BARS_PER_HOUR)

    fig, (ax_tbl, ax_txt) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]}
    )
    ax_tbl.axis("off")
    tbl = step[["bar", "close", "log_ret"]].copy()
    tbl["log_ret"] = tbl["log_ret"].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    tbl["close"] = tbl["close"].map(lambda x: f"{x:.4f}")
    highlight = step["is_current"].values
    table = ax_tbl.table(
        cellText=tbl.values,
        colLabels=["bar", "close", "log_ret"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.2)
    for i, is_cur in enumerate(highlight):
        if is_cur:
            for j in range(3):
                table[(i + 1, j)].set_facecolor("#fff3cd")

    ax_txt.axis("off")
    ax_txt.text(
        0.05,
        0.85,
        f"{symbol.upper()} ret_z explainer\n"
        f"lookback: {HMM_RET_Z_LOOKBACK_BARS} bars ({hours:.0f}h)\n\n"
        f"μ = mean(log_ret, last hour)\n"
        f"σ = std(log_ret, last hour)\n"
        f"ret_z = (log_ret_t − μ) / σ\n\n"
        f"μ = {mu:.6f}\n"
        f"σ = {sigma:.6f}\n"
        f"ret_z = {ret_z:.3f}",
        fontsize=11,
        va="top",
        family="monospace",
    )
    fig.suptitle(f"{symbol.upper()} ret_z (1-hour baseline)", fontsize=13)
    fig.tight_layout()
    out = _plots_dir() / f"ret_z_explainer_{symbol.upper()}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_hmm_regime_scatter(symbol: str) -> Path:
    """Scatter of HMM observations colored by dominant regime."""
    close = _load_close(symbol)
    feats = compute_bar_features(close.iloc[-HMM_BAR_PLOT_ROWS:])
    if feats.empty:
        raise ValueError(f"Insufficient features for {symbol}")

    prices = pd.DataFrame({symbol.upper(): close})
    regime = build_hmm_regime_frame(prices).reindex(feats.index)
    with pd.option_context("future.no_silent_downcasting", True):
        regime = regime.ffill()
    if hasattr(regime, "infer_objects"):
        regime = regime.infer_objects(copy=False)
    dom = regime.get(COL_HMM_DOMINANT, pd.Series(HMM_MEAN_REVERT, index=feats.index))

    colors = {
        HMM_IMPULSE: "#2ecc71",
        HMM_MEAN_REVERT: "#3498db",
        HMM_STRESS: "#e74c3c",
    }
    c = [colors.get(str(d), "#95a5a6") for d in dom]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        feats["hmm_ret"],
        feats["hmm_vol"],
        c=c,
        s=8,
        alpha=0.55,
        edgecolors="none",
    )
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("hmm_ret (ret_z)")
    ax.set_ylabel("hmm_vol (vol_z)")
    ax.set_title(f"{symbol.upper()} HMM observation scatter")
    for label, color in colors.items():
        ax.scatter([], [], c=color, label=label.replace("HMM_", ""))
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = _plots_dir() / f"hmm_scatter_{symbol.upper()}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_regime_timeline(symbol: str) -> Path:
    """Stacked regime probabilities over recent bars."""
    close = _load_close(symbol)
    tail = close.iloc[-HMM_BAR_PLOT_ROWS:]
    regime = build_hmm_regime_frame(pd.DataFrame({symbol.upper(): tail}))
    if regime.empty:
        raise ValueError(f"No regime frame for {symbol}")

    cols = [COL_PROB_HMM_IMPULSE, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_STRESS]
    with pd.option_context("future.no_silent_downcasting", True):
        probs = regime[cols].reindex(tail.index).ffill()
    if hasattr(probs, "infer_objects"):
        probs = probs.infer_objects(copy=False)
    probs = probs.fillna(1.0 / 3.0)

    fig, (ax_p, ax_c) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_p.stackplot(
        probs.index,
        probs[COL_PROB_HMM_IMPULSE],
        probs[COL_PROB_HMM_MEAN_REVERT],
        probs[COL_PROB_HMM_STRESS],
        labels=["impulse", "mean_revert", "stress"],
        colors=["#2ecc71", "#3498db", "#e74c3c"],
        alpha=0.75,
    )
    ax_p.set_ylim(0, 1)
    ax_p.set_ylabel("P(regime)")
    ax_p.legend(loc="upper left", ncol=3)
    ax_p.set_title(f"{symbol.upper()} HMM regime timeline")

    ax_c.plot(tail.index, tail.values, color="#2c3e50", lw=0.8)
    ax_c.set_ylabel("close")
    fig.autofmt_xdate()
    fig.tight_layout()
    out = _plots_dir() / f"hmm_timeline_{symbol.upper()}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_system_overview(symbol: str) -> list[Path]:
    """Generate ret_z explainer, scatter, and timeline plots."""
    return [
        plot_ret_z_explainer(symbol),
        plot_hmm_regime_scatter(symbol),
        plot_regime_timeline(symbol),
    ]
