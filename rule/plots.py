"""Visual artifacts for the HMM exhaustion rule strategy."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUT_DIR
from reporting.desk_reports import plot_monthly_returns_heatmap, plot_rolling_sharpe, plot_underwater_drawdown
from reporting.plots import plot_equity_curve, plot_return_density
from simulation.trade_survival import block_bootstrap_paths, survival_simulation

RULE_PLOTS_DIR = OUT_DIR / "plots" / "rule"


def _web_path(path: Path) -> str:
    """Convert filesystem plot path to web-servable ``/output/...`` URL."""
    try:
        rel = path.resolve().relative_to(OUT_DIR.resolve())
        return f"/output/{rel.as_posix()}"
    except ValueError:
        return str(path)


def _prepare_equity_frame(equity: pd.DataFrame) -> pd.DataFrame:
    """Ensure desk plot helpers receive a datetime-indexed equity frame."""
    eq = equity.copy()
    if isinstance(eq.index, pd.DatetimeIndex):
        return eq
    if "bar_time" in eq.columns:
        return eq.set_index(pd.to_datetime(eq["bar_time"]))
    if "date" in eq.columns:
        return eq.set_index(pd.to_datetime(eq["date"]))
    return eq


def _equity_series(equity: pd.DataFrame | None) -> pd.Series:
    if equity is None or equity.empty or "value" not in equity.columns:
        return pd.Series(dtype=float)
    if isinstance(equity.index, pd.DatetimeIndex):
        return equity["value"].astype(float).sort_index()
    if "bar_time" in equity.columns:
        idx = pd.to_datetime(equity["bar_time"])
    elif "date" in equity.columns:
        idx = pd.to_datetime(equity["date"])
    else:
        return pd.Series(dtype=float)
    return pd.Series(equity["value"].astype(float).values, index=idx).sort_index()


def plot_per_ticker_returns(
    per_ticker_bt: dict,
    out_path: Path,
    *,
    title: str = "Per-instrument: strategy vs buy & hold",
) -> Path:
    """Grouped bar chart of strategy and benchmark returns (%)."""
    rows = [
        (sym, stats)
        for sym, stats in sorted(per_ticker_bt.items())
        if isinstance(stats, dict) and not stats.get("skipped")
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    if not rows:
        ax.text(0.5, 0.5, "No per-ticker backtest data", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = [r[0] for r in rows]
        strat = [float(r[1].get("total_return_pct") or 0.0) for r in rows]
        bench = [float(r[1].get("benchmark_return_pct") or 0.0) for r in rows]
        x = np.arange(len(labels))
        width = 0.36
        ax.bar(x - width / 2, strat, width, label="Strategy", color="#c9a227")
        ax.bar(x + width / 2, bench, width, label="Buy & hold", color="#64748b", alpha=0.85)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Return, %")
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_ticker_excess(per_ticker_bt: dict, out_path: Path, *, title: str = "Excess return vs buy & hold") -> Path:
    rows = [
        (sym, float(stats.get("excess_return_pct") or 0.0))
        for sym, stats in sorted(per_ticker_bt.items())
        if isinstance(stats, dict) and not stats.get("skipped")
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    if not rows:
        ax.text(0.5, 0.5, "No excess return data", ha="center", va="center", transform=ax.transAxes)
    else:
        labels, vals = zip(*rows)
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in vals]
        ax.bar(labels, vals, color=colors)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel("Excess return, %")
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_ticker_sharpe(per_ticker_bt: dict, out_path: Path, *, title: str = "Per-instrument Sharpe") -> Path:
    rows = [
        (sym, float(stats.get("sharpe") or 0.0))
        for sym, stats in sorted(per_ticker_bt.items())
        if isinstance(stats, dict) and not stats.get("skipped") and stats.get("sharpe") is not None
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    if not rows:
        ax.text(0.5, 0.5, "No Sharpe data", ha="center", va="center", transform=ax.transAxes)
    else:
        labels, vals = zip(*rows)
        colors = ["#2563eb" if v >= 0 else "#dc2626" for v in vals]
        ax.bar(labels, vals, color=colors)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axhline(1.0, color="#16a34a", ls="--", lw=0.7, alpha=0.6)
        ax.set_ylabel("Sharpe (annualized)")
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_signal_distribution(
    per_ticker_sig: list[dict],
    out_path: Path,
    *,
    title: str = "Signal mix by instrument (%)",
) -> Path:
    """Stacked bars: dump-buy vs rally-sell vs other bars."""
    rows = [r for r in per_ticker_sig if r.get("ticker")]
    fig, ax = plt.subplots(figsize=(14, 6))
    if not rows:
        ax.text(0.5, 0.5, "No signal stats", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = [str(r["ticker"]) for r in rows]
        dump = [float(r.get("pct_dump_buy") or 0.0) for r in rows]
        rally = [float(r.get("pct_rally_sell") or 0.0) for r in rows]
        other = [max(0.0, 100.0 - d - s) for d, s in zip(dump, rally)]
        x = np.arange(len(labels))
        ax.bar(x, dump, label="Dump buy", color="#c9a227")
        ax.bar(x, rally, bottom=dump, label="Rally sell", color="#b83d5e")
        ax.bar(x, other, bottom=[d + s for d, s in zip(dump, rally)], label="Neutral / other", color="#334155", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("% of bars")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_monte_carlo_terminal(
    equity: pd.DataFrame | None,
    out_path: Path,
    *,
    title: str = "Monte Carlo terminal wealth distribution",
    n_paths: int = 2000,
    block_bars: int = 12,
) -> Path:
    """Histogram of bootstrap terminal wealth (start = 1.0)."""
    eq = _equity_series(equity)
    fig, ax = plt.subplots(figsize=(11, 5))
    if eq.empty or len(eq) < 20:
        ax.text(0.5, 0.5, "Insufficient equity for Monte Carlo", ha="center", va="center", transform=ax.transAxes)
    else:
        rets = eq.pct_change().dropna().values
        rets = rets[np.isfinite(rets)]
        horizon = len(rets)
        paths = block_bootstrap_paths(rets, n_paths, horizon, block_bars)
        terminal = paths[:, -1]
        surv = survival_simulation(eq, n_paths=min(n_paths, 500), block_bars=block_bars)
        ax.hist(terminal, bins=40, density=True, alpha=0.55, color="#8b2942", edgecolor="white", lw=0.4)
        ax.axvline(1.0, color="black", ls=":", lw=1.2, label="Break-even")
        p50 = surv.get("terminal_wealth_p50")
        if p50 is not None:
            ax.axvline(float(p50), color="#c9a227", ls="--", lw=1.4, label=f"p50 = {p50}")
        ax.set_xlabel("Terminal wealth (× start capital)")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_title(
            f"{title}\n"
            f"Survival={surv.get('survival_rate')} | P(loss)={surv.get('prob_terminal_loss')}"
        )
    if eq.empty or len(eq) < 20:
        ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_rule_desk_plots(
    report: dict,
    equity: pd.DataFrame | None,
    *,
    plots_dir: Path | None = None,
) -> dict[str, str]:
    """Generate rule-strategy chart pack; values are web paths ``/output/...``."""
    out_dir = plots_dir or RULE_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = report.get("backtest") or {}
    per_bt = report.get("per_ticker_backtest") or {}
    per_sig = report.get("per_ticker_signals") or []

    paths: dict[str, str] = {}
    eq_bt = {"equity": equity, "stats": stats}
    if equity is not None and not equity.empty:
        eq_dt = _prepare_equity_frame(equity)
        eq_path = out_dir / "rule_equity_curve.png"
        plot_equity_curve(eq_bt, eq_path, title="HMM exhaustion: portfolio equity")
        paths["rule_equity_curve"] = _web_path(eq_path)

        den_path = out_dir / "rule_return_density.png"
        bar_rets = equity["value"].astype(float).pct_change().dropna().to_numpy()
        plot_return_density(np.array([]), bar_rets, den_path, title="HMM exhaustion: return distribution")
        paths["rule_return_density"] = _web_path(den_path)

        uw_path = out_dir / "rule_underwater_drawdown.png"
        plot_underwater_drawdown(eq_dt, uw_path, title="HMM exhaustion: underwater drawdown")
        paths["rule_underwater_drawdown"] = _web_path(uw_path)

        rs_path = out_dir / "rule_rolling_sharpe.png"
        plot_rolling_sharpe(eq_dt, rs_path, title="HMM exhaustion: rolling Sharpe")
        paths["rule_rolling_sharpe"] = _web_path(rs_path)

        hm_path = out_dir / "rule_monthly_returns.png"
        plot_monthly_returns_heatmap(eq_dt, hm_path, title="HMM exhaustion: monthly returns")
        paths["rule_monthly_returns"] = _web_path(hm_path)

        mc_path = out_dir / "rule_monte_carlo_terminal.png"
        plot_monte_carlo_terminal(equity, mc_path)
        paths["rule_monte_carlo_terminal"] = _web_path(mc_path)

    if per_bt:
        ret_path = out_dir / "rule_per_ticker_returns.png"
        plot_per_ticker_returns(per_bt, ret_path)
        paths["rule_per_ticker_returns"] = _web_path(ret_path)

        exc_path = out_dir / "rule_per_ticker_excess.png"
        plot_per_ticker_excess(per_bt, exc_path)
        paths["rule_per_ticker_excess"] = _web_path(exc_path)

        sh_path = out_dir / "rule_per_ticker_sharpe.png"
        plot_per_ticker_sharpe(per_bt, sh_path)
        paths["rule_per_ticker_sharpe"] = _web_path(sh_path)

    if per_sig:
        sig_path = out_dir / "rule_signal_distribution.png"
        plot_signal_distribution(per_sig, sig_path)
        paths["rule_signal_distribution"] = _web_path(sig_path)

    return paths
