"""Plotting helpers for OOS backtest artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.benchmark_alpha import equal_weight_benchmark
import config as _cfg
from config import OUT_DIR

STANDARD_EQUITY_PLOT = OUT_DIR / "plots" / "equity_curve_oos.png"
STANDARD_DENSITY_PLOT = OUT_DIR / "plots" / "backtest_return_density.png"
_SYMBOL_COLORS = {
    "BTC": "#f7931a",
    "ETH": "#627eea",
    "SOL": "#14f195",
    "ETHBTC": "#a78bfa",
    "EW": "#64748b",
}


def _ew_label(symbols: list[str]) -> str:
  """Human label for equal-weight benchmark basket."""
  names = getattr(_cfg, "CRYPTO_DISPLAY_NAMES", {})
  legs = [names.get(s, s) for s in symbols]
  return f"EW ({'+'.join(legs)})" if legs else "EW basket"


def plot_equity_curve(bt: dict, out_path: Path, *, title: str) -> Path:
    """Plot normalized strategy equity and benchmark equity."""
    eq = bt.get("equity")
    bench = bt.get("benchmark")
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False

    if eq is not None and not eq.empty and "value" in eq.columns:
        eq_norm = eq["value"] / float(eq["value"].iloc[0])
        ax.plot(eq_norm.index, eq_norm.values, lw=1.8, color="tab:blue", label="Strategy")
        plotted = True

    if bench is not None and not bench.empty and "value" in bench.columns:
        bench_norm = bench["value"] / float(bench["value"].iloc[0])
        ax.plot(bench_norm.index, bench_norm.values, lw=1.2, color="tab:gray", alpha=0.8, label="Benchmark")
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No equity curve: no executed trades", ha="center", va="center", transform=ax.transAxes)

    stats = bt.get("stats") or {}
    sharpe = stats.get("sharpe_bar_annualized") or stats.get("sharpe")
    total_return = stats.get("total_return_pct")
    ax.set_title(f"{title}\nSharpe={sharpe} | Return={total_return}%")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized equity")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_standard_oos_plots(
    backtest: dict,
    trade_returns: np.ndarray,
    *,
    title_equity: str,
    title_density: str,
    equity_path: Path | None = None,
    density_path: Path | None = None,
    use_monthly_heatmap: bool = False,
) -> dict[str, str]:
    """Write canonical ``equity_curve_oos.png`` and return distribution plot."""
    eq_path = equity_path or STANDARD_EQUITY_PLOT
    den_path = density_path or STANDARD_DENSITY_PLOT
    eq = backtest.get("equity")
    bar_returns = np.array([], dtype=float)
    if eq is not None and not eq.empty and "value" in eq.columns:
        bar_returns = eq["value"].pct_change().dropna().to_numpy()
    plot_equity_curve(backtest, eq_path, title=title_equity)
    if use_monthly_heatmap:
        from reporting.desk_reports import plot_monthly_returns_heatmap

        plot_monthly_returns_heatmap(eq, den_path, title=title_density)
        return {"equity_curve": str(eq_path), "monthly_returns_heatmap": str(den_path)}
    plot_return_density(
        np.asarray(trade_returns, dtype=float),
        bar_returns,
        den_path,
        title=title_density,
    )
    return {"equity_curve": str(eq_path), "return_density": str(den_path)}


def _strategy_equity_series(equity: pd.DataFrame | pd.Series | None) -> pd.Series:
    """Extract a single rebased equity series from backtest equity frame."""
    if equity is None:
        return pd.Series(dtype=float)
    if isinstance(equity, pd.Series):
        s = equity.astype(float).dropna()
    elif equity.empty or "value" not in equity.columns:
        return pd.Series(dtype=float)
    else:
        s = equity["value"].astype(float).dropna()
    if s.empty:
        return s
    return (s / float(s.iloc[0])).rename("strategy")


def _aligned_buy_hold(prices: pd.DataFrame, symbol: str, index: pd.Index) -> pd.Series:
    if symbol not in prices.columns or len(index) == 0:
        return pd.Series(dtype=float)
    px = prices[symbol].astype(float).reindex(index).ffill().dropna()
    if px.empty:
        return pd.Series(dtype=float)
    common = px.index.intersection(index)
    if len(common) < 2:
        return pd.Series(dtype=float)
    px = px.loc[common]
    return (px / float(px.iloc[0])).rename(symbol)


def plot_equity_vs_benchmarks(
    equity: pd.DataFrame | pd.Series | None,
    prices: pd.DataFrame,
    symbols: list[str],
    out_path: Path,
    *,
    title: str,
    stats: dict | None = None,
) -> Path:
    """Strategy vs per-symbol buy-hold and equal-weight basket (rebased to 1.0)."""
    strat = _strategy_equity_series(equity)
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False

    if not strat.empty:
        ax.plot(strat.index, strat.values, lw=2.0, color="tab:blue", label="Strategy", zorder=5)
        plotted = True
        idx = strat.index
        for sym in symbols:
            bh = _aligned_buy_hold(prices, sym, idx)
            if not bh.empty:
                ax.plot(
                    bh.index, bh.values, lw=1.3, alpha=0.85,
                    color=_SYMBOL_COLORS.get(sym, "tab:gray"), label=f"{sym} buy-hold", zorder=3,
                )
                plotted = True
        ew = equal_weight_benchmark(prices, symbols)
        if not ew.empty:
            ew = ew.reindex(idx).ffill().dropna()
            if len(ew) >= 2:
                ew = ew / float(ew.iloc[0])
                ax.plot(ew.index, ew.values, lw=1.4, ls="--", color=_SYMBOL_COLORS["EW"],
                        label=_ew_label(symbols), zorder=2)
                plotted = True

    if not plotted:
        ax.text(
            0.5, 0.5, "No equity / price overlap for benchmark chart",
            ha="center", va="center", transform=ax.transAxes,
        )

    st = stats or {}
    sharpe = st.get("sharpe_bar_annualized") or st.get("sharpe")
    ret = st.get("total_return_pct")
    subtitle = f"Sharpe={sharpe} | Return={ret}%" if sharpe is not None or ret is not None else ""
    ax.set_title(f"{title}\n{subtitle}".rstrip("\n"))
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized equity (start = 1.0)")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _resolve_oos_index(
    strat: pd.Series,
    prices: pd.DataFrame,
    symbols: list[str],
    oos_index: pd.Index | None,
) -> pd.Index:
    """Pick a time axis: strategy index → explicit OOS index → benchmark prices."""
    if not strat.empty:
        return strat.index
    if oos_index is not None and len(oos_index) >= 2:
        return pd.DatetimeIndex(pd.to_datetime(oos_index)).sort_values().unique()
    for sym in symbols:
        if sym in prices.columns:
            idx = prices[sym].dropna().index
            if len(idx) >= 2:
                return idx
    return pd.Index([])


def plot_capital_over_time(
    equity: pd.DataFrame | pd.Series | None,
    prices: pd.DataFrame,
    symbols: list[str],
    out_path: Path,
    *,
    oos_index: pd.Index | None = None,
    start_capital: float = 10_000.0,
    title: str,
    stats: dict | None = None,
) -> Path:
    """Capital ($) over time: strategy vs buy-hold benchmarks.

    Unlike the rebased equity curve, this always renders the OOS window even when
    the strategy makes no trades — the strategy line stays flat at ``start_capital``
    while benchmark buy-hold capital still moves, making the no-trade outcome explicit.
    """
    strat = _strategy_equity_series(equity)
    idx = _resolve_oos_index(strat, prices, symbols, oos_index)
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False
    no_trade = strat.empty

    if not strat.empty:
        cap = strat * float(start_capital)
        ax.plot(cap.index, cap.values, lw=2.0, color="tab:blue", label="Strategy", zorder=5)
        plotted = True
    elif len(idx) >= 2:
        flat = pd.Series(float(start_capital), index=idx)
        ax.plot(
            flat.index, flat.values, lw=2.0, color="tab:blue", ls="--",
            label="Strategy (no trades — flat)", zorder=5,
        )
        plotted = True

    if len(idx) >= 2:
        for sym in symbols:
            bh = _aligned_buy_hold(prices, sym, idx)
            if not bh.empty:
                ax.plot(
                    bh.index, (bh * float(start_capital)).values, lw=1.3, alpha=0.85,
                    color=_SYMBOL_COLORS.get(sym, "tab:gray"), label=f"{sym} buy-hold", zorder=3,
                )
                plotted = True
        ew = equal_weight_benchmark(prices, symbols)
        if not ew.empty:
            ew = ew.reindex(idx).ffill().dropna()
            if len(ew) >= 2:
                ew = ew / float(ew.iloc[0]) * float(start_capital)
                ax.plot(ew.index, ew.values, lw=1.4, ls="--", color=_SYMBOL_COLORS["EW"],
                        label=_ew_label(symbols), zorder=2)
                plotted = True

    if not plotted:
        ax.text(
            0.5, 0.5, "No equity / price data for capital chart",
            ha="center", va="center", transform=ax.transAxes,
        )

    ax.axhline(float(start_capital), color="black", ls=":", lw=0.8, alpha=0.6)
    st = stats or {}
    ret = st.get("total_return_pct")
    final_cap = float(start_capital) * (1.0 + (float(ret) / 100.0)) if ret is not None else float(start_capital)
    tag = " | NO-TRADE" if no_trade else ""
    ax.set_title(
        f"{title}\nStart=${start_capital:,.0f} → End=${final_cap:,.0f} "
        f"(Return={ret if ret is not None else 0.0}%){tag}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Capital, $")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rolling_excess_vs_benchmark(
    equity: pd.DataFrame | pd.Series | None,
    prices: pd.DataFrame,
    benchmark_symbol: str,
    out_path: Path,
    *,
    title: str,
    window_bars: int | None = None,
) -> Path:
    """Rolling mean excess return (strategy − benchmark) in bps per bar."""
    strat = _strategy_equity_series(equity)
    bench = _aligned_buy_hold(prices, benchmark_symbol, strat.index)
    fig, ax = plt.subplots(figsize=(12, 4))
    plotted = False
    win = window_bars or max(int(getattr(_cfg, "BARS_PER_DAY", 78)) * 30, 20)

    if not strat.empty and not bench.empty:
        aligned = pd.concat([strat, bench], axis=1, join="inner").dropna()
        if len(aligned) >= 10:
            s_ret = aligned.iloc[:, 0].pct_change()
            b_ret = aligned.iloc[:, 1].pct_change()
            excess = (s_ret - b_ret).dropna()
            roll = excess.rolling(win, min_periods=max(win // 3, 5)).mean() * 10_000
            ax.plot(roll.index, roll.values, lw=1.5, color="#2563eb", label=f"vs {benchmark_symbol}")
            ax.axhline(0.0, color="black", ls=":", lw=0.8)
            ax.fill_between(roll.index, 0, roll.values, where=roll.values >= 0, alpha=0.2, color="#16a34a")
            ax.fill_between(roll.index, 0, roll.values, where=roll.values < 0, alpha=0.2, color="#dc2626")
            plotted = True

    if not plotted:
        ax.text(
            0.5, 0.5, "Insufficient overlap for rolling excess",
            ha="center", va="center", transform=ax.transAxes,
        )

    ax.set_title(f"{title}\n(window ≈ {win} bars)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling mean excess return, bps/bar")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_return_density(
    trade_returns: np.ndarray,
    bar_returns: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> Path:
    """Plot empirical return density with a normal fit overlay."""
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False
    source = np.asarray(trade_returns, dtype=float)
    source = source[np.isfinite(source)]
    label = "Trade net returns"

    if len(source) < 5:
        source = np.asarray(bar_returns, dtype=float)
        source = source[np.isfinite(source)]
        label = "Strategy bar returns"

    if len(source) >= 5:
        returns_bps = source * 10_000
        lo, hi = np.percentile(returns_bps, [2, 98])
        if np.isclose(lo, hi):
            lo, hi = lo - 5.0, hi + 5.0
        xs = np.linspace(lo, hi, 240)
        ax.hist(returns_bps, bins=35, density=True, alpha=0.35, color="tab:blue", label=label)
        if len(np.unique(np.round(returns_bps, 10))) > 2:
            kde = sp_stats.gaussian_kde(returns_bps)
            ax.plot(xs, kde(xs), lw=2, color="tab:blue", label="KDE")
        mu = float(np.mean(returns_bps))
        sigma = float(np.std(returns_bps, ddof=1))
        ax.plot(
            xs,
            sp_stats.norm.pdf(xs, loc=mu, scale=max(sigma, 1e-6)),
            lw=1.7,
            ls="--",
            color="tab:orange",
            label=f"Normal fit (mean={mu:.1f}, std={sigma:.1f} bps)",
        )
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "Insufficient returns for density plot", ha="center", va="center", transform=ax.transAxes)

    ax.axvline(0.0, color="black", ls=":", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Net return, bps")
    ax.set_ylabel("Probability density")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


TRADE_TP_DIST_PLOT = OUT_DIR / "plots" / "trade_tp_distribution.png"
TRADE_SL_DIST_PLOT = OUT_DIR / "plots" / "trade_sl_distribution.png"


def _plot_trade_side_distribution(
    returns_bps: np.ndarray,
    out_path: Path,
    *,
    title: str,
    color: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    data = np.asarray(returns_bps, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) >= 3:
        lo, hi = np.percentile(data, [2, 98])
        if np.isclose(lo, hi):
            lo, hi = lo - 5.0, hi + 5.0
        ax.hist(data, bins=35, density=True, alpha=0.4, color=color, label="Empirical")
        if len(np.unique(np.round(data, 10))) > 2:
            xs = np.linspace(lo, hi, 240)
            kde = sp_stats.gaussian_kde(data)
            ax.plot(xs, kde(xs), lw=2.0, color=color, label="KDE")
        med = float(np.median(data))
        ax.axvline(med, color="black", ls="--", lw=1.2, label=f"median={med:.1f} bps")
        ax.set_title(f"{title}\nn={len(data)} | median={med:.1f} bps | mean={float(np.mean(data)):.1f} bps")
    else:
        ax.text(0.5, 0.5, "Insufficient trades", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    ax.set_xlabel("Net return, bps")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    if len(data) >= 3:
        ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trade_tp_sl_distributions(
    trade_returns: np.ndarray,
    *,
    tp_path: Path | None = None,
    sl_path: Path | None = None,
) -> dict[str, str]:
    """Histograms of de-overlapped trade net returns split by TP / SL."""
    tp_out = tp_path or TRADE_TP_DIST_PLOT
    sl_out = sl_path or TRADE_SL_DIST_PLOT
    rets = np.asarray(trade_returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    bps = rets * 10_000.0
    tp = bps[bps > 0]
    sl = bps[bps <= 0]
    _plot_trade_side_distribution(tp, tp_out, title="Trade TP distribution (net bps)", color="tab:green")
    _plot_trade_side_distribution(sl, sl_out, title="Trade SL distribution (net bps)", color="tab:red")
    return {"trade_tp_distribution": str(tp_out), "trade_sl_distribution": str(sl_out)}


def write_fusion_extended_oos_plots(
    equity: pd.DataFrame | pd.Series | None,
    prices: pd.DataFrame,
    symbols: list[str],
    trade_returns: np.ndarray,
    plots_dir: Path,
    *,
    stats: dict | None = None,
    oos_index: pd.Index | None = None,
    start_capital: float = 10_000.0,
) -> dict[str, str]:
    """Benchmark + quant-desk risk plots (Sharpe, underwater, trade PnL, monthly heatmap)."""
    from reporting.desk_reports import write_quant_desk_plots

    paths: dict[str, str] = {}
    paths.update(write_quant_desk_plots(equity, trade_returns, plots_dir))
    if equity is not None and not (isinstance(equity, pd.DataFrame) and equity.empty):
        paths["equity_vs_benchmarks"] = str(plot_equity_vs_benchmarks(
            equity, prices, symbols,
            plots_dir / "fusion_equity_vs_benchmarks.png",
            title="Fusion OOS: Strategy vs BTC / ETH / EW",
            stats=stats or {},
        ))
    paths["capital_over_time"] = str(plot_capital_over_time(
        equity, prices, symbols,
        plots_dir / "fusion_capital_over_time.png",
        oos_index=oos_index,
        start_capital=start_capital,
        title="Fusion OOS: capital over time",
        stats=stats or {},
    ))
    return paths


def _slice_buy_hold_return(
    prices: pd.DataFrame,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float | None:
    if symbol not in prices.columns:
        return None
    px = prices[symbol].astype(float)
    px = px[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))].dropna()
    if len(px) < 2:
        return None
    return float(px.iloc[-1] / px.iloc[0] - 1.0) * 100.0


def _slice_ew_return(
    prices: pd.DataFrame,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float | None:
    sub = prices[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]
    ew = equal_weight_benchmark(sub, symbols)
    ew = ew.dropna()
    if len(ew) < 2:
        return None
    return float(ew.iloc[-1] / ew.iloc[0] - 1.0) * 100.0


def build_fold_benchmark_table(
    report: dict,
    prices: pd.DataFrame | None = None,
) -> list[dict]:
    """Per rolling-walk-forward fold: strategy vs BTC / ETH / EW buy-hold."""
    symbols = list(report.get("tickers") or report.get("symbols") or [])
    folds = report.get("folds") or report.get("walk_forward_folds") or []
    rows: list[dict] = []
    for fold in folds:
        if fold.get("skipped"):
            rows.append({
                "fold": fold.get("fold"),
                "test_start": str(fold.get("test_start", ""))[:10],
                "test_end": str(fold.get("test_end", ""))[:10],
                "skipped": True,
                "reason": fold.get("reason"),
            })
            continue
        bt = fold.get("backtest") or {}
        test_start = fold.get("test_start")
        test_end = fold.get("test_end")
        row = {
            "fold": int(fold.get("fold", 0)),
            "test_start": str(test_start)[:10],
            "test_end": str(test_end)[:10],
            "skipped": False,
            "disable_trading": bool(fold.get("disable_trading")),
            "signal_rows": int(fold.get("signal_rows") or 0),
            "strategy_return_pct": bt.get("total_return_pct"),
            "benchmark_btc_pct": bt.get("benchmark_return_pct"),
            "excess_vs_btc_pct": bt.get("excess_return_pct"),
            "sharpe": bt.get("sharpe_bar_annualized") or bt.get("sharpe"),
            "max_drawdown_pct": bt.get("max_drawdown_pct"),
            "flat_no_signal_fold": bool(bt.get("flat_no_signal_fold")),
        }
        if prices is not None and test_start and test_end:
            ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
            if "ETH" in symbols:
                eth_ret = _slice_buy_hold_return(prices, "ETH", ts, te)
                if eth_ret is not None:
                    row["benchmark_eth_pct"] = round(eth_ret, 3)
            if len(symbols) >= 2:
                ew = _slice_ew_return(prices, symbols, ts, te)
                if ew is not None:
                    row["benchmark_ew_pct"] = round(ew, 3)
                    strat = row.get("strategy_return_pct")
                    if strat is not None:
                        row["excess_vs_ew_pct"] = round(float(strat) - ew, 3)
        rows.append(row)
    return rows


def plot_fold_benchmark_comparison(
    table: list[dict],
    out_path: Path,
    *,
    title: str = "Rolling walk-forward: strategy vs benchmarks by fold",
) -> Path:
    """Grouped bar chart of per-fold returns (%)."""
    active = [r for r in table if not r.get("skipped")]
    if not active:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "No completed folds", ha="center", va="center", transform=ax.transAxes)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return out_path

    labels = [f"F{r['fold']}\n{r['test_start'][:7]}" for r in active]
    strat = [float(r.get("strategy_return_pct") or 0.0) for r in active]
    btc = [float(r.get("benchmark_btc_pct") or 0.0) for r in active]
    has_eth = any(r.get("benchmark_eth_pct") is not None for r in active)
    has_ew = any(r.get("benchmark_ew_pct") is not None for r in active)

    x = np.arange(len(active))
    n_series = 2 + int(has_eth) + int(has_ew)
    width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    offset = -0.4 + width / 2
    ax.bar(x + offset, strat, width, label="Strategy", color="#2563eb")
    offset += width
    ax.bar(x + offset, btc, width, label="BTC buy-hold", color="#f7931a")
    offset += width
    if has_eth:
        eth = [float(r.get("benchmark_eth_pct") or 0.0) for r in active]
        ax.bar(x + offset, eth, width, label="ETH buy-hold", color="#627eea")
        offset += width
    if has_ew:
        ew_vals = [float(r.get("benchmark_ew_pct") or 0.0) for r in active]
        ax.bar(x + offset, ew_vals, width, label="EW basket", color="#64748b")

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Return, %")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_fold_benchmark_markdown(table: list[dict], out_path: Path, *, summary: dict | None = None) -> Path:
    """Write per-fold vs benchmark table as markdown."""
    lines = [
        "# Rolling walk-forward: per-fold vs benchmark",
        "",
        "| Fold | Test period | Strategy % | BTC % | ETH % | EW % | Excess vs BTC % | Excess vs EW % | Sharpe | Signals |",
        "|------|-------------|------------|-------|-------|------|-----------------|----------------|--------|---------|",
    ]
    for r in table:
        if r.get("skipped"):
            lines.append(
                f"| {r.get('fold')} | {r.get('test_start')} → {r.get('test_end')} | — | — | — | — | — | — | — | skipped |"
            )
            continue
        lines.append(
            "| {fold} | {start} → {end} | {strat} | {btc} | {eth} | {ew} | {exb} | {exe} | {sh} | {sig} |".format(
                fold=r.get("fold"),
                start=r.get("test_start"),
                end=r.get("test_end"),
                strat=r.get("strategy_return_pct"),
                btc=r.get("benchmark_btc_pct"),
                eth=r.get("benchmark_eth_pct", "—"),
                ew=r.get("benchmark_ew_pct", "—"),
                exb=r.get("excess_vs_btc_pct"),
                exe=r.get("excess_vs_ew_pct", "—"),
                sh=r.get("sharpe"),
                sig=r.get("signal_rows"),
            )
        )
    if summary:
        lines.extend([
            "",
            "## Stitched OOS summary (all folds compounded)",
            "",
            f"- Strategy return: **{summary.get('total_return_pct')}%**",
            f"- BTC buy-hold (stitched): **{summary.get('benchmark_return_pct')}%**",
            f"- Excess vs BTC: **{summary.get('excess_return_pct')}%**",
            f"- Max drawdown: **{summary.get('max_drawdown_pct')}%**",
            f"- Sharpe (fold-compounded): **{summary.get('fold_compounded_sharpe')}**",
        ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


FOLD_ANOMALY_PLOT = OUT_DIR / "plots" / "fusion_fold_anomaly.png"
FOLD_ANOMALY_JSON = OUT_DIR / "fusion_fold_anomaly.json"


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD z-score — robust to a few extreme folds (unlike mean/std)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    if mad > 1e-12:
        return 0.6745 * (arr - med) / mad
    std = np.nanstd(arr)
    return np.zeros_like(arr) if std <= 1e-12 else (arr - med) / std


def compute_fold_anomaly_table(
    equity: pd.DataFrame | None,
    fold_signals: pd.DataFrame | None,
    fold_map: pd.DataFrame | None,
    *,
    commission_bps: float,
    slippage_bps: float = 0.0,
    z_threshold: float = 3.5,
) -> list[dict]:
    """Per-fold realized stats + robust-z anomaly flags.

    ``fold_map`` maps ``bar_time`` -> ``wf_fold`` (from the OOS cache). Anomalies:
    - return / trade-count robust |z| above threshold, or
    - structural: fold equity return negative while de-overlapped trade edge positive
      (the commission-churn signature).
    """
    if equity is None or equity.empty or fold_map is None or fold_map.empty:
        return []
    fm = fold_map.dropna(subset=["wf_fold"]).copy()
    fm["bar_time"] = pd.to_datetime(fm["bar_time"])
    fm = fm.drop_duplicates("bar_time").sort_values("bar_time")

    eq = equity.copy()
    if "date" not in eq.columns:
        eq = eq.reset_index()
        first_col = eq.columns[0]
        if first_col != "date":
            eq = eq.rename(columns={first_col: "date"})
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")
    val_col = "value" if "value" in eq.columns else eq.columns[-1]
    eq_m = pd.merge_asof(eq, fm, left_on="date", right_on="bar_time", direction="backward")

    rt_cost = (2.0 * float(commission_bps) + 2.0 * float(slippage_bps)) / 10_000.0
    trades = None
    if fold_signals is not None and not fold_signals.empty and "fwd_ret_entry" in fold_signals.columns:
        trades = fold_signals.copy()
        trades["date"] = pd.to_datetime(trades["date"])
        trades = trades.sort_values("date")
        trades["net_bps"] = trades["fwd_ret_entry"].astype(float) * 10_000.0 - rt_cost * 10_000.0
        trades = pd.merge_asof(trades, fm, left_on="date", right_on="bar_time", direction="backward")

    rows: list[dict] = []
    for fold, grp in eq_m.groupby("wf_fold"):
        grp = grp.sort_values("date")
        v = grp[val_col].astype(float)
        if len(v) < 2 or v.iloc[0] <= 0:
            continue
        ret_pct = (v.iloc[-1] / v.iloc[0] - 1.0) * 100.0
        tr = trades[trades["wf_fold"] == fold] if trades is not None else None
        n_tr = int(len(tr)) if tr is not None else 0
        avg_net = float(tr["net_bps"].mean()) if n_tr else float("nan")
        sum_net = float(tr["net_bps"].sum()) if n_tr else 0.0
        rows.append({
            "fold": int(fold),
            "start": str(grp["date"].iloc[0].date()),
            "end": str(grp["date"].iloc[-1].date()),
            "equity_return_pct": round(ret_pct, 3),
            "n_trades": n_tr,
            "avg_net_bps": round(avg_net, 2) if n_tr else None,
            "sum_net_bps": round(sum_net, 1),
        })
    if not rows:
        return rows

    rets = np.array([r["equity_return_pct"] for r in rows])
    n_trs = np.array([r["n_trades"] for r in rows], dtype=float)
    z_ret = _robust_z(rets)
    z_tr = _robust_z(n_trs)
    for r, zr, zt in zip(rows, z_ret, z_tr):
        reasons = []
        if r["n_trades"] == 0:
            reasons.append("zero OOS trades")
        if abs(zr) > z_threshold:
            reasons.append(f"return z={zr:.1f}")
        if abs(zt) > z_threshold:
            reasons.append(f"trades z={zt:.1f}")
        if r["equity_return_pct"] < 0 and (r["sum_net_bps"] or 0) > 0:
            reasons.append("neg equity vs positive trade edge (churn)")
        r["return_z"] = round(float(zr), 2)
        r["trades_z"] = round(float(zt), 2)
        r["anomaly"] = bool(reasons)
        r["anomaly_reason"] = "; ".join(reasons)
    return rows


def plot_fold_anomaly_analysis(
    equity: pd.DataFrame | None,
    fold_signals: pd.DataFrame | None,
    fold_map: pd.DataFrame | None,
    out_path: Path | None = None,
    *,
    commission_bps: float,
    slippage_bps: float = 0.0,
    title: str = "Per-fold anomaly analysis",
    json_path: Path | None = None,
) -> dict:
    """Render per-fold return / trades / edge with anomalies highlighted; write JSON."""
    import json as _json

    out_path = out_path or FOLD_ANOMALY_PLOT
    jpath = json_path or FOLD_ANOMALY_JSON
    table = compute_fold_anomaly_table(
        equity, fold_signals, fold_map,
        commission_bps=commission_bps, slippage_bps=slippage_bps,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not table:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "No per-fold data for anomaly analysis", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        jpath.write_text(_json.dumps({"folds": [], "anomalies": []}, indent=2), encoding="utf-8")
        return {"plot": str(out_path), "json": str(jpath), "n_anomalies": 0, "folds": []}

    labels = [f"F{r['fold']}\n{r['start'][:7]}" for r in table]
    x = np.arange(len(table))
    rets = [r["equity_return_pct"] for r in table]
    trs = [r["n_trades"] for r in table]
    nets = [r["avg_net_bps"] if r["avg_net_bps"] is not None else 0.0 for r in table]
    anom = [r["anomaly"] for r in table]
    base = "#2563eb"
    ret_colors = ["#dc2626" if a else base for a in anom]
    tr_colors = ["#dc2626" if a else "#64748b" for a in anom]
    net_colors = ["#16a34a" if (n or 0) >= 0 else "#dc2626" for n in nets]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].bar(x, rets, color=ret_colors)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("Equity return, %")
    axes[0].set_title(f"{title} — anomalous folds in red (robust MAD z>3.5)")
    axes[0].grid(axis="y", alpha=0.3)
    for xi, r in zip(x, table):
        if r["anomaly"]:
            axes[0].annotate(r["anomaly_reason"], (xi, rets[int(xi)]), fontsize=6,
                             ha="center", va="bottom" if rets[int(xi)] >= 0 else "top", rotation=90, color="#7f1d1d")

    axes[1].bar(x, trs, color=tr_colors)
    axes[1].set_ylabel("De-overlapped trades")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, nets, color=net_colors)
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_ylabel("Avg net per trade, bps")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    anomalies = [r for r in table if r["anomaly"]]
    jpath.write_text(_json.dumps({"folds": table, "anomalies": anomalies}, indent=2), encoding="utf-8")
    return {"plot": str(out_path), "json": str(jpath), "n_anomalies": len(anomalies), "folds": table}
