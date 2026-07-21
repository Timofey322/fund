"""Causality guards: label embargo and train/calibration splits."""

from __future__ import annotations

import pandas as pd

from research.features.entry_ml import FWD_HORIZON_BARS, effective_purge_sessions


def resolve_label_horizon_bars() -> int:
    try:
        from strategy.target_opt import applied_hold_default

        return int(applied_hold_default(FWD_HORIZON_BARS))
    except Exception:
        return int(FWD_HORIZON_BARS)


def trim_label_embargo(
    df: pd.DataFrame,
    *,
    test_start: pd.Timestamp,
    horizon_bars: int,
    bar_minutes: int = 5,
) -> pd.DataFrame:
    """Drop rows whose forward label would use prices from the test window."""
    if df.empty or "bar_time" not in df.columns:
        return df
    cutoff = pd.Timestamp(test_start) - pd.Timedelta(minutes=int(horizon_bars) * int(bar_minutes))
    return df.loc[pd.to_datetime(df["bar_time"]) < cutoff].copy()


def split_fit_calibration(
    train: pd.DataFrame,
    *,
    cal_frac: float,
    horizon_bars: int,
    test_start: pd.Timestamp,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological fit/calibration split with label embargo and tail session purge."""
    import config as _cfg

    work = trim_label_embargo(train, test_start=test_start, horizon_bars=horizon_bars)
    purge = effective_purge_sessions(horizon_bars, purge=getattr(_cfg, "PURGE_PERIODS", None))
    if "session" in work.columns:
        sessions = sorted(work["session"].unique())
        if len(sessions) > purge:
            work = work[work["session"].isin(sessions[:-purge])].copy()
    if work.empty or target_col not in work.columns:
        empty = train.iloc[0:0]
        return empty, empty

    frac = min(max(float(cal_frac), 0.0), 0.4)
    cut = max(1, int(len(work) * (1.0 - frac)))
    tr_fit = work.iloc[:cut]
    va = work.iloc[cut:]
    if len(va) < 20 or va[target_col].nunique() < 2:
        return work, work.iloc[0:0]
    return tr_fit, va


def resolve_walk_forward_oos_cutoff(
    panel: pd.DataFrame,
    *,
    train_days: int | None = None,
    backtest_years: int | None = None,
    test_months: int | None = None,
) -> pd.Timestamp | None:
    """``test_start`` of the first stitched walk-forward OOS window."""
    import config as _cfg
    from strategy.pipeline import causal_model_opt_cutoff

    return causal_model_opt_cutoff(
        panel,
        train_days=int(train_days if train_days is not None else getattr(_cfg, "FUSION_WF_TRAIN_DAYS", 365)),
        backtest_years=int(
            backtest_years if backtest_years is not None else getattr(_cfg, "FUSION_WF_BACKTEST_YEARS", 4)
        ),
        test_months=int(test_months if test_months is not None else getattr(_cfg, "FUSION_WF_TEST_MONTHS", 1)),
    )


def pre_oos_panel(
    panel: pd.DataFrame,
    *,
    train_days: int,
    backtest_years: int,
    test_months: int,
) -> pd.DataFrame:
    """Rows strictly before the first walk-forward OOS month."""
    cut = resolve_walk_forward_oos_cutoff(
        panel,
        train_days=train_days,
        backtest_years=backtest_years,
        test_months=test_months,
    )
    if cut is None or "bar_time" not in panel.columns:
        return panel
    out = panel.loc[pd.to_datetime(panel["bar_time"]) < cut].copy()
    return out if not out.empty else panel


def panel_for_causal_target_opt(
    panel: pd.DataFrame,
    *,
    train_days: int | None = None,
    backtest_years: int | None = None,
    test_months: int | None = None,
    min_rows: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Panel slice for offline ``target-opt``: excludes stitched OOS evaluation dates.

    Target specification must not be tuned on bars that later appear in the
    walk-forward backtest window.
    """
    import config as _cfg

    if panel.empty or "bar_time" not in panel.columns:
        raise ValueError("target-opt: panel is empty or missing bar_time")

    td = int(train_days if train_days is not None else getattr(_cfg, "FUSION_WF_TRAIN_DAYS", 365))
    by = int(backtest_years if backtest_years is not None else getattr(_cfg, "FUSION_WF_BACKTEST_YEARS", 4))
    tm = int(test_months if test_months is not None else getattr(_cfg, "FUSION_WF_TEST_MONTHS", 1))
    floor = int(min_rows if min_rows is not None else getattr(_cfg, "FUSION_WF_MIN_TRAIN_ROWS", 20_000))

    cutoff = resolve_walk_forward_oos_cutoff(panel, train_days=td, backtest_years=by, test_months=tm)
    if cutoff is None:
        raise ValueError("target-opt: cannot resolve walk-forward OOS cutoff from panel dates")

    work = panel.copy()
    work["bar_time"] = pd.to_datetime(work["bar_time"])
    causal = work.loc[work["bar_time"] < cutoff].copy()
    meta = {
        "causal": True,
        "oos_cutoff": str(cutoff.date()),
        "rows_full": int(len(work)),
        "rows_causal": int(len(causal)),
        "train_days": td,
        "backtest_years": by,
        "test_months": tm,
    }
    if len(causal) < floor:
        raise ValueError(
            f"target-opt: only {len(causal):,} rows before OOS {cutoff.date()} "
            f"(need >= {floor:,}); widen history or lower FUSION_WF_MIN_TRAIN_ROWS"
        )
    return causal, meta
