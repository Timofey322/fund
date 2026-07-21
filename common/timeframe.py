"""
Intraday session handling for bar data.

Alpaca bar timestamps are naive UTC. US regular trading hours (RTH) are
09:30–16:00 America/New_York. Overnight gaps (close→next open) must not be
treated as normal bar returns, or volatility/HMM features get contaminated.

Pure scaling math (bars/year, day→bar) lives in config; this module adds the
pandas/session utilities that need a DatetimeIndex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (  # noqa: F401  (re-export scaling constants)
    BARS_PER_DAY,
    BARS_PER_YEAR,
    BAR_MINUTES,
    BAR_TIMEFRAME,
    IS_INTRADAY,
    days_to_bars,
)

NY_TZ = "America/New_York"
RTH_OPEN = (9, 30)
RTH_CLOSE = (16, 0)


def _to_ny(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Interpret naive index as UTC and convert to New York wall-clock."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(NY_TZ)


def filter_session(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """RTH for US ETFs; 24/7 passthrough for crypto."""
    if df.empty:
        return df
    if symbol:
        try:
            from data_platform.universe import is_crypto_symbol
            if is_crypto_symbol(symbol):
                return df
        except ImportError:
            pass
    return filter_rth(df)


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-hours bars (drops pre/post-market and weekends)."""
    if df.empty or not IS_INTRADAY:
        return df
    ny = _to_ny(df.index)
    minutes = ny.hour * 60 + ny.minute
    open_m = RTH_OPEN[0] * 60 + RTH_OPEN[1]
    close_m = RTH_CLOSE[0] * 60 + RTH_CLOSE[1]
    # bar timestamped at its open: [09:30, 16:00); weekday < 5
    mask = (minutes >= open_m) & (minutes < close_m) & (ny.weekday < 5)
    return df.loc[mask]


def session_id(index: pd.DatetimeIndex, symbol: str | None = None) -> pd.Series:
    """Trading-day label — NY date for ETFs, UTC date for crypto (24/7)."""
    if symbol:
        try:
            from data_platform.universe import is_crypto_symbol
            if is_crypto_symbol(symbol):
                idx = pd.DatetimeIndex(index)
                return pd.Series(idx.normalize(), index=index, name="session")
        except ImportError:
            pass
    ny = _to_ny(index)
    return pd.Series(ny.normalize().tz_localize(None), index=index, name="session")


def is_session_open_bar(index: pd.DatetimeIndex) -> pd.Series:
    """True for the first bar of each session (the overnight-gap bar)."""
    sess = session_id(index)
    return pd.Series(sess.values != sess.shift(1).values, index=index, name="session_open")


def overnight_safe_log_returns(close: pd.Series) -> pd.Series:
    """
    Log returns with overnight gaps masked out (intraday only).

    The first bar of each session would otherwise encode the close→open jump;
    we set it to NaN so rolling vol/HMM features reflect intraday dynamics.
    """
    lr = np.log(close / close.shift(1))
    if not IS_INTRADAY or close.empty:
        return lr
    open_bar = is_session_open_bar(close.index)
    lr = lr.copy()
    lr[open_bar.values] = np.nan
    return lr


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate OHLCV bars to a coarser frequency (e.g. intraday → '1D')."""
    if df.empty:
        return df
    agg = {}
    if "open" in df:
        agg["open"] = "first"
    if "high" in df:
        agg["high"] = "max"
    if "low" in df:
        agg["low"] = "min"
    if "close" in df:
        agg["close"] = "last"
    if "volume" in df:
        agg["volume"] = "sum"
    return df.resample(freq).agg(agg).dropna(how="all")


def session_close_series(close: pd.Series) -> pd.Series:
    """Last price of each trading session, indexed by the NY session date."""
    if not IS_INTRADAY:
        return close
    sess = session_id(close.index)
    grouped = close.groupby(sess.values).last()
    grouped.index = pd.to_datetime(grouped.index)
    return grouped.sort_index()


def session_last_close(close: pd.Series) -> pd.Series:
    """
    Daily close series indexed by the *actual last-bar timestamp* of each session.

    Crypto (24/7): last bar per UTC calendar day.
    Stocks: last bar per NY trading session.
    """
    if not IS_INTRADAY or close.empty:
        return close
    sym = close.name if isinstance(close.name, str) else None
    try:
        from data_platform.universe import is_crypto_symbol
        if sym and is_crypto_symbol(sym):
            sess = session_id(close.index, symbol=sym)
            frame = pd.DataFrame({"close": close.to_numpy(), "session": sess.to_numpy()}, index=close.index)
            last = frame.groupby("session", sort=True).tail(1)
            return pd.Series(last["close"].to_numpy(), index=last.index, name=close.name).sort_index()
    except ImportError:
        pass
    sess = session_id(close.index)
    frame = pd.DataFrame({"close": close.to_numpy(), "session": sess.to_numpy()}, index=close.index)
    last = frame.groupby("session", sort=True).tail(1)
    return pd.Series(last["close"].to_numpy(), index=last.index, name=close.name).sort_index()
