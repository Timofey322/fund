"""Configuration for the rule-based (non-ML) strategy pipeline."""

from __future__ import annotations

from pathlib import Path

from config import (  # noqa: F401 — documented re-exports
    BAR_TIMEFRAME,
    BENCHMARK,
    COMMISSION_BPS_PER_SIDE,
    OUT_DIR,
    REGIME_TICKER,
    SCORE_BUY,
    SCORE_HOLD,
    TRADFI_UNIVERSE,
)

RULE_NAME = "hmm_nw_buy"
RULE_REPORT_PATH = OUT_DIR / "rule_pipeline_report.json"
RULE_EQUITY_CACHE = OUT_DIR / "cache" / "rule_bt_equity.parquet"
RULE_MONTE_CARLO_PATH = OUT_DIR / "rule_monte_carlo_report.json"
RULE_SUMMARY_PATH = OUT_DIR / "rule_summary.md"

RULE_DEFAULT_TICKERS: list[str] = list(TRADFI_UNIVERSE)

RULE_FORECAST_HORIZON_BARS = 10_000  # hold forever; buy-only book
RULE_STOP_LOSS_BPS = 0.0
RULE_USE_VOL_TARGETING = False
RULE_EQUAL_WEIGHT = True
RULE_MIN_HOLD_BARS = 3
RULE_REBALANCE_BAND = 0.08
RULE_REENTRY_COOLDOWN_BARS = 3
RULE_BUY_ONLY = True
RULE_BUY_MAINLY = True
RULE_PARTIAL_SELL_FRAC = 0.0  # never sell
RULE_ALWAYS_INVESTED = False
RULE_TACTICAL_BUDGET = 0.0
RULE_INITIAL_CAPITAL_USD = 100_000.0
RULE_REBALANCE_FREQ = "D"
RULE_SCORE_ENTER = 50.0
RULE_SCORE_EXIT = 40.0

# Nadaraya-Watson envelope (daily bars)
RULE_NW_LOOKBACK = 80
RULE_NW_BANDWIDTH = 24.0
RULE_NW_BAND_MULT = 2.0
RULE_NW_BAND_STD_WINDOW = 40
RULE_NW_TOUCH_MAX = 0.12  # price in bottom 12% of envelope = touch lower wave
RULE_HMM_GROWTH_MIN = 0.32  # min HMM P(upside bounce after touch)

RULE_EXHAUSTION_MIN = RULE_HMM_GROWTH_MIN  # legacy alias
RULE_NW_MIN_DEV = RULE_NW_TOUCH_MAX  # legacy alias
RULE_NEUTRAL_SCORE = 45.0

# Legacy impulse params (tests / diagnostics)
RULE_DUMP_LOOKBACK_BARS = 5
RULE_IMPULSE_VOL_MULT = 0.90

RULE_BAR_TIMEFRAME = "1Day"
RULE_HISTORY_DAYS = 7300  # ~20y
RULE_INTRADAY_DAYS = RULE_HISTORY_DAYS

RULE_WEB_DIR = OUT_DIR / "rule"
RULE_WEB_MANIFEST = OUT_DIR / "rule" / "manifest.json"
