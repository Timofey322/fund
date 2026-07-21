"""Universe and daily/intraday strategy parameters.

Daily mode (BAR_TIMEFRAME="1Day"): monthly rebalance, calendar windows.
Intraday mode (5Min…1Hour): all bar-count constants are rescaled from the
timeframe at the bottom of this file (see "Intraday derivation"). HMM regimes
are computed on a daily resample of intraday bars; signal exits act per bar.
"""

import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT_DIR.parent
OUT_DIR = WORKSPACE_ROOT / "output"
VALIDATION_DIR = OUT_DIR / "validation"
DATA_DIR = WORKSPACE_ROOT / "data"

# --- Rebalance / period (daily defaults; overridden for intraday below) ---
REBALANCE_FREQ = "ME"
HMM_FREQ = "ME"          # frequency at which HMM regime features are computed
PERIODS_PER_YEAR = 12
PERIOD_NAME = "month"
OOS_TRAIN_END = "2020-12-31"

TRAIN_PERIODS = 36
HMM_TRAIN_PERIODS = 60
MIN_TRAIN_PERIODS = 40
ML_WARMUP_PERIODS = 12
REOPTIMIZE_EVERY = 6
PURGE_PERIODS = 1
HMM_ROLLING_PERIODS = 12
HMM_ROLLING_MIN = 6

# HMM + rule fusion (fixes z-score mislabels without crushing OOS returns)
HMM_RULE_BLEND = 0.28  # light touch: raw HMM dominates; rules fix edge cases
HMM_BLEND_STRESS_FLOOR = 0.50  # λ when clear stress (loss / high vol)
HMM_BLEND_RALLY_FLOOR = 0.45  # λ when strong rally
HMM_VOL_CRISIS_RATIO = 1.25
HMM_MAX_TREND_IF_RISK_OFF = 0.15  # drawdown month below SMA200
HMM_MAX_TREND_RECOVERY_RISK_OFF = 0.35  # positive month but still below SMA200
HMM_MAX_TREND_ON_LOSS_MONTH = 0.20

# Crypto HMM (24/7, higher baseline vol — softer crisis floors, wider range band)
CRYPTO_TRADING_DAYS_PER_YEAR = 365
HMM_VOL_CRISIS_RATIO_CRYPTO = 1.65   # vol/median > this → stress (was 1.25 for stocks)
HMM_STRESS_FLOOR_LOSS_CRYPTO = 0.12  # min P(crisis) on bad day (stocks implicit ~0.28)
HMM_STRESS_FLOOR_VOL_CRYPTO = 0.20   # min P(crisis) on vol spike (stocks ~0.38)
HMM_RULE_BLEND_CRYPTO = 0.12         # less rule override — trust Gaussian filter more
HMM_FLAT_RET_SCALE_CRYPTO = 0.045    # wider "stagnation" band for daily crypto moves
HMM_RET_Z_TREND_CRYPTO = 0.55        # |ret_z| above → less range, more trend/crisis

# 5Min crypto bar grid (24/7): 12 bars = 1 hour, 288 bars = 1 day
CRYPTO_BARS_PER_HOUR = 12
CRYPTO_BARS_PER_DAY = 288

# HMM on 5Min bars — baseline = last 1 hour (not 7 days)
HMM_FREQUENCY = "bar"                # "bar" | "daily"
HMM_RET_Z_LOOKBACK_BARS = CRYPTO_BARS_PER_HOUR       # 1 hour
HMM_BAR_ROLLING_BARS = CRYPTO_BARS_PER_HOUR        # HMM emissions: same 1h window
HMM_BAR_ROLLING_MIN = 6                            # 30 min warmup
HMM_BAR_VOL_WINDOW_BARS = CRYPTO_BARS_PER_HOUR     # 1h realized vol
HMM_BAR_VOL_MEDIAN_BARS = CRYPTO_BARS_PER_DAY      # 1d median = "normal" vol level
HMM_BAR_RISK_SMA_BARS = CRYPTO_BARS_PER_DAY        # risk_on: price vs 1-day SMA
HMM_BAR_TRAIN_BARS = CRYPTO_BARS_PER_DAY * 7       # 7 days history for EM fit
HMM_BAR_REFIT_EVERY = 72   # refit every 6h @ 5min (was 12 = 1h → ~6x faster HMM)
# Cap bar-level HMM input (~100k bars ≈ 11 months @ 5Min). None = full history (very slow).
HMM_BAR_MAX_ROWS = 100_000
HMM_STATUS_MAX_ROWS = 12_000
HMM_BAR_EM_ITERS = 10
HMM_BAR_PLOT_ROWS = 3_000
HMM_REGIME_CACHE_PATH = OUT_DIR / "cache" / "hmm_regime.parquet"

# Rule fusion scales for a single 5Min bar (not daily log_ret)
HMM_BAR_FLAT_RET_SCALE_CRYPTO = 0.002    # ~0.2%/bar = quiet range
HMM_BAR_RET_SIGNAL_SCALE_CRYPTO = 0.004  # tanh scale for bar log_ret
HMM_BAR_LOSS_SCALE_CRYPTO = 0.005        # bad 5m bar → crisis floor
HMM_BAR_RALLY_LOG_RET_CRYPTO = 0.006     # strong 5m rally bar
HMM_BAR_RALLY_RET_Z_CRYPTO = 0.75

USE_SIGNAL_EXIT = True
FORECAST_HORIZON_BARS = 21  # trading days ≈ 1-month forward horizon

# No-trade band: skip re-weighting when target weight drifts less than this fraction.
# Suppresses vol-target micro-rebalances that dominated commission drag in 48-fold OOS.
BACKTEST_REBALANCE_BAND = 0.20
# Hold entry weight; re-weight weekly only (not daily vol-target churn on open positions).
BACKTEST_HOLD_ENTRY_WEIGHT = True
BACKTEST_REWEIGHT_FREQ = "W"
# Min bars before signal-dead exit; cooldown before re-entry (5m bars: 72=6h, 36=3h).
BACKTEST_MIN_HOLD_BARS = 96
BACKTEST_REENTRY_COOLDOWN_BARS = 36

# Robustness: vol spike → cash; hysteresis reduces whipsaw / commissions
TAIL_VOL_RATIO_FREEZE = 1.5
SCORE_ENTER = 50   # enter / rebalance-in threshold
SCORE_EXIT = 40    # hold / signal-exit threshold (lower than enter)

# Fusion optimization — tuned for law-of-large-numbers (more independent bets).
FUSION_MIN_SIGNAL_ROWS = 30
FUSION_MIN_ACTIVE_REBALANCES = 3
FUSION_MIN_AVG_EXPOSURE_PCT = 0.5
FUSION_EXPOSURE_PENALTY = 5.0
FUSION_ENTRY_TARGET = "label_entry"
FUSION_EXPECTED_MOVE_BPS = 20.0
# Gross stop-loss distance (bps). Lower SL => lower edge floor, more entries.
FUSION_STOP_LOSS_BPS = 45.0
# Global edge fallback only when per-instrument floor/policy is unset (prefer ticker cost/target-opt).
FUSION_MIN_EXPECTED_EDGE_BPS = 0.0
FUSION_EDGE_BUFFER_BPS = 2.0
# TP/SL regressor calibration: enforce min ratio and cap wide targets (same rule, per-row values).
FUSION_TP_SL_MIN_RATIO = 1.0
FUSION_TP_SL_MAX_RATIO = 2.0
FUSION_TP_SL_TP_FLOOR_BPS = 40.0
FUSION_TP_SL_SL_CAP_BPS = 75.0
# Left-tail: skip entries only when stress AND high vol (keeps trade count).
FUSION_TAIL_ENTRY_FILTER = True
FUSION_TAIL_STRESS_ENTRY_MAX = 0.42
FUSION_TAIL_HIGH_VOL_ANN = 0.22
FUSION_TAIL_HIGH_VOL_RATIO = 1.35
# Tighten SL in elevated stress/vol (time-stop proxy via faster stop-out).
FUSION_TAIL_SL_TIGHTEN = True
FUSION_TAIL_STRESS_SL_MIN = 0.35
FUSION_TAIL_SL_TIGHTEN_IN_STRESS = 0.70
FUSION_TAIL_SL_TIGHTEN_HIGH_VOL = 0.80
FUSION_TAIL_SL_TIGHTEN_HIGH_VOL_RATIO = 0.85
FUSION_TAIL_SL_MIN_BPS = 12.0
# Gate economics (one story): label TP, edge floor, and ML edge gate all use full RT.
FUSION_EDGE_GATE_FLOOR_MODE = "full_round_trip"
FUSION_LABEL_TP_MODE = "full_round_trip"
FUSION_EDGE_FLOOR_MODE = "full_round_trip"
# Default take-profit when row has no pred_tp (bps); target-opt threshold preferred.
FUSION_DEFAULT_TAKE_PROFIT_BPS = 47.5
# Calibrate min |edge| near top quintile of train |expected_edge| panel.
FUSION_EDGE_CALIBRATION_QUANTILE = 0.80
FUSION_EDGE_CALIBRATION_CAP_BPS = 60.0
# Decile gate: block symbol when top-decile net edge below this (bps).
FUSION_MIN_TOP_DECILE_NET_BPS = 5.0
# Per-fold threshold opt: disable trading when CV objective falls below this.
FUSION_THRESHOLD_NO_TRADE_OBJECTIVE = -2.0
# HMM hard gate off by default with multi-asset panel (macro ETFs need stress entries).
FUSION_HMM_HARD_GATE = False
FUSION_RETURN_OBJECTIVE_SCALE = 10.0
FUSION_TURNOVER_PENALTY = 0.04
FUSION_NEGATIVE_RETURN_PENALTY = 2.0
# Bonus in threshold-opt objective for high signal count (LLN).
FUSION_FREQUENCY_OBJECTIVE_WEIGHT = 0.0
FUSION_TARGET_TRADES_PER_YEAR = 30
FUSION_EXPECTANCY_OBJECTIVE_WEIGHT = 0.5
FUSION_HMM_CONFIDENCE_MIN = 0.30
FUSION_HMM_ENTROPY_MAX = 0.80
# Regime-adaptive holding horizon: pick exit horizon per dominant HMM regime by
# de-overlapped after-cost EV (candidates = HOLD_BUCKETS). Off => fixed horizon.
FUSION_ADAPTIVE_HORIZON = False
# Impulse grid parallelism: None => max(1, cpu_count() - 1); override via FUSION_GRID_WORKERS env.
FUSION_GRID_WORKERS = None
# Per-symbol panel build / target-opt pools (None => cpu_count - 1). Env: FUSION_PANEL_BUILD_WORKERS, FUSION_TARGET_OPT_WORKERS.
FUSION_PANEL_BUILD_WORKERS = None
FUSION_TARGET_OPT_WORKERS = None
# Optuna trial parallelism per fold (None => cpu_count-1). Set LGBM threads to 1
# when Optuna runs multiple trials so cores are not oversubscribed.
FUSION_OPTUNA_N_JOBS = None
FUSION_LGBM_N_JOBS = 1
# Walk-forward evaluation modes:
# - "legacy": existing fast session-by-session OOS.
# - "adaptive": 12m rolling train -> 6m OOS; recency weights + warm-start LightGBM.
# - "monthly4y": 1y trailing train -> 1 calendar month OOS (legacy monthly).
FUSION_WF_MODE = "adaptive"
# 10y Finam 5Min: 5y train -> 3m OOS, retrain each fold (~20 folds over remaining 5y).
FUSION_WF_TRAIN_DAYS = 1825
FUSION_WF_BACKTEST_YEARS = 5
FUSION_WF_TEST_MONTHS = 3
# Cap monthly WF folds (None = all windows from backtest_years / test_months).
FUSION_WF_MAX_FOLDS = None
FUSION_WF_MIN_TRAIN_ROWS = 50_000
FUSION_WF_CALIBRATION_FRACTION = 0.15

# Adaptive learning (rolling window + recency weights + warm-start trees).
FUSION_ADAPTIVE_RECENCY_HALFLIFE_DAYS = 120.0
FUSION_ADAPTIVE_INCREMENTAL_TREES = 72
FUSION_ADAPTIVE_INCREMENTAL_LR = 0.025
FUSION_ADAPTIVE_L1 = 0.15
FUSION_ADAPTIVE_L2 = 1.5
FUSION_ADAPTIVE_FEATURE_FRACTION = 0.8
FUSION_ADAPTIVE_BAGGING_FRACTION = 0.75
FUSION_ADAPTIVE_BAGGING_FREQ = 1

# Monthly4y backtests are much larger than the legacy 120-session OOS; use a
# compact impulse grid so the full 4y report is runnable in hours, not days.
FUSION_MONTHLY_COMPACT_GRID = True
# Avoid Windows multiprocessing pickle failures on multi-year OOS DataFrames.
FUSION_MONTHLY_GRID_WORKERS = 1
# Full monthly4y OOS is large enough that impulse-grid search is usually the
# bottleneck. Use a fixed conservative impulse policy for the final stitched
# report; keep grid search available in legacy/smoke modes.
FUSION_MONTHLY_SKIP_GRID = True
# Per-fold (monthly4y): optimize inside each 1y train window; test month never used.
FUSION_MONTHLY_OPTIMIZE_PER_FOLD = True
# Legacy global tune (single pre-OOS slice) — disabled when per-fold is on.
FUSION_MONTHLY_OPTIMIZE_MODEL = True
FUSION_FOLD_OPT_MAX_TRAIN_ROWS = None
# Per-fold LightGBM hyperparameter search (Optuna TPE on purged session CV).
FUSION_MODEL_OPTUNA_TRIALS = 150
FUSION_MODEL_OPTUNA_SEED = 42
FUSION_MODEL_OPT_PROGRESS_EVERY = 5
# Ranges expanded where prior run pinned best values at bounds (monthly_fold_optimizations).
FUSION_MODEL_NUM_LEAVES_RANGE = (5, 48)
FUSION_MODEL_MAX_DEPTH_RANGE = (3, 10)
FUSION_MODEL_LEARNING_RATE_RANGE = (0.015, 0.12)
FUSION_MODEL_N_ESTIMATORS_RANGE = (60, 400)
FUSION_MODEL_MIN_CHILD_SAMPLES_RANGE = (25, 220)
FUSION_MODEL_LAMBDA_L1_RANGE = (0.0, 0.65)
FUSION_MODEL_LAMBDA_L2_RANGE = (0.3, 3.5)
FUSION_MODEL_FEATURE_FRACTION_RANGE = (0.45, 0.98)
FUSION_MODEL_BAGGING_FRACTION_RANGE = (0.45, 0.95)
FUSION_MODEL_BAGGING_FREQ_RANGE = (1, 7)
FUSION_MODEL_MIN_SPLIT_GAIN_RANGE = (0.0, 1.0)
FUSION_MODEL_MAX_BIN_RANGE = (127, 511)
FUSION_MODEL_PATH_SMOOTH_RANGE = (0.0, 1.0)
FUSION_MONTHLY_OPTIMIZE_THRESHOLDS_PER_FOLD = True
# Per-fold entry-target grid (horizon/label_type/threshold) on train-only slice before model opt.
# On: labels track regime; global cache is only the seed / fallback.
FUSION_MONTHLY_OPTIMIZE_TARGETS_PER_FOLD = True
FUSION_FOLD_TARGET_OPT_MIN_TRAIN_ROWS = 5_000
FUSION_FOLD_TARGET_OPT_MAX_TRAIN_ROWS = 40_000
FUSION_FOLD_TARGET_OPT_COMPACT_GRID = True
FUSION_FOLD_TARGET_OPT_HORIZON_GRID = (24, 48, 72, 96)
FUSION_FOLD_TARGET_OPT_THRESHOLD_MULTS = (1.0, 1.5)
# Per-fold policy: off | fixed | calibrated | per_ticker_calibrated | optuna | optuna_per_ticker
# per_ticker_calibrated: same calibration concept, per-instrument buy/edge from fold train.
FUSION_THRESHOLD_POLICY_MODE = "per_ticker_calibrated"
# Per-ticker threshold model (gate-aligned top-decile + optional isotonic policy)
FUSION_THRESHOLD_CAL_TARGET_NET_BPS = 0.0
FUSION_THRESHOLD_CAL_BUY_LO = 30
FUSION_THRESHOLD_CAL_BUY_HI = 55
FUSION_THRESHOLD_CAL_EDGE_HI_BPS = 12.0
FUSION_THRESHOLD_CAL_MIN_TRADES = 200
FUSION_THRESHOLD_CAL_MIN_ROWS = 200
FUSION_THRESHOLD_CAL_REQUIRE_POSITIVE_SIGNAL = True
FUSION_THRESHOLD_CAL_CV_FOLDS = 3
FUSION_THRESHOLD_CAL_FALLBACK_QUANTILE = 0.85
# Legacy grid keys (unused by isotonic model; kept for older scripts)
FUSION_THRESHOLD_CAL_BUY_GRID = (32, 36, 40, 44)
FUSION_THRESHOLD_CAL_EDGE_OFFSETS_BPS = (0.0, 1.0, 2.0, 4.0)
# Cap rows for per-fold Optuna threshold search (optuna mode only).
FUSION_THRESHOLD_OPT_MAX_TRAIN_ROWS = 80_000
FUSION_THRESHOLD_OPT_PROGRESS_EVERY = 5
# Per-fold gate/threshold search (Optuna TPE on train-only CV).
FUSION_THRESHOLD_OPTUNA_TRIALS = 25
FUSION_THRESHOLD_OPTUNA_TRIALS_PER_TICKER = 20
FUSION_THRESHOLD_OPTUNA_SEED = 42
FUSION_THRESHOLD_OPT_MIN_TRAIN_ROWS_PER_TICKER = 200
FUSION_THRESHOLD_OPT_MIN_TRAIN_SESSIONS_PER_TICKER = 4
# Relaxed edge floor during CV search (production gate stays FUSION_EDGE_GATE_FLOOR_MODE).
FUSION_THRESHOLD_OPT_EDGE_FLOOR_MODE = "full_round_trip"
# Backtest-heavy; None => min(8, cpu_count-1). Model opt uses FUSION_OPTUNA_N_JOBS.
FUSION_THRESHOLD_OPTUNA_N_JOBS = None
# Persist every Optuna trial (for narrowing ranges in config.py later).
FUSION_OPTIMIZATION_SAVE_ALL_TRIALS = True
FUSION_OPTIMIZATION_SUMMARY_TOP_FRAC = 0.25
# Threshold search bounds — symmetric around score=50 (buy>50, sell<50).
FUSION_THRESHOLD_BUY_RANGE = (51, 58)
FUSION_THRESHOLD_HOLD_RANGE = (40, 48)
FUSION_THRESHOLD_GAIN_RANGE = (60, 100)
FUSION_THRESHOLD_IMPULSE_MIN_RANGE = (0.01, 0.06)
FUSION_THRESHOLD_W_ML_RANGE = (0.35, 0.55)
# None => derive from panel max edge + heuristic floor (avoid fixed 25bps blocking search).
FUSION_THRESHOLD_EDGE_RANGE = None
FUSION_THRESHOLD_EDGE_SPAN_BPS = 10.0
FUSION_THRESHOLD_OPT_MIN_SIGNAL_ROWS = 5
FUSION_THRESHOLD_OPT_MIN_ACTIVE_REBALANCES = 1
FUSION_THRESHOLD_OPT_MIN_AVG_EXPOSURE_PCT = 0.05
# Relaxed anti-churn for short purged-CV windows (production backtest keeps stricter settings).
FUSION_THRESHOLD_OPT_MIN_HOLD_BARS = 12
FUSION_THRESHOLD_OPT_REBALANCE_BAND = 0.02
FUSION_THRESHOLD_OPT_HOLD_ENTRY_WEIGHT = False
FUSION_THRESHOLD_OPT_REENTRY_COOLDOWN_BARS = 12
FUSION_THRESHOLD_OPT_MIN_TRAIN_ROWS = 500
FUSION_THRESHOLD_OPT_MIN_TRAIN_SESSIONS = 4
FUSION_THRESHOLD_STOP_LOSS_RANGE = (45, 45)
FUSION_THRESHOLD_EDGE_SCALE = (1.0, 1.25)

# Monte Carlo (final pipeline stage).
MONTE_CARLO_PATHS = 2000
MONTE_CARLO_BLOCK_BARS = 12
MONTE_CARLO_MAX_DD_THRESHOLD = 0.20

# Panel cache version — bump when feature schema / universe changes (forces rebuild).
# v14: structure/path features + expanded CS ranks + spec_dominant_period in ML
PANEL_CACHE_VERSION = 14
# Per-instrument parquet caches (no shared fusion_panel_v*.parquet when True).
FUSION_PANEL_ISOLATED = True
# Pick tradfi TF with the largest bar count (~1Day ≈ 5k bars / 20y on ETFs).
FUSION_AUTO_TIMEFRAME_MAX_BARS = False
# Deprecated: median-threshold labels force ~50% positive (artificial + lookahead).
FUSION_USE_BALANCED_ENTRY = False
FUSION_BALANCED_TARGET_RATE = 0.5
# Default economic entry labels when target-opt cache is absent (fixed bps thresholds).
FUSION_DEFAULT_ENTRY_LABEL_TYPE = "triple_barrier"
FUSION_DEFAULT_ENTRY_HORIZON = 48
FUSION_DEFAULT_ENTRY_THRESHOLD_BPS = 0.0
# LightGBM: auto scale_pos_weight from train positive rate.
FUSION_CLASS_WEIGHT_AUTO = True
# Per-ticker LightGBM regressors for take-profit / stop-loss (bps).
FUSION_USE_TP_SL_REGRESSOR = True
# Crypto: require tick flow; tradfi: candle-imputed flow (see market.universe).
FUSION_TICK_ONLY_CRYPTO = True

# ----------------------------------------------------------------------------
# Per-instrument target optimization (experiment-driven option).
# Each crypto has different volatility, so a single global label
# (label_12_after_costs) over- or under-shoots the cost floor per symbol.
# `ml target-opt` sweeps (horizon, label_type, threshold) per symbol and can
# persist the winners; when enabled, build_fusion_panel attaches a unified
# `label_entry` column where every symbol is labeled by its own best spec.
# False = use applied target cache; run `optimize_targets_per_instrument` offline to refresh.
FUSION_IGNORE_APPLIED_TARGETS = False
# Inline overrides (symbol -> {"horizon": int, "label_type": str, "threshold_bps": float}).
USE_PER_INSTRUMENT_TARGETS = False
# Inline overrides (symbol -> {"horizon": int, "label_type": str, "threshold_bps": float}).
# Empty by default; `ml target-opt --apply` writes results to cache instead of here.
PER_INSTRUMENT_TARGETS: dict[str, dict] = {}
# Min TP floor (bps): SL + round-trip commission; None => derive via min_tp_gross_bps().
TARGET_OPT_COST_FLOOR_BPS = None
# Experiment search grids — shorter horizons for more de-overlapped trades (LLN).
TARGET_OPT_HORIZON_GRID = (12, 24, 36, 48, 72, 96)
TARGET_OPT_MAX_HORIZON_BARS = 96
# Direction mode: skip horizons below this (5m bars: 24 ≈ 2h hold).
TARGET_OPT_DIRECTION_MIN_HORIZON_BARS = 24
TARGET_OPT_LABEL_TYPES = ("triple_barrier", "after_costs")
TARGET_OPT_THRESHOLD_MULTS = (1.0, 1.25, 1.5)
# Acceptable positive-class share; wider low bound => more labels.
TARGET_OPT_BALANCE_RANGE = (0.08, 0.55)
# Cap train rows per CV fold to bound experiment runtime (None => no cap).
TARGET_OPT_MAX_TRAIN_ROWS = 60_000
# Composite: learnability + economics + balance + trade frequency (LLN).
# Economics weight raised so RT-aligned top-decile dominates pick (ranking gate).
TARGET_OPT_SCORE_WEIGHTS = {"cv": 0.25, "economics": 0.45, "balance": 0.15, "frequency": 0.15}
TARGET_OPT_TRADES_PER_YEAR_TARGET = 30
# Prefer shorter horizons when economics tie (more independent bets).
TARGET_OPT_HORIZON_PENALTY_PER_BAR = 0.0008
# Prefer targets whose top-decile net edge is positive; fall back to least-negative if none.
TARGET_OPT_REQUIRE_POSITIVE_EDGE = True
# Target selection: "entry" (binary label_entry, RT-aligned top-decile) | "direction" (3-class).
TARGET_OPT_SCORING_MODE = "entry"
# Acceptable flat/long/short shares for direction-aware scoring.
TARGET_OPT_DIRECTION_CLASS_RANGES = {
    "flat": (0.15, 0.45),
    "long": (0.20, 0.45),
    "short": (0.20, 0.45),
}
TARGET_OPT_DIRECTION_SCORE_WEIGHTS = {
    "accuracy": 0.35,
    "economics": 0.35,
    "balance": 0.20,
    "frequency": 0.10,
}
# Direction mode: require positive long and short economic edge (bps) on the label.
TARGET_OPT_DIRECTION_REQUIRE_EDGE = True
# Reject overly flat labels in direction mode (long+short must clear this share).
TARGET_OPT_DIRECTION_MIN_DIRECTIONAL_RATE = 0.35
# Penalize / filter when flat share exceeds this in direction scoring.
TARGET_OPT_DIRECTION_MAX_FLAT_RATE = 0.50

# Separate LightGBM per asset class (crypto vs tradfi) — disabled when per-ticker only.
FUSION_ASSET_CLASS_MODELS = False
FUSION_ASSET_CLASS_MIN_ROWS = 500
# Per-ticker models + Optuna (takes precedence over asset-class when True).
FUSION_PER_TICKER_MODELS = True
FUSION_PER_TICKER_MIN_ROWS = 200
# Apply target-opt cache specs for label_entry even when USE_PER_INSTRUMENT_TARGETS is False.
FUSION_APPLY_TARGET_CACHE = True
# Decile gate: strict = stitched pass OR fold-history with positive stitched OOS net.
FUSION_DECILE_GATE_MODE = "strict"
FUSION_GATE_RECENT_FOLDS = 0
FUSION_GATE_MIN_OK_FOLDS = 2
# Legacy absolute bps (unused); fold median uses economics_floor × FRAC.
FUSION_GATE_FOLD_MEDIAN_NET_BPS = 3.0
FUSION_GATE_FOLD_MEDIAN_FLOOR_FRAC = 0.6
FUSION_GATE_STRICT_MIN_STITCHED_NET_BPS = 0.0
# Drop stitched-passers only when fold SQ pass-rate AND mean holdout fail economics floor.
FUSION_GATE_REQUIRE_SIGNAL_QUALITY = True
# Fail-closed: quality/decile gates on; soft-size hard-zeros failed names.
FUSION_DISABLE_QUALITY_GATE = False
# Soft-size exposure_cap from CV net / inverted |edge| alignment (universal rule).
FUSION_QUALITY_SOFT_SIZE = True
# Quality fail or holdout_net < 0 → exposure_cap=0 (not fail_cap).
FUSION_SOFT_SIZE_HARD_ZERO = True
FUSION_SOFT_SIZE_MIN = 0.10
FUSION_SOFT_SIZE_QUALITY_FAIL = 0.35
FUSION_SOFT_SIZE_INVERTED_CAP = 0.25
FUSION_SOFT_SIZE_CV_LO_BPS = -10.0
FUSION_SOFT_SIZE_CV_HI_BPS = 5.0
# Per-ticker allowed sides (asymmetry from OOS desk: NASDAQ shorts / IMOEX longs drag).
FUSION_SIDE_POLICY: dict[str, str] = {
    "NASDAQ": "long_only",
    "IMOEX": "short_only",
    "GAZP": "both",
    "SBER": "both",
    "SP500": "both",
}
# Live book: inverse-vol exposure budget across tradeable set (cap = max weight).
FUSION_PER_TICKER_EXPOSURE_BUDGET = True
FUSION_PER_TICKER_MAX_WEIGHT = 0.35
# Economics floor = RT(vol) × OVER_RT_MULT × vol_scale^VOL_EXP (no absolute bps per ticker).
FUSION_ECONOMICS_OVER_RT_MULT = 1.25
FUSION_ECONOMICS_VOL_EXP = 0.5
FUSION_VOL_SCALE_CLIP_LO = 0.5
FUSION_VOL_SCALE_CLIP_HI = 2.5
# Soft-size CV band = [-LO_OVER_FLOOR × floor, +floor].
FUSION_SOFT_SIZE_CV_LO_OVER_FLOOR = 2.0
# SQ filter: keep stitched-passer if fold pass_rate ≥ this OR mean holdout ≥ economics floor.
FUSION_SQ_MIN_PASS_RATE = 0.40
# Per-fold target-opt: cap horizon search (helps US indices vs very long H=96).
FUSION_FOLD_TARGET_OPT_MAX_HORIZON_BARS = 72
# Legacy alias kept for reporting fallbacks; live gates use economics_floor_bps().
FUSION_MIN_TOP_DECILE_NET_BPS = 5.0
# Threshold calibrator: floors come from instrument_economics (RT×mult×vol).
FUSION_THRESHOLD_OOS_HOLDOUT_MIN_NET_BPS = 5.0  # unused when economics module active
FUSION_THRESHOLD_CAL_MIN_CV_NET_BPS = 5.0  # unused when economics module active
# Penalize train>>val top-decile net gap in Optuna composite (overfit guard).
FUSION_TRAIN_OOS_GAP_WEIGHT = 0.45
FUSION_TRAIN_OOS_GAP_SCALE_BPS = 8.0

# Entry model: LightGBM (purged CV tunes hyperparameters only).
FUSION_ENTRY_MODEL = "lightgbm"
ENTRY_MODEL_CANDIDATES = (FUSION_ENTRY_MODEL,)
# Classifier Optuna: profit-first; accuracy as secondary guard.
FUSION_OPTUNA_OBJECTIVE = "profit"
# Hard-reject only clearly uneconomic trials (gross cannot clear RT).
# Do NOT hard-require +5 net here — that starved Optuna (all profit=-1).
# Live gate (+5) stays in decile/quality gates after model selection.
FUSION_OPTUNA_REJECT_GROSS_BELOW_RT = True
FUSION_OPTUNA_MIN_GROSS_OVER_RT_BPS = 0.0
# None => no hard net floor during search (soft ranking via profitability_score).
FUSION_OPTUNA_MIN_NET_BPS = None
# Soft gap penalty in profit objective; no hard gap kill during search.
FUSION_OPTUNA_APPLY_GAP_PENALTY = True
FUSION_OPTUNA_MAX_TRAIN_OOS_GAP_BPS = None
# Weight Optuna fits by |fwd_ret| so ranking economics enter the loss.
FUSION_OPTUNA_PROFIT_SAMPLE_WEIGHT = True
# Separate Optuna for short classifiers (do not reuse long HPs).
FUSION_OPTUNA_SHORT_MODELS = True
FUSION_MODEL_OPTUNA_SHORT_TRIALS = 60
ENTRY_MODEL_CRITERIA_WEIGHTS = {
    "top_decile_net_bps": 0.60,
    "accuracy": 0.12,
    "precision": 0.08,
    "recall": 0.08,
    "auc": 0.04,
    "pr_auc": 0.02,
    "edge_corr": 0.06,
    "ic_spearman": 0.00,
    "log_loss": 0.00,
    "brier": 0.00,
    "calibration_mae": 0.00,
    "fold_stability": 0.00,
}
# Threshold Optuna: fail constraints when mean deoverlapped trade net < 0.
FUSION_THRESHOLD_OPT_REQUIRE_POSITIVE_TRADE_NET = True
# Primary equity path includes commission+slippage (optimize what we report as net).
FUSION_BACKTEST_GROSS_ONLY = False
FUSION_METRICS_GROSS_ONLY = False
# Optional side report with friction already in primary; keep gross companion if needed.
FUSION_BACKTEST_REPORT_NET = True
# TP/SL regressor: weight training rows by |forward return| (profit proxy).
FUSION_TP_SL_PROFIT_WEIGHTS = True
# Long + short: separate short classifier; signed fusion score in backtest.
FUSION_ALLOW_SHORT = True
# None => derive sell as 100 - buy (symmetric around score=50).
FUSION_SELL_THRESHOLD = None
FUSION_SYMMETRIC_THRESHOLDS = True
# Min long/short edge gap before assigning direction (symmetric scoring).
FUSION_MIN_DIRECTION_EDGE = 0.01
# Feature groups excluded from LightGBM (kept for impulse/HMM gates).
ML_EXCLUDED_GROUPS: tuple[str, ...] = ("hmm_regime", "hmm_dynamics", "vp_hmm_blend")

TRADING_DAYS_PER_YEAR = 252

FUSION_HISTORY_DAYS = 3650  # 10y via Finam Trade API / MOEX

# 10y Finam Trade API 5Min desk — 4 instruments with full history.
FINAM_UNIVERSE = ["GAZP", "SBER", "IMOEX", "NASDAQ", "SP500"]


def _resolve_fusion_bar_timeframe() -> str:
    """Tradfi TF with the most bars for the configured history window."""
    if not FUSION_AUTO_TIMEFRAME_MAX_BARS:
        return "5Min"
    try:
        from common.timeframe_policy import best_tradfi_timeframe_for_max_bars

        tf, _ = best_tradfi_timeframe_for_max_bars(FUSION_HISTORY_DAYS)
        return str(tf)
    except Exception:
        return "1Day"


# ML fusion: auto-pick timeframe that maximizes bar count (1Day for US ETFs).
BAR_TIMEFRAME = _resolve_fusion_bar_timeframe()

# --- Universe: Finam desk (5Min) — MOEX RU + yfinance US/EU indices ---
CRYPTO_UNIVERSE = ["BTC"]
TRADFI_UNIVERSE = list(FINAM_UNIVERSE)
TRADFI_INTRADAY_DAYS = 59  # legacy default (5Min yfinance cap)
# yfinance intraday history limits (calendar days) by bar size.
TRADFI_INTRADAY_DAYS_BY_TF: dict[str, int] = {
    "5Min": 3650,  # Finam Trade API ~10y
    "15Min": 60,
    "30Min": 60,
    "1Hour": 730,
    "1Day": 7300,  # ~20 calendar years (yfinance daily max via start/end)
}


def tradfi_max_days(timeframe: str) -> int:
    return int(TRADFI_INTRADAY_DAYS_BY_TF.get(timeframe, TRADFI_INTRADAY_DAYS))
FLOW_DEFAULT_TICKERS: list[str] = list(TRADFI_UNIVERSE)  # ML index desk: tradfi only
CRYPTO_DISPLAY_NAMES = {
    "BTC": "Bitcoin / USD",
}
TRADFI_DISPLAY_NAMES = {
    "GAZP": "Gazprom (MOEX)",
    "SBER": "Sberbank (MOEX)",
    "IMOEX": "MOEX Russia Index",
    "NASDAQ": "NASDAQ Composite",
    "SP500": "S&P 500",
    "DAX": "DAX (Germany)",
}
DISPLAY_NAMES = {**CRYPTO_DISPLAY_NAMES, **TRADFI_DISPLAY_NAMES}
REGIME_TICKERS = ["IMOEX", "SP500", "NASDAQ"]
REGIME_TICKER_WEIGHTS = {"IMOEX": 0.34, "SP500": 0.33, "NASDAQ": 0.33}
BENCHMARK = "SP500"
REGIME_TICKER = "SP500"

SMA_LONG = 200
SMA_SHORT = 50
PRICE_TREND_LOOKBACK_DAYS = 252  # 12-1 price trend lookback (trading days)
PRICE_TREND_SKIP_DAYS = 21  # skip recent month (Jegadeesh-Titman)
VOL_WINDOW = 20
TARGET_VOL_ANN = 0.12
MAX_WEIGHT = 0.20

SCORE_BUY = 65
SCORE_HOLD = 45

# Slippage per side (bps). Backtest: RT cost = 2×(commission + slippage_per_side).
SLIPPAGE_BPS_CRYPTO = 1.5
SLIPPAGE_BPS_TRADFI = 2.0
SLIPPAGE_BPS_DEFAULT = 2.0
SLIPPAGE_VOL_SCALE = True
SLIPPAGE_VOL_REF_CRYPTO = 0.80   # 80% ann vol reference
SLIPPAGE_VOL_REF_TRADFI = 0.18   # 18% ann vol reference
SLIPPAGE_VOL_REF_DEFAULT = 0.50
# Legacy alias (total RT slippage if added once — prefer execution_costs module)
DEFAULT_SLIPPAGE_BPS = SLIPPAGE_BPS_CRYPTO
COMMISSION_PCT_CRYPTO = 0.011
COMMISSION_PCT_TRADFI = 0.04
COMMISSION_BPS_CRYPTO = 1.1   # 0.011% × 100
COMMISSION_BPS_TRADFI = 0.5   # liquid ETF; was 4.0 (killed SPY net)
COMMISSION_BPS_BASE = COMMISSION_BPS_CRYPTO
COMMISSION_BPS_PER_SIDE = COMMISSION_BPS_CRYPTO  # default when ticker unknown
# Notional starting capital for backtest equity / capital-over-time plots.
BACKTEST_START_CAPITAL = 10_000.0


# ============================================================================
# Intraday derivation — rescale bar-count constants from BAR_TIMEFRAME.
# Runs after all daily base constants so it overrides them when intraday.
# ============================================================================
RTH_MINUTES = 390  # US regular trading hours 09:30–16:00 ET = 6.5h


def _tf_minutes(tf: str) -> int:
    m = re.fullmatch(r"(\d+)(Min|Hour|Day|Week|Month)", tf)
    if not m:
        return RTH_MINUTES  # treat unknown as 1 session
    n, unit = int(m.group(1)), m.group(2)
    return {
        "Min": n,
        "Hour": 60 * n,
        "Day": RTH_MINUTES * n,
        "Week": RTH_MINUTES * 5 * n,
        "Month": RTH_MINUTES * 21 * n,
    }[unit]


BAR_MINUTES = _tf_minutes(BAR_TIMEFRAME)
IS_INTRADAY = BAR_TIMEFRAME not in ("1Day", "1Week", "1Month")
_DEFAULTS_TO_CRYPTO = set(FLOW_DEFAULT_TICKERS).issubset(set(CRYPTO_UNIVERSE))
BARS_PER_DAY = (
    CRYPTO_BARS_PER_DAY
    if IS_INTRADAY and _DEFAULTS_TO_CRYPTO
    else max(1, RTH_MINUTES // BAR_MINUTES) if IS_INTRADAY else 1
)
BARS_PER_YEAR = float(
    BARS_PER_DAY * (CRYPTO_TRADING_DAYS_PER_YEAR if IS_INTRADAY and _DEFAULTS_TO_CRYPTO else TRADING_DAYS_PER_YEAR)
    if IS_INTRADAY
    else TRADING_DAYS_PER_YEAR
)


def days_to_bars(n_days: float) -> int:
    """Convert a calendar/trading-day window into bar units for the active TF."""
    return max(1, int(round(n_days * BARS_PER_DAY))) if IS_INTRADAY else max(1, int(n_days))


if IS_INTRADAY:
    # Rebalance once per session (avoid per-5min turnover); exits act per bar.
    _INTRADAY_REBALANCE_FREQ = "D"
    _INTRADAY_HMM_FREQ = "D"
    _INTRADAY_PERIOD_NAME = "bar"
    _INTRADAY_PERIODS_PER_YEAR = BARS_PER_YEAR
    _INTRADAY_FORECAST_HORIZON_BARS = BARS_PER_DAY
    _INTRADAY_SMA_LONG = days_to_bars(200)
    _INTRADAY_SMA_SHORT = days_to_bars(50)
    _INTRADAY_PRICE_TREND_LOOKBACK_DAYS = days_to_bars(252)
    _INTRADAY_PRICE_TREND_SKIP_DAYS = days_to_bars(21)
    _INTRADAY_VOL_WINDOW = days_to_bars(20)
    _INTRADAY_HMM_TRAIN_PERIODS = 252
    _INTRADAY_HMM_ROLLING_PERIODS = 63
    _INTRADAY_HMM_ROLLING_MIN = 21
    _INTRADAY_TRAIN_PERIODS = 252
    _INTRADAY_MIN_TRAIN_PERIODS = 180
    _INTRADAY_ML_WARMUP_PERIODS = 63
    _INTRADAY_REOPTIMIZE_EVERY = 21
    _INTRADAY_PURGE_PERIODS = 5

    REBALANCE_FREQ = _INTRADAY_REBALANCE_FREQ
    HMM_FREQ = _INTRADAY_HMM_FREQ
    PERIOD_NAME = _INTRADAY_PERIOD_NAME
    PERIODS_PER_YEAR = _INTRADAY_PERIODS_PER_YEAR
    FORECAST_HORIZON_BARS = _INTRADAY_FORECAST_HORIZON_BARS
    SMA_LONG = _INTRADAY_SMA_LONG
    SMA_SHORT = _INTRADAY_SMA_SHORT
    PRICE_TREND_LOOKBACK_DAYS = _INTRADAY_PRICE_TREND_LOOKBACK_DAYS
    PRICE_TREND_SKIP_DAYS = _INTRADAY_PRICE_TREND_SKIP_DAYS
    VOL_WINDOW = _INTRADAY_VOL_WINDOW
    HMM_TRAIN_PERIODS = _INTRADAY_HMM_TRAIN_PERIODS
    HMM_ROLLING_PERIODS = _INTRADAY_HMM_ROLLING_PERIODS
    HMM_ROLLING_MIN = _INTRADAY_HMM_ROLLING_MIN
    TRAIN_PERIODS = _INTRADAY_TRAIN_PERIODS
    MIN_TRAIN_PERIODS = _INTRADAY_MIN_TRAIN_PERIODS
    ML_WARMUP_PERIODS = _INTRADAY_ML_WARMUP_PERIODS
    REOPTIMIZE_EVERY = _INTRADAY_REOPTIMIZE_EVERY
    PURGE_PERIODS = _INTRADAY_PURGE_PERIODS


def apply_bar_timeframe(tf: str) -> None:
    """Re-derive bar-count and rebalance constants (rule pipeline daily vs fusion intraday)."""
    global BAR_TIMEFRAME, BAR_MINUTES, IS_INTRADAY, BARS_PER_DAY, BARS_PER_YEAR
    global REBALANCE_FREQ, HMM_FREQ, PERIOD_NAME, PERIODS_PER_YEAR, FORECAST_HORIZON_BARS
    global SMA_LONG, SMA_SHORT, PRICE_TREND_LOOKBACK_DAYS, PRICE_TREND_SKIP_DAYS, VOL_WINDOW
    global HMM_TRAIN_PERIODS, HMM_ROLLING_PERIODS, HMM_ROLLING_MIN
    global TRAIN_PERIODS, MIN_TRAIN_PERIODS, ML_WARMUP_PERIODS, REOPTIMIZE_EVERY, PURGE_PERIODS

    BAR_TIMEFRAME = tf
    BAR_MINUTES = _tf_minutes(tf)
    IS_INTRADAY = tf not in ("1Day", "1Week", "1Month")
    BARS_PER_DAY = (
        CRYPTO_BARS_PER_DAY
        if IS_INTRADAY and _DEFAULTS_TO_CRYPTO
        else max(1, RTH_MINUTES // BAR_MINUTES) if IS_INTRADAY else 1
    )
    BARS_PER_YEAR = float(
        BARS_PER_DAY * (CRYPTO_TRADING_DAYS_PER_YEAR if IS_INTRADAY and _DEFAULTS_TO_CRYPTO else TRADING_DAYS_PER_YEAR)
        if IS_INTRADAY
        else TRADING_DAYS_PER_YEAR
    )
    if IS_INTRADAY:
        REBALANCE_FREQ = _INTRADAY_REBALANCE_FREQ
        HMM_FREQ = _INTRADAY_HMM_FREQ
        PERIOD_NAME = _INTRADAY_PERIOD_NAME
        PERIODS_PER_YEAR = BARS_PER_YEAR
        FORECAST_HORIZON_BARS = max(1, int(round(BARS_PER_DAY)))
        SMA_LONG = days_to_bars(200)
        SMA_SHORT = days_to_bars(50)
        PRICE_TREND_LOOKBACK_DAYS = days_to_bars(252)
        PRICE_TREND_SKIP_DAYS = days_to_bars(21)
        VOL_WINDOW = days_to_bars(20)
        HMM_TRAIN_PERIODS = _INTRADAY_HMM_TRAIN_PERIODS
        HMM_ROLLING_PERIODS = _INTRADAY_HMM_ROLLING_PERIODS
        HMM_ROLLING_MIN = _INTRADAY_HMM_ROLLING_MIN
        TRAIN_PERIODS = _INTRADAY_TRAIN_PERIODS
        MIN_TRAIN_PERIODS = _INTRADAY_MIN_TRAIN_PERIODS
        ML_WARMUP_PERIODS = _INTRADAY_ML_WARMUP_PERIODS
        REOPTIMIZE_EVERY = _INTRADAY_REOPTIMIZE_EVERY
        PURGE_PERIODS = _INTRADAY_PURGE_PERIODS
    else:
        REBALANCE_FREQ = "ME"
        HMM_FREQ = "ME"
        PERIOD_NAME = "month"
        PERIODS_PER_YEAR = 12
        FORECAST_HORIZON_BARS = 21
        SMA_LONG = 200
        SMA_SHORT = 50
        PRICE_TREND_LOOKBACK_DAYS = 252
        PRICE_TREND_SKIP_DAYS = 21
        VOL_WINDOW = 20
        HMM_TRAIN_PERIODS = 60
        HMM_ROLLING_PERIODS = 12
        HMM_ROLLING_MIN = 6
        TRAIN_PERIODS = 36
        MIN_TRAIN_PERIODS = 40
        ML_WARMUP_PERIODS = 12
        REOPTIMIZE_EVERY = 6
        PURGE_PERIODS = 1

# Binance public REST only — no API keys required for OHLCV download.