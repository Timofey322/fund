"""Quick backtest param sweep (reuses one HMM signal build)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from config import apply_bar_timeframe
from data_platform.bars import load_closes
from rule.backtest import equal_weight_universe_return_pct, run_portfolio_backtest
from rule.config import RULE_BAR_TIMEFRAME, RULE_DEFAULT_TICKERS
from rule.signals import build_hmm_exhaustion_signal_frame
from simulation.engine import run_backtest_signal_exit
from rule.config import (
    RULE_EQUAL_WEIGHT,
    RULE_MIN_HOLD_BARS,
    RULE_REBALANCE_BAND,
    RULE_REENTRY_COOLDOWN_BARS,
    RULE_STOP_LOSS_BPS,
    RULE_USE_VOL_TARGETING,
)
from data_platform.universe import commission_bps_for_ticker


def main() -> None:
    apply_bar_timeframe(RULE_BAR_TIMEFRAME)
    tickers = list(RULE_DEFAULT_TICKERS)
    prices = load_closes(tickers, RULE_BAR_TIMEFRAME)
    print("building signals...", flush=True)
    signals = build_hmm_exhaustion_signal_frame(prices)
    comm = float(sum(commission_bps_for_ticker(t) for t in tickers) / len(tickers))
    best = None
    for horizon in (126, 189, 252, 315):
        for enter in (44, 45, 46, 48):
            for sl in (45, 55, 70):
                cfg.SCORE_ENTER = float(enter)
                cfg.SCORE_EXIT = 38.0
                bt = run_backtest_signal_exit(
                    prices,
                    signals,
                    score_col="score",
                    use_dynamic_thresholds=False,
                    use_vol_targeting=RULE_USE_VOL_TARGETING,
                    equal_weight=RULE_EQUAL_WEIGHT,
                    commission_bps=comm,
                    horizon_bars=horizon,
                    stop_loss_bps=sl,
                    min_hold_bars=RULE_MIN_HOLD_BARS,
                    rebalance_band=RULE_REBALANCE_BAND,
                    hold_entry_weight=False,
                    reentry_cooldown_bars=RULE_REENTRY_COOLDOWN_BARS,
                )
                stats = bt.get("stats") or {}
                ps, pe = stats.get("period_start"), stats.get("period_end")
                ew = equal_weight_universe_return_pct(prices, ps, pe)
                ret = stats.get("total_return_pct")
                excess = round(float(ret) - float(ew), 2) if ret is not None and ew is not None else None
                row = (excess, ret, ew, horizon, enter, sl, stats.get("avg_exposure_pct"))
                if best is None or (excess is not None and best[0] is not None and excess > best[0]):
                    best = row
                if excess is not None and excess > 0:
                    print(f"POSITIVE alpha {row}", flush=True)
    print("BEST", best, flush=True)


if __name__ == "__main__":
    main()
