"""Unit tests for HMM exhaustion rule signals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common.naming import COL_PROB_HMM_IMPULSE, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_STRESS
from rule.backtest import expected_move_pct
from rule.config import RULE_EXHAUSTION_MIN, RULE_IMPULSE_VOL_MULT, RULE_NEUTRAL_SCORE
from rule.signals import (
    build_hmm_exhaustion_signal_frame,
    exhaustion_score,
    expected_move_frac,
    impulse_stopping_probability,
    is_price_impulse,
)


def test_expected_move_scales_with_vol():
    low = expected_move_frac(0.10, 12)
    high = expected_move_frac(0.30, 12)
    assert high > low


def test_is_price_impulse_when_move_exceeds_vol():
    vol_ann = 0.20
    thr = expected_move_frac(vol_ann, 12) * RULE_IMPULSE_VOL_MULT
    assert is_price_impulse(-thr * 1.5, vol_ann, 12, RULE_IMPULSE_VOL_MULT)
    assert not is_price_impulse(-thr * 0.5, vol_ann, 12, RULE_IMPULSE_VOL_MULT)


def test_impulse_stopping_rises_when_mean_revert_dominates():
    low = impulse_stopping_probability(0.55, 0.25, 0.20, None)
    high = impulse_stopping_probability(0.15, 0.70, 0.15, None)
    assert high > low


def test_exhaustion_score_buy_after_vol_impulse_dump():
    vol_ann = 0.20
    thr = expected_move_frac(vol_ann, 12) * RULE_IMPULSE_VOL_MULT
    score, side = exhaustion_score(-thr * 1.5, vol_ann, 0.65)
    assert side == "buy_dump_exhaustion"
    assert score > 50.0


def test_exhaustion_score_sell_after_vol_impulse_rally(monkeypatch):
    monkeypatch.setattr("rule.signals.RULE_PARTIAL_SELL_FRAC", 0.10)
    vol_ann = 0.20
    thr = expected_move_frac(vol_ann, 12) * RULE_IMPULSE_VOL_MULT
    score, side = exhaustion_score(thr * 1.5, vol_ann, 0.65)
    assert side == "sell_rally_exhaustion"
    assert score < 40.0


def test_exhaustion_score_rally_neutral_when_partial_sell_disabled(monkeypatch):
    monkeypatch.setattr("rule.signals.RULE_PARTIAL_SELL_FRAC", 0.0)
    vol_ann = 0.20
    thr = expected_move_frac(vol_ann, 12) * RULE_IMPULSE_VOL_MULT
    score, side = exhaustion_score(thr * 1.5, vol_ann, 0.65)
    assert side == "sell_rally_exhaustion"
    assert score == RULE_NEUTRAL_SCORE


def test_exhaustion_score_neutral_without_impulse():
    vol_ann = 0.20
    thr = expected_move_frac(vol_ann, 12) * RULE_IMPULSE_VOL_MULT
    score, side = exhaustion_score(thr * 0.3, vol_ann, 0.80)
    assert side == "neutral"
    assert score == RULE_NEUTRAL_SCORE


def test_exhaustion_score_neutral_when_exhaustion_too_low():
    vol_ann = 0.20
    thr = expected_move_frac(vol_ann, 12) * RULE_IMPULSE_VOL_MULT
    score, side = exhaustion_score(-thr * 2, vol_ann, RULE_EXHAUSTION_MIN - 0.05)
    assert side == "neutral"
    assert score == RULE_NEUTRAL_SCORE


def test_expected_move_pct_helper():
    assert expected_move_pct(0.20, 12) > 0


def test_build_hmm_exhaustion_signal_frame_with_mock_hmm(monkeypatch):
    idx = pd.date_range("2024-01-02 09:35", periods=150, freq="5min")
    close = 100.0 * (1.0 + np.linspace(-0.02, 0.01, len(idx)))
    prices = pd.DataFrame({"SPY": close}, index=idx)

    def _fake_hmm(prices: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
        tcol = "bar_time"
        n = len(prices)
        return pd.DataFrame(
            {
                tcol: prices.index,
                COL_PROB_HMM_IMPULSE: np.linspace(0.55, 0.20, n),
                COL_PROB_HMM_MEAN_REVERT: np.linspace(0.25, 0.65, n),
                COL_PROB_HMM_STRESS: np.full(n, 0.15),
                "risk_on": np.ones(n, dtype=bool),
                "vol_ratio": np.ones(n),
            }
        )

    monkeypatch.setattr("rule.signals.build_hmm_regime_frame", _fake_hmm)
    sig = build_hmm_exhaustion_signal_frame(prices)
    assert not sig.empty
    assert {"date", "ticker", "score", "signal_side", "p_stop", "impulse_thr_pct"}.issubset(sig.columns)
    assert sig["ticker"].eq("SPY").all()
    assert sig["score"].between(0, 100).all()
