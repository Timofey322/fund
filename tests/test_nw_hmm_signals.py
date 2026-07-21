"""Unit tests for NW + HMM buy-only rule signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.naming import COL_PROB_HMM_IMPULSE, COL_PROB_HMM_MEAN_REVERT, COL_PROB_HMM_STRESS
from rule.config import RULE_HMM_GROWTH_MIN, RULE_NEUTRAL_SCORE, RULE_NW_TOUCH_MAX
from rule.nw_hmm_signals import (
    build_nw_hmm_buy_signal_frame,
    hmm_growth_probability,
    nw_envelope_frame,
    nw_hmm_buy_score,
)


def test_nw_envelope_produces_bands():
    idx = pd.bdate_range("2020-01-01", periods=200)
    close = pd.Series(100.0 + np.sin(np.linspace(0, 6, len(idx))) * 5, index=idx)
    env = nw_envelope_frame(close)
    assert not env.empty
    assert "nw_touches_lower" in env.columns
    valid = env.dropna(subset=["nw_upper", "nw_lower"])
    assert (valid["nw_upper"] >= valid["nw_lower"]).all()


def test_hmm_growth_rises_when_mean_revert_dominates():
    low = hmm_growth_probability(0.55, 0.25, 0.20, None, risk_on=True)
    high = hmm_growth_probability(0.15, 0.70, 0.15, 0.55, risk_on=True)
    assert high > low


def test_nw_hmm_buy_score_on_lower_touch_with_growth():
    score, side = nw_hmm_buy_score(
        env_pos=0.05,
        touches_lower=True,
        p_growth=0.65,
    )
    assert side == "buy_nw_hmm"
    assert score > 50.0


def test_nw_hmm_buy_score_neutral_above_band():
    score, side = nw_hmm_buy_score(
        env_pos=0.80,
        touches_lower=False,
        p_growth=0.90,
    )
    assert side == "neutral"
    assert score == RULE_NEUTRAL_SCORE


def test_nw_hmm_buy_score_neutral_without_hmm_growth():
    score, side = nw_hmm_buy_score(
        env_pos=0.02,
        touches_lower=True,
        p_growth=RULE_HMM_GROWTH_MIN - 0.05,
    )
    assert side == "neutral_hmm"
    assert score == RULE_NEUTRAL_SCORE


def test_nw_hmm_buy_score_never_sells():
    score, side = nw_hmm_buy_score(
        env_pos=0.90,
        touches_lower=False,
        p_growth=0.10,
    )
    assert score >= RULE_NEUTRAL_SCORE
    assert "sell" not in side


def test_build_nw_hmm_buy_signal_frame_with_mock_hmm(monkeypatch):
    idx = pd.bdate_range("2020-01-01", periods=220)
    rng = np.random.default_rng(0)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0002, 0.012, len(idx)))
    prices = pd.DataFrame({"SPY": close}, index=idx)

    def _fake_hmm(prices: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
        n = len(prices)
        return pd.DataFrame(
            {
                "date": prices.index,
                COL_PROB_HMM_IMPULSE: np.full(n, 0.20),
                COL_PROB_HMM_MEAN_REVERT: np.full(n, 0.60),
                COL_PROB_HMM_STRESS: np.full(n, 0.20),
                "risk_on": np.ones(n, dtype=bool),
                "vol_ratio": np.ones(n),
            }
        )

    monkeypatch.setattr("rule.nw_hmm_signals.build_hmm_regime_frame", _fake_hmm)
    sig = build_nw_hmm_buy_signal_frame(prices)
    assert not sig.empty
    assert {"date", "ticker", "score", "signal_side", "p_growth", "nw_touches_lower"}.issubset(
        sig.columns
    )
    assert sig["score"].between(0, 100).all()
    assert (sig["score"] >= RULE_NEUTRAL_SCORE).all()
