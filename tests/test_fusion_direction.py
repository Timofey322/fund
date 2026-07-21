"""Tests for long/short fusion scoring."""

from __future__ import annotations

import numpy as np

from strategy.fusion_direction import fusion_signed_scores, signed_ml_edges


def test_signed_scores_long_dominates_high_proba_long():
    pl = np.array([0.8, 0.2])
    ps = np.array([0.2, 0.8])
    imp = np.array([0.5, 0.5])
    scores, sides = fusion_signed_scores(pl, ps, imp, baseline=0.5, buy_threshold=55.0)
    assert scores[0] > 50.0
    assert scores[1] < 50.0
    assert sides[0] == 1
    assert sides[1] == -1


def test_signed_scores_symmetric_around_fifty():
    pl = np.array([0.55, 0.45])
    ps = np.array([0.45, 0.55])
    imp = np.array([0.3, 0.3])
    scores, _ = fusion_signed_scores(pl, ps, imp, baseline=0.5, buy_threshold=55.0)
    assert scores[0] > 50.0
    assert scores[1] < 50.0
    assert abs(scores[0] + scores[1] - 100.0) < 1e-6


def test_signed_edges_non_negative():
    long_e, short_e = signed_ml_edges(np.array([0.7]), np.array([0.6]), 0.5)
    assert long_e[0] >= 0.0
    assert short_e[0] >= 0.0
