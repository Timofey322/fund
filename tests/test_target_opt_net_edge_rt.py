"""Target-opt economics must use RT costs, not TP floor."""

from __future__ import annotations

import numpy as np

from strategy.target_opt import _top_decile_net_edge_bps


def test_top_decile_net_edge_uses_round_trip_not_tp_floor():
    # Strong ranking: top decile mean fwd ≈ +30 bps gross.
    n = 200
    proba = np.linspace(0.0, 1.0, n)
    fwd = np.full(n, -0.001)  # -10 bps
    fwd[-20:] = 0.003  # +30 bps on top proba

    # commission 0.5 + slip 2.0 → RT = 5.0 bps (tradfi-like)
    net = _top_decile_net_edge_bps(proba, fwd, commission_bps=0.5, slippage_bps=2.0)
    assert net > 20.0, f"expected ~25 bps after RT, got {net}"

    # Old bug: subtracting TP floor (~47.5) would flip this negative.
    assert net > 0.0
