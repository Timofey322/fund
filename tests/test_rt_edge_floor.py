"""RT-aware edge floor and abs-edge calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.entry_signals import edge_floor_bps, round_trip_cost_bps
from strategy.edge_gate import heuristic_gate_floor_bps, panel_abs_edge_stats


def test_full_round_trip_floor_covers_rt_plus_buffer(monkeypatch):
    monkeypatch.setattr("config.FUSION_EDGE_BUFFER_BPS", 2.0)
    comm, slip = 0.5, 1.5
    floor = edge_floor_bps(comm, slip, mode="full_round_trip", buffer_bps=2.0)
    assert floor == round_trip_cost_bps(comm, slip) + 2.0
    assert floor >= 2.0 * (comm + slip)


def test_heuristic_gate_uses_full_round_trip_when_configured(monkeypatch):
    monkeypatch.setattr("config.FUSION_EDGE_GATE_FLOOR_MODE", "full_round_trip")
    monkeypatch.setattr("config.FUSION_EDGE_BUFFER_BPS", 2.0)
    floor = heuristic_gate_floor_bps(0.5, 1.5)
    assert abs(floor - (2.0 * (0.5 + 1.5) + 2.0)) < 1e-9


def test_panel_abs_edge_stats_uses_magnitude():
    edges = np.array([10.0, -12.0, 3.0, -4.0, np.nan])
    mx, q65, n = panel_abs_edge_stats(edges)
    assert n == 4
    assert mx == 12.0
    assert q65 > 0
