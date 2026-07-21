"""Optuna edge bounds must respect panel ceiling from search_space."""

from __future__ import annotations

from strategy.threshold_opt import _edge_bounds_from_search_space, _search_space_spec


def test_params_from_trial_uses_panel_capped_edge_spec():
    search_space = _search_space_spec(
        calibrated_edge=5.0,
        commission_bps=1.1,
        panel_max_edge_bps=6.2,
    )
    lo, hi = _edge_bounds_from_search_space(search_space, commission_bps=1.1, calibrated_edge=5.0)
    assert hi <= 6.5
    assert lo <= hi
    spec_hi = float(search_space["params"]["min_expected_edge_bps"]["high"])
    assert hi == spec_hi


def test_edge_bounds_from_search_space_falls_back_without_spec():
    lo, hi = _edge_bounds_from_search_space({}, commission_bps=1.1, calibrated_edge=5.0)
    assert lo < hi
