"""Instrument adapter registry tests."""

from __future__ import annotations

import config as cfg
from strategy.instrument_adapter import (
    get_instrument_adapter,
    instrument_registry,
    onboarding_checklist,
    per_ticker_exposure_budget,
)


def test_registry_has_side_policy():
    reg = instrument_registry(["NASDAQ", "IMOEX"])
    assert reg["NASDAQ"].side_policy == "long_only"
    assert reg["IMOEX"].side_policy == "short_only"


def test_per_ticker_exposure_budget_inverse_vol(monkeypatch):
    monkeypatch.setattr(cfg, "FUSION_PER_TICKER_EXPOSURE_BUDGET", True)
    monkeypatch.setattr(cfg, "FUSION_PER_TICKER_MAX_WEIGHT", 0.9)
    from strategy.instrument_economics import inverse_vol_exposure_budget

    budget = inverse_vol_exposure_budget(
        ["NASDAQ", "SP500"],
        vol_by_ticker={"NASDAQ": 0.20, "SP500": 0.10},
    )
    assert len(budget) == 2
    assert budget["SP500"] > budget["NASDAQ"]
    assert abs(sum(budget.values()) - 1.0) < 1e-9


def test_onboarding_checklist_mentions_gate():
    steps = onboarding_checklist("TEST")
    assert any("top-decile" in s.lower() for s in steps)


def test_exposure_budget_disabled_returns_unity(monkeypatch):
    monkeypatch.setattr(cfg, "FUSION_PER_TICKER_EXPOSURE_BUDGET", False)
    budget = per_ticker_exposure_budget(["A", "B"])
    assert budget["A"] == 1.0
    assert budget["B"] == 1.0


def test_adapter_round_trip_positive():
    ad = get_instrument_adapter("NASDAQ")
    assert ad.round_trip_cost_bps > 0
    assert ad.default_horizon > 0
