"""Instrument-relative economics: floors, SQ vote, inverse-vol budgets."""

from __future__ import annotations

import math

import config as cfg
from strategy.instrument_economics import (
    economics_floor_bps,
    filter_tradeable_by_signal_quality_v2,
    inverse_vol_exposure_budget,
    preferred_trade_side,
    soft_size_cv_band_bps,
    vol_scale,
)


def test_economics_floor_scales_with_vol(monkeypatch):
    monkeypatch.setattr(cfg, "FUSION_ECONOMICS_OVER_RT_MULT", 1.25)
    monkeypatch.setattr(cfg, "FUSION_ECONOMICS_VOL_EXP", 0.5)

    low = economics_floor_bps("NASDAQ", vol_ann=0.12)
    high = economics_floor_bps("NASDAQ", vol_ann=0.36)
    assert high > low
    assert low > 0
    assert math.isfinite(high)


def test_soft_size_band_is_relative_to_floor(monkeypatch):
    monkeypatch.setattr(cfg, "FUSION_SOFT_SIZE_CV_LO_OVER_FLOOR", 2.0)
    floor = economics_floor_bps("SBER", vol_ann=0.25)
    lo, hi = soft_size_cv_band_bps("SBER", vol_ann=0.25)
    assert hi == floor
    assert lo == -2.0 * floor


def test_vol_scale_clipped(monkeypatch):
    monkeypatch.setattr(cfg, "FUSION_VOL_SCALE_CLIP_LO", 0.5)
    monkeypatch.setattr(cfg, "FUSION_VOL_SCALE_CLIP_HI", 2.5)
    # Force tradfi ref via monkeypatch of helper path: pass extreme vols.
    assert vol_scale("NASDAQ", vol_ann=0.01) == 0.5
    assert vol_scale("NASDAQ", vol_ann=10.0) == 2.5


def test_sq_majority_keeps_stitched_passer(monkeypatch):
    monkeypatch.setattr(cfg, "FUSION_GATE_REQUIRE_SIGNAL_QUALITY", True)
    monkeypatch.setattr(cfg, "FUSION_SQ_MIN_PASS_RATE", 0.40)

    # 2/3 folds OK → pass_rate 0.667 ≥ 0.40 even if last fold fails.
    folds = [
        {
            "threshold_optimization": {
                "cv": {
                    "by_ticker": {
                        "SBER": {
                            "signal_quality_ok": True,
                            "holdout_top_decile_net_bps": 8.0,
                        }
                    }
                }
            }
        },
        {
            "threshold_optimization": {
                "cv": {
                    "by_ticker": {
                        "SBER": {
                            "signal_quality_ok": True,
                            "holdout_top_decile_net_bps": 6.0,
                        }
                    }
                }
            }
        },
        {
            "threshold_optimization": {
                "cv": {
                    "by_ticker": {
                        "SBER": {
                            "signal_quality_ok": False,
                            "holdout_top_decile_net_bps": -1.0,
                        }
                    }
                }
            }
        },
    ]
    kept, removed, detail = filter_tradeable_by_signal_quality_v2(folds, ["SBER"])
    assert kept == ["SBER"]
    assert removed == []
    assert abs(detail["SBER"]["pass_rate"] - 2 / 3) < 1e-3
    assert detail["SBER"]["ok"] is True


def test_sq_mean_holdout_rescues_low_pass_rate(monkeypatch):
    monkeypatch.setattr(cfg, "FUSION_GATE_REQUIRE_SIGNAL_QUALITY", True)
    monkeypatch.setattr(cfg, "FUSION_SQ_MIN_PASS_RATE", 0.80)

    fixed_floor = 4.0

    def _floor(_ticker, **_kwargs):
        return fixed_floor

    monkeypatch.setattr(
        "strategy.instrument_economics.economics_floor_bps",
        _floor,
    )
    folds = [
        {
            "threshold_optimization": {
                "cv": {
                    "by_ticker": {
                        "SBER": {
                            "signal_quality_ok": False,
                            "holdout_top_decile_net_bps": 5.0,
                        }
                    }
                }
            }
        },
        {
            "threshold_optimization": {
                "cv": {
                    "by_ticker": {
                        "SBER": {
                            "signal_quality_ok": False,
                            "holdout_top_decile_net_bps": 6.0,
                        }
                    }
                }
            }
        },
    ]
    kept, removed, detail = filter_tradeable_by_signal_quality_v2(folds, ["SBER"])
    assert detail["SBER"]["pass_rate"] == 0.0
    assert detail["SBER"]["mean_holdout_net_bps"] == 5.5
    assert kept == ["SBER"]
    assert removed == []


def test_sq_removes_structurally_weak(monkeypatch):
    monkeypatch.setattr(cfg, "FUSION_GATE_REQUIRE_SIGNAL_QUALITY", True)
    monkeypatch.setattr(cfg, "FUSION_SQ_MIN_PASS_RATE", 0.40)

    folds = [
        {
            "threshold_optimization": {
                "cv": {
                    "by_ticker": {
                        "GAZP": {
                            "signal_quality_ok": False,
                            "holdout_top_decile_net_bps": -5.0,
                        }
                    }
                }
            }
        },
        {
            "threshold_optimization": {
                "cv": {
                    "by_ticker": {
                        "GAZP": {
                            "signal_quality_ok": False,
                            "holdout_top_decile_net_bps": -4.0,
                        }
                    }
                }
            }
        },
        {
            "threshold_optimization": {
                "cv": {
                    "by_ticker": {
                        "GAZP": {
                            "signal_quality_ok": True,
                            "holdout_top_decile_net_bps": 1.0,
                        }
                    }
                }
            }
        },
    ]
    kept, removed, detail = filter_tradeable_by_signal_quality_v2(folds, ["GAZP"])
    assert detail["GAZP"]["pass_rate"] < 0.40
    assert "GAZP" in removed
    assert kept == []


def test_soft_size_keeps_sq_v2_passer():
    from strategy.soft_sizing import soft_size_block_reason, soft_size_multiplier

    pol = {
        "ticker": "SBER",
        "sq_soft_keep": True,
        "sq_pass_rate": 0.66,
        "signal_quality_ok": False,
        "holdout_top_decile_net_bps": -1.0,
    }
    assert soft_size_block_reason(pol) is None
    assert soft_size_multiplier(pol) > 0.0


def test_inverse_vol_respects_max_weight(monkeypatch):
    monkeypatch.setattr(cfg, "FUSION_PER_TICKER_EXPOSURE_BUDGET", True)
    monkeypatch.setattr(cfg, "FUSION_PER_TICKER_MAX_WEIGHT", 0.35)
    budget = inverse_vol_exposure_budget(
        ["A", "B", "C"],
        vol_by_ticker={"A": 0.05, "B": 0.40, "C": 0.40},
    )
    assert all(v <= 0.35 + 1e-9 for v in budget.values())
    assert abs(sum(budget.values()) - 1.0) < 1e-6


def test_preferred_side_picks_clearing_floor(monkeypatch):
    monkeypatch.setattr(cfg, "FUSION_ECONOMICS_OVER_RT_MULT", 1.0)
    monkeypatch.setattr(cfg, "FUSION_ECONOMICS_VOL_EXP", 0.0)
    floor = economics_floor_bps("IMOEX", vol_ann=0.20)
    side = preferred_trade_side(
        {
            "long": {"top_decile_net_bps": floor - 1.0, "monotonic": True},
            "short": {"top_decile_net_bps": floor + 2.0, "monotonic": True},
        },
        ticker="IMOEX",
    )
    assert side == "short"


def test_desk_go_no_go_union_tradeable():
    from reporting.desk_reports import desk_go_no_go

    report = {
        "impulse_optimization": {
            "best": {
                "disable_trading": True,
                "decile_gate_blocked": True,
                "tradeable_tickers": ["SBER"],
            }
        },
        "decile_audit": {
            "tradeable": True,
            "tradeable_tickers": ["SBER"],
            "reasons": ["portfolio_top_weak"],
            "top_decile_net_bps": -3,
        },
        "backtest_walk_forward_oos": {"total_return_pct": 0},
        "ml_diagnostics": {},
        "walk_forward_folds": [],
    }
    out = desk_go_no_go(report)
    assert out["tradeable"] is True
    assert out["tradeable_tickers"] == ["SBER"]
    assert out["disable_trading"] is False
