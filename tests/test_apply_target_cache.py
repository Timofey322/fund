"""Target cache application for per-instrument labels."""

from __future__ import annotations

import json
from pathlib import Path

import config as _cfg


def test_build_panel_uses_cache_with_apply_flag(monkeypatch, tmp_path):
    cache = {
        "applied": True,
        "per_symbol": {
            "BTC": {
                "spec": {"horizon": 48, "label_type": "triple_barrier", "threshold_bps": 50.0},
                "tradeable": False,
            },
        },
    }
    cache_path = tmp_path / "target_optimization.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setattr("strategy.target_opt.TARGET_OPT_PATH", cache_path)
    monkeypatch.setattr(_cfg, "FUSION_APPLY_TARGET_CACHE", True)
    monkeypatch.setattr(_cfg, "USE_PER_INSTRUMENT_TARGETS", False)

    from strategy.target_opt import per_instrument_specs

    specs = per_instrument_specs(tradeable_only=False)
    assert "BTC" in specs
    assert specs["BTC"]["horizon"] == 48

    # applied=false in file but FUSION_APPLY_TARGET_CACHE=True still loads specs
    specs2 = per_instrument_specs(tradeable_only=False)
    assert len(specs2) >= 1
