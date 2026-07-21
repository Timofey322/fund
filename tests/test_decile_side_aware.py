"""Side-aware decile gate tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.diagnostics.decile_audit import (
    decile_audit_by_ticker,
    decile_audit_for_ticker,
    tradeable_tickers_from_audit,
)


def _short_only_panel(n: int = 400) -> pd.DataFrame:
    proba_long = np.linspace(0.1, 0.9, n)
    proba_short = np.linspace(0.9, 0.1, n)
    fwd = np.where(proba_short > np.quantile(proba_short, 0.9), -0.004, 0.0002)
    return pd.DataFrame({
        "ticker": ["IMOEX"] * n,
        "ml_proba": proba_long,
        "ml_proba_short": proba_short,
        "fwd_ret_entry": fwd,
    })


def test_short_only_uses_short_proba_not_long():
    df = _short_only_panel()
    audit = decile_audit_for_ticker(df, "IMOEX", ret_col="fwd_ret_entry", min_top_decile_net_bps=5.0)
    assert audit.get("active_side") == "short"
    assert audit.get("proba_col") == "ml_proba_short"
    long_side = (audit.get("sides") or {}).get("long") or {}
    short_side = (audit.get("sides") or {}).get("short") or {}
    assert float(short_side.get("top_decile_net_bps", -999)) > float(
        long_side.get("top_decile_net_bps", -999)
    )


def test_short_only_tradeable_when_short_decile_clears_gate(monkeypatch):
    import config as cfg

    monkeypatch.setitem(cfg.FUSION_SIDE_POLICY, "IMOEX", "short_only")
    df = _short_only_panel()
    by_t = decile_audit_by_ticker(df, ret_col="fwd_ret_entry", min_top_decile_net_bps=5.0)
    tradeable = tradeable_tickers_from_audit(by_t, min_top_decile_net_bps=5.0)
    assert "IMOEX" in tradeable


def test_long_only_ignores_short_proba(monkeypatch):
    import config as cfg

    monkeypatch.setitem(cfg.FUSION_SIDE_POLICY, "TEST", "long_only")
    n = 400
    proba = np.linspace(0.1, 0.9, n)
    fwd = np.where(proba > np.quantile(proba, 0.9), 0.004, -0.0002)
    df = pd.DataFrame({
        "ticker": ["TEST"] * n,
        "ml_proba": proba,
        "ml_proba_short": 1.0 - proba,
        "fwd_ret_entry": fwd,
    })
    audit = decile_audit_for_ticker(df, "TEST", ret_col="fwd_ret_entry", min_top_decile_net_bps=5.0)
    assert audit.get("active_side") == "long"
    assert audit.get("tradeable") is True
