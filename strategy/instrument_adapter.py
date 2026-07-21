"""Per-instrument adapter registry — side policy, labels, costs, onboarding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import config as _cfg
from data_platform.universe import commission_bps_for_ticker, is_tradfi_symbol
from simulation.execution_costs import round_trip_cost_bps_for_ticker, slippage_bps_per_side
from strategy.side_policy import allowed_sides_for_ticker


@dataclass(frozen=True)
class InstrumentAdapter:
    """Unified per-symbol trading configuration."""

    symbol: str
    side_policy: str
    default_horizon: int
    default_label_type: str
    default_threshold_bps: float
    commission_bps_per_side: float
    slippage_bps_per_side: float
    round_trip_cost_bps: float
    min_rows: int
    asset_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side_policy": self.side_policy,
            "default_horizon": self.default_horizon,
            "default_label_type": self.default_label_type,
            "default_threshold_bps": self.default_threshold_bps,
            "commission_bps_per_side": self.commission_bps_per_side,
            "slippage_bps_per_side": self.slippage_bps_per_side,
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "min_rows": self.min_rows,
            "asset_class": self.asset_class,
        }


def _default_entry_spec(symbol: str) -> dict[str, Any]:
    from research.labels.trade import default_entry_spec, resolve_entry_spec

    try:
        return dict(resolve_entry_spec(symbol))
    except Exception:
        return dict(default_entry_spec(symbol))


def get_instrument_adapter(symbol: str) -> InstrumentAdapter:
    """Build adapter for one symbol from config + target cache."""
    sym = str(symbol).upper()
    spec = _default_entry_spec(sym)
    comm = float(commission_bps_for_ticker(sym))
    slip = float(slippage_bps_per_side(sym))
    min_rows = int(getattr(_cfg, "FUSION_PER_TICKER_MIN_ROWS", 200))
    asset = "tradfi" if is_tradfi_symbol(sym) else "crypto"
    return InstrumentAdapter(
        symbol=sym,
        side_policy=allowed_sides_for_ticker(sym),
        default_horizon=int(spec.get("horizon", getattr(_cfg, "FUSION_DEFAULT_ENTRY_HORIZON", 48))),
        default_label_type=str(spec.get("label_type", getattr(_cfg, "FUSION_DEFAULT_ENTRY_LABEL_TYPE", "triple_barrier"))),
        default_threshold_bps=float(spec.get("threshold_bps", getattr(_cfg, "FUSION_DEFAULT_ENTRY_THRESHOLD_BPS", 0.0))),
        commission_bps_per_side=comm,
        slippage_bps_per_side=slip,
        round_trip_cost_bps=float(round_trip_cost_bps_for_ticker(sym)),
        min_rows=min_rows,
        asset_class=asset,
    )


def instrument_registry(symbols: list[str]) -> dict[str, InstrumentAdapter]:
    """Registry for all symbols in a run."""
    return {str(s).upper(): get_instrument_adapter(s) for s in symbols}


def sync_side_policy_from_registry(symbols: list[str]) -> dict[str, str]:
    """Return effective side policy map (config overrides preserved)."""
    base = dict(getattr(_cfg, "FUSION_SIDE_POLICY", {}) or {})
    for sym in symbols:
        sym_u = str(sym).upper()
        if sym_u not in base:
            base[sym_u] = allowed_sides_for_ticker(sym_u)
    return base


def per_ticker_exposure_budget(tradeable_tickers: list[str]) -> dict[str, float]:
    """Inverse-vol exposure budget per live tradeable symbol."""
    from strategy.instrument_economics import inverse_vol_exposure_budget

    return inverse_vol_exposure_budget(tradeable_tickers)


def onboarding_checklist(symbol: str) -> list[str]:
    """Human-readable onboarding steps for a new instrument."""
    sym = str(symbol).upper()
    return [
        f"Add {sym} to universe + data cache",
        f"Run: python run.py target-opt --tickers {sym} --apply",
        "Smoke: 2–3 walk-forward folds with FUSION_WF_MAX_FOLDS=3",
        f"Pass: stitched top-decile net >= {getattr(_cfg, 'FUSION_MIN_TOP_DECILE_NET_BPS', 5.0)} bps",
        "Full pipeline run + verify decile gate tradeable=True",
    ]
