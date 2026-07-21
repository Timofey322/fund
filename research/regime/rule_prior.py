"""
Rule-based HMM regime prior — continuous formula, not if-else tables.

    P(regime | x) = softmax( L_regime(x) )

Features x = (risk_on, vol_ratio, log_ret, ret_z) are mapped to bounded signals,
then combined with signed weights per regime (see RegimePriorFormula).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import math
from dataclasses import dataclass

from config import (
    HMM_BLEND_RALLY_FLOOR,
    HMM_BLEND_STRESS_FLOOR,
    HMM_STRESS_FLOOR_LOSS_CRYPTO,
    HMM_STRESS_FLOOR_VOL_CRYPTO,
    HMM_FLAT_RET_SCALE_CRYPTO,
    HMM_MAX_TREND_IF_RISK_OFF,
    HMM_MAX_TREND_ON_LOSS_MONTH,
    HMM_MAX_TREND_RECOVERY_RISK_OFF,
    HMM_RULE_BLEND,
    HMM_RULE_BLEND_CRYPTO,
    HMM_VOL_CRISIS_RATIO,
    HMM_VOL_CRISIS_RATIO_CRYPTO,
    IS_INTRADAY,
    REGIME_TICKERS,
    CRYPTO_UNIVERSE,
)
from common.naming import HMM_STRESS, HMM_MEAN_REVERT, HMM_REGIMES, HMM_IMPULSE


def _safe_float(x: float, default: float = 0.0) -> float:
    return default if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    s = sum(float(probs.get(r, 0.0)) for r in HMM_REGIMES)
    if s <= 0:
        u = 1.0 / len(HMM_REGIMES)
        return {r: u for r in HMM_REGIMES}
    return {r: float(probs.get(r, 0.0)) / s for r in HMM_REGIMES}


@dataclass(frozen=True)
class RegimeObservables:
    risk_on: bool
    vol_ratio: float
    log_ret: float
    ret_z: float

    @classmethod
    def from_raw(
        cls,
        risk_on: bool,
        vol_ratio: float,
        log_ret: float,
        ret_z: float,
    ) -> RegimeObservables:
        return cls(
            risk_on=bool(risk_on),
            vol_ratio=_safe_float(vol_ratio, 1.0),
            log_ret=_safe_float(log_ret, 0.0),
            ret_z=_safe_float(ret_z, 0.0),
        )


@dataclass(frozen=True)
class RegimePriorFormula:
    """
    Softmax logits: L_k = bias_k + Σ w_k,i * feature_i.

    Features (see regime_features):
      risk_on, vol_stress, ret_signal, z_signal, flat_signal, loss_signal
    """

    bias_trend: float = -0.35
    bias_range: float = 0.10
    bias_crisis: float = -0.55

    # trend: risk-on rally, penalized by vol stress / risk-off
    w_trend_risk: float = 2.20
    w_trend_ret: float = 2.80
    w_trend_z: float = 1.40
    w_trend_vol: float = -2.50
    w_trend_risk_off: float = -3.20

    # range: flat month + calm vol
    w_range_flat: float = 2.40
    w_range_calm_vol: float = 0.80
    w_range_ret_pen: float = -1.20
    w_range_z_pen: float = -0.60

    # crisis: risk-off, vol spike, drawdown (recovery bounce below SMA reduces crisis)
    w_crisis_risk_off: float = 2.40
    w_crisis_vol: float = 2.20
    w_crisis_loss: float = 2.40
    w_crisis_z: float = 1.00
    w_crisis_recovery: float = -2.20

    temperature: float = 1.0


# Softer crisis / wider range for 24/7 crypto
CRYPTO_PRIOR_FORMULA = RegimePriorFormula(
    bias_range=0.22,
    bias_crisis=-0.75,
    w_range_flat=2.90,
    w_crisis_vol=1.35,
    w_crisis_loss=1.85,
    w_trend_ret=3.10,
)


def is_crypto_regime() -> bool:
    """True when regime universe is BTC/ETH (or subset of CRYPTO_UNIVERSE)."""
    return bool(CRYPTO_UNIVERSE) and set(REGIME_TICKERS).issubset(set(CRYPTO_UNIVERSE))


def is_bar_hmm_mode() -> bool:
    """5Min bar HMM (hour baseline) vs daily/monthly regime features."""
    return IS_INTRADAY and getattr(__import__("config", fromlist=["HMM_FREQUENCY"]), "HMM_FREQUENCY", "daily") == "bar"


def _bar_rule_scales(crypto: bool) -> dict[str, float]:
    """Thresholds for one 5Min bar (import config lazily for test overrides)."""
    import config as cfg

    if crypto and is_bar_hmm_mode():
        return {
            "flat": getattr(cfg, "HMM_BAR_FLAT_RET_SCALE_CRYPTO", 0.002),
            "ret_signal": getattr(cfg, "HMM_BAR_RET_SIGNAL_SCALE_CRYPTO", 0.004),
            "loss": getattr(cfg, "HMM_BAR_LOSS_SCALE_CRYPTO", 0.005),
            "rally_ret": getattr(cfg, "HMM_BAR_RALLY_LOG_RET_CRYPTO", 0.006),
            "rally_z": getattr(cfg, "HMM_BAR_RALLY_RET_Z_CRYPTO", 0.75),
            "loss_day": -getattr(cfg, "HMM_BAR_LOSS_SCALE_CRYPTO", 0.005),
        }
    return {
        "flat": HMM_FLAT_RET_SCALE_CRYPTO if crypto else 0.025,
        "ret_signal": 0.055 if crypto else 0.04,
        "loss": 0.08 if crypto else 0.06,
        "rally_ret": 0.05,
        "rally_z": 0.75,
        "loss_day": -0.06 if crypto else -0.05,
    }


def regime_features(obs: RegimeObservables, *, crypto: bool = False) -> dict[str, float]:
    """Bounded transforms of raw market observables."""
    risk = 1.0 if obs.risk_on else 0.0
    vol_ratio_thr = HMM_VOL_CRISIS_RATIO_CRYPTO if crypto else HMM_VOL_CRISIS_RATIO
    scales = _bar_rule_scales(crypto)
    flat_scale = scales["flat"]
    vol_stress = _clip((obs.vol_ratio - 1.0) / max(vol_ratio_thr - 1.0, 1e-6), 0.0, 2.0)
    ret_signal = math.tanh(obs.log_ret / scales["ret_signal"])
    z_signal = math.tanh(obs.ret_z)
    flat_signal = math.exp(-abs(obs.log_ret) / flat_scale) * math.exp(-abs(obs.ret_z) / 0.6)
    loss_signal = _clip(-obs.log_ret / scales["loss"], 0.0, 2.0)
    return {
        "risk_on": risk,
        "vol_stress": vol_stress,
        "ret_signal": ret_signal,
        "z_signal": z_signal,
        "flat_signal": flat_signal,
        "loss_signal": loss_signal,
    }


def regime_logits(obs: RegimeObservables, formula: RegimePriorFormula | None = None) -> dict[str, float]:
    """L_regime = bias + weighted feature sum."""
    f = formula or (CRYPTO_PRIOR_FORMULA if is_crypto_regime() else RegimePriorFormula())
    x = regime_features(obs, crypto=is_crypto_regime())
    risk_off = 1.0 - x["risk_on"]
    calm_vol = 1.0 - _clip(x["vol_stress"], 0.0, 1.0)

    return {
        HMM_IMPULSE: (
            f.bias_trend
            + f.w_trend_risk * x["risk_on"]
            + f.w_trend_ret * x["ret_signal"]
            + f.w_trend_z * x["z_signal"]
            + f.w_trend_vol * x["vol_stress"]
            + f.w_trend_risk_off * risk_off
        ),
        HMM_MEAN_REVERT: (
            f.bias_range
            + f.w_range_flat * x["flat_signal"]
            + f.w_range_calm_vol * calm_vol
            + f.w_range_ret_pen * abs(x["ret_signal"])
            + f.w_range_z_pen * abs(x["z_signal"])
        ),
        HMM_STRESS: (
            f.bias_crisis
            + f.w_crisis_risk_off * risk_off
            + f.w_crisis_vol * x["vol_stress"]
            + f.w_crisis_loss * x["loss_signal"]
            + f.w_crisis_z * max(0.0, -x["z_signal"])
            + f.w_crisis_recovery * max(0.0, x["ret_signal"]) * risk_off
        ),
    }


def softmax_logits(logits: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    t = max(temperature, 1e-6)
    scaled = {k: v / t for k, v in logits.items()}
    m = max(scaled.values())
    expv = {k: math.exp(v - m) for k, v in scaled.items()}
    s = sum(expv.values()) or 1.0
    return {k: expv[k] / s for k in HMM_REGIMES}


def rule_regime_prior(
    risk_on: bool,
    vol_ratio: float,
    log_ret: float,
    ret_z: float,
    formula: RegimePriorFormula | None = None,
) -> dict[str, float]:
    """P(regime | observables) from closed-form softmax."""
    f = formula or RegimePriorFormula()
    obs = RegimeObservables.from_raw(risk_on, vol_ratio, log_ret, ret_z)
    logits = regime_logits(obs, f)
    return softmax_logits(logits, f.temperature)


def effective_rule_blend(obs: RegimeObservables, base_blend: float | None = None) -> float:
    """Raise λ only on unambiguous stress or rally — not every risk_off day."""
    crypto = is_crypto_regime()
    blend = (HMM_RULE_BLEND_CRYPTO if crypto else HMM_RULE_BLEND) if base_blend is None else base_blend
    vol_thr = HMM_VOL_CRISIS_RATIO_CRYPTO if crypto else HMM_VOL_CRISIS_RATIO
    scales = _bar_rule_scales(crypto)
    loss_thr = scales["loss_day"]
    stress = (
        obs.log_ret <= loss_thr
        or obs.vol_ratio > vol_thr
        or (not obs.risk_on and obs.log_ret < 0)
    )
    rally = (
        obs.log_ret >= scales["rally_ret"]
        and obs.ret_z >= scales["rally_z"]
        and obs.risk_on
        and obs.vol_ratio < 1.15
    )
    if stress:
        blend = max(blend, HMM_BLEND_STRESS_FLOOR)
    if rally:
        blend = max(blend, HMM_BLEND_RALLY_FLOOR)
    return blend


def _redistribute(from_reg: str, amount: float, probs: dict[str, float], to_crisis: float = 0.7) -> None:
    if amount <= 0:
        return
    probs[from_reg] = max(0.0, probs[from_reg] - amount)
    probs[HMM_STRESS] += amount * to_crisis
    probs[HMM_MEAN_REVERT] += amount * (1.0 - to_crisis)


def apply_economic_caps(probs: dict[str, float], obs: RegimeObservables) -> dict[str, float]:
    """Hard portfolio sanity constraints — only on clear stress, not recovery bounces."""
    crypto = is_crypto_regime()
    vol_thr = HMM_VOL_CRISIS_RATIO_CRYPTO if crypto else HMM_VOL_CRISIS_RATIO
    crisis_floor_loss = HMM_STRESS_FLOOR_LOSS_CRYPTO if crypto else 0.28
    crisis_floor_vol = HMM_STRESS_FLOOR_VOL_CRYPTO if crypto else 0.38
    scales = _bar_rule_scales(crypto)
    loss_bar = scales["loss_day"]
    rally_ret = scales["rally_ret"]
    rally_z = scales["rally_z"]
    out = dict(probs)

    if not obs.risk_on:
        trend_cap = (
            HMM_MAX_TREND_RECOVERY_RISK_OFF
            if obs.log_ret >= 0
            else HMM_MAX_TREND_IF_RISK_OFF
        )
        _redistribute(HMM_IMPULSE, max(0.0, out[HMM_IMPULSE] - trend_cap), out)
    if obs.log_ret <= loss_bar:
        _redistribute(HMM_IMPULSE, max(0.0, out[HMM_IMPULSE] - HMM_MAX_TREND_ON_LOSS_MONTH), out)
        if out[HMM_STRESS] < crisis_floor_loss:
            need = crisis_floor_loss - out[HMM_STRESS]
            _redistribute(HMM_IMPULSE, min(need, out[HMM_IMPULSE] * 0.5), out)
    if obs.vol_ratio > vol_thr:
        _redistribute(HMM_IMPULSE, max(0.0, out[HMM_IMPULSE] - 0.15), out)
        if out[HMM_STRESS] < crisis_floor_vol:
            need = crisis_floor_vol - out[HMM_STRESS]
            _redistribute(HMM_MEAN_REVERT, min(need, out[HMM_MEAN_REVERT] * 0.6), out, to_crisis=1.0)
    if obs.log_ret >= rally_ret and obs.ret_z >= rally_z and obs.risk_on and obs.vol_ratio < 1.15:
        _redistribute(HMM_MEAN_REVERT, max(0.0, out[HMM_MEAN_REVERT] - 0.30), out, to_crisis=0.15)
        if out[HMM_IMPULSE] < 0.40:
            need = 0.40 - out[HMM_IMPULSE]
            take = min(need, out[HMM_MEAN_REVERT])
            out[HMM_MEAN_REVERT] -= take
            out[HMM_IMPULSE] += take

    return normalize_probs(out)


def fuse_hmm_probs(
    hmm_probs: dict[str, float],
    *,
    risk_on: bool,
    vol_ratio: float,
    log_ret: float,
    ret_z: float,
    rule_blend: float | None = None,
) -> dict[str, float]:
    """
    Fuse HMM filter output with rule prior:

        P* = normalize( (1-λ) · P_hmm + λ · P_rule )
        P  = apply_economic_caps(P*, observables)
    """
    obs = RegimeObservables.from_raw(risk_on, vol_ratio, log_ret, ret_z)
    blend = effective_rule_blend(obs, rule_blend)
    prior = rule_regime_prior(obs.risk_on, obs.vol_ratio, obs.log_ret, obs.ret_z)
    raw = normalize_probs(hmm_probs)
    mixed = {r: (1.0 - blend) * raw[r] + blend * prior[r] for r in HMM_REGIMES}
    return apply_economic_caps(normalize_probs(mixed), obs)
