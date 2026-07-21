"""
Gaussian Hidden Markov Model (Ang & Bekaert, 2002).

Скрытая марковская цепь S_t ∈ {bull, flat, crisis}:
  - Наблюдения y_t = (return, vol) ~ N(μ_{S_t}, Σ_{S_t})
  - Переходы P(S_{t+1}=j | S_t=i) = A_ij  (оценка EM / Baum-Welch)
  - Фильтр: α_t(k) = P(S_t=k | y_1:t)  (forward algorithm, без look-ahead)
  - Fusion: смешивание α_t с rule-prior (risk_on, vol_ratio, абсолютный ret)
  - Параметры score: Σ_k P(regime=k) × profile_k  (мягкий вывод)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import math

import numpy as np
import pandas as pd

import config as _cfg
from config import (
    HMM_ROLLING_MIN,
    HMM_ROLLING_PERIODS,
    HMM_TRAIN_PERIODS,
    IS_INTRADAY,
    REGIME_TICKER,
    REGIME_TICKERS,
    REGIME_TICKER_WEIGHTS,
    REBALANCE_FREQ,
    REOPTIMIZE_EVERY,
    SMA_LONG,
    TRADING_DAYS_PER_YEAR,
)
from common.naming import (
    COL_HMM_DOMINANT,
    COL_PROB_HMM_STRESS,
    COL_PROB_HMM_MEAN_REVERT,
    COL_PROB_HMM_IMPULSE,
    COL_WEIGHT_MEAN_REV,
    COL_WEIGHT_RISK,
    COL_WEIGHT_TREND,
    HMM_STRESS,
    HMM_MEAN_REVERT,
    HMM_REGIMES,
    HMM_IMPULSE,
)
from research.regime.rule_prior import fuse_hmm_probs
from research.regime.profiles import REGIMES
from operations.scoring import ann_vol, log_returns, vol_regime_ratio

HMM_TRAIN_MONTHS = HMM_TRAIN_PERIODS  # alias: rebalance-period units
HMM_N_STATES = 3
HMM_EM_ITERS = 20

# Cache last fitted model metadata for reports
LAST_HMM_META: dict = {}


def _softmax(logp: np.ndarray) -> np.ndarray:
    logp = logp - logp.max()
    p = np.exp(logp)
    return p / p.sum()


def _logsumexp(x: np.ndarray) -> float:
    m = x.max()
    return m + math.log(np.exp(x - m).sum())


def _logsumexp_axis(x: np.ndarray, axis: int) -> np.ndarray:
    """Vectorized logsumexp collapsing one axis (matches scalar _logsumexp)."""
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


class GaussianHMM:
    """СММ с Gaussian emissions: π, A, (μ_k, Σ_k)."""

    def __init__(self, n_states: int = 3, n_iter: int = 20, seed: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)
        self.pi = np.ones(n_states) / n_states
        self.A = np.full((n_states, n_states), 1.0 / n_states)
        self.means: np.ndarray | None = None
        self.covs: np.ndarray | None = None
        self.state_map: dict[int, str] = {}

    def _log_gaussian(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        d = len(x)
        cov = cov + np.eye(d) * 1e-6
        diff = x - mean
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return -1e9
        inv = np.linalg.inv(cov)
        return -0.5 * (d * math.log(2 * math.pi) + logdet + diff @ inv @ diff)

    def _emission_log_probs_matrix(self, X: np.ndarray) -> np.ndarray:
        """Batched Gaussian log-densities: returns (T, n_states).

        Identical numerics to per-row ``_log_gaussian`` (same 1e-6 jitter,
        slogdet, inverse), but the expensive T loop is replaced by vectorized
        Mahalanobis distances; the tiny state loop (K<=4) stays.
        """
        assert self.means is not None and self.covs is not None
        X = np.atleast_2d(np.asarray(X, dtype=float))
        T, D = X.shape
        K = self.n_states
        logB = np.empty((T, K))
        const = D * math.log(2 * math.pi)
        for k in range(K):
            cov = self.covs[k] + np.eye(D) * 1e-6
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                logB[:, k] = -1e9
                continue
            inv = np.linalg.inv(cov)
            diff = X - self.means[k]
            maha = np.einsum("ti,ij,tj->t", diff, inv, diff)
            logB[:, k] = -0.5 * (const + logdet + maha)
        return logB

    def _emission_log_probs(self, x: np.ndarray) -> np.ndarray:
        return self._emission_log_probs_matrix(np.asarray(x, dtype=float)[None, :])[0]

    def fit(self, X: np.ndarray) -> None:
        """Baum-Welch EM: оценка π, A, μ, Σ."""
        T, D = X.shape
        if T < 24:
            self.means = np.zeros((self.n_states, D))
            self.covs = np.array([np.eye(D) for _ in range(self.n_states)])
            self._assign_state_names()
            return

        idx = self.rng.choice(T, size=self.n_states, replace=False)
        self.means = X[idx].copy()
        self.covs = np.array([np.cov(X.T) + np.eye(D) * 1e-3 for _ in range(self.n_states)])

        K = self.n_states
        for _ in range(self.n_iter):
            logB = self._emission_log_probs_matrix(X)  # (T, K)
            logA = np.log(self.A + 1e-12)
            logpi = np.log(self.pi + 1e-12)

            # Forward / backward: the t-recursion is sequential, but each step is
            # now a single vectorized logsumexp over states (no inner state loop).
            alpha = np.empty((T, K))
            alpha[0] = logpi + logB[0]
            for t in range(1, T):
                alpha[t] = logB[t] + _logsumexp_axis(alpha[t - 1][:, None] + logA, axis=0)
            beta = np.zeros((T, K))
            for t in range(T - 2, -1, -1):
                beta[t] = _logsumexp_axis(logA + (logB[t + 1] + beta[t + 1])[None, :], axis=1)

            # gamma = softmax(alpha + beta) per row
            g = alpha + beta
            g -= g.max(axis=1, keepdims=True)
            gamma = np.exp(g)
            gamma /= gamma.sum(axis=1, keepdims=True)

            # xi[t, i, j] ∝ alpha[t,i] + logA[i,j] + logB[t+1,j] + beta[t+1,j]
            log_xi = alpha[:-1, :, None] + logA[None, :, :] + (logB[1:] + beta[1:])[:, None, :]
            flat = log_xi.reshape(T - 1, -1)
            flat -= flat.max(axis=1, keepdims=True)
            xi = np.exp(flat)
            xi /= xi.sum(axis=1, keepdims=True)
            xi = xi.reshape(T - 1, K, K)

            self.pi = gamma[0]
            self.A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
            self.A = self.A / self.A.sum(axis=1, keepdims=True)

            for k in range(K):
                w = gamma[:, k]
                wsum = w.sum()
                if wsum < 1e-8:
                    continue
                self.means[k] = (w[:, None] * X).sum(axis=0) / wsum
                diff = X - self.means[k]
                self.covs[k] = np.einsum("t,ti,tj->ij", w, diff, diff) / wsum + np.eye(D) * 1e-3

        self._assign_state_names()

    def _assign_state_names(self) -> None:
        assert self.means is not None
        # means[:, 0] is the directional dimension (trend_z when present), so the
        # ordering reflects persistent drift rather than single-bar return noise.
        ret = self.means[:, 0]
        vol = self.means[:, 1] if self.means.shape[1] > 1 else np.zeros(self.n_states)
        order_ret = np.argsort(ret)
        
        # For 3 states: map to TREND/RANGE/CRISIS
        if self.n_states == 3:
            low, mid, high = int(order_ret[0]), int(order_ret[1]), int(order_ret[2])
            stress_s, mr_s = (low, mid) if vol[low] >= vol[mid] else (mid, low)
            self.state_map = {high: HMM_IMPULSE, mr_s: HMM_MEAN_REVERT, stress_s: HMM_STRESS}
        # For 4+ states: assign TREND, RANGE, CRISIS, HYPER_TREND, etc.
        elif self.n_states >= 4:
            self.state_map = {}
            for idx in range(self.n_states):
                if idx == order_ret[-1]:
                    self.state_map[idx] = HMM_IMPULSE  # Highest return
                elif idx == order_ret[0]:
                    self.state_map[idx] = HMM_STRESS  # Lowest return
                else:
                    self.state_map[idx] = HMM_MEAN_REVERT  # Middle regimes
        else:
            # 2 states: impulse vs defensive (stress if low-ret is high-vol, else quiet)
            low, high = int(order_ret[0]), int(order_ret[1])
            defensive = HMM_STRESS if vol[low] >= vol[high] else HMM_MEAN_REVERT
            self.state_map = {high: HMM_IMPULSE, low: defensive}
    
    def compute_log_likelihood(self, X: np.ndarray) -> float:
        """Compute log-likelihood of observations under fitted model."""
        T = len(X)
        if T < 2 or self.means is None or self.covs is None:
            return -1e9
        
        logB = self._emission_log_probs_matrix(X)
        logA = np.log(self.A + 1e-12)
        logpi = np.log(self.pi + 1e-12)

        # Forward pass to get log-likelihood (vectorized over states per step)
        alpha = np.empty((T, self.n_states))
        alpha[0] = logpi + logB[0]
        for t in range(1, T):
            alpha[t] = logB[t] + _logsumexp_axis(alpha[t - 1][:, None] + logA, axis=0)

        return float(_logsumexp(alpha[-1]))
    
    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion: -2*LL + k*log(n)."""
        ll = self.compute_log_likelihood(X)
        T, D = X.shape
        # Number of parameters: π (n_states-1) + A ((n_states-1)*n_states) + means (n_states*D) + covs (n_states*D*(D+1)/2)
        n_params = (self.n_states - 1) + (self.n_states - 1) * self.n_states + self.n_states * D + self.n_states * D * (D + 1) // 2
        return -2 * ll + n_params * np.log(T)
    
    def aic(self, X: np.ndarray) -> float:
        """Akaike Information Criterion: -2*LL + 2*k."""
        ll = self.compute_log_likelihood(X)
        T, D = X.shape
        n_params = (self.n_states - 1) + (self.n_states - 1) * self.n_states + self.n_states * D + self.n_states * D * (D + 1) // 2
        return -2 * ll + 2 * n_params

    def forward_filter(self, prev_alpha: np.ndarray | None, x: np.ndarray) -> np.ndarray:
        """
        Forward step (Markov property):
        P(S_t=j | y_1:t) ∝ P(y_t | S_t=j) × Σ_i P(S_t=j | S_{t-1}=i) × P(S_{t-1}=i | y_1:t-1)
        """
        logB = self._emission_log_probs(x)
        logA = np.log(self.A + 1e-12)
        if prev_alpha is None:
            logp = np.log(self.pi + 1e-12) + logB
        else:
            logp = logB + _logsumexp_axis(np.log(prev_alpha + 1e-12)[:, None] + logA, axis=0)
        return _softmax(logp)

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Causal filter over y_1:T → α_T."""
        alpha: np.ndarray | None = None
        for t in range(len(X)):
            alpha = self.forward_filter(alpha, X[t])
        return alpha if alpha is not None else np.ones(self.n_states) / self.n_states

    def regime_name(self, state_idx: int) -> str:
        return self.state_map.get(state_idx, HMM_MEAN_REVERT)

    def economic_probs(self, alpha: np.ndarray) -> dict[str, float]:
        out = {r: 0.0 for r in HMM_REGIMES}
        for k, p in enumerate(alpha):
            out[self.regime_name(k)] += float(p)
        return out

    def transition_matrix_labeled(self) -> dict[str, dict[str, float]]:
        labels = [self.regime_name(k) for k in range(self.n_states)]
        return {
            labels[i]: {labels[j]: round(float(self.A[i, j]), 3) for j in range(self.n_states)}
            for i in range(self.n_states)
        }

    def expected_durations(self) -> dict[str, float]:
        labels = [self.regime_name(k) for k in range(self.n_states)]
        out = {}
        for k, lab in enumerate(labels):
            p_stay = float(self.A[k, k])
            out[lab] = round(1.0 / (1.0 - p_stay), 1) if p_stay < 0.999 else 999.0
        return out



def blend_regime_profile(probs: dict[str, float]) -> dict:
    """Мягкий вывод HMM: θ = Σ_k P(S_t=k) · θ_k."""
    wm = wmr = wr = buy = hold = cap = 0.0
    label_parts = []
    for name, p in probs.items():
        if p < 1e-6 or name not in REGIMES:
            continue
        prof = REGIMES[name]
        wm += p * prof.weight_factor_trend
        wmr += p * prof.weight_factor_mean_rev
        wr += p * prof.weight_factor_risk
        buy += p * prof.buy_threshold
        hold += p * prof.hold_threshold
        cap += p * prof.exposure_cap
        if p >= 0.20:
            label_parts.append(f"{prof.label_ru} ({p:.0%})")
    wm = max(0.05, wm)
    wmr = max(0.05, wmr)
    wr = max(0.05, wr)
    s = wm + wmr + wr
    if s > 0:
        wm, wmr, wr = wm / s, wmr / s, wr / s
    dominant = max(probs, key=probs.get)
    return {
        COL_HMM_DOMINANT: dominant,
        "regime_label": ", ".join(label_parts) if label_parts else REGIMES[dominant].label_ru,
        COL_WEIGHT_TREND: round(wm, 3),
        COL_WEIGHT_MEAN_REV: round(wmr, 3),
        COL_WEIGHT_RISK: round(wr, 3),
        "buy_threshold": round(buy, 1),
        "hold_threshold": round(hold, 1),
        "exposure_cap": round(cap, 3),
        COL_PROB_HMM_IMPULSE: round(probs.get(HMM_IMPULSE, 0), 3),
        COL_PROB_HMM_MEAN_REVERT: round(probs.get(HMM_MEAN_REVERT, 0), 3),
        COL_PROB_HMM_STRESS: round(probs.get(HMM_STRESS, 0), 3),
    }


# Daily regime windows (used for the intraday→daily HMM, in trading-day units)
_HMM_DAILY_VOL_WIN = 20
_HMM_DAILY_VOL_MED = 252


def _is_crypto_series(close: pd.Series) -> bool:
    name = close.name if isinstance(close.name, str) else None
    if not name:
        return False
    try:
        from data_platform.binance import is_crypto_symbol
        return is_crypto_symbol(name)
    except ImportError:
        return False


def _hmm_close(close: pd.Series) -> pd.Series:
    """Series the HMM runs on: intraday → one close per session; daily → as-is."""
    if IS_INTRADAY:
        from common.timeframe import session_last_close

        return session_last_close(close.dropna())
    return close


def _period_features(close: pd.Series) -> pd.DataFrame:
    """
    HMM observation features (ret_z, vol_z) at the regime frequency.

    Daily strategy: resample to HMM_FREQ (month) and z-score over rolling periods.
    Intraday: collapse to one bar per session first (HMM on daily regimes built
    from intraday data), so overnight gaps are part of the daily return, not noise.
    """
    if IS_INTRADAY:
        c = _hmm_close(close)
        lr = np.log(c / c.shift(1))
        crypto = _is_crypto_series(close)
        ann_days = (
            _cfg.CRYPTO_TRADING_DAYS_PER_YEAR if crypto and hasattr(_cfg, "CRYPTO_TRADING_DAYS_PER_YEAR")
            else TRADING_DAYS_PER_YEAR
        )
        vol = lr.rolling(_HMM_DAILY_VOL_WIN).std() * math.sqrt(ann_days)
        vol_ratio = vol / vol.rolling(_HMM_DAILY_VOL_MED, min_periods=_HMM_DAILY_VOL_WIN).median().replace(0, np.nan)
        period = pd.DataFrame({"log_ret": lr, "vol_ratio": vol_ratio}).dropna()
    else:
        lr = log_returns(close)
        daily = pd.DataFrame({"log_ret": lr, "vol_ratio": vol_regime_ratio(ann_vol(lr))}, index=close.index)
        period = daily.resample(_cfg.HMM_FREQ).agg({"log_ret": "sum", "vol_ratio": "last"}).dropna()

    pret, pvol = period["log_ret"], period["vol_ratio"]
    roll = HMM_ROLLING_PERIODS
    roll_min = HMM_ROLLING_MIN
    period["ret_z"] = (pret - pret.rolling(roll, min_periods=roll_min).mean()) / pret.rolling(
        roll, min_periods=roll_min
    ).std().replace(0, np.nan)
    period["vol_z"] = (pvol - pvol.rolling(roll, min_periods=roll_min).mean()) / pvol.rolling(
        roll, min_periods=roll_min
    ).std().replace(0, np.nan)
    return period.dropna(subset=["ret_z", "vol_z"])


def _monthly_features(close: pd.Series) -> pd.DataFrame:
    return _period_features(close)


def _resolve_regime_tickers(prices: pd.DataFrame, ticker: str | None) -> list[str]:
    if ticker and ticker in prices.columns:
        return [ticker]
    ordered = [t for t in REGIME_TICKERS if t in prices.columns]
    if ordered:
        return ordered
    if REGIME_TICKER in prices.columns:
        return [REGIME_TICKER]
    return [prices.columns[0]]


def _ticker_weight(ticker: str, tickers: list[str]) -> float:
    raw = {t: float(REGIME_TICKER_WEIGHTS.get(t, 1.0)) for t in tickers}
    total = sum(raw.values()) or 1.0
    return raw.get(ticker, 1.0 / max(len(tickers), 1)) / total


def _hmm_bar_table(close: pd.Series, max_bars: int | None = None) -> pd.DataFrame:
    """HMM probs per 5Min bar (causal walk-forward)."""
    from research.features.hmm_observations import compute_bar_features, hmm_observation_matrix

    c = close.dropna().astype(float)
    cap = max_bars if max_bars is not None else getattr(_cfg, "HMM_BAR_MAX_ROWS", None)
    if cap and len(c) > cap:
        c = c.iloc[-int(cap):]

    feats = compute_bar_features(c)
    if feats.empty:
        return pd.DataFrame()

    n = len(feats)
    train_w = int(getattr(_cfg, "HMM_BAR_TRAIN_BARS", 2016))
    train_w = min(train_w, max(int(getattr(_cfg, "CRYPTO_BARS_PER_DAY", 288)) * 2, n - 50))
    refit_stride = int(getattr(_cfg, "HMM_BAR_REFIT_EVERY", 12))
    hour = int(getattr(_cfg, "HMM_RET_Z_LOOKBACK_BARS", 12))
    min_train = max(hour * 4, int(getattr(_cfg, "HMM_BAR_ROLLING_MIN", 6)) * 4)

    risk_sma = int(getattr(_cfg, "HMM_BAR_RISK_SMA_BARS", 288))
    sma = c.rolling(risk_sma, min_periods=hour).mean()
    hmm_arr = hmm_observation_matrix(feats)
    vol_ratio_arr = feats["vol_ratio"].values
    log_ret_arr = feats["log_ret"].values
    ret_z_arr = feats["ret_z"].values
    times = feats.index.to_numpy()
    close_arr = c.reindex(feats.index).values
    sma_arr = sma.reindex(feats.index).values

    hmm: GaussianHMM | None = None
    prev_alpha: np.ndarray | None = None
    rows: list[dict] = []
    n = len(feats)

    from operations.progress import track

    indices = list(range(n))
    for i in track(indices, total=n, label="HMM bar regime"):
        start = max(0, i - train_w)
        x_hist = hmm_arr[start:i]
        x_cur = hmm_arr[i]
        if len(x_hist) < min_train:
            continue
        if hmm is None or i % refit_stride == 0:
            n_iter = int(getattr(_cfg, "HMM_BAR_EM_ITERS", HMM_EM_ITERS))
            hmm = GaussianHMM(n_states=HMM_N_STATES, n_iter=n_iter)
            hmm.fit(x_hist)
            alpha = hmm.forward_pass(x_hist)
            alpha = hmm.forward_filter(alpha, x_cur)
        else:
            assert hmm is not None
            alpha = hmm.forward_filter(prev_alpha, x_cur)
        prev_alpha = alpha
        probs = hmm.economic_probs(alpha)
        ro = bool(close_arr[i] > sma_arr[i]) if np.isfinite(sma_arr[i]) else False
        rows.append(
            {
                "bar_time": pd.Timestamp(times[i]),
                "period": pd.Timestamp(times[i]),
                COL_PROB_HMM_IMPULSE: probs.get(HMM_IMPULSE, 0.0),
                COL_PROB_HMM_MEAN_REVERT: probs.get(HMM_MEAN_REVERT, 0.0),
                COL_PROB_HMM_STRESS: probs.get(HMM_STRESS, 0.0),
                "hmm_confidence": float(max(probs.values())),
                "vol_ratio": float(vol_ratio_arr[i]),
                "log_ret": float(log_ret_arr[i]),
                "ret_z": float(ret_z_arr[i]),
                "risk_on": ro,
                "hmm_state": int(np.argmax(alpha)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("period")


def _hmm_monthly_table(close: pd.Series) -> pd.DataFrame:
    """HMM probs + vol/risk features per regime period (month daily / session intraday)."""
    monthly = _period_features(close)
    # Run SMA / observations on the regime-frequency close (daily for intraday)
    hmm_close = _hmm_close(close)
    sma_win = 200 if IS_INTRADAY else SMA_LONG
    sma200 = hmm_close.rolling(sma_win).mean()
    # Intraday has thousands of periods → refit periodically and filter in between
    refit_stride = REOPTIMIZE_EVERY if IS_INTRADAY else 1
    hmm: GaussianHMM | None = None
    prev_alpha: np.ndarray | None = None
    rows: list[dict] = []

    from operations.progress import track

    for i, dt in track(
        list(enumerate(monthly.index)), total=len(monthly.index), label="HMM regime fit"
    ):
        hist = monthly.iloc[max(0, i - HMM_TRAIN_MONTHS) : i]
        if len(hist) < 24:
            continue
        x_hist = hist[["ret_z", "vol_z"]].values
        x_cur = monthly.iloc[i][["ret_z", "vol_z"]].values.astype(float)
        if hmm is None or i % refit_stride == 0:
            hmm = GaussianHMM(n_states=HMM_N_STATES, n_iter=HMM_EM_ITERS)
            hmm.fit(x_hist)
            alpha = hmm.forward_pass(x_hist)
            alpha = hmm.forward_filter(alpha, x_cur)
        else:
            alpha = hmm.forward_filter(prev_alpha, x_cur)
        prev_alpha = alpha
        probs = hmm.economic_probs(alpha)
        obs_dt = hmm_close.index[hmm_close.index <= dt][-1]
        ro = bool(hmm_close.loc[obs_dt] > sma200.loc[obs_dt]) if pd.notna(sma200.loc[obs_dt]) else False
        log_ret = float(monthly.loc[dt, "log_ret"])
        ret_z = float(monthly.loc[dt, "ret_z"])
        vol_ratio = float(monthly.loc[dt, "vol_ratio"])
        rows.append(
            {
                "period": pd.Timestamp(dt),
                "date": pd.Timestamp(obs_dt),
                COL_PROB_HMM_IMPULSE: probs.get(HMM_IMPULSE, 0.0),
                COL_PROB_HMM_MEAN_REVERT: probs.get(HMM_MEAN_REVERT, 0.0),
                COL_PROB_HMM_STRESS: probs.get(HMM_STRESS, 0.0),
                "hmm_confidence": float(max(probs.values())),
                "vol_ratio": vol_ratio,
                "log_ret": log_ret,
                "ret_z": ret_z,
                "risk_on": ro,
                "hmm_state": int(np.argmax(alpha)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("period")


def _blend_hmm_tables(
    tables: list[tuple[str, float, pd.DataFrame]],
    tickers: list[str],
    prices: pd.DataFrame,
    *,
    time_col: str,
) -> tuple[list[dict], pd.Series, dict[str, dict]]:
    """Fuse per-ticker HMM tables into regime rows."""
    per_ticker_meta: dict[str, dict] = {}
    if len(tables) == 1:
        t, _, tbl = tables[0]
        monthly_rows = []
        for dt in tbl.index:
            row = tbl.loc[dt]
            probs = fuse_hmm_probs(
                {
                    HMM_IMPULSE: row[COL_PROB_HMM_IMPULSE],
                    HMM_MEAN_REVERT: row[COL_PROB_HMM_MEAN_REVERT],
                    HMM_STRESS: row[COL_PROB_HMM_STRESS],
                },
                risk_on=bool(row["risk_on"]),
                vol_ratio=float(row["vol_ratio"]),
                log_ret=float(row["log_ret"]),
                ret_z=float(row["ret_z"]),
            )
            blended = blend_regime_profile(probs)
            ts = row.get("bar_time", row.get("date", dt))
            monthly_rows.append(
                {
                    time_col: ts,
                    "date": ts,
                    "bar_time": ts,
                    "hmm_state": int(np.argmax([probs[r] for r in HMM_REGIMES])),
                    "hmm_confidence": round(float(max(probs.values())), 3),
                    "vol_ratio": round(float(row["vol_ratio"]), 2),
                    "risk_on": bool(row["risk_on"]),
                    **blended,
                }
            )
        close = prices[t].dropna()
        per_ticker_meta[t] = {"source": t}
        return monthly_rows, close, per_ticker_meta

    idx = tables[0][2].index
    for _, _, tbl in tables[1:]:
        idx = idx.union(tbl.index)
    idx = idx.sort_values()
    monthly_rows = []
    weights = {t: _ticker_weight(t, tickers) for t, _, _ in tables}
    for dt in idx:
        p_m = p_mr = p_st = 0.0
        lr_w = rz_w = 0.0
        c_w = v_max = 0.0
        ro_flags: list[bool] = []
        w_used = 0.0
        obs_times = []
        for t, _, tbl in tables:
            if dt not in tbl.index:
                continue
            wt = weights[t]
            row = tbl.loc[dt]
            p_m += wt * row[COL_PROB_HMM_IMPULSE]
            p_mr += wt * row[COL_PROB_HMM_MEAN_REVERT]
            p_st += wt * row[COL_PROB_HMM_STRESS]
            lr_w += wt * float(row["log_ret"])
            rz_w += wt * float(row["ret_z"])
            c_w += wt * row["hmm_confidence"]
            v_max = max(v_max, float(row["vol_ratio"]))
            ro_flags.append(bool(row["risk_on"]))
            obs_times.append(row.get("bar_time", row.get("date", dt)))
            w_used += wt
        if w_used <= 0:
            continue
        scale = 1.0 / w_used
        risk_on = all(ro_flags)
        vol_ratio = v_max
        log_ret = lr_w * scale
        ret_z = rz_w * scale
        probs = fuse_hmm_probs(
            {
                HMM_IMPULSE: p_m * scale,
                HMM_MEAN_REVERT: p_mr * scale,
                HMM_STRESS: p_st * scale,
            },
            risk_on=risk_on,
            vol_ratio=vol_ratio,
            log_ret=log_ret,
            ret_z=ret_z,
        )
        blended = blend_regime_profile(probs)
        ts = max(obs_times)
        monthly_rows.append(
            {
                time_col: ts,
                "date": ts,
                "bar_time": ts,
                "hmm_state": int(np.argmax([probs[r] for r in HMM_REGIMES])),
                "hmm_confidence": round(max(probs.values()), 3),
                "vol_ratio": round(vol_ratio, 2),
                "risk_on": risk_on,
                "regime_sources": "+".join(t for t, _, tbl in tables if dt in tbl.index),
                **blended,
            }
        )
    close = prices[tickers[0]].dropna()
    per_ticker_meta = {t: {"weight": _ticker_weight(t, tickers)} for t, _, _ in tables}
    return monthly_rows, close, per_ticker_meta


def build_hmm_regime_frame(
    prices: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    global LAST_HMM_META
    use_bar = getattr(_cfg, "HMM_FREQUENCY", "daily") == "bar" and IS_INTRADAY
    tickers = _resolve_regime_tickers(prices, ticker)
    tables: list[tuple[str, float, pd.DataFrame]] = []
    per_ticker_meta: dict[str, dict] = {}

    for t in tickers:
        s = prices[t].dropna()
        s.name = t
        tbl = _hmm_bar_table(s) if use_bar else _hmm_monthly_table(s)
        if tbl.empty:
            continue
        w = _ticker_weight(t, tickers)
        tables.append((t, w, tbl))

    if not tables:
        return pd.DataFrame()

    time_col = "bar_time" if use_bar else "date"
    monthly_rows, close, per_ticker_meta = _blend_hmm_tables(
        tables, tickers, prices, time_col=time_col
    )

    LAST_HMM_META.clear()
    LAST_HMM_META.update(
        {
            "hmm_frequency": "bar" if use_bar else "daily",
            "regime_tickers": [t for t, _, _ in tables],
            "regime_weights": {t: _ticker_weight(t, tickers) for t in tickers if t in [x[0] for x in tables]},
            "per_ticker": per_ticker_meta,
        }
    )

    mdf = pd.DataFrame(monthly_rows)
    if mdf.empty:
        return mdf
    mdf["date"] = pd.to_datetime(mdf["date"])
    if "bar_time" in mdf.columns:
        mdf["bar_time"] = pd.to_datetime(mdf["bar_time"])

    if use_bar:
        return mdf.sort_values("bar_time").reset_index(drop=True)

    if IS_INTRADAY:
        return mdf.sort_values("date").reset_index(drop=True)

    daily_idx = pd.DataFrame({"date": pd.to_datetime(close.index)})
    return pd.merge_asof(daily_idx.sort_values("date"), mdf.sort_values("date"), on="date", direction="backward")
