"""
Minimal Gaussian Hidden Markov Model for unsupervised regime detection.

Sentinel's existing regime classifier (``strategy.market_regime``) uses
hand-tuned rules over ADX / BB-width / ATR. That's interpretable but
fixed — a new market regime (e.g. summer-2022-style chop) is invisible
to it until someone codes a new branch. An unsupervised HMM learns
regime structure directly from the return series: two or three latent
states whose means, variances, and transition frequencies are fit to
the data.

This is a **minimal** implementation — Baum-Welch EM for parameter
estimation, Viterbi for decoding the most-likely state sequence. Pure
NumPy, no ``hmmlearn`` dependency. It's intentionally scope-limited to
1D Gaussian emissions (daily or hourly returns); higher dimensional
features should use a proper HMM library.

Intended use: feed a price-return series, train on 6–12 months of
history, predict the regime on new observations in real time, and
route trade decisions through a regime-aware policy (e.g. size down
in high-vol states, stop trading when transition probability to
bearish exceeds a threshold). The HMM output is **one more input** to
an aggregator, not a standalone decision.

Pitfalls locked in by tests:

* Numerical stability via log-sum-exp everywhere.
* Optional floor on emission variance prevents collapse.
* Random multi-start to avoid local optima.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


_MIN_VAR = 1e-8     # floor to prevent σ² → 0 collapse
_LOG_SUM_EPS = 1e-300


def _log_sum_exp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-sum-exp along ``axis``."""
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    return (a_max_safe + np.log(np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True) + _LOG_SUM_EPS)).squeeze(axis)


def _gaussian_log_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """log N(x | μ, σ²) for a 1D Gaussian, vectorised over x."""
    var = max(var, _MIN_VAR)
    return -0.5 * (math.log(2 * math.pi * var) + (x - mu) ** 2 / var)


@dataclass(slots=True)
class GaussianHMMFit:
    """Fitted Gaussian HMM parameters + training diagnostics."""
    n_states: int
    pi: np.ndarray          # shape (K,): initial state distribution
    transmat: np.ndarray    # shape (K, K): A[i, j] = P(state_t=j | state_{t-1}=i)
    means: np.ndarray       # shape (K,)
    variances: np.ndarray   # shape (K,)
    log_likelihood: float   # training log-likelihood at convergence
    n_iter: int             # iterations actually run
    converged: bool

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Viterbi decoding — most likely state sequence given observations.

        Returns integer array of shape ``(T,)``, values in ``[0, K)``.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        T = x.size
        K = self.n_states
        if T == 0:
            return np.array([], dtype=np.int64)

        log_emission = np.column_stack([
            _gaussian_log_pdf(x, float(self.means[k]), float(self.variances[k]))
            for k in range(K)
        ])
        log_pi = np.log(np.clip(self.pi, 1e-300, 1.0))
        log_A = np.log(np.clip(self.transmat, 1e-300, 1.0))

        delta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=np.int64)
        delta[0] = log_pi + log_emission[0]

        for t in range(1, T):
            # scores[j] = max_i (delta[t-1, i] + log_A[i, j]) + log_emission[t, j]
            scores = delta[t - 1, :, None] + log_A
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_emission[t]

        path = np.zeros(T, dtype=np.int64)
        path[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = int(psi[t + 1, path[t + 1]])
        return path

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Posterior state probabilities γ[t, k] = P(s_t = k | x).

        Returns shape ``(T, K)`` array. Sum across k axis equals 1 per row.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        T = x.size
        K = self.n_states
        if T == 0:
            return np.zeros((0, K))

        log_emission = np.column_stack([
            _gaussian_log_pdf(x, float(self.means[k]), float(self.variances[k]))
            for k in range(K)
        ])
        log_alpha, log_beta, _ = _forward_backward(
            log_emission, np.log(np.clip(self.pi, 1e-300, 1.0)),
            np.log(np.clip(self.transmat, 1e-300, 1.0)),
        )
        log_gamma = log_alpha + log_beta
        # Normalise per timestep
        log_gamma -= _log_sum_exp(log_gamma, axis=1)[:, None]
        return np.exp(log_gamma)


def _forward_backward(
    log_emission: np.ndarray,
    log_pi: np.ndarray,
    log_A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run forward + backward in log space. Returns (log_alpha, log_beta, log_likelihood)."""
    T, K = log_emission.shape

    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_pi + log_emission[0]
    for t in range(1, T):
        # log_alpha[t, j] = log_sum_exp(log_alpha[t-1] + log_A[:, j]) + log_emission[t, j]
        tmp = log_alpha[t - 1, :, None] + log_A
        log_alpha[t] = _log_sum_exp(tmp, axis=0) + log_emission[t]

    log_beta = np.zeros((T, K))
    # log_beta[T-1] = 0
    for t in range(T - 2, -1, -1):
        tmp = log_A + log_emission[t + 1][None, :] + log_beta[t + 1][None, :]
        log_beta[t] = _log_sum_exp(tmp, axis=1)

    log_likelihood = float(_log_sum_exp(log_alpha[-1], axis=-1).item())
    return log_alpha, log_beta, log_likelihood


def fit_gaussian_hmm(
    x: np.ndarray,
    *,
    n_states: int = 2,
    max_iter: int = 50,
    tol: float = 1e-4,
    n_starts: int = 3,
    seed: int = 42,
) -> GaussianHMMFit:
    """Baum-Welch fit of a ``n_states`` Gaussian HMM to a 1D observation series.

    Args:
        x: 1D observations (e.g. log-returns).
        n_states: Number of latent states. 2 ≈ low/high vol or
            bull/bear. 3 adds a neutral / transition state. Rarely
            worth more than 4 on a return series.
        max_iter: EM iterations ceiling.
        tol: Stop when the relative change in log-likelihood is below
            this threshold.
        n_starts: Independent random restarts; returns the best-
            log-likelihood fit. Protects against local optima.
        seed: Base RNG seed (each restart uses seed + restart_index).

    Returns:
        ``GaussianHMMFit`` with parameters and training diagnostics.
        State indices are NOT guaranteed to map to a particular regime
        meaning — callers should inspect ``means`` / ``variances`` and
        label the states themselves.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    T = x.size
    if T < 4 * n_states:
        raise ValueError(
            f"need ≥ {4 * n_states} observations for n_states={n_states}, got {T}"
        )

    best: Optional[GaussianHMMFit] = None
    for restart in range(n_starts):
        rng = np.random.default_rng(seed + restart)

        # Random initialisation. Quantile-based means give EM a head-start
        # on roughly-separated states without requiring the caller to
        # hand-tune anything.
        means = np.quantile(x, np.linspace(0.1, 0.9, n_states)).astype(np.float64)
        means += rng.normal(0, float(x.std()) * 0.05, size=n_states)
        variances = np.full(n_states, max(float(x.var()), _MIN_VAR))
        # Favour self-transition to bias toward persistent regimes
        transmat = np.full((n_states, n_states), 0.1 / (n_states - 1)) if n_states > 1 else np.array([[1.0]])
        if n_states > 1:
            np.fill_diagonal(transmat, 0.9)
        pi = np.full(n_states, 1.0 / n_states)

        prev_ll = -np.inf
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            # --- E-step ---
            log_emission = np.column_stack([
                _gaussian_log_pdf(x, float(means[k]), float(variances[k]))
                for k in range(n_states)
            ])
            log_alpha, log_beta, log_ll = _forward_backward(
                log_emission, np.log(np.clip(pi, 1e-300, 1.0)),
                np.log(np.clip(transmat, 1e-300, 1.0)),
            )
            # γ[t, k]
            log_gamma = log_alpha + log_beta
            log_gamma -= _log_sum_exp(log_gamma, axis=1)[:, None]
            gamma = np.exp(log_gamma)

            # ξ[t, i, j] = P(s_t=i, s_{t+1}=j | x)
            log_xi_num = (
                log_alpha[:-1, :, None] + np.log(np.clip(transmat, 1e-300, 1.0))[None, :, :]
                + log_emission[1:, None, :] + log_beta[1:, None, :]
            )
            log_xi = log_xi_num - log_ll

            # --- M-step ---
            pi_new = gamma[0]
            pi_new = pi_new / (pi_new.sum() + _LOG_SUM_EPS)

            # Transition matrix
            xi_sum = np.exp(_log_sum_exp(log_xi, axis=0))   # (K, K)
            transmat_new = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + _LOG_SUM_EPS)

            # Means and variances
            gamma_sum = gamma.sum(axis=0)
            means_new = (gamma * x[:, None]).sum(axis=0) / (gamma_sum + _LOG_SUM_EPS)
            diff2 = (x[:, None] - means_new[None, :]) ** 2
            variances_new = (gamma * diff2).sum(axis=0) / (gamma_sum + _LOG_SUM_EPS)
            variances_new = np.maximum(variances_new, _MIN_VAR)

            pi, transmat, means, variances = pi_new, transmat_new, means_new, variances_new

            if abs(log_ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
                converged = True
                break
            prev_ll = log_ll

        fit = GaussianHMMFit(
            n_states=n_states,
            pi=pi.copy(), transmat=transmat.copy(),
            means=means.copy(), variances=variances.copy(),
            log_likelihood=float(prev_ll),
            n_iter=it, converged=converged,
        )
        if best is None or fit.log_likelihood > best.log_likelihood:
            best = fit

    assert best is not None
    return best
