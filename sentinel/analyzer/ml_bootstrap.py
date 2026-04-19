"""
ML Bootstrap — Monte Carlo confidence intervals for evaluation metrics.

A single train/test run produces one point estimate per metric. "Precision
= 0.67" from one holdout tells us nothing about whether the true precision
is 0.55 or 0.78. With N=200 test samples both are consistent with the
observed result. Trading decisions made on point estimates are trading
decisions made on noise.

This module bootstraps evaluation metrics by resampling the test set with
replacement and recomputing each metric per resample. The resulting
distribution gives:

- A central point estimate (p50 — the bootstrap median, more robust than
  the mean when the metric is bounded).
- Symmetric or asymmetric CIs depending on where the operator cares —
  we default to a 90% CI (p5, p95) because 95% is noisy on small N and
  falsely narrow on large N without bias correction.
- ``probability_above_baseline`` — the fraction of bootstrap samples where
  AUC exceeds a user-specified baseline (e.g. 0.5 for "better than random",
  or the previous model's AUC for regression testing).

Bootstrap is NOT a substitute for walk-forward validation. It tells you
how noisy a single fold's metrics are; it does NOT tell you the metrics
generalise to other windows. Use both: WF for time-robustness, bootstrap
for within-window noise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BootstrapCI:
    """Confidence interval for a single metric.

    ``p5`` / ``p95`` are the 5th and 95th percentiles of the bootstrap
    distribution, so ``[p5, p95]`` is the 90% CI. For a 95% CI, read
    ``ci_95_lower`` / ``ci_95_upper`` from :meth:`summary`.
    """
    metric: str
    mean: float
    p5: float
    p50: float
    p95: float
    std: float
    n_samples: int
    n_simulations: int

    def summary(self) -> dict[str, float | int | str]:
        return {
            "metric": self.metric,
            "mean": round(self.mean, 4),
            "p5": round(self.p5, 4),
            "p50": round(self.p50, 4),
            "p95": round(self.p95, 4),
            "std": round(self.std, 4),
            "n_samples": self.n_samples,
            "n_simulations": self.n_simulations,
        }


class MLBootstrap:
    """Monte Carlo bootstrap for classification metrics.

    Defaults to **moving-block bootstrap** (Künsch 1989) because trade-level
    samples are temporally autocorrelated: the same regime, same volatility
    cluster, and same instrument feed adjacent rows. iid resampling breaks
    that structure and reports CIs that are too narrow — the model looks
    more certain than it is. Block size follows the cube-root rule
    ``b = max(5, n^(1/3))`` (Hall, Horowitz & Jing 1995); pass
    ``block_bootstrap=False`` to recover the older iid behaviour for
    genuinely independent samples.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        seed: int = 42,
        block_bootstrap: bool = True,
        block_size: Optional[int] = None,
    ) -> None:
        if n_simulations < 50:
            raise ValueError(f"n_simulations must be >= 50, got {n_simulations}")
        self.n_simulations = n_simulations
        # Round-8 §3.3: store the seed, not a shared RNG. Every method that
        # draws resamples creates a fresh default_rng(seed) so call order
        # doesn't change the numbers. A single shared RNG would mean
        # ``bootstrap_metrics`` followed by ``probability_above_baseline``
        # produces different results than calling them in reverse — reports
        # should be reproducible regardless of how the caller sequences
        # the helpers.
        self._seed = seed
        self._block_bootstrap = bool(block_bootstrap)
        # Override only when the caller has a domain-specific block length;
        # otherwise the cube-root rule is computed per-call from len(y).
        self._block_size_override = block_size

    def _resample_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a length-n index array.

        Block-bootstrap path (default): pick contiguous blocks of length
        ``b ≈ n^(1/3)`` with replacement, concatenate, trim to n. Preserves
        local autocorrelation so the resulting CIs reflect time-series noise
        rather than iid noise.

        IID path (``block_bootstrap=False``): legacy ``rng.integers(0, n, n)``.
        """
        if not self._block_bootstrap:
            return rng.integers(0, n, size=n)
        b = self._block_size_override or max(5, int(round(n ** (1.0 / 3.0))))
        b = max(1, min(b, n))
        n_blocks = max(1, (n + b - 1) // b)
        starts = rng.integers(0, n - b + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + b) for s in starts])
        return idx[:n]

    def bootstrap_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, BootstrapCI]:
        """Resample the test set and compute CI for each metric.

        Args:
            y_true: Binary labels, shape (n,).
            y_proba: Predicted probabilities, shape (n,).
            threshold: Decision threshold — rows with proba ≥ threshold are
                       classified as the positive class.

        Returns:
            Dict with keys ``precision``, ``recall``, ``roc_auc``, each a
            :class:`BootstrapCI`. Returns an empty dict if inputs are too
            small (n < 20) — reliable bootstrap needs enough samples to
            have a chance of seeing both classes in each resample.
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_proba = np.asarray(y_proba, dtype=np.float64)
        n = len(y_true)
        if n < 20:
            logger.warning("MLBootstrap: n=%d too small, skipping CI computation", n)
            return {}
        if len(y_proba) != n:
            raise ValueError(f"y_true/y_proba length mismatch: {n} vs {len(y_proba)}")

        try:
            from sklearn.metrics import precision_score, recall_score, roc_auc_score
        except ImportError:
            logger.warning("MLBootstrap: sklearn missing, skipping")
            return {}

        precs = np.zeros(self.n_simulations)
        recs = np.zeros(self.n_simulations)
        aucs = np.zeros(self.n_simulations)

        # Fresh RNG per call (see __init__ docstring) keeps outputs
        # reproducible independent of method call order.
        rng = np.random.default_rng(self._seed)
        for i in range(self.n_simulations):
            idx = self._resample_indices(n, rng)
            y_s, p_s = y_true[idx], y_proba[idx]
            pred_s = (p_s >= threshold).astype(int)
            # Single-class resamples collapse the metrics to 0/undefined; we
            # treat them as neutral so the distribution's tail stays honest
            # instead of getting artificially skewed by NaN-filtering.
            if len(np.unique(y_s)) < 2:
                precs[i] = 0.0
                recs[i] = 0.0
                aucs[i] = 0.5
                continue
            precs[i] = precision_score(y_s, pred_s, zero_division=0)
            recs[i] = recall_score(y_s, pred_s, zero_division=0)
            try:
                aucs[i] = roc_auc_score(y_s, p_s)
            except ValueError:
                aucs[i] = 0.5

        return {
            "precision": self._summarize("precision", precs, n),
            "recall": self._summarize("recall", recs, n),
            "roc_auc": self._summarize("roc_auc", aucs, n),
        }

    def probability_above_baseline(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        baseline_auc: float = 0.5,
    ) -> float:
        """Fraction of bootstrap resamples whose AUC exceeds ``baseline_auc``.

        Useful framing: "with 93% probability, this model beats random" is
        more informative for production gating than "AUC is 0.67" alone.
        Values near 0.5 signal no evidence either way; values ≥ 0.95 warrant
        shipping.
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_proba = np.asarray(y_proba, dtype=np.float64)
        n = len(y_true)
        if n < 20:
            return 0.5

        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return 0.5

        wins = 0
        rng = np.random.default_rng(self._seed)
        for _ in range(self.n_simulations):
            idx = self._resample_indices(n, rng)
            y_s, p_s = y_true[idx], y_proba[idx]
            if len(np.unique(y_s)) < 2:
                continue
            try:
                auc = roc_auc_score(y_s, p_s)
            except ValueError:
                continue
            if auc > baseline_auc:
                wins += 1
        return wins / self.n_simulations

    # ──────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────

    def _summarize(self, name: str, samples: np.ndarray, n_test: int) -> BootstrapCI:
        return BootstrapCI(
            metric=name,
            mean=float(samples.mean()),
            p5=float(np.percentile(samples, 5)),
            p50=float(np.percentile(samples, 50)),
            p95=float(np.percentile(samples, 95)),
            std=float(samples.std(ddof=1)) if len(samples) > 1 else 0.0,
            n_samples=n_test,
            n_simulations=self.n_simulations,
        )
