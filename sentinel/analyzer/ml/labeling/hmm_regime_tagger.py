"""
HMM Regime Tagger — consumable wrapper around the pure Gaussian HMM.

The raw ``GaussianHMMFit`` returns integer state indices with no
semantics — state 0 might be low-vol on one fit and high-vol on the
next. A trading pipeline needs stable *labels* ("low_vol", "high_vol")
that survive restarts. This wrapper fits once, sorts states by variance
(lowest-var → "low_vol", highest → "high_vol", middle → "neutral"), and
exposes ``predict_regime(returns) -> list[str]`` for live use.

Two-state model is the default because it's what most operators want
first (is the market calm or violent?); the three-state variant adds a
"neutral" / transition regime that's occasionally useful.

Intended deployment:

1. Training script fits the tagger on 6–12 months of 1h returns.
2. Pickles the tagger alongside the ML predictor bundle.
3. Inference loop calls ``predict_regime([last_return])`` each bar and
   feeds the label into the existing regime-routing logic — or tags
   ``FeatureVector.market_regime`` with an "ml_" prefix so dashboards
   can compare the rule-based and HMM-based views side-by-side.

The wrapper is deliberately opt-in. The upstream rule-based regime
detector (``strategy.market_regime.detect_regime``) stays the primary
source until an operator validates HMM labels against their own data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from analyzer.ml.domain.hmm_regime import GaussianHMMFit, fit_gaussian_hmm


# Label vocabulary used by the tagger. Kept deliberately small so
# downstream logic (dashboard filters, regime routing) has a stable
# enum to switch on.
LABEL_LOW_VOL: str = "hmm_low_vol"
LABEL_HIGH_VOL: str = "hmm_high_vol"
LABEL_NEUTRAL: str = "hmm_neutral"
LABEL_UNKNOWN: str = "hmm_unknown"


@dataclass(slots=True)
class HMMRegimeTagger:
    """Fit once, tag many.

    Attributes:
        n_states: 2 for low/high vol; 3 adds a middle "neutral" bucket.
        fit_: The underlying ``GaussianHMMFit`` (``None`` until ``fit()``
            has been called).
        state_to_label_: Map from internal state index to stable label.
            Computed at fit time by sorting states by variance.
    """
    n_states: int = 2
    fit_: Optional[GaussianHMMFit] = None
    state_to_label_: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_states not in (2, 3):
            raise ValueError(
                f"n_states must be 2 or 3 (got {self.n_states}). "
                "Larger models fit fine from the pure API but have no "
                "canonical label vocabulary here."
            )

    def fit(
        self,
        returns: np.ndarray | list[float],
        *,
        n_starts: int = 3,
        max_iter: int = 50,
        seed: int = 42,
    ) -> "HMMRegimeTagger":
        """Fit the HMM and establish the variance-ordered label mapping."""
        arr = np.asarray(returns, dtype=np.float64).ravel()
        self.fit_ = fit_gaussian_hmm(
            arr, n_states=self.n_states,
            n_starts=n_starts, max_iter=max_iter, seed=seed,
        )
        # Sort states by variance — lowest gets "low_vol", highest gets "high_vol",
        # the middle (if any) gets "neutral". This gives stable labels across
        # refits that would otherwise shuffle state indices arbitrarily.
        order = np.argsort(self.fit_.variances)
        if self.n_states == 2:
            self.state_to_label_ = {
                int(order[0]): LABEL_LOW_VOL,
                int(order[-1]): LABEL_HIGH_VOL,
            }
        else:   # n_states == 3
            self.state_to_label_ = {
                int(order[0]): LABEL_LOW_VOL,
                int(order[1]): LABEL_NEUTRAL,
                int(order[2]): LABEL_HIGH_VOL,
            }
        return self

    def predict_regime(
        self, returns: np.ndarray | list[float],
    ) -> list[str]:
        """Return stable labels for each observation in ``returns``.

        Before ``fit()`` has been called, or when the fit failed, returns
        ``["hmm_unknown", ...]`` so live callers can cleanly fall back
        to the rule-based regime classifier.
        """
        arr = np.asarray(returns, dtype=np.float64).ravel()
        if self.fit_ is None:
            return [LABEL_UNKNOWN] * arr.size
        states = self.fit_.predict(arr)
        return [
            self.state_to_label_.get(int(s), LABEL_UNKNOWN)
            for s in states
        ]

    def current_regime(
        self, returns: np.ndarray | list[float],
    ) -> str:
        """Convenience: label of the *most recent* observation."""
        labels = self.predict_regime(returns)
        return labels[-1] if labels else LABEL_UNKNOWN

    def regime_statistics(self) -> dict:
        """Introspection: per-label mean return, variance, stationary prob.

        Useful for dashboards to show "state 'hmm_high_vol' has mean -0.002
        and σ 0.04" alongside raw state counts. Returns empty dict before fit.
        """
        if self.fit_ is None:
            return {}
        stats: dict = {}
        for idx, label in self.state_to_label_.items():
            stats[label] = {
                "mean_return": float(self.fit_.means[idx]),
                "std_return": float(np.sqrt(self.fit_.variances[idx])),
                "stationary_pi": float(self.fit_.pi[idx]),
                "self_transition": float(self.fit_.transmat[idx, idx]),
            }
        return stats
