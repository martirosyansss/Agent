"""Pure scoring helpers — composite skill score + Wilson CI lower bound.

Used by both training (to gate model deployment) and runtime drift
detection (to test whether live precision is significantly below
training precision). Kept framework-free so the same arithmetic is
trivially unit-testable without importing sklearn.

Moved out of ``analyzer.ml_predictor`` during the round-10 refactor.
The public names ``compute_skill_score`` and ``wilson_lower_bound``
are re-exported from the old module path for backwards compatibility.
"""
from __future__ import annotations

from .constants import (
    _SKILL_W_PRECISION,
    _SKILL_W_RECALL,
    _SKILL_W_ROC_AUC,
    _SKILL_W_PROFIT_FACTOR,
)


def compute_skill_score(
    precision: float,
    recall: float,
    roc_auc: float,
    profit_factor_score: float,
) -> float:
    """Weighted composite skill score used for model selection and gating.

    All four inputs should be normalised to [0, 1]. ``profit_factor_score``
    is typically ``min(profit_factor / 3.0, 1.0)``.
    """
    return (
        _SKILL_W_PRECISION * precision
        + _SKILL_W_RECALL * recall
        + _SKILL_W_ROC_AUC * roc_auc
        + _SKILL_W_PROFIT_FACTOR * profit_factor_score
    )


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """95%-CI lower bound on a binomial proportion (Wilson score interval).

    Used to test whether an observed success rate is *significantly above*
    a target. The Wilson interval is well-behaved at small N and at
    extreme p, unlike the naive normal approximation (which fails when
    p≈0 or p≈1).

    Args:
        successes: Observed successes (e.g., true-positive predictions)
        trials:    Total trials (e.g., total predicted-positive events)
        z:         z-score for the desired confidence (1.96 = 95%, 1.645 = 90%)

    Returns:
        Lower bound of the confidence interval in [0, 1]. Returns 0 if
        ``trials == 0``.
    """
    if trials <= 0:
        return 0.0
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = p + z2 / (2.0 * trials)
    half_width = z * ((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) ** 0.5
    return max(0.0, (center - half_width) / denom)
