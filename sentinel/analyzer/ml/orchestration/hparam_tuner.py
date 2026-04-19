"""
Hyper-parameter tuner — lightweight grid / random search without external
dependencies (no Optuna, no Ray). Written so operators can tune the RF /
LGBM / XGB gyroscopes without pulling a SQL backend into the container.

The tuner takes an objective callable of shape
``(params: dict) -> float`` (higher is better — e.g. mean walk-forward
AUC, or a custom risk-adjusted score), a search space description, and
a budget. It returns the top-K parameter sets sorted by score, plus the
full evaluation history for reproducibility.

Two modes are supported:

* **Grid** — exhaustive over the Cartesian product of the space. Good
  when the space is small and every cell matters.
* **Random** — ``n_trials`` uniform samples from the space. Strictly
  better than grid once any axis has more than 4–5 candidates because
  random search hits "useful" regions with fewer trials (Bergstra &
  Bengio, 2012).

The evaluation is serial-by-default. Parallelising training folds lives
at a lower layer (the trainer itself); a hyper-parameter tuner that
spawns its own pool on top usually blows memory on Kaggle-style tree
ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import itertools
import logging
import math
import random as _random
import time

import numpy as np


logger = logging.getLogger(__name__)


ObjectiveFn = Callable[[dict[str, Any]], float]


@dataclass(slots=True)
class TuneTrial:
    """One evaluated parameter set."""
    params: dict[str, Any]
    score: float
    duration_s: float
    error: Optional[str] = None


@dataclass(slots=True)
class TuneResult:
    """Full output of a tuning run."""
    best_params: dict[str, Any]
    best_score: float
    trials: list[TuneTrial] = field(default_factory=list)
    n_trials: int = 0
    duration_s: float = 0.0

    def top_k(self, k: int = 5) -> list[TuneTrial]:
        return sorted(
            (t for t in self.trials if t.error is None),
            key=lambda t: t.score, reverse=True,
        )[:k]


def _cartesian_grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Enumerate the full Cartesian product of ``space``."""
    if not space:
        return [{}]
    keys = list(space.keys())
    values = [space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _random_sample(
    space: dict[str, list[Any]],
    rng: _random.Random,
) -> dict[str, Any]:
    """Pick one random value per axis."""
    return {k: rng.choice(v) for k, v in space.items()}


def grid_search(
    objective: ObjectiveFn,
    space: dict[str, list[Any]],
    *,
    max_trials: Optional[int] = None,
    verbose: bool = True,
) -> TuneResult:
    """Exhaustive grid search over ``space``.

    Args:
        objective: Function mapping a parameter dict to a score (higher
            is better). Exceptions are caught and recorded as trial
            errors rather than propagated — tuning runs are long and
            one bad config shouldn't kill the session.
        space: ``{param_name: [candidate_values, ...]}``.
        max_trials: Optional ceiling on evaluations. If the grid has
            more cells than ``max_trials``, a ``ValueError`` is raised
            so the caller knows to switch to ``random_search``.
        verbose: Log progress at INFO level.

    Returns:
        ``TuneResult`` with the best params and full trial history.
    """
    grid = _cartesian_grid(space)
    if max_trials is not None and len(grid) > max_trials:
        raise ValueError(
            f"grid has {len(grid)} cells > max_trials={max_trials}; "
            "use random_search or shrink the space"
        )
    if verbose:
        logger.info("grid_search: %d cells", len(grid))
    return _run_trials(objective, grid, verbose=verbose)


def random_search(
    objective: ObjectiveFn,
    space: dict[str, list[Any]],
    *,
    n_trials: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> TuneResult:
    """Random search over ``space``.

    Args:
        n_trials: Number of samples. Bergstra & Bengio (2012) show that
            after ~60 trials you're within 5% of the best grid cell for
            most realistic spaces, so 50 is the common default.
        seed: RNG seed for reproducibility.
    """
    rng = _random.Random(seed)
    samples = [_random_sample(space, rng) for _ in range(n_trials)]
    # Deduplicate — random sampling can hit the same cell twice; we'd
    # rather spend the budget exploring.
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for s in samples:
        key = tuple(sorted(s.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    if verbose:
        logger.info(
            "random_search: %d unique samples (budget %d)",
            len(deduped), n_trials,
        )
    return _run_trials(objective, deduped, verbose=verbose)


def _run_trials(
    objective: ObjectiveFn,
    candidates: list[dict[str, Any]],
    *,
    verbose: bool,
) -> TuneResult:
    """Evaluate every candidate serially. Exceptions are captured, not raised."""
    start = time.time()
    trials: list[TuneTrial] = []
    best_params: dict[str, Any] = {}
    best_score = -math.inf
    for i, params in enumerate(candidates, start=1):
        t0 = time.time()
        try:
            score = float(objective(params))
            trials.append(TuneTrial(
                params=params, score=score,
                duration_s=round(time.time() - t0, 3),
            ))
            if score > best_score:
                best_score = score
                best_params = params
            if verbose:
                logger.info("  trial %d/%d: score=%.4f params=%s",
                            i, len(candidates), score, params)
        except Exception as exc:  # never break the tuner on one bad config
            trials.append(TuneTrial(
                params=params, score=-math.inf,
                duration_s=round(time.time() - t0, 3),
                error=str(exc),
            ))
            if verbose:
                logger.warning(
                    "  trial %d/%d FAILED: %s params=%s",
                    i, len(candidates), exc, params,
                )

    return TuneResult(
        best_params=best_params,
        best_score=best_score if best_score > -math.inf else 0.0,
        trials=trials,
        n_trials=len(trials),
        duration_s=round(time.time() - start, 3),
    )


# ────────────────────────────────────────────────────────────────
# Halving (Successive-Halving / SHA) — trial-budget allocator
# ────────────────────────────────────────────────────────────────


def successive_halving(
    objective: ObjectiveFn,
    space: dict[str, list[Any]],
    *,
    n_initial: int = 16,
    rounds: int = 3,
    halving_ratio: float = 2.0,
    seed: int = 42,
    verbose: bool = True,
) -> TuneResult:
    """Successive halving (Jamieson & Talwalkar, 2015).

    Starts with ``n_initial`` random candidates. After each round, keeps
    the top ``1/halving_ratio`` and eliminates the rest. With budget
    fixed, halving concentrates compute on promising configs faster
    than random search — especially useful when ``objective`` is
    expensive (full walk-forward ≈ minutes).

    Note: the ``objective`` is assumed to return a comparable score
    regardless of round. If your objective does *partial* evaluation
    (fewer folds on early rounds, more on later rounds) pass a closure
    that accepts ``params`` and reads its own round-budget from outside.
    """
    rng = _random.Random(seed)
    survivors = [_random_sample(space, rng) for _ in range(n_initial)]
    all_trials: list[TuneTrial] = []
    start = time.time()
    for r in range(rounds):
        if verbose:
            logger.info("successive_halving round %d/%d: %d candidates",
                        r + 1, rounds, len(survivors))
        round_result = _run_trials(objective, survivors, verbose=verbose)
        all_trials.extend(round_result.trials)
        if not round_result.trials:
            break
        ranked = sorted(
            (t for t in round_result.trials if t.error is None),
            key=lambda t: t.score, reverse=True,
        )
        if not ranked:
            break
        keep = max(1, int(len(ranked) / halving_ratio))
        survivors = [t.params for t in ranked[:keep]]
        if len(survivors) <= 1:
            break

    best = max(
        (t for t in all_trials if t.error is None),
        key=lambda t: t.score, default=None,
    )
    return TuneResult(
        best_params=best.params if best else {},
        best_score=best.score if best else 0.0,
        trials=all_trials,
        n_trials=len(all_trials),
        duration_s=round(time.time() - start, 3),
    )
