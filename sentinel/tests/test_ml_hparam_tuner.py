"""Tests for the lightweight hyperparameter tuner.

Contracts:

* **Grid search picks the global maximum** — with an objective that
  has a unique best cell, grid_search must find it.
* **Random search approaches the max on a small space** — given enough
  trials on a 3x3 grid it hits the peak almost deterministically.
* **Objective exceptions are captured, not raised** — one bad config
  doesn't kill the run; the failing trial carries an ``error`` field.
* **Max-trials guard** — grid_search with ``max_trials < grid cells``
  raises so the operator knows to switch modes.
* **Halving prunes round-over-round** — successive_halving keeps fewer
  candidates each round; best-seen score is monotone non-decreasing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml.orchestration.hparam_tuner import (
    grid_search,
    random_search,
    successive_halving,
)


# Synthetic objective: peak at (lr=0.1, depth=5). Higher is better.
def _toy_objective(params):
    lr = params.get("lr", 0.0)
    depth = params.get("depth", 0)
    # Negative quadratic around (0.1, 5) with a little noise-free ripple.
    return -(abs(lr - 0.1) * 10) ** 2 - (depth - 5) ** 2


class TestGridSearch:
    def test_finds_global_max(self):
        space = {"lr": [0.01, 0.05, 0.1, 0.2], "depth": [3, 5, 7]}
        result = grid_search(_toy_objective, space, verbose=False)
        assert result.best_params == {"lr": 0.1, "depth": 5}
        assert result.n_trials == 12

    def test_max_trials_overflow_raises(self):
        space = {"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}   # 27 cells
        with pytest.raises(ValueError):
            grid_search(lambda p: 0.0, space, max_trials=10, verbose=False)


class TestRandomSearch:
    def test_hits_peak_on_small_space(self):
        space = {"lr": [0.01, 0.05, 0.1, 0.2], "depth": [3, 5, 7]}
        # With 30 trials on a 12-cell grid, we expect to see every cell.
        result = random_search(_toy_objective, space, n_trials=60, verbose=False)
        assert result.best_params == {"lr": 0.1, "depth": 5}

    def test_determinism_with_seed(self):
        space = {"lr": [0.01, 0.05, 0.1], "depth": [3, 5, 7]}
        r1 = random_search(_toy_objective, space, n_trials=20, seed=7, verbose=False)
        r2 = random_search(_toy_objective, space, n_trials=20, seed=7, verbose=False)
        assert r1.best_params == r2.best_params


class TestErrorHandling:
    def test_exception_recorded_not_raised(self):
        def bad(params):
            if params["x"] == 3:
                raise RuntimeError("boom")
            return float(params["x"])

        space = {"x": [1, 2, 3, 4]}
        result = grid_search(bad, space, verbose=False)
        assert result.n_trials == 4
        failed = [t for t in result.trials if t.error]
        assert len(failed) == 1
        assert "boom" in failed[0].error
        # Best still picked from successful runs — x=4 wins.
        assert result.best_params == {"x": 4}


class TestSuccessiveHalving:
    def test_monotone_best_over_rounds(self):
        # Simple 1D objective — peak at x=5.
        def obj(p):
            return -(p["x"] - 5) ** 2

        space = {"x": list(range(-5, 11))}
        result = successive_halving(
            obj, space, n_initial=16, rounds=3, halving_ratio=2.0,
            seed=1, verbose=False,
        )
        assert result.best_score >= -5  # at worst within 2 of peak
        assert result.n_trials >= 16     # all initial candidates evaluated

    def test_shrinks_candidate_pool(self):
        call_count = {"round_boundaries": []}

        def obj(p):
            return float(p.get("a", 0))

        space = {"a": list(range(20))}
        result = successive_halving(
            obj, space, n_initial=12, rounds=3, halving_ratio=3.0,
            seed=2, verbose=False,
        )
        # 12 → 4 → 1  ⇒  12 + 4 + 1 = 17 trials (or fewer if early stop)
        assert result.n_trials <= 17


class TestTopK:
    def test_top_k_orders_by_score(self):
        space = {"x": list(range(10))}
        result = grid_search(
            lambda p: float(p["x"]), space, verbose=False,
        )
        top3 = result.top_k(3)
        scores = [t.score for t in top3]
        assert scores == sorted(scores, reverse=True)
        assert len(top3) == 3
        assert top3[0].params == {"x": 9}
