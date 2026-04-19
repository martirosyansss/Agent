"""Tests for the champion-challenger A/B framework.

Locks in the contracts:

* Below ``min_samples`` evaluate() returns None (no premature verdicts).
* Identical models → verdict == "hold" (no false promotions).
* Strong improvement → verdict == "promote" with low p-value and positive
  Wilson lower bound.
* Strong regression → verdict == "demote".
* McNemar's test handles small discordant counts via exact binomial
  fallback.
* Wilson lower bound is correctly negative when CI brackets zero.
* Comparator is thread-safe under concurrent record() calls.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from analyzer.ml.ab import (
    ChampionChallengerComparator,
    PredictionPair,
    mcnemars_test,
    wilson_lift_lower_bound,
)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


class TestMcNemar:
    def test_zero_disagreements_returns_one(self):
        chi2, p = mcnemars_test(0, 0)
        assert p == 1.0

    def test_perfectly_symmetric_disagreement_high_p_value(self):
        chi2, p = mcnemars_test(50, 50)
        assert p > 0.5  # no evidence of asymmetry

    def test_strongly_asymmetric_disagreement_low_p_value(self):
        chi2, p = mcnemars_test(0, 60)
        assert p < 0.001

    def test_small_n_uses_exact_binomial(self):
        # Below the n=25 cutoff, function should use exact test instead
        # of the chi² approximation. We check by passing very small disc
        # counts and verifying we get a finite p-value (not NaN from a
        # broken approximation).
        chi2, p = mcnemars_test(2, 3)
        assert 0.0 < p <= 1.0


class TestWilsonLiftLowerBound:
    def test_zero_when_either_n_is_zero(self):
        assert wilson_lift_lower_bound(0, 0, 5, 10) == 0.0
        assert wilson_lift_lower_bound(5, 10, 0, 0) == 0.0

    def test_strong_improvement_positive_lower_bound(self):
        # Challenger 90/100, champion 50/100 — clearly better.
        lo = wilson_lift_lower_bound(50, 100, 90, 100)
        assert lo > 0.20

    def test_negative_lower_bound_when_ci_brackets_zero(self):
        # Tiny lift on small N — CI should include zero (negative lower).
        lo = wilson_lift_lower_bound(50, 100, 52, 100)
        assert lo < 0.0


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


def _seed_pairs(comp: ChampionChallengerComparator, pairs):
    for p in pairs:
        comp.record(p.champion_pred, p.challenger_pred, p.actual_win)


class TestChampionChallengerComparator:
    def test_validates_window_vs_min_samples(self):
        with pytest.raises(ValueError):
            ChampionChallengerComparator("a", "b", window=10, min_samples=50)

    def test_returns_none_below_min_samples(self):
        comp = ChampionChallengerComparator("a", "b", min_samples=50)
        for _ in range(20):
            comp.record(True, True, True)
        assert comp.evaluate() is None

    def test_identical_models_verdict_hold(self):
        # Both models always make the same prediction.
        comp = ChampionChallengerComparator("a", "b", min_samples=50, window=200)
        rng = np.random.default_rng(0)
        for _ in range(120):
            same = bool(rng.integers(0, 2))
            actual = bool(rng.integers(0, 2))
            comp.record(same, same, actual)
        decision = comp.evaluate()
        assert decision is not None
        assert decision.verdict == "hold"
        assert abs(decision.precision_lift) < 0.05  # ~0 lift

    def test_strong_challenger_promoted(self):
        """Challenger picks winners far more reliably than champion.

        Setup: ~50% of trades are real wins. Champion predicts "win"
        randomly (precision ≈ 0.50). Challenger predicts "win" only when
        the trade is actually a winner 90% of the time (precision ≈ 0.85).
        Wide gap, large N → significance + practical lift → promote.
        """
        comp = ChampionChallengerComparator(
            "champ", "chall", min_samples=50, window=500,
            alpha=0.05, min_lift=0.05,
        )
        rng = np.random.default_rng(1)
        for _ in range(300):
            actual = int(rng.random() < 0.5)
            # Champion predicts 1 with 50% probability — uninformative.
            champ = int(rng.random() < 0.5)
            # Challenger predicts 1 reliably when actual=1, rarely when 0.
            if actual == 1:
                chall = int(rng.random() < 0.90)
            else:
                chall = int(rng.random() < 0.10)
            comp.record(bool(champ), bool(chall), bool(actual))
        decision = comp.evaluate()
        assert decision is not None, "expected a verdict at n=300"
        assert decision.precision_lift > 0.20, (
            f"expected large positive lift, got {decision.precision_lift:.3f}"
        )
        assert decision.mcnemar_p_value < 0.01, (
            f"expected p < 0.01, got {decision.mcnemar_p_value:.4f}"
        )
        assert decision.verdict == "promote", (
            f"got {decision.verdict}: {decision.reason}"
        )

    def test_strong_regression_demoted(self):
        """Challenger systematically WORSE than champion."""
        comp = ChampionChallengerComparator(
            "champ", "chall", min_samples=50, window=500,
            alpha=0.05, min_lift=0.05,
        )
        rng = np.random.default_rng(2)
        for _ in range(300):
            actual = int(rng.random() < 0.5)
            # Reverse roles from the previous test: champion is the
            # informed one, challenger is uninformative — challenger
            # should be demoted.
            if actual == 1:
                champ = int(rng.random() < 0.90)
            else:
                champ = int(rng.random() < 0.10)
            chall = int(rng.random() < 0.5)
            comp.record(bool(champ), bool(chall), bool(actual))
        decision = comp.evaluate()
        assert decision is not None
        assert decision.precision_lift < -0.10, (
            f"expected strong negative lift, got {decision.precision_lift:.3f}"
        )
        assert decision.mcnemar_p_value < 0.05
        assert decision.verdict == "demote", (
            f"got {decision.verdict}: {decision.reason}"
        )

    def test_marginal_lift_held_not_promoted(self):
        """Real but tiny lift — significance reached but Wilson lower
        bound below practical threshold → hold, not promote."""
        comp = ChampionChallengerComparator(
            "champ", "chall", min_samples=50, window=500,
            alpha=0.05, min_lift=0.10,  # demand 10% lift to promote
        )
        rng = np.random.default_rng(3)
        for _ in range(300):
            actual = 1
            # Both around 50% but challenger 1pp better — too small.
            champ = int(rng.random() < 0.50)
            chall = int(rng.random() < 0.51)
            comp.record(bool(champ), bool(chall), bool(actual))
        decision = comp.evaluate()
        assert decision is not None
        assert decision.verdict == "hold"

    def test_record_is_thread_safe(self):
        comp = ChampionChallengerComparator("a", "b", min_samples=50, window=2000)
        rng = np.random.default_rng(4)

        def worker():
            for _ in range(500):
                comp.record(
                    bool(rng.integers(0, 2)),
                    bool(rng.integers(0, 2)),
                    bool(rng.integers(0, 2)),
                )

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 4 × 500 records, but window cap keeps it ≤ 2000.
        assert comp.n_pairs == 2000
        # And evaluate must not raise.
        assert comp.evaluate() is not None
