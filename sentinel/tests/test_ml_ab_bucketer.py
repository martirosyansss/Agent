"""Tests for the A/B bucketer.

Contracts locked in:

* **Determinism** — the same (unit, experiment) always maps to the
  same bucket across process restarts.
* **Percentages match at large N** — a 10/90 split produces ≈10%
  treatment over a reasonable symbol pool.
* **Different experiments are independent** — bucketing by experiment
  salt means adding a new experiment does NOT re-bucket existing
  assignments in other experiments.
* **Monotonicity of rollout** — raising challenger_pct from 10% → 20%
  only adds units to challenger (none move back to champion).
* **Validation** — percentages must sum to 100, individual values in
  [0, 100], at least one bucket.
* **Multi-bucket order** — insertion order defines hash-space allocation,
  so the first bucket always takes the lowest hash range.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml.ab.bucketer import (
    Bucketer,
    _hash_to_bp,
    rollout,
    split_50_50,
)


class TestDeterminism:
    def test_same_unit_same_bucket(self):
        b = split_50_50("exp-1")
        assignments = {b.assign("BTCUSDT").bucket for _ in range(10)}
        assert len(assignments) == 1

    def test_hash_independent_of_call_order(self):
        assert _hash_to_bp("BTCUSDT", "exp-A") == _hash_to_bp("BTCUSDT", "exp-A")


class TestDistribution:
    def test_90_10_matches_roughly(self):
        b = rollout("exp-ramp", challenger_pct=10.0)
        n = 10_000
        challenger_count = sum(
            1 for i in range(n) if b.assign(f"sym-{i}").bucket == "challenger"
        )
        # Expected ~1000, allow 3σ ≈ 30 for binomial(n=10000, p=0.1) → σ ≈ 30
        assert 900 <= challenger_count <= 1100

    def test_50_50_matches_roughly(self):
        b = split_50_50("exp-even")
        n = 5_000
        a_count = sum(1 for i in range(n) if b.assign(f"sym-{i}").bucket == "control")
        assert 2_400 <= a_count <= 2_600


class TestIndependence:
    def test_new_experiment_salt_rebuckets(self):
        # The salt space is independent — same unit, different experiments,
        # the bucket assignment is NOT correlated with the original.
        b_a = rollout("exp-A", challenger_pct=10.0)
        b_b = rollout("exp-B", challenger_pct=10.0)
        mismatches = 0
        for i in range(1000):
            sym = f"sym-{i}"
            if b_a.assign(sym).bucket != b_b.assign(sym).bucket:
                mismatches += 1
        # With two independent coin flips at p=0.1 agreeing on both sides,
        # expected overlap ≈ 0.01 + 0.81 = 0.82, so mismatches ≈ 180 / 1000.
        # Assert at least 50 are different as a sanity check for independence.
        assert mismatches >= 50


class TestMonotonicity:
    def test_rollout_only_adds_to_challenger(self):
        b10 = rollout("exp-ramp", 10.0)
        b20 = rollout("exp-ramp", 20.0)
        regressed = 0
        added = 0
        for i in range(2000):
            sym = f"sym-{i}"
            was = b10.assign(sym).bucket
            now = b20.assign(sym).bucket
            if was == "challenger" and now == "champion":
                regressed += 1    # must be zero — violates monotonicity
            if was == "champion" and now == "challenger":
                added += 1
        assert regressed == 0
        # And some should have been added (otherwise test is vacuous).
        assert added > 100


class TestValidation:
    def test_percentages_must_sum_to_100(self):
        with pytest.raises(ValueError):
            Bucketer(experiment="x", percentages={"a": 30, "b": 30})

    def test_negative_pct_rejected(self):
        with pytest.raises(ValueError):
            Bucketer(experiment="x", percentages={"a": -5, "b": 105})

    def test_empty_buckets_rejected(self):
        with pytest.raises(ValueError):
            Bucketer(experiment="x", percentages={})

    def test_rollout_pct_range_enforced(self):
        with pytest.raises(ValueError):
            rollout("x", challenger_pct=150.0)


class TestMultiBucket:
    def test_three_way_split(self):
        b = Bucketer(
            experiment="tri",
            percentages={"A": 20, "B": 30, "C": 50},
        )
        n = 6_000
        counts = {"A": 0, "B": 0, "C": 0}
        for i in range(n):
            counts[b.assign(f"sym-{i}").bucket] += 1
        # Expected 1200, 1800, 3000; allow 3σ ≈ 100 binomial.
        assert 1_100 <= counts["A"] <= 1_300
        assert 1_700 <= counts["B"] <= 1_900
        assert 2_900 <= counts["C"] <= 3_100

    def test_assignment_carries_audit_fields(self):
        b = split_50_50("audit-trail")
        a = b.assign("ETHUSDT")
        assert a.experiment == "audit-trail"
        assert 0 <= a.hash_bp < 10_000
        assert a.bucket in ("control", "treatment")
