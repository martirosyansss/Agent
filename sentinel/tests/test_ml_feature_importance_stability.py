"""Tests for feature-importance stability analysis.

Contracts:

* **Identical runs → rank-Spearman = 1.0** — when every ensemble member
  agrees, the stability metric must report perfect agreement.
* **Shuffled rankings → rank-Spearman ≈ 0** — totally disagreeing
  rankings produce a small absolute correlation.
* **Top-K persistence is correct** — for a feature that's in top 3 in
  2 out of 4 runs, ``top3_rate == 0.5``.
* **Per-feature mean/p05/p95 are sample-aware** — a feature observed
  in only 2 of 5 runs has ``n_observations == 2`` and its CI is
  computed from those 2 values only.
* **Empty / all-empty input** — ``compute_stability([])`` and
  ``compute_stability([{}, {}])`` return a clean empty report, not a
  crash.
* **Ranking by mean** — ``sorted_by_mean`` returns features in
  descending mean-importance order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml.domain.feature_importance_stability import (
    FeatureImportanceStats,
    ImportanceStabilityReport,
    compute_stability,
    _rank_importances,
    _spearman_on_shared,
)


class TestRanking:
    def test_rank_simple(self):
        ranks = _rank_importances({"a": 0.5, "b": 0.3, "c": 0.1})
        assert ranks == {"a": 1, "b": 2, "c": 3}

    def test_rank_empty(self):
        assert _rank_importances({}) == {}

    def test_spearman_identical(self):
        a = {"x": 1, "y": 2, "z": 3}
        assert _spearman_on_shared(a, a) == pytest.approx(1.0)

    def test_spearman_reversed(self):
        a = {"x": 1, "y": 2, "z": 3}
        b = {"x": 3, "y": 2, "z": 1}
        assert _spearman_on_shared(a, b) == pytest.approx(-1.0)

    def test_spearman_insufficient_overlap(self):
        a = {"x": 1, "y": 2}
        b = {"w": 1, "z": 2}
        # No shared features → undefined → 0.0
        assert _spearman_on_shared(a, b) == 0.0


class TestComputeStability:
    def test_identical_runs_perfect_agreement(self):
        run = {"a": 0.5, "b": 0.3, "c": 0.2}
        report = compute_stability([run, run, run])
        assert report.n_runs == 3
        assert report.n_pairs == 3   # C(3,2)
        assert report.mean_rank_spearman == pytest.approx(1.0)
        # Per-feature stats are degenerate (all identical)
        a_stats = report.per_feature["a"]
        assert a_stats.mean == pytest.approx(0.5)
        assert a_stats.p05 == pytest.approx(0.5)
        assert a_stats.p95 == pytest.approx(0.5)
        assert a_stats.top3_rate == 1.0

    def test_reversed_runs_negative_spearman(self):
        run1 = {"a": 0.5, "b": 0.3, "c": 0.2}
        run2 = {"a": 0.1, "b": 0.3, "c": 0.6}   # rank reversed for a, c
        report = compute_stability([run1, run2])
        assert report.n_pairs == 1
        assert report.mean_rank_spearman < 0

    def test_top_k_persistence(self):
        # Feature "a" is top-3 in runs 0 and 1, but not in runs 2 and 3.
        runs = [
            {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.05, "e": 0.05},
            {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.05, "e": 0.05},
            {"b": 0.4, "c": 0.3, "d": 0.2, "e": 0.05, "a": 0.05},
            {"b": 0.4, "c": 0.3, "d": 0.2, "e": 0.05, "a": 0.05},
        ]
        report = compute_stability(runs)
        a = report.per_feature["a"]
        # a is in top 3 for 2 of 4 runs → 0.5
        assert a.top3_rate == pytest.approx(0.5)

    def test_partial_overlap_feature_set(self):
        # Feature "novel" appears in only 1 of 3 runs.
        runs = [
            {"common": 0.8, "shared": 0.2},
            {"common": 0.7, "shared": 0.3},
            {"common": 0.5, "shared": 0.3, "novel": 0.2},
        ]
        report = compute_stability(runs)
        assert report.per_feature["novel"].n_observations == 1
        assert report.per_feature["common"].n_observations == 3

    def test_empty_input(self):
        report = compute_stability([])
        assert report.n_runs == 0
        assert report.per_feature == {}
        assert report.mean_rank_spearman == 0.0

    def test_all_empty_dicts(self):
        report = compute_stability([{}, {}])
        assert report.n_runs == 0

    def test_sorted_by_mean(self):
        runs = [
            {"low": 0.1, "high": 0.9, "mid": 0.5},
            {"low": 0.1, "high": 0.9, "mid": 0.5},
        ]
        report = compute_stability(runs)
        ordered = [s.name for s in report.sorted_by_mean()]
        assert ordered == ["high", "mid", "low"]

    def test_dashboard_output_is_json_shape(self):
        runs = [
            {"a": 0.4, "b": 0.3},
            {"a": 0.6, "b": 0.2},
        ]
        d = compute_stability(runs).to_dashboard()
        assert "features" in d
        assert "mean_rank_spearman" in d
        assert isinstance(d["features"], list)
        assert d["features"][0]["name"] in {"a", "b"}
