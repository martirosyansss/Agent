"""Tests for the PSI-based feature drift monitor.

Locks in the contracts:
* Identical distributions → PSI ≈ 0.
* Shifted distributions → PSI grows monotonically with the shift.
* Quantile bins are derived from the *reference* alone (otherwise PSI ≡ 0).
* Live values outside the reference range count toward the extreme bins
  (they do NOT silently disappear).
* The monitor is thread-safe (no exceptions under concurrent record/report).
* Insufficient live samples → ``report()`` returns ``[]`` (no spurious alarms).
* Schema mismatch (vector length ≠ trained features) is logged and skipped,
  not silently truncated/padded.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from analyzer.ml.monitoring.feature_drift import (
    FeatureDriftMonitor,
    PSI_MINOR_SHIFT,
    PSI_NO_SHIFT,
    population_stability_index,
)


# ---------------------------------------------------------------------------
# population_stability_index — pure function
# ---------------------------------------------------------------------------


class TestPopulationStabilityIndex:
    def test_identical_distributions_psi_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, size=2000)
        psi, edges = population_stability_index(x, x)
        # Same data both sides → PSI must be essentially zero.
        assert psi < 0.001
        assert edges.size > 1

    def test_resampled_same_distribution_psi_below_no_shift(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, size=2000)
        live = rng.normal(0, 1, size=2000)  # same dist, fresh sample
        psi, _ = population_stability_index(ref, live)
        # Sampling noise alone should not exceed the "no shift" cutoff.
        assert psi < PSI_NO_SHIFT

    def test_mean_shift_pushes_psi_above_minor(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, size=2000)
        live = rng.normal(0.5, 1, size=2000)  # 0.5σ shift
        psi, _ = population_stability_index(ref, live)
        assert psi > PSI_NO_SHIFT, f"0.5σ shift should be detectable, PSI={psi:.4f}"

    def test_large_shift_above_major_threshold(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(0, 1, size=2000)
        live = rng.normal(2.0, 1, size=2000)  # 2σ shift = strong drift
        psi, _ = population_stability_index(ref, live)
        assert psi > PSI_MINOR_SHIFT, f"2σ shift should trigger major drift, PSI={psi:.4f}"

    def test_psi_monotone_in_shift_magnitude(self):
        """The bigger the mean shift, the larger PSI must be (modulo
        sampling noise). This is the core property of any drift score."""
        rng = np.random.default_rng(4)
        ref = rng.normal(0, 1, size=3000)
        psis = []
        for shift in (0.0, 0.5, 1.0, 1.5, 2.0):
            live = rng.normal(shift, 1, size=3000)
            psi, _ = population_stability_index(ref, live)
            psis.append(psi)
        # Allow tiny tie-breaks from sampling noise via a 5% tolerance.
        for a, b in zip(psis, psis[1:]):
            assert b > a * 0.95, f"PSI not monotone: {psis}"

    def test_bin_edges_reused_for_consistency(self):
        """Calling PSI a second time with the returned edges must give a
        result identical to passing the same data without edges (bins are
        derived from reference only, so reuse is identity)."""
        rng = np.random.default_rng(5)
        ref = rng.normal(0, 1, size=1000)
        live = rng.normal(0.3, 1, size=1000)
        psi_a, edges = population_stability_index(ref, live)
        psi_b, _ = population_stability_index(ref, live, bin_edges=edges)
        assert psi_a == pytest.approx(psi_b)

    def test_handles_nan_input_without_blowing_up(self):
        ref = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        live = np.array([1.5, np.nan, 2.5, 3.5, 4.5])
        psi, _ = population_stability_index(ref, live, n_bins=3)
        assert np.isfinite(psi)

    def test_degenerate_reference_returns_zero(self):
        # All-constant reference → bin edges collapse → PSI undefined.
        # We agreed to return 0 (and an empty/singleton edges array)
        # rather than blow up — caller filters by edges.size.
        ref = np.full(100, 3.14)
        live = np.array([1.0, 2.0, 3.0, 4.0])
        psi, edges = population_stability_index(ref, live)
        assert psi == 0.0
        assert edges.size < 2

    def test_empty_inputs_return_zero(self):
        psi, _ = population_stability_index(np.array([]), np.array([1.0]))
        assert psi == 0.0
        psi, _ = population_stability_index(np.array([1.0]), np.array([]))
        assert psi == 0.0

    def test_out_of_range_live_values_count_in_extreme_bins(self):
        """A live observation outside the reference range MUST contribute
        to PSI (otherwise we'd silently miss the most dangerous drift —
        a feature exploding to a value never seen in training)."""
        rng = np.random.default_rng(6)
        ref = rng.normal(0, 1, size=2000)
        # Live points all sit at +10σ — far outside reference range.
        live = np.full(500, 10.0)
        psi, _ = population_stability_index(ref, live)
        assert psi > PSI_MINOR_SHIFT, (
            f"Out-of-range live values should trigger major drift, PSI={psi:.4f}"
        )


# ---------------------------------------------------------------------------
# FeatureDriftMonitor — stateful API
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor():
    return FeatureDriftMonitor(window=200, min_live_samples=50, n_bins=5)


def _train_matrix(n: int, seed: int = 7) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.normal(0, 1, size=n),     # feat_0 — Gaussian
        rng.uniform(0, 100, size=n),  # feat_1 — uniform
        rng.exponential(1, size=n),   # feat_2 — skewed
    ])
    return X, ["feat_0", "feat_1", "feat_2"]


class TestFeatureDriftMonitor:
    def test_validates_window_vs_min_live(self):
        with pytest.raises(ValueError):
            FeatureDriftMonitor(window=10, min_live_samples=50)

    def test_no_report_before_fit_reference(self, monitor):
        assert monitor.report() == []
        assert monitor.summary()["status"] == "insufficient_data"

    def test_fit_reference_validates_shape(self, monitor):
        X = np.zeros((100, 3))
        with pytest.raises(ValueError):
            monitor.fit_reference(X, ["a", "b"])  # 2 names but 3 cols

    def test_no_report_until_min_live_samples(self, monitor):
        X, names = _train_matrix(500)
        monitor.fit_reference(X, names)
        # Push 30 samples (< min_live=50) — report should be empty.
        rng = np.random.default_rng(100)
        for _ in range(30):
            monitor.record(rng.normal(size=3))
        assert monitor.report() == []
        assert monitor.summary()["status"] == "insufficient_data"

    def test_stable_distribution_reports_no_drift(self, monitor):
        X, names = _train_matrix(2000)
        monitor.fit_reference(X, names)
        # Feed live samples drawn from the SAME distribution.
        rng = np.random.default_rng(101)
        for _ in range(200):
            monitor.record([
                rng.normal(0, 1),
                rng.uniform(0, 100),
                rng.exponential(1),
            ])
        reps = monitor.report()
        assert len(reps) == 3
        for r in reps:
            assert r.severity == "stable", f"{r.feature} false-positive: PSI={r.psi:.3f}"
        assert monitor.summary()["status"] == "ok"

    def test_shifted_distribution_flagged_as_major(self, monitor):
        X, names = _train_matrix(2000)
        monitor.fit_reference(X, names)
        # feat_0 shifted by 2σ; others stable.
        rng = np.random.default_rng(102)
        for _ in range(200):
            monitor.record([
                rng.normal(2.0, 1),         # SHIFTED
                rng.uniform(0, 100),
                rng.exponential(1),
            ])
        rep = monitor.report()
        # Sorted by PSI desc — shifted feature should top the list.
        assert rep[0].feature == "feat_0"
        assert rep[0].severity == "major", f"PSI for shifted feat={rep[0].psi:.3f}"
        # Others should still be stable.
        for r in rep[1:]:
            assert r.severity in ("stable", "minor")
        summary = monitor.summary()
        assert summary["status"] == "major_drift"
        assert "feat_0" in summary["drifting"]

    def test_schema_mismatch_skipped_not_truncated(self, monitor, caplog):
        import logging
        X, names = _train_matrix(500)
        monitor.fit_reference(X, names)
        with caplog.at_level(logging.WARNING):
            monitor.record([1.0, 2.0])  # only 2 values, expects 3
        # Live deques untouched — confirms no silent truncation.
        assert monitor.n_live_samples() == 0
        # And we logged about it so operators notice.
        assert any("doesn't match" in m or "match" in m or "skipping" in m.lower()
                   for m in (r.getMessage() for r in caplog.records))

    def test_nan_live_values_not_recorded(self, monitor):
        X, names = _train_matrix(500)
        monitor.fit_reference(X, names)
        for _ in range(60):
            monitor.record([1.0, float("nan"), 0.5])
        # feat_1 saw only NaNs → empty live → still insufficient data overall.
        assert min(len(monitor._live[n]) for n in names) == 0

    def test_record_no_op_when_reference_unset(self):
        m = FeatureDriftMonitor(window=200, min_live_samples=50)
        # Should not raise — production safety: predict() can fire before
        # the first retrain has populated the monitor.
        m.record([1.0, 2.0, 3.0])
        assert m.n_live_samples() == 0

    def test_pickle_roundtrip_preserves_reference_and_live(self):
        """The trainer persists the monitor in the model pickle. After a
        process restart + load, PSI scoring must still work and the live
        window must be preserved up to the restart moment."""
        import pickle
        m = FeatureDriftMonitor(window=200, min_live_samples=50, n_bins=5)
        X, names = _train_matrix(2000)
        m.fit_reference(X, names)
        rng = np.random.default_rng(99)
        for _ in range(120):
            m.record([rng.normal(2.0, 1), rng.uniform(0, 100), rng.exponential(1)])
        rep_before = m.report()
        assert rep_before, "expected non-empty report before pickle"

        blob = pickle.dumps(m)
        m2: FeatureDriftMonitor = pickle.loads(blob)
        # Lock survives the round-trip (re-instantiated by __setstate__).
        assert m2._lock is not None
        # Live deques and reference snapshots survive.
        assert m2.n_live_samples() == m.n_live_samples()
        rep_after = m2.report()
        assert len(rep_after) == len(rep_before)
        # Monitor stays usable post-load: a new record() actually appends
        # (one more sample than the pre-pickle snapshot), and report() is
        # still callable without raising on the rebuilt lock.
        n_before_extra = m2.n_live_samples()
        m2.record([0.0, 50.0, 1.0])
        assert m2.n_live_samples() == n_before_extra + 1
        assert m2.report(), "report() should still work after pickle round-trip"

    def test_thread_safety_under_concurrent_record(self, monitor):
        X, names = _train_matrix(500)
        monitor.fit_reference(X, names)
        rng = np.random.default_rng(103)
        sample = lambda: rng.normal(size=3).tolist()

        def worker():
            for _ in range(500):
                monitor.record(sample())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Each deque is bounded to window=200, so we can't exceed that
        # regardless of how many writers we had.
        for name in names:
            assert len(monitor._live[name]) <= 200
        # And report() must still work (no race-condition exception).
        reps = monitor.report()
        assert len(reps) == 3
