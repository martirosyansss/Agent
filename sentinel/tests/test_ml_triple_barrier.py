"""Tests for triple-barrier labelling + meta-labels (López de Prado).

Contracts:

* **Upper / lower / vertical** — the three barrier cases each produce
  the right label, exit reason, and duration.
* **Direction flip** — a short position with otherwise-identical inputs
  produces the opposite labels; barriers swap semantics cleanly.
* **Vertical terminal sign** — when no barrier hits within ``horizon``,
  the label is the sign of the terminal-vs-entry return (+1, 0, or -1).
* **Degenerate inputs** — zero/negative barriers, out-of-range indices,
  zero volatility all return ``label=0, exit_reason="invalid"`` without
  crashing.
* **Meta-labels** — ``build_meta_labels`` produces 1 only when the
  primary fired AND was correct in sign; 0 when it didn't fire at all
  or fired wrongly.
* **Meta-filter** — ``meta_filter`` preserves the primary's sign when
  secondary clears the threshold; zeros it out otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml.domain.triple_barrier import (
    BarrierResult,
    build_meta_labels,
    meta_filter,
    triple_barrier_label,
    triple_barrier_labels_batch,
)


class TestSingleBarrier:
    def test_upper_barrier_long(self):
        # Price climbs from 100 → 110 over 5 bars, pt=5 should fire by bar 3.
        prices = [100, 101, 103, 106, 110, 112]
        r = triple_barrier_label(prices, 0, pt_abs=5.0, sl_abs=3.0, horizon=5)
        assert r.label == +1
        assert r.exit_reason == "upper"
        assert r.exit_idx == 3
        assert r.duration == 3

    def test_lower_barrier_long(self):
        prices = [100, 99, 97, 94, 92, 95]
        r = triple_barrier_label(prices, 0, pt_abs=5.0, sl_abs=5.0, horizon=5)
        assert r.label == -1
        assert r.exit_reason == "lower"
        assert r.exit_idx == 3   # first index where price ≤ 95

    def test_vertical_positive_terminal(self):
        # Neither barrier hits; terminal price > entry → label +1.
        prices = [100, 101, 101, 102, 102, 102.5]
        r = triple_barrier_label(prices, 0, pt_abs=10.0, sl_abs=10.0, horizon=5)
        assert r.label == +1
        assert r.exit_reason == "vertical"

    def test_vertical_flat_terminal(self):
        prices = [100, 101, 99, 100, 101, 100]
        r = triple_barrier_label(prices, 0, pt_abs=10.0, sl_abs=10.0, horizon=5)
        assert r.label == 0
        assert r.exit_reason == "vertical"

    def test_short_direction_reverses(self):
        # Same price path as the "upper barrier long" case, but short.
        prices = [100, 101, 103, 106, 110, 112]
        r = triple_barrier_label(
            prices, 0, pt_abs=5.0, sl_abs=3.0, horizon=5, direction=-1,
        )
        # For a short, price rising 3 points first = stop-loss hit.
        assert r.label == -1
        assert r.exit_reason == "lower"


class TestDegenerateInputs:
    def test_empty_prices(self):
        r = triple_barrier_label([], 0, pt_abs=1, sl_abs=1, horizon=5)
        assert r.label == 0
        assert r.exit_reason == "invalid"

    def test_negative_barriers(self):
        r = triple_barrier_label([100, 101], 0, pt_abs=-1, sl_abs=1, horizon=2)
        assert r.exit_reason == "invalid"

    def test_zero_horizon(self):
        r = triple_barrier_label([100, 101], 0, pt_abs=1, sl_abs=1, horizon=0)
        assert r.exit_reason == "invalid"

    def test_out_of_range_entry_idx(self):
        r = triple_barrier_label([100, 101], 10, pt_abs=1, sl_abs=1, horizon=2)
        assert r.exit_reason == "invalid"


class TestBatch:
    def test_batch_matches_individual(self):
        prices = list(range(100, 200))
        vol = [2.0] * len(prices)
        entries = [0, 10, 50]
        batch = triple_barrier_labels_batch(
            prices, entries, volatility=vol,
            pt_mult=3.0, sl_mult=1.5, horizon=10,
        )
        for k, idx in enumerate(entries):
            expected = triple_barrier_label(
                prices, idx, pt_abs=6.0, sl_abs=3.0, horizon=10,
            )
            assert batch[k].label == expected.label
            assert batch[k].exit_idx == expected.exit_idx

    def test_batch_zero_volatility_invalidates(self):
        prices = [100, 101, 102]
        batch = triple_barrier_labels_batch(
            prices, [0, 1], volatility=[0.0, 0.0],
            pt_mult=3.0, sl_mult=1.5, horizon=2,
        )
        assert all(b.exit_reason == "invalid" for b in batch)


class TestMetaLabels:
    def test_build_meta_labels_positive_match(self):
        primary = np.array([+1, -1, +1, 0])
        realised = np.array([+1, -1, -1, +1])
        meta = build_meta_labels(primary, realised)
        # Correct sign match: [1, 1, 0, 0] — last is 0 because primary didn't fire.
        assert list(meta) == [1, 1, 0, 0]

    def test_build_meta_labels_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_meta_labels([1, 2, 3], [1, 2])


class TestMetaFilter:
    def test_meta_filter_keeps_above_threshold(self):
        primary = np.array([+1, -1, +1, 0])
        probs = np.array([0.8, 0.4, 0.9, 0.6])
        filtered = meta_filter(primary, probs, threshold=0.5)
        # primary=+1 with prob 0.8 → kept; primary=-1 with prob 0.4 → zeroed;
        # primary=+1 with prob 0.9 → kept; primary=0 stays 0 regardless.
        assert list(filtered) == [1, 0, 1, 0]

    def test_meta_filter_all_below_threshold(self):
        primary = np.array([+1, -1, +1])
        probs = np.array([0.1, 0.1, 0.1])
        filtered = meta_filter(primary, probs, threshold=0.5)
        assert np.all(filtered == 0)

    def test_meta_filter_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            meta_filter([1, 2], [0.5, 0.6, 0.7])
