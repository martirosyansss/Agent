"""Tests for purged / embargoed walk-forward splits.

Contracts:

* **Backward compatibility** — ``purge=0, embargo=0`` (the defaults)
  produces identical splits to the legacy validator, bit-for-bit. No
  existing training run changes behaviour on upgrade.
* **Purge shrinks train from the tail** — a positive ``purge`` removes
  exactly that many samples from the end of the training window;
  ``test_start`` and ``test_end`` are untouched.
* **Embargo creates a gap** — a positive ``embargo`` enlarges the gap
  between ``train_end`` and ``test_start`` by exactly that many rows
  (implementation shrinks train_end because test positions are fixed
  by the budgeting logic).
* **Folds dropped when train becomes too small** — if ``purge + embargo``
  eats enough of the training window, the fold is skipped rather than
  producing a sub-``min_train_size`` fit.
* **Input validation** — negative ``purge`` / ``embargo`` raise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml_walk_forward import MLWalkForwardValidator


class TestBackwardCompat:
    def test_zero_purge_zero_embargo_matches_legacy(self):
        v_legacy = MLWalkForwardValidator(n_folds=3, min_train_size=50, min_test_size=20)
        v_new = MLWalkForwardValidator(
            n_folds=3, min_train_size=50, min_test_size=20,
            purge=0, embargo=0,
        )
        assert v_legacy.generate_splits(500) == v_new.generate_splits(500)


class TestPurge:
    def test_purge_shrinks_train_end(self):
        n = 500
        v = MLWalkForwardValidator(
            n_folds=3, min_train_size=50, min_test_size=20, purge=10,
        )
        splits = v.generate_splits(n)
        assert len(splits) >= 1
        for train_start, train_end, test_start, test_end in splits:
            # test_start must exceed train_end by at least the purge amount
            # because both purge and embargo=0 — only the tail of train is cut.
            assert test_start - train_end >= 10

    def test_purge_does_not_move_test_boundary(self):
        v_no = MLWalkForwardValidator(
            n_folds=3, min_train_size=50, min_test_size=20,
        )
        v_purge = MLWalkForwardValidator(
            n_folds=3, min_train_size=50, min_test_size=20, purge=15,
        )
        splits_no = v_no.generate_splits(500)
        splits_purge = v_purge.generate_splits(500)
        for s_no, s_p in zip(splits_no, splits_purge):
            # (train_start, train_end, test_start, test_end)
            assert s_no[2] == s_p[2]   # test_start unchanged
            assert s_no[3] == s_p[3]   # test_end unchanged


class TestEmbargo:
    def test_embargo_creates_gap_between_train_and_test(self):
        n = 500
        v = MLWalkForwardValidator(
            n_folds=3, min_train_size=50, min_test_size=20, embargo=20,
        )
        splits = v.generate_splits(n)
        assert len(splits) >= 1
        for train_start, train_end, test_start, test_end in splits:
            assert test_start - train_end >= 20

    def test_purge_and_embargo_are_additive(self):
        n = 800
        v = MLWalkForwardValidator(
            n_folds=3, min_train_size=80, min_test_size=30,
            purge=15, embargo=25,
        )
        splits = v.generate_splits(n)
        assert len(splits) >= 1
        for train_start, train_end, test_start, test_end in splits:
            # Both purge (15) and embargo (25) shrink train_end → gap ≥ 40.
            assert test_start - train_end >= 40


class TestFoldDroppingSafety:
    def test_excessive_purge_drops_folds(self):
        # Train pool is small; a very large purge should cause folds to be
        # rejected rather than producing a training window below min_train_size.
        n = 300
        v = MLWalkForwardValidator(
            n_folds=3, min_train_size=100, min_test_size=20, purge=200,
        )
        splits = v.generate_splits(n)
        # Every fold that survives has at least min_train_size rows.
        for s in splits:
            assert s[1] - s[0] >= 100


class TestValidation:
    def test_negative_purge_raises(self):
        with pytest.raises(ValueError):
            MLWalkForwardValidator(n_folds=3, purge=-1)

    def test_negative_embargo_raises(self):
        with pytest.raises(ValueError):
            MLWalkForwardValidator(n_folds=3, embargo=-5)
