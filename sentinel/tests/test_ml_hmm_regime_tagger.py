"""Tests for HMMRegimeTagger — the consumable HMM wrapper.

Contracts:

* **Labels are stable across fits** — two independent fits on similar
  data produce the same label set (``LABEL_LOW_VOL`` maps to the
  lower-variance state every time, not a random state index).
* **Variance ordering** — on a clearly two-regime synthetic series,
  the low-vol label picks up the low-vol data and the high-vol label
  picks up the high-vol data.
* **Unknown fallback** — ``predict_regime`` before ``fit()`` returns
  ``LABEL_UNKNOWN`` for every observation, no crash.
* **n_states validation** — 1 and 4+ states are rejected (vocabulary
  only covers 2 and 3).
* **Three-state vocabulary** — with n_states=3, the middle state
  resolves to ``LABEL_NEUTRAL``.
* **regime_statistics** — reports mean/std/stationary per label after
  a fit; empty before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml.labeling.hmm_regime_tagger import (
    HMMRegimeTagger,
    LABEL_HIGH_VOL,
    LABEL_LOW_VOL,
    LABEL_NEUTRAL,
    LABEL_UNKNOWN,
)


def _two_regime_series(n_per: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (series, truth) where truth is the ground-truth regime label."""
    rng = np.random.default_rng(seed)
    low = rng.normal(0, 1.0, size=n_per)
    high = rng.normal(0, 5.0, size=n_per)
    series = np.concatenate([low, high])
    truth = np.array([LABEL_LOW_VOL] * n_per + [LABEL_HIGH_VOL] * n_per)
    return series, truth


class TestStableLabels:
    def test_labels_are_variance_ordered(self):
        s, _ = _two_regime_series(seed=1)
        tagger = HMMRegimeTagger(n_states=2).fit(s, n_starts=3, seed=1)
        stats = tagger.regime_statistics()
        assert stats[LABEL_LOW_VOL]["std_return"] < stats[LABEL_HIGH_VOL]["std_return"]

    def test_refit_produces_same_label_vocabulary(self):
        s, _ = _two_regime_series(seed=2)
        t1 = HMMRegimeTagger(n_states=2).fit(s, seed=1)
        t2 = HMMRegimeTagger(n_states=2).fit(s, seed=2)
        # Both have the same label *vocabulary* regardless of internal
        # state-index shuffle.
        assert set(t1.state_to_label_.values()) == set(t2.state_to_label_.values())
        assert LABEL_LOW_VOL in t1.state_to_label_.values()
        assert LABEL_HIGH_VOL in t1.state_to_label_.values()


class TestPredictionQuality:
    def test_high_vol_section_dominates_high_vol_label(self):
        s, truth = _two_regime_series(seed=3)
        tagger = HMMRegimeTagger(n_states=2).fit(s, n_starts=3, seed=3)
        labels = tagger.predict_regime(s)
        # Most of the last half (generated as high-vol) is labelled high-vol.
        second_half = labels[len(s) // 2 :]
        high_count = sum(1 for lbl in second_half if lbl == LABEL_HIGH_VOL)
        assert high_count / len(second_half) > 0.7

    def test_current_regime_reads_last_obs(self):
        s, _ = _two_regime_series(seed=4)
        tagger = HMMRegimeTagger(n_states=2).fit(s, seed=4)
        last = tagger.current_regime(s)
        assert last in (LABEL_LOW_VOL, LABEL_HIGH_VOL)


class TestFallbacks:
    def test_predict_before_fit_returns_unknown(self):
        tagger = HMMRegimeTagger(n_states=2)
        labels = tagger.predict_regime([0.1, 0.2, -0.1])
        assert labels == [LABEL_UNKNOWN, LABEL_UNKNOWN, LABEL_UNKNOWN]

    def test_current_regime_before_fit(self):
        tagger = HMMRegimeTagger(n_states=2)
        assert tagger.current_regime([0.1, -0.2]) == LABEL_UNKNOWN

    def test_regime_statistics_before_fit(self):
        tagger = HMMRegimeTagger(n_states=2)
        assert tagger.regime_statistics() == {}


class TestValidation:
    @pytest.mark.parametrize("bad", [1, 0, 4, 5, -1])
    def test_invalid_n_states_raises(self, bad):
        with pytest.raises(ValueError):
            HMMRegimeTagger(n_states=bad)


class TestThreeState:
    def test_three_state_vocabulary_includes_neutral(self):
        rng = np.random.default_rng(7)
        s = np.concatenate([
            rng.normal(0, 0.5, 300),
            rng.normal(0, 2.0, 300),
            rng.normal(0, 6.0, 300),
        ])
        tagger = HMMRegimeTagger(n_states=3).fit(s, n_starts=3, seed=7)
        vocab = set(tagger.state_to_label_.values())
        assert vocab == {LABEL_LOW_VOL, LABEL_NEUTRAL, LABEL_HIGH_VOL}
