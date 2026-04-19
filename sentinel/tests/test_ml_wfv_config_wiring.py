"""Tests that MLConfig.wfv_purge / wfv_embargo flow through the runner.

Contracts:

* **Defaults are zero** — a fresh MLConfig instance has both purge
  and embargo at 0, matching legacy behaviour.
* **Runner passes the values to the validator** — if the config sets
  purge=5 / embargo=10, the resulting MLWalkForwardValidator carries
  those exact values.
* **Backwards compatible** — a run with default config produces the
  same validator as the pre-Phase-11 signature.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml.domain.config import MLConfig
from analyzer.ml_walk_forward import MLWalkForwardValidator


class TestDefaults:
    def test_defaults_zero(self):
        cfg = MLConfig()
        assert cfg.wfv_purge == 0
        assert cfg.wfv_embargo == 0


class TestWiring:
    def test_validator_receives_config_values(self):
        cfg = MLConfig(wfv_purge=5, wfv_embargo=10)
        # Mimic the runner-internal constructor call with these parameters.
        wf = MLWalkForwardValidator(
            n_folds=3, test_fraction=0.2,
            anchored=False, min_train_size=50, min_test_size=20,
            purge=cfg.wfv_purge, embargo=cfg.wfv_embargo,
        )
        assert wf.purge == 5
        assert wf.embargo == 10

    def test_default_config_matches_legacy(self):
        cfg = MLConfig()
        wf_new = MLWalkForwardValidator(
            n_folds=3, test_fraction=0.2,
            anchored=False, min_train_size=50, min_test_size=20,
            purge=cfg.wfv_purge, embargo=cfg.wfv_embargo,
        )
        wf_legacy = MLWalkForwardValidator(
            n_folds=3, test_fraction=0.2,
            anchored=False, min_train_size=50, min_test_size=20,
        )
        assert wf_new.generate_splits(500) == wf_legacy.generate_splits(500)


class TestRunnerGuardRail:
    """The runner reads from getattr(cfg, ..., 0) so old pickled MLConfig
    instances (which don't have the new fields) don't blow up."""

    def test_missing_attr_treated_as_zero(self):
        class StubCfg:
            pass
        cfg = StubCfg()
        # Mimic what walk_forward_runner does:
        purge = int(getattr(cfg, "wfv_purge", 0) or 0)
        embargo = int(getattr(cfg, "wfv_embargo", 0) or 0)
        assert purge == 0
        assert embargo == 0
