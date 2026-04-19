"""Tests for scripts/tune_ml.py — the operator-facing tuning CLI.

Focused on the bits that don't require training a real ML model: CLI
argument parsing, search-space sanity, dry-run end-to-end, output
persistence. A full training-integration test would take minutes per
config and is left to the operator.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT = BASE_DIR / "scripts" / "tune_ml.py"


def _run(*extra_args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run the script from the sentinel root and return the completed process."""
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.tune_ml", *extra_args],
        cwd=str(BASE_DIR),
        capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"tune_ml exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


class TestSearchSpace:
    def test_default_space_shape(self):
        # Import directly so we validate the structure without spawning.
        sys.path.insert(0, str(BASE_DIR))
        from scripts.tune_ml import DEFAULT_SPACE

        assert "n_estimators" in DEFAULT_SPACE
        assert "learning_rate" in DEFAULT_SPACE
        assert all(isinstance(v, list) and len(v) >= 2 for v in DEFAULT_SPACE.values())


class TestDryRun:
    def test_dry_run_random_writes_output(self, tmp_path):
        out = tmp_path / "tune.json"
        _run(
            "--dry-run", "--mode", "random",
            "--n-trials", "10", "--seed", "1",
            "--output", str(out),
        )
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["mode"] == "random"
        assert data["n_trials"] <= 10
        assert "best_params" in data
        assert isinstance(data["top_k"], list)

    def test_dry_run_halving_writes_output(self, tmp_path):
        out = tmp_path / "tune.json"
        _run(
            "--dry-run", "--mode", "halving",
            "--n-initial", "6", "--rounds", "2",
            "--halving-ratio", "2.0", "--seed", "2",
            "--output", str(out),
        )
        data = json.loads(out.read_text())
        assert data["mode"] == "halving"
        assert data["n_trials"] >= 1

    def test_dry_run_grid_errors_on_overflow(self, tmp_path):
        # Default space has ~6912 cells; max_trials=50 must raise.
        proc = _run(
            "--dry-run", "--mode", "grid", "--n-trials", "50",
            "--output", str(tmp_path / "tune.json"),
            check=False,
        )
        # ValueError is raised inside the tuner; script propagates non-zero
        # only when it explicitly returns it, so grid-overflow bubbles up
        # as a Python traceback (returncode != 0). Either way must surface.
        combined = proc.stdout + proc.stderr
        assert "grid has" in combined or proc.returncode != 0


class TestMissingTrades:
    def test_error_when_trades_missing_and_not_dry_run(self, tmp_path):
        proc = _run(
            "--mode", "random", "--n-trials", "2",
            "--output", str(tmp_path / "tune.json"),
            check=False,
        )
        assert proc.returncode != 0
        assert "--trades is required" in (proc.stdout + proc.stderr)
