"""Tests for the SQL-backed ML model registry.

Locks in the contracts:

* Schema is created idempotently on first connection.
* Runs and model versions are persisted across connections.
* Auto-incrementing version numbers per model name.
* Promotion to ``production`` atomically demotes the existing champion.
* ``get_production`` returns the latest active production version, never
  an archived one.
* register_model rejects unknown run_ids (foreign-key sanity).
* Concurrent writers can't corrupt the version sequence (lock test).
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from analyzer.ml.registry import (
    MLRegistry,
    STAGE_ARCHIVED,
    STAGE_NONE,
    STAGE_PRODUCTION,
    STAGE_STAGING,
)


@pytest.fixture
def registry(tmp_path: Path) -> MLRegistry:
    return MLRegistry(tmp_path / "test_registry.db")


# ---------------------------------------------------------------------------
# Schema + connection
# ---------------------------------------------------------------------------


class TestSchema:
    def test_creates_tables_on_construction(self, tmp_path: Path):
        db_path = tmp_path / "fresh.db"
        MLRegistry(db_path)
        # Connect manually and check the tables exist.
        conn = sqlite3.connect(db_path)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "ml_runs" in names
        assert "ml_model_versions" in names

    def test_schema_creation_is_idempotent(self, tmp_path: Path):
        db_path = tmp_path / "twice.db"
        MLRegistry(db_path)
        # Second construction must not raise.
        MLRegistry(db_path)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class TestRuns:
    def test_start_and_finish_run(self, registry: MLRegistry):
        run = registry.start_run("trainer", params={"lr": 0.05})
        assert run.run_id.startswith("run_")
        assert run.status == "running"

        registry.finish_run(run.run_id, status="finished")
        loaded = registry.get_run(run.run_id)
        assert loaded.status == "finished"
        assert loaded.finished_at is not None
        assert loaded.params == {"lr": 0.05}

    def test_finish_run_validates_status(self, registry: MLRegistry):
        run = registry.start_run("trainer")
        with pytest.raises(ValueError):
            registry.finish_run(run.run_id, status="bogus")

    def test_log_metrics_merges_into_existing(self, registry: MLRegistry):
        run = registry.start_run("trainer")
        registry.log_metrics(run.run_id, {"precision": 0.7})
        registry.log_metrics(run.run_id, {"recall": 0.6, "auc": 0.8})
        loaded = registry.get_run(run.run_id)
        assert loaded.metrics == {"precision": 0.7, "recall": 0.6, "auc": 0.8}

    def test_log_metrics_unknown_run_raises(self, registry: MLRegistry):
        with pytest.raises(ValueError):
            registry.log_metrics("nonexistent", {"x": 1})

    def test_log_artifact_records_path(self, registry: MLRegistry):
        run = registry.start_run("trainer")
        registry.log_artifact(run.run_id, "model_pkl", "/tmp/model.pkl")
        registry.log_artifact(run.run_id, "report", "/tmp/report.html")
        loaded = registry.get_run(run.run_id)
        assert loaded.artifacts == {
            "model_pkl": "/tmp/model.pkl",
            "report": "/tmp/report.html",
        }

    def test_finish_run_with_error(self, registry: MLRegistry):
        run = registry.start_run("trainer")
        registry.finish_run(run.run_id, status="failed", error="OOM")
        loaded = registry.get_run(run.run_id)
        assert loaded.status == "failed"
        assert loaded.error == "OOM"

    def test_list_runs_orders_by_start_desc(self, registry: MLRegistry):
        for i in range(3):
            registry.start_run("trainer", params={"i": i})
            time.sleep(0.005)  # ensure distinct started_at
        runs = registry.list_runs("trainer")
        assert len(runs) == 3
        assert runs[0].started_at >= runs[1].started_at >= runs[2].started_at


# ---------------------------------------------------------------------------
# Model versions
# ---------------------------------------------------------------------------


class TestModelVersions:
    def test_register_model_assigns_v1_first(self, registry: MLRegistry):
        run = registry.start_run("trainer")
        mv = registry.register_model("ensemble", run.run_id, "/tmp/m.pkl")
        assert mv.version == 1
        assert mv.stage == STAGE_NONE

    def test_register_model_auto_increments_per_name(self, registry: MLRegistry):
        for i in range(3):
            run = registry.start_run("trainer")
            registry.register_model("ensemble", run.run_id, f"/tmp/m{i}.pkl")
        # Independent name has its own counter.
        run = registry.start_run("trainer")
        regime_mv = registry.register_model("regime_router", run.run_id, "/tmp/r.pkl")

        versions = registry.list_versions("ensemble")
        assert sorted(v.version for v in versions) == [1, 2, 3]
        assert regime_mv.version == 1

    def test_register_model_rejects_unknown_run_id(self, registry: MLRegistry):
        with pytest.raises(ValueError):
            registry.register_model("ensemble", "nonexistent_run", "/tmp/m.pkl")

    def test_transition_stage_validates_stage(self, registry: MLRegistry):
        run = registry.start_run("trainer")
        registry.register_model("ensemble", run.run_id, "/tmp/m.pkl")
        with pytest.raises(ValueError):
            registry.transition_stage("ensemble", 1, "bogus")

    def test_transition_stage_unknown_version(self, registry: MLRegistry):
        with pytest.raises(ValueError):
            registry.transition_stage("ensemble", 999, STAGE_STAGING)

    def test_promote_to_production_archives_old_champion(self, registry: MLRegistry):
        # Set up: register two versions, promote v1 to production, then
        # promote v2. v1 should be auto-archived.
        for i in range(2):
            run = registry.start_run("trainer")
            registry.register_model("ensemble", run.run_id, f"/tmp/m{i}.pkl")
        registry.transition_stage("ensemble", 1, STAGE_PRODUCTION)
        assert registry.get_production("ensemble").version == 1
        registry.transition_stage("ensemble", 2, STAGE_PRODUCTION)
        prod = registry.get_production("ensemble")
        assert prod.version == 2
        # Old champion now archived.
        v1 = next(v for v in registry.list_versions("ensemble") if v.version == 1)
        assert v1.stage == STAGE_ARCHIVED

    def test_promote_with_archive_disabled_keeps_old(self, registry: MLRegistry):
        for i in range(2):
            run = registry.start_run("trainer")
            registry.register_model("ensemble", run.run_id, f"/tmp/m{i}.pkl")
        registry.transition_stage("ensemble", 1, STAGE_PRODUCTION)
        registry.transition_stage("ensemble", 2, STAGE_PRODUCTION,
                                  archive_existing=False)
        # Both will now be production — get_production picks the latest.
        prod = registry.get_production("ensemble")
        assert prod.version == 2
        all_versions = registry.list_versions("ensemble", stage=STAGE_PRODUCTION)
        assert len(all_versions) == 2

    def test_get_production_returns_none_when_unset(self, registry: MLRegistry):
        run = registry.start_run("trainer")
        registry.register_model("ensemble", run.run_id, "/tmp/m.pkl")
        # Stage is "none" — no production yet.
        assert registry.get_production("ensemble") is None

    def test_archived_excluded_from_get_production(self, registry: MLRegistry):
        run = registry.start_run("trainer")
        registry.register_model("ensemble", run.run_id, "/tmp/m.pkl")
        registry.transition_stage("ensemble", 1, STAGE_PRODUCTION)
        registry.transition_stage("ensemble", 1, STAGE_ARCHIVED)
        assert registry.get_production("ensemble") is None

    def test_list_versions_filters_by_stage(self, registry: MLRegistry):
        for i in range(3):
            run = registry.start_run("trainer")
            registry.register_model("ensemble", run.run_id, f"/tmp/m{i}.pkl")
        registry.transition_stage("ensemble", 1, STAGE_STAGING)
        registry.transition_stage("ensemble", 2, STAGE_PRODUCTION)
        # v3 stays at "none"

        none_only = registry.list_versions("ensemble", stage=STAGE_NONE)
        staging_only = registry.list_versions("ensemble", stage=STAGE_STAGING)
        prod_only = registry.list_versions("ensemble", stage=STAGE_PRODUCTION)
        all_versions = registry.list_versions("ensemble")
        assert {v.version for v in none_only} == {3}
        assert {v.version for v in staging_only} == {1}
        assert {v.version for v in prod_only} == {2}
        assert len(all_versions) == 3


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrentWrites:
    def test_parallel_register_assigns_unique_versions(
        self, registry: MLRegistry,
    ):
        """Two threads registering the same model name in parallel must
        get distinct version numbers — not the same v1 because both read
        MAX(version)=0 before either insert."""
        run = registry.start_run("trainer")

        results: list[int] = []
        lock = threading.Lock()

        def worker():
            mv = registry.register_model("ensemble", run.run_id, "/tmp/m.pkl")
            with lock:
                results.append(mv.version)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == list(range(1, 9)), (
            f"version numbers collided: {sorted(results)}"
        )
