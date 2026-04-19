"""Tests for the lightweight DAG orchestrator.

Locks in the contracts:

* Topological sort orders stages so each runs after its dependencies.
* Cycles raise ``ValueError`` (don't deadlock or loop).
* Unknown dependency names raise ``ValueError`` immediately.
* Stage failure marks downstream stages as skipped, not failed.
* Retries respect ``Stage.retries`` and the runner sleeps ``backoff_sec``.
* Idempotent stages skip when output already in Context.
* Pipeline writes summary metrics to the registry when one is wired.
* Successful pipeline returns ``status="succeeded"`` and all stage
  outputs are accessible via ``Context.data``.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from analyzer.ml.orchestration import (
    Context,
    PipelineRunner,
    Stage,
    topological_order,
)
from analyzer.ml.registry import MLRegistry


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


class TestTopologicalOrder:
    def test_linear_chain(self):
        a = Stage("a", run=lambda ctx: 1)
        b = Stage("b", run=lambda ctx: 2, depends_on=["a"])
        c = Stage("c", run=lambda ctx: 3, depends_on=["b"])
        ordered = topological_order([c, b, a])
        names = [s.name for s in ordered]
        assert names == ["a", "b", "c"]

    def test_diamond(self):
        # a → b, c; b, c → d
        a = Stage("a", run=lambda ctx: 0)
        b = Stage("b", run=lambda ctx: 0, depends_on=["a"])
        c = Stage("c", run=lambda ctx: 0, depends_on=["a"])
        d = Stage("d", run=lambda ctx: 0, depends_on=["b", "c"])
        ordered = [s.name for s in topological_order([d, c, b, a])]
        # a first, d last; b/c order doesn't matter as long as they're
        # between a and d.
        assert ordered[0] == "a"
        assert ordered[-1] == "d"
        assert set(ordered[1:3]) == {"b", "c"}

    def test_cycle_raises(self):
        a = Stage("a", run=lambda ctx: 0, depends_on=["b"])
        b = Stage("b", run=lambda ctx: 0, depends_on=["a"])
        with pytest.raises(ValueError, match="cycle"):
            topological_order([a, b])

    def test_unknown_dep_raises(self):
        a = Stage("a", run=lambda ctx: 0, depends_on=["nonexistent"])
        with pytest.raises(ValueError, match="unknown stage"):
            topological_order([a])

    def test_duplicate_names_raise(self):
        a1 = Stage("a", run=lambda ctx: 0)
        a2 = Stage("a", run=lambda ctx: 0)
        with pytest.raises(ValueError, match="duplicate"):
            topological_order([a1, a2])


# ---------------------------------------------------------------------------
# Runner — happy path
# ---------------------------------------------------------------------------


class TestRunnerHappyPath:
    def test_linear_pipeline_runs_in_order_and_collects_outputs(self):
        log = []

        def make(name):
            def fn(ctx):
                log.append(name)
                return f"out_{name}"
            return fn

        stages = [
            Stage("a", run=make("a")),
            Stage("b", run=make("b"), depends_on=["a"]),
            Stage("c", run=make("c"), depends_on=["b"]),
        ]
        runner = PipelineRunner()
        result = runner.run("test", stages)
        assert result.succeeded
        assert log == ["a", "b", "c"]
        outputs = {s.name: s.output for s in result.stages}
        assert outputs == {"a": "out_a", "b": "out_b", "c": "out_c"}

    def test_outputs_accessible_through_context(self):
        seen_in_b = {}

        def stage_a(ctx: Context):
            return {"value": 42}

        def stage_b(ctx: Context):
            seen_in_b.update(ctx.data["a"])
            return None

        runner = PipelineRunner()
        runner.run("test", [
            Stage("a", run=stage_a),
            Stage("b", run=stage_b, depends_on=["a"]),
        ])
        assert seen_in_b == {"value": 42}


# ---------------------------------------------------------------------------
# Runner — failure handling
# ---------------------------------------------------------------------------


class TestRunnerFailureHandling:
    def test_stage_failure_skips_downstream(self):
        def fail(ctx):
            raise RuntimeError("kaboom")

        called = []

        def downstream(ctx):
            called.append("d")

        stages = [
            Stage("a", run=lambda ctx: "ok"),
            Stage("b", run=fail, depends_on=["a"]),
            Stage("c", run=downstream, depends_on=["b"]),
        ]
        runner = PipelineRunner()
        result = runner.run("test", stages)
        assert not result.succeeded
        assert result.failed_stage == "b"
        assert called == []  # c never ran
        statuses = {s.name: s.status for s in result.stages}
        assert statuses == {"a": "succeeded", "b": "failed", "c": "skipped"}

    def test_retries_succeed_within_budget(self):
        attempts = [0]

        def flaky(ctx):
            attempts[0] += 1
            if attempts[0] < 3:
                raise RuntimeError("transient")
            return "ok"

        runner = PipelineRunner()
        result = runner.run("test", [
            Stage("a", run=flaky, retries=3, backoff_sec=0.0),
        ])
        assert result.succeeded
        assert result.stages[0].attempts == 3

    def test_retries_exhausted_marks_failed(self):
        def always_fail(ctx):
            raise RuntimeError("permanent")

        runner = PipelineRunner()
        result = runner.run("test", [
            Stage("a", run=always_fail, retries=2, backoff_sec=0.0),
        ])
        assert not result.succeeded
        assert result.stages[0].attempts == 3  # 1 + 2 retries

    def test_backoff_actually_waits_between_attempts(self):
        attempts = [0]

        def flaky(ctx):
            attempts[0] += 1
            if attempts[0] < 2:
                raise RuntimeError("transient")
            return "ok"

        runner = PipelineRunner()
        t0 = time.time()
        result = runner.run("test", [
            Stage("a", run=flaky, retries=1, backoff_sec=0.1),
        ])
        elapsed = time.time() - t0
        assert result.succeeded
        # We slept 0.1s between the failed attempt and the retry.
        assert elapsed >= 0.09


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_idempotent_stage_skipped_when_cached(self):
        called = [0]

        def fn(ctx):
            called[0] += 1
            return "x"

        stage = Stage("a", run=fn, idempotent=True)
        runner = PipelineRunner()

        # First run — executes.
        result = runner.run("test", [stage])
        assert called[0] == 1
        assert result.stages[0].status == "succeeded"

        # Manually pre-populate context (simulating resume from prior run)
        # by injecting a stage whose output overlaps. We do this through
        # a wrapper stage that primes Context.data["a"] before "a" runs.
        called[0] = 0

        def primer(ctx):
            ctx.data["a"] = "cached"
            return None

        primer_stage = Stage("primer", run=primer)
        result2 = runner.run("test", [primer_stage, stage])
        # The idempotent "a" stage sees its key already in ctx.data and
        # skips execution.
        assert called[0] == 0
        a_result = next(s for s in result2.stages if s.name == "a")
        assert a_result.status == "skipped"
        assert a_result.output == "cached"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_pipeline_logs_run_to_registry(self, tmp_path: Path):
        registry = MLRegistry(tmp_path / "reg.db")
        runner = PipelineRunner(registry=registry)
        result = runner.run("retrain", [
            Stage("a", run=lambda ctx: "ok"),
        ])
        assert result.succeeded

        runs = registry.list_runs("retrain")
        assert len(runs) == 1
        run = runs[0]
        assert run.status == "finished"
        assert run.metrics["n_succeeded"] == 1
        assert run.metrics["n_failed"] == 0

    def test_failed_pipeline_recorded_with_failed_stage(self, tmp_path: Path):
        registry = MLRegistry(tmp_path / "reg.db")
        runner = PipelineRunner(registry=registry)

        def boom(ctx):
            raise RuntimeError("kaboom")

        runner.run("retrain", [
            Stage("a", run=lambda ctx: "ok"),
            Stage("b", run=boom, depends_on=["a"]),
        ])
        run = registry.list_runs("retrain")[0]
        assert run.status == "failed"
        assert run.error == "b"
        assert run.metrics["n_failed"] == 1
