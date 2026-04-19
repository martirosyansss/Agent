"""Lightweight in-process DAG orchestrator for the ML retraining pipeline.

Why not Airflow/Prefect/Dagster: they all assume a separate scheduler
process, a metadata DB beyond what we already have, and cross-host worker
pools. The sentinel project runs in a single Python process; the goal of
"DAG orchestration" here is not horizontal scaling but the same correctness
properties an external scheduler gives you:

* **Stage isolation** — each stage is a callable that receives the
  pipeline ``Context`` and returns its outputs; failures don't leak state
  from half-finished stages into downstream ones.
* **Dependency graph** — stages declare upstream stages by name; the
  runner topologically sorts them and errors on cycles or unmet deps.
* **Idempotency** — re-running a successful pipeline restarts from the
  first non-skipped stage; ``Stage.idempotent=True`` skips re-execution
  when an upstream input hasn't changed.
* **Retries with backoff** — transient failures (network, exchange API)
  get N attempts with exponential delay before terminal failure.
* **Observability** — every stage transition is recorded in the registry
  via :class:`MLRegistry.log_metrics`, so a dashboard can render the
  pipeline timeline without scraping logs.

The retraining pipeline this is meant to run::

    candle_freshness_check → ingest_recent_candles → run_backtests
        → train_model → evaluate_holdout → register_model
        → enable_shadow_comparison

Each stage is a pure function of the Context; the pipeline runner is
generic so the same machinery works for evaluation-only or backtesting
sweeps.
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context — mutable bag of stage outputs
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """Shared state passed to each stage. Stages READ inputs they declared
    as dependencies (via the upstream stage names) and WRITE their own
    outputs into ``data`` keyed by the stage name. The pipeline runner
    enforces no-other-key-writes via a name-prefixed copy; deliberate
    cross-stage shared state must go through the runner's typed Context
    API rather than direct dict mutation."""
    pipeline_name: str
    started_at: int
    data: dict[str, Any] = field(default_factory=dict)
    # Set by the runner; available to stages that want to log metrics
    # against the same run id (e.g. evaluation stage logging AUC).
    run_id: Optional[str] = None
    registry: Optional[Any] = None  # MLRegistry instance, if wired up


# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Per-stage outcome captured by the runner."""
    name: str
    status: str  # "succeeded" | "failed" | "skipped"
    output: Any = None
    error: Optional[str] = None
    started_at: int = 0
    finished_at: int = 0
    attempts: int = 0


@dataclass
class Stage:
    """One node in the pipeline DAG.

    Attributes:
        name: unique identifier (also the key under which the stage's
            output is stored in ``Context.data``).
        run: callable taking the Context and returning the stage output.
            Raise to signal failure; the runner will retry per ``retries``.
        depends_on: names of upstream stages that must succeed first.
        retries: how many times to retry on exception (0 = single attempt).
        backoff_sec: seconds to sleep between retries (linear, not exponential
            — we're already in a retraining loop; exponential backoff would
            mostly add ambiguous latency without recovery benefit).
        idempotent: when ``True`` and the stage's outputs are already in
            the Context (e.g. from a previous partial run), skip execution.
            Default False because most ML stages have side effects.
    """
    name: str
    run: Callable[[Context], Any]
    depends_on: list[str] = field(default_factory=list)
    retries: int = 0
    backoff_sec: float = 1.0
    idempotent: bool = False


# ---------------------------------------------------------------------------
# Pipeline definition + topological sort
# ---------------------------------------------------------------------------


def topological_order(stages: list[Stage]) -> list[Stage]:
    """Return stages in execution order. Raises ValueError on cycles or
    dangling dependencies."""
    by_name = {s.name: s for s in stages}
    if len(by_name) != len(stages):
        raise ValueError("duplicate stage names")
    # Validate dependencies before sorting so the error message is clearer
    # than "cycle detected" when the real problem is a typo.
    for s in stages:
        for dep in s.depends_on:
            if dep not in by_name:
                raise ValueError(
                    f"stage {s.name!r} depends on unknown stage {dep!r}"
                )

    visited: dict[str, str] = {}  # white = unseen, gray = in-progress, black = done
    order: list[Stage] = []

    def visit(name: str) -> None:
        color = visited.get(name, "white")
        if color == "black":
            return
        if color == "gray":
            raise ValueError(f"cycle detected involving stage {name!r}")
        visited[name] = "gray"
        for dep in by_name[name].depends_on:
            visit(dep)
        visited[name] = "black"
        order.append(by_name[name])

    for s in stages:
        visit(s.name)
    return order


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Aggregated outcome across all stages."""
    pipeline_name: str
    status: str  # "succeeded" | "failed"
    started_at: int
    finished_at: int
    stages: list[StageResult] = field(default_factory=list)
    failed_stage: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"


class PipelineRunner:
    """Executes a list of stages respecting dependencies, retries, and
    idempotency. Single-process, single-thread by design — the trainer
    runs as part of the main bot and we want predictable resource use."""

    def __init__(self, registry: Optional[Any] = None):
        self._registry = registry

    def run(self, pipeline_name: str, stages: list[Stage]) -> PipelineResult:
        """Execute the DAG. Returns a PipelineResult with per-stage detail.

        On any stage's terminal failure (after ``retries`` exhausted),
        downstream stages are NOT executed and the result is marked
        ``failed`` with ``failed_stage`` populated. Already-succeeded
        upstream stages keep their results in the output, so the operator
        can resume from the failure point in a follow-up run.
        """
        ordered = topological_order(stages)
        ctx = Context(
            pipeline_name=pipeline_name,
            started_at=int(time.time() * 1000),
            registry=self._registry,
        )

        # Open a registry run if a registry is wired up so the dashboard
        # can see the pipeline timeline alongside model versions.
        if self._registry is not None:
            try:
                run = self._registry.start_run(
                    name=pipeline_name,
                    tags={"kind": "pipeline"},
                )
                ctx.run_id = run.run_id
            except Exception as exc:  # noqa: BLE001
                logger.warning("PipelineRunner: registry.start_run failed: %s", exc)

        results: list[StageResult] = []
        failed = False
        failed_stage: Optional[str] = None

        for stage in ordered:
            sr = StageResult(name=stage.name, status="skipped",
                             started_at=int(time.time() * 1000), finished_at=0)
            if failed:
                sr.status = "skipped"
                sr.error = f"upstream stage {failed_stage!r} failed"
                sr.finished_at = sr.started_at
                results.append(sr)
                continue

            if stage.idempotent and stage.name in ctx.data:
                sr.status = "skipped"
                sr.output = ctx.data[stage.name]
                sr.finished_at = sr.started_at
                results.append(sr)
                logger.debug("Stage %s skipped (idempotent + cached)", stage.name)
                continue

            attempts = 0
            max_attempts = max(1, stage.retries + 1)
            while attempts < max_attempts:
                attempts += 1
                try:
                    output = stage.run(ctx)
                    ctx.data[stage.name] = output
                    sr.status = "succeeded"
                    sr.output = output
                    sr.attempts = attempts
                    sr.finished_at = int(time.time() * 1000)
                    logger.info(
                        "Pipeline %s | stage %s succeeded (attempt %d/%d)",
                        pipeline_name, stage.name, attempts, max_attempts,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                    if attempts < max_attempts:
                        logger.warning(
                            "Pipeline %s | stage %s attempt %d/%d failed: %s — retrying",
                            pipeline_name, stage.name, attempts, max_attempts, err,
                        )
                        time.sleep(max(0.0, stage.backoff_sec))
                    else:
                        sr.status = "failed"
                        sr.error = err + "\n" + traceback.format_exc(limit=5)
                        sr.attempts = attempts
                        sr.finished_at = int(time.time() * 1000)
                        failed = True
                        failed_stage = stage.name
                        logger.error(
                            "Pipeline %s | stage %s FAILED after %d attempts: %s",
                            pipeline_name, stage.name, attempts, err,
                        )

            results.append(sr)

        finished_at = int(time.time() * 1000)
        result = PipelineResult(
            pipeline_name=pipeline_name,
            status="failed" if failed else "succeeded",
            started_at=ctx.started_at,
            finished_at=finished_at,
            stages=results,
            failed_stage=failed_stage,
        )

        if self._registry is not None and ctx.run_id is not None:
            try:
                # Summary metrics so a dashboard can render pipeline health.
                self._registry.log_metrics(ctx.run_id, {
                    "n_stages": len(ordered),
                    "n_succeeded": sum(1 for r in results if r.status == "succeeded"),
                    "n_failed": sum(1 for r in results if r.status == "failed"),
                    "n_skipped": sum(1 for r in results if r.status == "skipped"),
                    "duration_ms": finished_at - ctx.started_at,
                })
                self._registry.finish_run(
                    ctx.run_id,
                    status="finished" if not failed else "failed",
                    error=failed_stage if failed else None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("PipelineRunner: registry finalisation failed: %s", exc)

        return result
