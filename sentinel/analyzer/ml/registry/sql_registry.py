"""SQLite-backed model registry with full MLOps lifecycle.

Replaces the file-only ``model_registry.json`` (which the dashboard can't
query, can't compare runs, and silently lost data after schema migrations)
with a proper relational store on the existing ``sentinel.db`` SQLite file.

Lifecycle stages match MLflow conventions so this layer is a drop-in
replacement once the team decides to migrate to a managed tracker:

    none → staging → production → archived

Promotion rules:

* ``staging`` is the default for a freshly-trained model.
* Exactly **one** model per ``(name, stage="production")`` at any time —
  promoting a new model to production atomically demotes the previous
  champion to ``archived``.
* ``archived`` models are never deleted (audit trail), but are excluded
  from ``get_production`` lookups.

Why a custom registry instead of installing MLflow:

* MLflow drags ~100MB of transitive deps (sqlalchemy, alembic, gunicorn)
  for what amounts to "log a row when a model is trained".
* The project already runs on a single SQLite file — adding a separate
  MLflow tracking server doubles the operational surface for one user.
* We expose the same five primitives MLflow uses (run, params, metrics,
  artifacts, stages), so a future migration is mechanical.

Schema is created on first use via ``ensure_schema()`` — idempotent and
safe to call from every process startup.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle stages
# ---------------------------------------------------------------------------

# Allowed stages; matches MLflow's vocabulary so a downstream migration
# is a column rename, not a semantic rewrite.
STAGE_NONE = "none"
STAGE_STAGING = "staging"
STAGE_PRODUCTION = "production"
STAGE_ARCHIVED = "archived"

ALL_STAGES = (STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Two tables:
#   ml_runs       — every training attempt (params + metrics + artifacts)
#   ml_model_versions — registered models (links to a run, has a stage)
# We intentionally do NOT touch the existing ``ml_model_registry`` table
# (its schema is rigid and the dashboard reads it); the new tables live
# alongside as a clean v2 surface.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ml_runs (
    run_id          TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    started_at      INTEGER NOT NULL,        -- ms epoch
    finished_at     INTEGER,                 -- ms epoch, NULL while in-flight
    status          TEXT NOT NULL DEFAULT 'running',  -- running|finished|failed
    params_json     TEXT NOT NULL DEFAULT '{}',
    metrics_json    TEXT NOT NULL DEFAULT '{}',
    artifacts_json  TEXT NOT NULL DEFAULT '{}',  -- {name: path}
    tags_json       TEXT NOT NULL DEFAULT '{}',
    error           TEXT
);
CREATE INDEX IF NOT EXISTS ix_ml_runs_name_started
    ON ml_runs(name, started_at DESC);

CREATE TABLE IF NOT EXISTS ml_model_versions (
    name            TEXT NOT NULL,
    version         INTEGER NOT NULL,
    run_id          TEXT NOT NULL,
    stage           TEXT NOT NULL DEFAULT 'none',
    artifact_path   TEXT NOT NULL,
    description     TEXT,
    created_at      INTEGER NOT NULL,
    promoted_at     INTEGER,
    metrics_json    TEXT NOT NULL DEFAULT '{}',
    tags_json       TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (name, version),
    FOREIGN KEY (run_id) REFERENCES ml_runs(run_id)
);
CREATE INDEX IF NOT EXISTS ix_ml_versions_stage
    ON ml_model_versions(name, stage);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the registry tables if they don't exist. Idempotent."""
    conn.executescript(_SCHEMA)
    conn.commit()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """One training attempt. Maps 1:1 with a row in ``ml_runs``."""
    run_id: str
    name: str
    started_at: int
    finished_at: Optional[int] = None
    status: str = "running"
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)  # {logical_name: file path}
    tags: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ModelVersion:
    """A version of a registered model. Always linked to a run."""
    name: str
    version: int
    run_id: str
    stage: str
    artifact_path: str
    description: Optional[str]
    created_at: int
    promoted_at: Optional[int]
    metrics: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MLRegistry:
    """Thread-safe, single-process model registry on top of SQLite.

    Concurrency model: a per-instance ``threading.Lock`` serialises writes
    that need atomicity (e.g. demote-then-promote during champion change).
    SQLite's own locking handles cross-process concurrency, but is too
    coarse to make the demote-promote pair atomic across threads — hence
    the explicit lock here.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        # Validate eagerly so misconfiguration surfaces at construction,
        # not on the first ``log_run`` call deep in a training loop.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            ensure_schema(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        # ON DELETE / FK behaviour off SQLite's default — matches the
        # rest of sentinel.db's connection style for consistency.
        try:
            yield conn
        finally:
            conn.close()

    # -------------------- runs --------------------

    def start_run(self, name: str, params: Optional[dict] = None,
                  tags: Optional[dict] = None) -> Run:
        """Open a new run with status='running'. Returns the Run handle —
        caller fills metrics/artifacts and finishes via ``finish_run``."""
        run = Run(
            run_id=f"run_{int(time.time() * 1000)}_{name}",
            name=name,
            started_at=int(time.time() * 1000),
            status="running",
            params=dict(params or {}),
            tags=dict(tags or {}),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO ml_runs (run_id, name, started_at, status, params_json, tags_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run.run_id, run.name, run.started_at, run.status,
                 json.dumps(run.params), json.dumps(run.tags)),
            )
            conn.commit()
        return run

    def log_metrics(self, run_id: str, metrics: dict) -> None:
        """Merge ``metrics`` into the run's stored metrics dict."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT metrics_json FROM ml_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown run_id: {run_id}")
            current = json.loads(row["metrics_json"] or "{}")
            current.update(metrics)
            conn.execute(
                "UPDATE ml_runs SET metrics_json = ? WHERE run_id = ?",
                (json.dumps(current), run_id),
            )
            conn.commit()

    def log_artifact(self, run_id: str, name: str, path: str | Path) -> None:
        """Record that ``run_id`` produced an artifact named ``name`` at
        ``path``. We DO NOT copy the file — the registry is a metadata
        store, not a blob store. Path is stored verbatim, so callers
        should pass a stable absolute path (or one anchored to a deploy
        directory)."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT artifacts_json FROM ml_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown run_id: {run_id}")
            current = json.loads(row["artifacts_json"] or "{}")
            current[name] = str(path)
            conn.execute(
                "UPDATE ml_runs SET artifacts_json = ? WHERE run_id = ?",
                (json.dumps(current), run_id),
            )
            conn.commit()

    def finish_run(self, run_id: str, status: str = "finished",
                   error: Optional[str] = None) -> None:
        """Mark the run terminal. ``status`` is 'finished' on success or
        'failed' when the trainer caught an exception (pass the message
        in ``error``)."""
        if status not in ("finished", "failed"):
            raise ValueError(f"status must be finished|failed, got {status}")
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE ml_runs SET finished_at = ?, status = ?, error = ? WHERE run_id = ?",
                (int(time.time() * 1000), status, error, run_id),
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[Run]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ml_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return _row_to_run(row) if row else None

    def list_runs(self, name: str, limit: int = 50) -> list[Run]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ml_runs WHERE name = ? ORDER BY started_at DESC LIMIT ?",
                (name, limit),
            ).fetchall()
            return [_row_to_run(r) for r in rows]

    # -------------------- model versions --------------------

    def register_model(
        self,
        name: str,
        run_id: str,
        artifact_path: str | Path,
        description: Optional[str] = None,
        metrics: Optional[dict] = None,
        tags: Optional[dict] = None,
    ) -> ModelVersion:
        """Register a new version of ``name`` linked to ``run_id``. The
        version number is auto-incremented per ``name``. Stage starts at
        ``"none"`` — promote separately via :meth:`transition_stage`.
        """
        if not self.get_run(run_id):
            raise ValueError(f"cannot register model — unknown run_id: {run_id}")
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS next FROM ml_model_versions WHERE name = ?",
                (name,),
            ).fetchone()
            version = int(row["next"])
            now = int(time.time() * 1000)
            conn.execute(
                "INSERT INTO ml_model_versions "
                "(name, version, run_id, stage, artifact_path, description, "
                " created_at, metrics_json, tags_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, version, run_id, STAGE_NONE, str(artifact_path),
                 description, now,
                 json.dumps(metrics or {}), json.dumps(tags or {})),
            )
            conn.commit()
        return ModelVersion(
            name=name, version=version, run_id=run_id, stage=STAGE_NONE,
            artifact_path=str(artifact_path), description=description,
            created_at=now, promoted_at=None,
            metrics=metrics or {}, tags=tags or {},
        )

    def transition_stage(
        self,
        name: str,
        version: int,
        stage: str,
        archive_existing: bool = True,
    ) -> ModelVersion:
        """Move ``(name, version)`` to ``stage``. When promoting to
        ``production`` and ``archive_existing=True`` (the default), any
        currently-production version of the same model is atomically
        archived in the same transaction — guaranteeing exactly one
        ``production`` row per name."""
        if stage not in ALL_STAGES:
            raise ValueError(f"unknown stage: {stage}")
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM ml_model_versions WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
            if existing is None:
                raise ValueError(f"unknown model version: {name} v{version}")

            now = int(time.time() * 1000)
            try:
                conn.execute("BEGIN")
                if stage == STAGE_PRODUCTION and archive_existing:
                    conn.execute(
                        "UPDATE ml_model_versions SET stage = ?, promoted_at = ? "
                        "WHERE name = ? AND stage = ? AND version != ?",
                        (STAGE_ARCHIVED, now, name, STAGE_PRODUCTION, version),
                    )
                conn.execute(
                    "UPDATE ml_model_versions SET stage = ?, promoted_at = ? "
                    "WHERE name = ? AND version = ?",
                    (stage, now, name, version),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

            row = conn.execute(
                "SELECT * FROM ml_model_versions WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
            return _row_to_model_version(row)

    def get_production(self, name: str) -> Optional[ModelVersion]:
        """Return the active production version for ``name``, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ml_model_versions WHERE name = ? AND stage = ? "
                "ORDER BY version DESC LIMIT 1",
                (name, STAGE_PRODUCTION),
            ).fetchone()
            return _row_to_model_version(row) if row else None

    def list_versions(self, name: str, stage: Optional[str] = None) -> list[ModelVersion]:
        with self._connect() as conn:
            if stage is not None:
                rows = conn.execute(
                    "SELECT * FROM ml_model_versions WHERE name = ? AND stage = ? "
                    "ORDER BY version DESC",
                    (name, stage),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ml_model_versions WHERE name = ? ORDER BY version DESC",
                    (name,),
                ).fetchall()
            return [_row_to_model_version(r) for r in rows]


# ---------------------------------------------------------------------------
# Row → dataclass adapters
# ---------------------------------------------------------------------------


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        run_id=row["run_id"],
        name=row["name"],
        started_at=int(row["started_at"]),
        finished_at=int(row["finished_at"]) if row["finished_at"] is not None else None,
        status=row["status"],
        params=json.loads(row["params_json"] or "{}"),
        metrics=json.loads(row["metrics_json"] or "{}"),
        artifacts=json.loads(row["artifacts_json"] or "{}"),
        tags=json.loads(row["tags_json"] or "{}"),
        error=row["error"],
    )


def _row_to_model_version(row: sqlite3.Row) -> ModelVersion:
    return ModelVersion(
        name=row["name"],
        version=int(row["version"]),
        run_id=row["run_id"],
        stage=row["stage"],
        artifact_path=row["artifact_path"],
        description=row["description"],
        created_at=int(row["created_at"]),
        promoted_at=int(row["promoted_at"]) if row["promoted_at"] is not None else None,
        metrics=json.loads(row["metrics_json"] or "{}"),
        tags=json.loads(row["tags_json"] or "{}"),
    )
