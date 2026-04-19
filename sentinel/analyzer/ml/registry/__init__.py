"""SQL-backed ML model registry with MLflow-style lifecycle.

Public API:
    MLRegistry            — primary façade
    Run, ModelVersion     — read-only data classes
    STAGE_*               — lifecycle constants
"""
from analyzer.ml.registry.sql_registry import (
    ALL_STAGES,
    MLRegistry,
    ModelVersion,
    Run,
    STAGE_ARCHIVED,
    STAGE_NONE,
    STAGE_PRODUCTION,
    STAGE_STAGING,
)

__all__ = [
    "MLRegistry",
    "Run",
    "ModelVersion",
    "ALL_STAGES",
    "STAGE_NONE",
    "STAGE_STAGING",
    "STAGE_PRODUCTION",
    "STAGE_ARCHIVED",
]
