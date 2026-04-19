"""Model version registry — rolling JSON audit log of saved models.

Separate from the codec so corruption of the JSON file never blocks a
save. The registry is informational — operators + the dashboard look
at it; the bot itself never reads it back.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["append_registry_entry"]


def append_registry_entry(
    model_path: Path,
    model_version: str,
    checksum: str,
    n_members: int,
    calibrated_threshold: float,
    metrics: Optional[Any] = None,
    max_entries: int = 50,
) -> None:
    """Append one entry to ``<model_dir>/model_registry.json``.

    Silent-fails on JSON parse / write errors: the goal is an audit
    trail, not a correctness gate. If the registry is unreadable we
    log at DEBUG and move on rather than aborting the save that just
    succeeded.

    ``metrics`` is the predictor's ``MLMetrics`` dataclass (or None).
    Individual fields are read via ``getattr`` so a shorter / longer
    metrics object doesn't break the registry write.
    """
    registry_path = model_path.parent / "model_registry.json"

    entry: dict[str, Any] = {
        "version": model_version,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "saved_ts": int(time.time()),
        "checksum_sha256": checksum,
        "ensemble_members": n_members,
        "calibrated_threshold": round(calibrated_threshold, 4),
        "metrics": {},
    }

    if metrics is not None:
        _p_ci = getattr(metrics, "precision_ci_95", (0.0, 0.0))
        _a_ci = getattr(metrics, "auc_ci_95", (0.0, 0.0))
        entry["metrics"] = {
            "precision": round(getattr(metrics, "precision", 0.0), 4),
            "recall": round(getattr(metrics, "recall", 0.0), 4),
            "roc_auc": round(getattr(metrics, "roc_auc", 0.0), 4),
            "accuracy": round(getattr(metrics, "accuracy", 0.0), 4),
            "skill_score": round(getattr(metrics, "skill_score", 0.0), 4),
            "train_samples": getattr(metrics, "train_samples", 0),
            "test_samples": getattr(metrics, "test_samples", 0),
            "precision_ci_95": [round(v, 4) for v in _p_ci],
            "auc_ci_95": [round(v, 4) for v in _a_ci],
            "baseline_win_rate": round(getattr(metrics, "baseline_win_rate", 0.0), 4),
            "precision_lift": round(getattr(metrics, "precision_lift", 0.0), 4),
            "auc_lift": round(getattr(metrics, "auc_lift", 0.0), 4),
            "oot_auc": (
                round(getattr(metrics, "oot_auc"), 4)
                if getattr(metrics, "oot_auc", None) is not None
                else None
            ),
        }

    try:
        registry: list = []
        if registry_path.exists():
            with registry_path.open("r", encoding="utf-8") as f:
                registry = json.load(f)
        registry.append(entry)
        # Bound the audit log so it doesn't grow forever.
        registry = registry[-max_entries:]
        with registry_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        logger.info("ML registry updated: %s (%d entries)", registry_path, len(registry))
    except Exception as exc:  # noqa: BLE001
        logger.debug("ML registry write failed: %s", exc)
