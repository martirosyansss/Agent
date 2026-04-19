"""
Deterministic A/B bucketing.

Champion-challenger (already present in ``ab/champion_challenger.py``)
compares two models *after* the fact using McNemar / Wilson. This
module handles the step *before*: deciding, at signal time, which
model should score each trading opportunity.

Requirements the bucketer enforces:

1. **Deterministic** — the same unit (symbol, trace_id, …) always lands
   in the same bucket across restarts, rollouts and dashboards. Two
   services that re-implement bucketing with drifting RNG state produce
   impossible-to-debug drift.
2. **Hash-based split** — no centralised assignment table. Each caller
   hashes locally and reads the bucket; the config only needs to carry
   percentages.
3. **Salt-able** — adding a new experiment must not re-bucket existing
   users. Each experiment takes a salt so its hash space is
   independent from any previous split.
4. **Monotone on the percentage axis** — bumping ``challenger_pct`` from
   10% to 20% only moves units *into* the challenger bucket; nothing
   flips back to champion, so the comparison stays valid across rollout
   ramps.

Hash choice: SHA-256 truncated to 8 bytes as an unsigned int, modulo
10 000 for basis-point resolution. 8 bytes is overkill for cryptographic
purposes but lets the bucketer share semantics with log lines that
already carry these IDs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional


_HASH_BASIS = 10_000  # basis-point granularity — 1 bp = 1 / 10 000


def _hash_to_bp(unit: str, salt: str = "") -> int:
    """Return an integer in ``[0, _HASH_BASIS)`` derived from ``unit + salt``.

    SHA-256 on the concatenated bytes, take the first 8 bytes as a big-
    endian unsigned int, modulo _HASH_BASIS. Deterministic, low-collision,
    salt-independent from other experiments.
    """
    digest = hashlib.sha256(f"{salt}::{unit}".encode("utf-8")).digest()
    n = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return n % _HASH_BASIS


@dataclass(slots=True, frozen=True)
class BucketAssignment:
    """Result of a single bucket lookup."""
    bucket: str           # e.g. "champion", "challenger_A"
    hash_bp: int          # bucket threshold input, useful for audit
    experiment: str       # experiment id (same as salt)


@dataclass(slots=True)
class Bucketer:
    """Percentage-based split with named buckets.

    ``percentages`` is ``{bucket_name: pct}`` where ``0 <= pct <= 100``
    and the sum equals 100. The buckets are assigned in insertion order
    — this matters for monotonicity: the first bucket always takes the
    low end of the hash space, so raising the second bucket's share
    only removes mass from *later* buckets.

    Usage::

        b = Bucketer(
            experiment="ml-v3-rollout",
            percentages={"champion": 80, "challenger": 20},
        )
        assignment = b.assign("BTCUSDT")
        if assignment.bucket == "challenger":
            ...
    """
    experiment: str
    percentages: dict[str, float]

    def __post_init__(self) -> None:
        if not self.percentages:
            raise ValueError("Bucketer needs at least one bucket")
        total = sum(self.percentages.values())
        if abs(total - 100.0) > 1e-6:
            raise ValueError(
                f"bucket percentages must sum to 100, got {total}"
            )
        for name, pct in self.percentages.items():
            if pct < 0 or pct > 100:
                raise ValueError(
                    f"bucket {name!r} pct out of [0, 100]: {pct}"
                )

    def assign(self, unit: str) -> BucketAssignment:
        """Return the bucket for ``unit`` under this experiment."""
        hbp = _hash_to_bp(unit, salt=self.experiment)
        cum = 0.0
        for name, pct in self.percentages.items():
            cum += pct
            # Cumulative percentage → basis-point threshold
            threshold_bp = int(round(cum * _HASH_BASIS / 100.0))
            if hbp < threshold_bp:
                return BucketAssignment(
                    bucket=name, hash_bp=hbp, experiment=self.experiment,
                )
        # Floating-point rounding might leave one edge case at the very top —
        # fall back to the last bucket by construction.
        last = next(reversed(self.percentages))
        return BucketAssignment(
            bucket=last, hash_bp=hbp, experiment=self.experiment,
        )


def split_50_50(
    experiment: str,
    a: str = "control",
    b: str = "treatment",
) -> Bucketer:
    """Convenience factory for the most common case."""
    return Bucketer(experiment=experiment, percentages={a: 50.0, b: 50.0})


def rollout(
    experiment: str,
    challenger_pct: float,
    *,
    champion_name: str = "champion",
    challenger_name: str = "challenger",
) -> Bucketer:
    """Single-treatment rollout. Bump ``challenger_pct`` monotonically."""
    if not 0.0 <= challenger_pct <= 100.0:
        raise ValueError(f"challenger_pct must be in [0, 100], got {challenger_pct}")
    return Bucketer(
        experiment=experiment,
        percentages={
            champion_name: 100.0 - challenger_pct,
            challenger_name: challenger_pct,
        },
    )
