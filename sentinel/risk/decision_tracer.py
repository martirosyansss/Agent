"""
Decision tracer — structured record of every gate evaluation in the risk pipeline.

The legacy ``RiskCheckResult(approved, reason)`` is enough for the executor
but throws away everything an analyst needs later:

- which gate fired first vs. which would have fired second?
- what were the actual feature values when the decision was made?
- how long did each gate take?

The tracer accumulates one ``GateVerdict`` per gate run, including in
``shadow`` mode where every gate is evaluated even if an earlier one would
have rejected the trade. Shadow runs are how you answer "the multi-TF
gate rejected this trade — would the regime gate have rejected it too?"
without instrumenting code at the call site.

Output is serialisable: ``trace.to_dict()`` produces a JSON-friendly
record suitable for ``EventLog.emit("signal_decision", **trace.to_dict())``
and for storage in the ``decision_audit`` table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class GateOutcome(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"     # gate not configured / N/A for this signal
    ERROR = "error"          # gate raised an exception (logged, treated as APPROVED)


@dataclass
class GateVerdict:
    """One gate's contribution to the decision trace."""
    gate: str                # short id e.g. "drawdown", "regime", "multi_tf"
    outcome: GateOutcome
    reason: str = ""
    latency_us: int = 0       # measured wall-clock time of the gate call
    payload: dict[str, Any] = field(default_factory=dict)  # gate-specific extras

    def to_dict(self) -> dict:
        return {
            "gate": self.gate,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "latency_us": self.latency_us,
            **({"payload": self.payload} if self.payload else {}),
        }


@dataclass
class DecisionTrace:
    """Full trace of a single signal evaluation through the gate pipeline."""
    signal_id: str = ""
    symbol: str = ""
    strategy: str = ""
    direction: str = ""           # "BUY" / "SELL"
    confidence: float = 0.0
    feature_snapshot: dict[str, Any] = field(default_factory=dict)
    gates: list[GateVerdict] = field(default_factory=list)
    final_outcome: GateOutcome = GateOutcome.APPROVED
    final_reason: str = ""
    short_circuit: bool = True    # True = stopped at first REJECTED; False = shadow mode

    def add(self, verdict: GateVerdict) -> None:
        self.gates.append(verdict)

    def first_rejection(self) -> Optional[GateVerdict]:
        for v in self.gates:
            if v.outcome == GateOutcome.REJECTED:
                return v
        return None

    def all_rejections(self) -> list[GateVerdict]:
        return [v for v in self.gates if v.outcome == GateOutcome.REJECTED]

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "final_outcome": self.final_outcome.value,
            "final_reason": self.final_reason,
            "short_circuit": self.short_circuit,
            "gates": [v.to_dict() for v in self.gates],
            "feature_snapshot": self.feature_snapshot,
        }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Best-effort attribute access (FeatureVector may be None / missing fields)."""
    try:
        v = getattr(obj, attr, default)
        if v is None:
            return default
        # Round floats for compactness in storage; ints/bools/strings pass through.
        if isinstance(v, float):
            return round(v, 6)
        return v
    except Exception:
        return default


# Fields we care about for post-mortem analysis. Adding fields is cheap;
# removing them later breaks downstream notebooks.
_FEATURE_SNAPSHOT_FIELDS = (
    # market context
    "symbol", "close", "atr", "market_regime",
    # trend
    "ema_9", "ema_21", "ema_50", "ema_50_daily", "trend_alignment",
    "macd_line", "macd_signal", "macd_histogram",
    # momentum / strength
    "rsi_14", "adx", "dmi_spread",
    "bb_upper", "bb_middle", "bb_lower",
    "ichimoku_senkou_a", "ichimoku_senkou_b",
    # volume / liquidity
    "volume", "volume_ratio", "obv", "vwap",
    # news
    "news_composite_score", "news_signal_strength",
    "news_critical_alert", "news_actionable", "news_dominant_category",
    "fear_greed_index",
)


def feature_snapshot_dict(features: Any) -> dict[str, Any]:
    """Pull the analytics-relevant fields off a FeatureVector into a flat dict.

    Failure-tolerant: missing attributes become None rather than raising.
    """
    if features is None:
        return {}
    return {f: _safe_get(features, f) for f in _FEATURE_SNAPSHOT_FIELDS}


# ──────────────────────────────────────────────
# Convenience: time a single gate call
# ──────────────────────────────────────────────

class GateTimer:
    """Context-manager helper to record a gate's verdict.

        with GateTimer(trace, "multi_tf") as t:
            ok, reason = gate.check(...)
            t.record(ok, reason, payload={...})
    """

    def __init__(self, trace: DecisionTrace, gate_id: str) -> None:
        self._trace = trace
        self._gate_id = gate_id
        self._start = 0
        self._verdict: Optional[GateVerdict] = None

    def __enter__(self) -> "GateTimer":
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed_us = max(0, (time.perf_counter_ns() - self._start) // 1000)
        if exc_type is not None:
            # Treat a gate exception as APPROVED (fail-open) but record it.
            verdict = GateVerdict(
                gate=self._gate_id,
                outcome=GateOutcome.ERROR,
                reason=f"{exc_type.__name__}: {exc_val}",
                latency_us=elapsed_us,
            )
            self._trace.add(verdict)
            return True  # swallow — gate errors must not crash the pipeline
        if self._verdict is None:
            # Caller forgot to .record(); default to APPROVED.
            self._verdict = GateVerdict(
                gate=self._gate_id,
                outcome=GateOutcome.APPROVED,
                latency_us=elapsed_us,
            )
        else:
            self._verdict.latency_us = elapsed_us
        self._trace.add(self._verdict)
        return False

    def record(
        self,
        approved: bool,
        reason: str = "",
        outcome: Optional[GateOutcome] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        self._verdict = GateVerdict(
            gate=self._gate_id,
            outcome=outcome if outcome is not None else (GateOutcome.APPROVED if approved else GateOutcome.REJECTED),
            reason=reason,
            payload=payload or {},
        )

    def skipped(self, reason: str = "not configured") -> None:
        self._verdict = GateVerdict(
            gate=self._gate_id,
            outcome=GateOutcome.SKIPPED,
            reason=reason,
        )
