"""
Post-mortem analytics over the decision_audit + signal_executions tables.

This module is THE answer to "I've been running for a week — what was the bot
actually doing?". Every helper returns a list of dicts (or a dict), so it
plugs straight into pandas:

    from analyzer.decision_analytics import DecisionAnalytics
    da = DecisionAnalytics(repo)
    df = pd.DataFrame(da.gate_rejection_breakdown(hours=168))

Helpers cover the questions you actually need to answer to tune a system:

- Which gate rejected most signals last week?
- For each gate, what feature distribution led to its rejections?
- Of the trades that DID execute, what was the outcome by strategy / regime?
- For rejected signals, what was the price 1h / 4h / 24h later? (counterfactual)
- Latency budget — which gate is slow?

The queries assume sqlite3 (Sentinel's runtime store). Switching to Postgres
later would need only minor rewrites of the date-bucket expressions.

This module imports nothing from the live runtime — it reads only the DB.
Safe to call from notebooks, dashboards, or a separate analytics process
while the bot runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class GateRejectionBreakdown:
    gate: str
    rejection_count: int
    pct_of_rejections: float
    sample_reasons: list[str] = field(default_factory=list)


@dataclass
class StrategyOutcome:
    strategy: str
    total_signals: int
    approved: int
    rejected: int
    rejection_rate_pct: float
    rejected_by_gate: dict[str, int]


class DecisionAnalytics:
    """Read-only analytics layer over decision_audit / signal_executions."""

    def __init__(self, repo) -> None:
        # Duck-typed: needs ._db with .fetchall(sql, params) → list[dict].
        self._db = repo._db if hasattr(repo, "_db") else repo

    # ──────────────────────────────────────────────
    # Time helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _ms_cutoff(hours: float) -> int:
        return int(time.time() * 1000) - int(hours * 3_600_000)

    # ──────────────────────────────────────────────
    # Gate-level rejection analysis
    # ──────────────────────────────────────────────

    def gate_rejection_breakdown(
        self,
        hours: float = 168,
        max_sample_reasons: int = 3,
    ) -> list[dict]:
        """For each gate, how many signals it rejected in the window.

        Returns a list of dicts ordered by descending count, ready for
        ``pd.DataFrame()`` or printing.
        """
        cutoff = self._ms_cutoff(hours)
        rows = self._db.fetchall(
            "SELECT rejected_by AS gate, COUNT(*) AS n, "
            "       GROUP_CONCAT(final_reason, ' || ') AS reasons "
            "FROM decision_audit "
            "WHERE ts >= ? AND final_outcome = 'rejected' AND rejected_by != '' "
            "GROUP BY rejected_by "
            "ORDER BY n DESC",
            (cutoff,),
        ) or []
        total = sum(r["n"] for r in rows) or 1
        out = []
        for r in rows:
            sample = (r["reasons"] or "").split(" || ")[:max_sample_reasons]
            out.append({
                "gate": r["gate"],
                "rejection_count": r["n"],
                "pct_of_rejections": round(r["n"] / total * 100, 1),
                "sample_reasons": sample,
            })
        return out

    def signals_by_strategy(self, hours: float = 168) -> list[dict]:
        """Per-strategy approve/reject counts and the gate that blocked most often."""
        cutoff = self._ms_cutoff(hours)
        rows = self._db.fetchall(
            "SELECT strategy, final_outcome, rejected_by, COUNT(*) AS n "
            "FROM decision_audit "
            "WHERE ts >= ? "
            "GROUP BY strategy, final_outcome, rejected_by "
            "ORDER BY strategy",
            (cutoff,),
        ) or []
        agg: dict[str, dict] = {}
        for r in rows:
            s = r["strategy"]
            d = agg.setdefault(s, {
                "strategy": s, "total": 0, "approved": 0, "rejected": 0,
                "rejected_by_gate": {},
            })
            d["total"] += r["n"]
            if r["final_outcome"] == "approved":
                d["approved"] += r["n"]
            else:
                d["rejected"] += r["n"]
                if r["rejected_by"]:
                    d["rejected_by_gate"][r["rejected_by"]] = (
                        d["rejected_by_gate"].get(r["rejected_by"], 0) + r["n"]
                    )
        for d in agg.values():
            d["rejection_rate_pct"] = (
                round(d["rejected"] / d["total"] * 100, 1) if d["total"] else 0.0
            )
        return sorted(agg.values(), key=lambda x: -x["total"])

    def gate_latency_stats(self, hours: float = 24) -> list[dict]:
        """Average / max latency per gate. Pulls from gates_json which carries
        per-gate ``latency_us``. Slow because we parse JSON in Python — but
        24h of data is typically a few hundred rows, so it's fine."""
        cutoff = self._ms_cutoff(hours)
        rows = self._db.fetchall(
            "SELECT gates_json FROM decision_audit WHERE ts >= ?",
            (cutoff,),
        ) or []
        sums: dict[str, list[int]] = {}
        for r in rows:
            try:
                gates = json.loads(r["gates_json"])
            except Exception:
                continue
            for g in gates:
                gate = g.get("gate", "?")
                lat = int(g.get("latency_us", 0))
                sums.setdefault(gate, []).append(lat)
        out = []
        for gate, lats in sorted(sums.items()):
            n = len(lats)
            out.append({
                "gate": gate,
                "n_calls": n,
                "avg_us": round(sum(lats) / n, 1) if n else 0,
                "max_us": max(lats) if lats else 0,
                "p95_us": sorted(lats)[int(n * 0.95) - 1] if n >= 20 else max(lats, default=0),
            })
        return out

    # ──────────────────────────────────────────────
    # Feature-distribution conditioning on rejection
    # ──────────────────────────────────────────────

    def feature_distribution_for_gate(
        self,
        gate: str,
        feature: str,
        hours: float = 168,
    ) -> dict:
        """Distribution stats for one feature among signals rejected by ``gate``.

        Use to answer: "When the regime gate rejects a signal, what was
        the typical RSI / ADX / trend_alignment?" — quick way to see whether
        the gate is firing on the cases you intended.
        """
        cutoff = self._ms_cutoff(hours)
        rows = self._db.fetchall(
            "SELECT feature_snapshot_json FROM decision_audit "
            "WHERE ts >= ? AND rejected_by = ?",
            (cutoff, gate),
        ) or []
        values: list[float] = []
        for r in rows:
            try:
                fs = json.loads(r["feature_snapshot_json"])
                v = fs.get(feature)
                if isinstance(v, (int, float)):
                    values.append(float(v))
            except Exception:
                continue
        if not values:
            return {"gate": gate, "feature": feature, "n": 0}
        values.sort()
        n = len(values)
        return {
            "gate": gate,
            "feature": feature,
            "n": n,
            "min": round(values[0], 4),
            "p25": round(values[n // 4], 4),
            "median": round(values[n // 2], 4),
            "p75": round(values[3 * n // 4], 4),
            "max": round(values[-1], 4),
            "mean": round(sum(values) / n, 4),
        }

    # ──────────────────────────────────────────────
    # Per-gate evaluation (shadow-mode analytics)
    # ──────────────────────────────────────────────

    def gate_independent_block_rate(self, hours: float = 168) -> list[dict]:
        """Counts how often each gate INDEPENDENTLY would have rejected a signal,
        regardless of whether short-circuit by another gate first.

        Requires shadow-mode rows (short_circuit = 0) for full coverage —
        production runs are short-circuit so this metric only sees a partial
        view unless you periodically re-evaluate signals in shadow mode.
        """
        cutoff = self._ms_cutoff(hours)
        rows = self._db.fetchall(
            "SELECT short_circuit, gates_json FROM decision_audit WHERE ts >= ?",
            (cutoff,),
        ) or []
        per_gate_rejected: dict[str, int] = {}
        per_gate_evaluated: dict[str, int] = {}
        for r in rows:
            try:
                gates = json.loads(r["gates_json"])
            except Exception:
                continue
            for g in gates:
                gate = g.get("gate", "?")
                outcome = g.get("outcome", "")
                if outcome in ("approved", "rejected"):
                    per_gate_evaluated[gate] = per_gate_evaluated.get(gate, 0) + 1
                if outcome == "rejected":
                    per_gate_rejected[gate] = per_gate_rejected.get(gate, 0) + 1
        out = []
        for gate in sorted(per_gate_evaluated.keys()):
            ev = per_gate_evaluated[gate]
            rj = per_gate_rejected.get(gate, 0)
            out.append({
                "gate": gate,
                "evaluations": ev,
                "rejections": rj,
                "block_rate_pct": round(rj / ev * 100, 1) if ev else 0.0,
            })
        return sorted(out, key=lambda x: -x["block_rate_pct"])

    # ──────────────────────────────────────────────
    # Final-decision summary (one-shot)
    # ──────────────────────────────────────────────

    def summary(self, hours: float = 168) -> dict:
        """One-shot snapshot suitable for a daily / weekly Telegram digest."""
        cutoff = self._ms_cutoff(hours)
        totals = self._db.fetchone(
            "SELECT COUNT(*) AS n, "
            "       SUM(CASE WHEN final_outcome = 'approved' THEN 1 ELSE 0 END) AS approved, "
            "       SUM(CASE WHEN final_outcome = 'rejected' THEN 1 ELSE 0 END) AS rejected "
            "FROM decision_audit WHERE ts >= ?",
            (cutoff,),
        ) or {"n": 0, "approved": 0, "rejected": 0}
        return {
            "window_hours": hours,
            "total_signals": totals.get("n", 0),
            "approved": totals.get("approved", 0),
            "rejected": totals.get("rejected", 0),
            "approval_rate_pct": (
                round((totals.get("approved", 0) or 0) / totals["n"] * 100, 1)
                if totals.get("n") else 0.0
            ),
            "top_blocking_gates": self.gate_rejection_breakdown(hours=hours)[:3],
            "by_strategy": self.signals_by_strategy(hours=hours),
        }
