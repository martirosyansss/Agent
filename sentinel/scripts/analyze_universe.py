"""Universe-expansion audit for the ML model.

The single biggest improvement available to the trading model is *more
diverse training data* — currently 588 backtest-synthetic trades come
from only 2 symbols (BTC, ETH), which means the "ensemble" has effectively
learned BTC-correlated regime behaviour rather than a portable edge.

This script audits the current state and produces a concrete migration
plan. It does NOT mutate state — it only inspects and reports — so the
operator can review before approving:

  * What symbols are configured for trading?
  * What symbols actually have OHLC candles in the DB?
  * What's the data gap (configured ≠ have data)?
  * Which symbols would be most valuable to add (by liquidity tier and
    correlation with the current pair)?
  * What's the projected training-corpus size after expansion?

Usage::

    python scripts/analyze_universe.py
    python scripts/analyze_universe.py --target-symbols 8

Exits 0 always — this is an analysis tool, not a gate.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Recommended universe by liquidity tier (Binance top by spot volume, 2026Q1).
# Tiers reflect a standard "concentrate exposure where liquidity is real"
# heuristic: tier-1 alone gives a 5-symbol portfolio with deep books and
# correlated-but-not-identical regimes. Tier-2 is added only when the
# operator has bandwidth to monitor more failure modes.
# ---------------------------------------------------------------------------

TIER_1_LARGE_CAP = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
TIER_2_MID_CAP = ["ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT"]
TIER_3_HIGH_BETA = ["DOGEUSDT", "SHIBUSDT", "APTUSDT", "ARBUSDT", "OPUSDT"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SymbolStatus:
    symbol: str
    has_candles: bool
    candle_count: int
    earliest_ts: Optional[int]
    latest_ts: Optional[int]
    days_of_data: int

    def coverage_summary(self) -> str:
        if not self.has_candles:
            return "MISSING — needs ingestion"
        return f"{self.days_of_data:>4}d, {self.candle_count:>7} candles"


@dataclass
class UniverseReport:
    configured: list[str]
    available: list[str]
    missing_data: list[str]
    surplus_data: list[str]
    statuses: dict[str, SymbolStatus]

    def diversity_score(self) -> float:
        """A simple 0-1 score: 1.0 = ≥10 active symbols, 0.0 = ≤2.
        Below 0.5 is the "single-asset overfit" risk zone."""
        n = len([s for s in self.statuses.values() if s.has_candles
                 and s.symbol in self.configured])
        return min(1.0, max(0.0, (n - 2) / 8.0))


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


def load_configured_symbols(env_path: Path) -> list[str]:
    """Parse TRADING_SYMBOLS from .env. Falls back to config.py default
    if .env is missing the key (matching pydantic-settings behaviour)."""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("TRADING_SYMBOLS="):
                raw = line.split("=", 1)[1].strip()
                try:
                    return list(json.loads(raw))
                except json.JSONDecodeError:
                    pass
    # Conservative fallback — matches MLConfig default.
    return ["BTCUSDT", "ETHUSDT"]


def query_symbol_status(db_path: Path, symbols: list[str]) -> dict[str, SymbolStatus]:
    """Per-symbol candle counts + date range. Read-only, no schema needed."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    out: dict[str, SymbolStatus] = {}
    # Discover all symbols actually in the candles table — informs the
    # "surplus_data" list (symbols you have data for but aren't trading).
    in_db = {r[0] for r in cur.execute("SELECT DISTINCT symbol FROM candles")}

    universe = sorted(set(symbols) | in_db)
    for sym in universe:
        if sym in in_db:
            row = cur.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM candles WHERE symbol = ?", (sym,),
            ).fetchone()
            count, earliest, latest = row
            days = (
                int((latest - earliest) / (86_400_000)) if earliest and latest else 0
            )
            out[sym] = SymbolStatus(
                symbol=sym,
                has_candles=count > 0,
                candle_count=int(count),
                earliest_ts=earliest,
                latest_ts=latest,
                days_of_data=days,
            )
        else:
            out[sym] = SymbolStatus(
                symbol=sym, has_candles=False, candle_count=0,
                earliest_ts=None, latest_ts=None, days_of_data=0,
            )
    conn.close()
    return out


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------


def recommend_additions(current: list[str], target: int) -> list[str]:
    """Pick the next ``target - len(current)`` symbols from tier-1 then
    tier-2 then tier-3, skipping anything already in ``current``. The
    tier ordering encodes the project policy "broaden across deep-book
    names before reaching for high-beta alts"."""
    needed = target - len(current)
    if needed <= 0:
        return []
    proposed: list[str] = []
    for tier in (TIER_1_LARGE_CAP, TIER_2_MID_CAP, TIER_3_HIGH_BETA):
        for sym in tier:
            if sym not in current and sym not in proposed:
                proposed.append(sym)
                if len(proposed) >= needed:
                    return proposed
    return proposed


def estimate_corpus_size(
    statuses: dict[str, SymbolStatus],
    symbols: list[str],
    trades_per_day_per_symbol: float = 4.0,
    project_missing: bool = False,
) -> int:
    """Project training-corpus size under a given symbol set.

    The 4 trades/day constant comes from observed grid + bollinger combined
    output on BTC/ETH (29 trades / 4 days ≈ 7/day, but ~half rejected by
    risk gates, leaving ~4 reaching the trainer). Anything more precise
    would need a per-strategy backtest, which we intentionally skip — the
    point here is order-of-magnitude planning, not a calibrated forecast.

    When ``project_missing=True``, symbols without candles are imputed at
    the median day-coverage of symbols that DO have candles — answering
    "after we run the ingestion step, how big does the corpus get?"
    instead of "how big is it right now?".
    """
    if project_missing:
        with_data = [st.days_of_data for st in statuses.values() if st.has_candles]
        median_days = (sorted(with_data)[len(with_data) // 2]
                       if with_data else 365)
    else:
        median_days = 0
    total = 0
    for sym in symbols:
        st = statuses.get(sym)
        if st and st.has_candles:
            total += int(st.days_of_data * trades_per_day_per_symbol)
        elif project_missing:
            total += int(median_days * trades_per_day_per_symbol)
    return total


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit(
    db_path: Path,
    env_path: Path,
    target_symbols: int = 5,
) -> UniverseReport:
    configured = load_configured_symbols(env_path)
    statuses = query_symbol_status(db_path, configured)
    available = [s for s, st in statuses.items() if st.has_candles]
    missing_data = [s for s in configured if not statuses[s].has_candles]
    surplus_data = [s for s in available if s not in configured]
    return UniverseReport(
        configured=configured,
        available=available,
        missing_data=missing_data,
        surplus_data=surplus_data,
        statuses=statuses,
    )


def render_report(report: UniverseReport, target: int) -> str:
    out: list[str] = []
    out.append("=" * 70)
    out.append("ML UNIVERSE AUDIT")
    out.append("=" * 70)
    out.append("")
    out.append(f"Configured symbols ({len(report.configured)}): {', '.join(report.configured)}")
    out.append(f"Diversity score: {report.diversity_score():.2f}  "
               f"({'WARNING — single-asset risk' if report.diversity_score() < 0.5 else 'OK'})")
    out.append("")

    out.append("Per-symbol coverage")
    out.append("-" * 70)
    out.append(f"{'symbol':<12} {'configured':<12} {'coverage'}")
    for sym in sorted(report.statuses, key=lambda s: -report.statuses[s].days_of_data):
        st = report.statuses[sym]
        cfg_marker = "yes" if sym in report.configured else "no"
        out.append(f"{sym:<12} {cfg_marker:<12} {st.coverage_summary()}")
    out.append("")

    if report.missing_data:
        out.append("⚠ MISSING DATA (configured, no candles in DB):")
        for s in report.missing_data:
            out.append(f"  - {s}")
        out.append("  → run candle-ingestion CLI before adding to TRADING_SYMBOLS")
        out.append("")

    proposed = recommend_additions(report.configured, target)
    if proposed:
        out.append(f"Recommended additions to reach target of {target} symbols:")
        for sym in proposed:
            tier = ("tier-1" if sym in TIER_1_LARGE_CAP
                    else "tier-2" if sym in TIER_2_MID_CAP
                    else "tier-3")
            in_db = sym in report.available
            ready = "✓ candles ready" if in_db else "✗ needs ingestion first"
            out.append(f"  {sym:<10} {tier:<8} {ready}")
        out.append("")

        full_set = report.configured + proposed
        # ``project_missing=True`` so the projection assumes the operator
        # actually runs the ingestion step in the migration plan below —
        # otherwise new symbols always score 0% growth (because they have
        # no candles yet) and the audit looks pointless.
        projected = estimate_corpus_size(report.statuses, full_set, project_missing=True)
        current = estimate_corpus_size(report.statuses, report.configured)
        out.append("Projected training corpus size (rough order-of-magnitude):")
        out.append(f"  current:    ~{current:>6} trades  (existing data)")
        out.append(f"  post-plan:  ~{projected:>6} trades  "
                   f"(+{projected - current:,} = {(projected/max(current,1) - 1)*100:.0f}% growth, "
                   "after ingestion step 1)")
        out.append("")

        out.append("Migration plan:")
        needs_ingestion = [s for s in proposed if s not in report.available]
        if needs_ingestion:
            out.append("  1. Ingest candles for missing symbols (BLOCKING):")
            out.append(f"     python scripts/ingest_candles.py --symbols {','.join(needs_ingestion)} --days 730")
            out.append("     # waits for ~30 minutes per symbol on Binance public REST")
            out.append("")
        out.append("  2. Update TRADING_SYMBOLS in sentinel/.env:")
        symbol_list = json.dumps(full_set)
        out.append(f"     TRADING_SYMBOLS={symbol_list}")
        out.append("")
        out.append("  3. Retrain the unified model (models per-symbol get auto-built on next cycle):")
        out.append("     python scripts/train_ml.py --symbols all")
        out.append("")
        out.append("  4. Verify diversity gate cleared in trainer log:")
        out.append("     grep 'corpus diversity OK' sentinel/logs/sentinel.log")

    if report.surplus_data:
        out.append("")
        out.append("Bonus: candle data already loaded but not configured for trading:")
        for s in report.surplus_data:
            out.append(f"  + {s}  ({report.statuses[s].coverage_summary()})")
        out.append("  → adding these to TRADING_SYMBOLS is FREE (no ingestion needed)")

    out.append("")
    out.append("=" * 70)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-symbols", type=int, default=5,
        help="Target portfolio size (default: 5; survivorship gate fires below 5)",
    )
    parser.add_argument(
        "--db-path", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sentinel.db",
    )
    parser.add_argument(
        "--env-path", type=Path,
        default=Path(__file__).resolve().parent.parent / ".env",
    )
    args = parser.parse_args(argv)

    if not args.db_path.exists():
        print(f"ERROR: DB not found at {args.db_path}", file=sys.stderr)
        return 1

    report = audit(args.db_path, args.env_path, target_symbols=args.target_symbols)
    print(render_report(report, args.target_symbols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
