"""Backfill historical OHLC candles for new trading symbols.

Currently the project's candle store covers only BTCUSDT and ETHUSDT,
which is the bottleneck for ML training (single-asset overfitting risk
per the survivorship-bias gate). This script fetches the missing
history from Binance's public REST endpoint and writes it to the
existing ``candles`` SQLite table — no schema changes, no new
infrastructure.

Why a script and not a runtime collector:

* The live ``binance_ws`` collector keeps recent candles fresh but is
  designed for single-bar updates, not multi-year backfills. Reusing
  it for backfill conflates two very different rate-limit budgets.
* Manual scripted ingestion is auditable: an operator runs it once,
  reviews the output, and the corpus grows in a known step. A continuous
  collector would silently start training on partial data.

Binance public REST limits (as of 2026-Q1):

* /api/v3/klines returns max 1000 rows per call.
* IP-banded weight: 6000/min for 1.5 IP-bands; we cap at 6 req/sec to
  stay well under threshold even for parallel symbol fetches.

Usage::

    python scripts/ingest_candles.py --symbols SOLUSDT,BNBUSDT,XRPUSDT --days 730
    python scripts/ingest_candles.py --symbols ADAUSDT --intervals 1h,4h,1d
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

# Binance kline interval string → milliseconds. The values match what the
# rest of the codebase already uses (collector/binance_ws.py:INTERVAL_MS).
INTERVAL_MS: dict[str, int] = {
    "1m":     60_000,
    "5m":    300_000,
    "15m":   900_000,
    "1h":  3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# Binance returns at most this many rows per /klines call.
BINANCE_KLINES_LIMIT = 1000

# Sleep between calls. 0.18s ≈ 5.5 req/sec, comfortably below the public
# rate limit while still finishing 5 years of 1h candles (~44k bars =
# 44 calls = ~8 seconds) in well under a minute.
DEFAULT_INTER_CALL_SLEEP = 0.18

# Maximum retries on transient errors (HTTP 429, 5xx, network blips).
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = (1.0, 3.0, 8.0)  # exponential per retry

KLINES_URL = "https://api.binance.com/api/v3/klines"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Per-(symbol, interval) summary of what got fetched and stored."""
    symbol: str
    interval: str
    requested_from_ts: int
    requested_to_ts: int
    n_fetched: int
    n_inserted: int
    n_failed: int
    elapsed_sec: float


@dataclass
class IngestionReport:
    """Aggregate report rendered at the end of a run."""
    results: list[IngestionResult] = field(default_factory=list)

    def total_inserted(self) -> int:
        return sum(r.n_inserted for r in self.results)

    def render(self) -> str:
        out = ["=" * 70, "CANDLE INGESTION REPORT", "=" * 70,
               f"{'symbol':<12} {'interval':<6} {'fetched':>8} {'inserted':>9} "
               f"{'failed':>7} {'elapsed':>9}",
               "-" * 70]
        for r in self.results:
            out.append(f"{r.symbol:<12} {r.interval:<6} {r.n_fetched:>8} "
                       f"{r.n_inserted:>9} {r.n_failed:>7} {r.elapsed_sec:>8.1f}s")
        out.append("-" * 70)
        out.append(f"{'TOTAL inserted':<28} {self.total_inserted():>10}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Binance fetcher
# ---------------------------------------------------------------------------


def _http_get_json(url: str, timeout: float = 30.0) -> Any:
    """Plain stdlib HTTP GET → JSON. We avoid pulling ``requests`` here
    because this script must be runnable from a minimal install (it's
    intentionally placed BEFORE the trainer install path)."""
    req = urllib.request.Request(url, headers={"User-Agent": "sentinel-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_klines(
    symbol: str,
    interval: str,
    start_ts: int,
    end_ts: int,
    *,
    inter_call_sleep: float = DEFAULT_INTER_CALL_SLEEP,
    http_get: Optional[Callable[[str], Any]] = None,
) -> Iterator[list]:
    """Yield kline rows for ``[start_ts, end_ts]`` in ``BINANCE_KLINES_LIMIT``
    chunks. Each yielded row is the raw 12-element list from Binance.

    The caller decides what to do with the rows (persist, validate,
    discard) — keeping this function pure-fetch makes it trivially
    unit-testable with a stub ``http_get``.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"unknown interval: {interval}")
    if start_ts >= end_ts:
        return

    interval_ms = INTERVAL_MS[interval]
    cursor = start_ts
    http_get = http_get or _http_get_json

    while cursor < end_ts:
        # Calculate the slice's upper bound: at most LIMIT bars from
        # cursor, but never overshoot end_ts.
        slice_end = min(cursor + BINANCE_KLINES_LIMIT * interval_ms, end_ts)
        params = urllib.parse.urlencode({
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": cursor,
            "endTime": slice_end,
            "limit": BINANCE_KLINES_LIMIT,
        })
        url = f"{KLINES_URL}?{params}"

        rows = _retry_call(http_get, url)
        if not rows:
            # No more data in this range — break out so we don't loop.
            return
        for row in rows:
            yield row

        # Advance cursor past the last bar's open_time. +1ms avoids
        # re-requesting the boundary bar on the next iteration.
        last_open = int(rows[-1][0])
        cursor = last_open + interval_ms
        if inter_call_sleep > 0:
            time.sleep(inter_call_sleep)


def _retry_call(http_get: Callable[[str], Any], url: str) -> Any:
    """Retry on transient HTTP errors with backoff. Permanent errors
    (4xx other than 429) raise immediately so the caller can surface
    config bugs (typo in symbol, retired interval) right away."""
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return http_get(url)
        except urllib.error.HTTPError as exc:
            # 429 = rate limited, 5xx = server problem; anything else is fatal.
            if exc.code != 429 and not (500 <= exc.code < 600):
                raise
            last_exc = exc
        except urllib.error.URLError as exc:
            # Network blip — also transient.
            last_exc = exc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        if attempt < MAX_RETRIES:
            backoff = RETRY_BACKOFF_SEC[min(attempt, len(RETRY_BACKOFF_SEC) - 1)]
            logger.warning("HTTP retry %d/%d after %ss (%s)",
                           attempt + 1, MAX_RETRIES, backoff, last_exc)
            time.sleep(backoff)
    raise RuntimeError(f"failed after {MAX_RETRIES} retries: {last_exc}")


# ---------------------------------------------------------------------------
# Persistence adapter
# ---------------------------------------------------------------------------


def klines_to_candles(rows: list[list], symbol: str, interval: str) -> list:
    """Convert raw Binance kline rows to the project's ``Candle`` dataclass.

    Binance row format (positional):
      0  open_time (ms)
      1  open
      2  high
      3  low
      4  close
      5  volume
      6  close_time (ms)
      7  quote_asset_volume
      8  number_of_trades
      9..11 — taker buy fields, ignored
    """
    from core.models import Candle
    out = []
    for r in rows:
        try:
            out.append(Candle(
                timestamp=int(r[0]),
                symbol=symbol.upper(),
                interval=interval,
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
                trades_count=int(r[8]) if len(r) > 8 else 0,
            ))
        except (IndexError, ValueError, TypeError) as exc:
            logger.debug("skipping malformed kline row %s: %s", r, exc)
    return out


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class IngestionConfig:
    symbols: list[str]
    intervals: list[str]
    days: int
    db_path: Path
    inter_call_sleep: float = DEFAULT_INTER_CALL_SLEEP


def ingest(
    cfg: IngestionConfig,
    *,
    http_get: Optional[Callable[[str], Any]] = None,
    repo_factory: Optional[Callable[[Path], Any]] = None,
    now_ms: Optional[int] = None,
) -> IngestionReport:
    """Run the full ingestion pipeline. Returns an :class:`IngestionReport`.

    Both ``http_get`` and ``repo_factory`` are dependency-injection seams
    used by the unit tests so we don't need real network or DB during
    testing.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    if repo_factory is None:
        from database.db import Database
        from database.repository import Repository

        def _default_factory(path: Path):
            db = Database(str(path))
            db.connect()
            return Repository(db)

        repo_factory = _default_factory

    repo = repo_factory(cfg.db_path)
    report = IngestionReport()

    for symbol in cfg.symbols:
        for interval in cfg.intervals:
            t0 = time.time()
            interval_ms = INTERVAL_MS[interval]
            start_ts = now_ms - cfg.days * 86_400_000
            # Resume from latest candle to avoid re-downloading what we have.
            try:
                latest = repo.get_latest_candle(symbol.upper(), interval)
            except Exception as exc:  # noqa: BLE001
                logger.warning("repo.get_latest_candle failed for %s/%s: %s",
                               symbol, interval, exc)
                latest = None
            if latest is not None:
                start_ts = max(start_ts, int(latest["timestamp"]) + interval_ms)

            n_fetched = n_inserted = n_failed = 0
            buffer: list = []
            try:
                for row in fetch_klines(
                    symbol, interval, start_ts, now_ms,
                    inter_call_sleep=cfg.inter_call_sleep,
                    http_get=http_get,
                ):
                    n_fetched += 1
                    buffer.append(row)
                    if len(buffer) >= 500:
                        n_inserted += _flush(repo, buffer, symbol, interval)
                        buffer = []
                if buffer:
                    n_inserted += _flush(repo, buffer, symbol, interval)
            except Exception as exc:  # noqa: BLE001
                logger.error("fetch failed for %s/%s: %s", symbol, interval, exc)
                n_failed = max(n_fetched - n_inserted, 1)

            report.results.append(IngestionResult(
                symbol=symbol.upper(),
                interval=interval,
                requested_from_ts=start_ts,
                requested_to_ts=now_ms,
                n_fetched=n_fetched,
                n_inserted=n_inserted,
                n_failed=n_failed,
                elapsed_sec=time.time() - t0,
            ))
            logger.info("Ingest %s/%s: fetched=%d inserted=%d failed=%d in %.1fs",
                        symbol, interval, n_fetched, n_inserted, n_failed,
                        time.time() - t0)

    return report


def _flush(repo: Any, rows: list, symbol: str, interval: str) -> int:
    """Convert a buffer of raw kline rows to Candle objects and bulk-insert."""
    candles = klines_to_candles(rows, symbol, interval)
    if not candles:
        return 0
    try:
        return int(repo.upsert_candles_batch(candles) or len(candles))
    except Exception as exc:  # noqa: BLE001
        logger.error("DB upsert failed for %s/%s (%d candles): %s",
                     symbol, interval, len(candles), exc)
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols", required=True,
        help="comma-separated symbols (e.g. 'SOLUSDT,BNBUSDT,XRPUSDT')",
    )
    parser.add_argument(
        "--intervals", default="1h,4h,1d",
        help="comma-separated kline intervals (default: 1h,4h,1d). "
             "Supported: " + ",".join(INTERVAL_MS),
    )
    parser.add_argument(
        "--days", type=int, default=730,
        help="how many days of history to backfill per (symbol, interval). "
             "Default 730 (2 years) — Binance retains 1m for ~6 weeks but "
             "1h+ goes back to listing date.",
    )
    parser.add_argument(
        "--db-path", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sentinel.db",
    )
    parser.add_argument(
        "--inter-call-sleep", type=float, default=DEFAULT_INTER_CALL_SLEEP,
        help="seconds to wait between Binance REST calls (rate-limit budget). "
             f"Default {DEFAULT_INTER_CALL_SLEEP}s ≈ 5.5 req/sec.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    bad_intervals = [i for i in args.intervals.split(",") if i not in INTERVAL_MS]
    if bad_intervals:
        print(f"ERROR: unknown intervals {bad_intervals}; "
              f"supported: {','.join(INTERVAL_MS)}",
              file=sys.stderr)
        return 1

    if not args.db_path.exists():
        print(f"ERROR: DB not found at {args.db_path}", file=sys.stderr)
        return 1

    cfg = IngestionConfig(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        intervals=[i.strip() for i in args.intervals.split(",") if i.strip()],
        days=args.days,
        db_path=args.db_path,
        inter_call_sleep=args.inter_call_sleep,
    )
    report = ingest(cfg)
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
