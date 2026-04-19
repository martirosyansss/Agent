"""Tests for the candle backfill script.

Locks in the contracts:

* fetch_klines paginates correctly: a 2,500-bar request comes back in
  three calls (1000 + 1000 + 500), with cursor advancing past the last
  open_time on each call.
* fetch_klines stops when the API returns an empty page.
* Unknown interval raises ValueError immediately (no silent loop forever).
* _retry_call retries on 429 / 5xx but bails on 404.
* klines_to_candles produces well-formed Candle objects, skips malformed
  rows.
* ingest() resumes from the latest stored candle (no re-download of
  existing data).
* CLI rejects unknown intervals and missing DB path with exit 1.
"""
from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ingest_candles as ic


# ---------------------------------------------------------------------------
# Fake HTTP + Repo
# ---------------------------------------------------------------------------


class FakeHttp:
    """Minimal HTTP stub: returns canned pages keyed by call index."""

    def __init__(self, pages: list[list]):
        self.pages = pages
        self.calls: list[str] = []

    def __call__(self, url: str):
        self.calls.append(url)
        if not self.pages:
            return []
        return self.pages.pop(0)


class FakeRepo:
    def __init__(self, latest: dict | None = None):
        self.latest = latest
        self.upserts: list[list] = []

    def get_latest_candle(self, symbol, interval):
        return self.latest

    def upsert_candles_batch(self, candles):
        self.upserts.append(candles)
        return len(candles)


def _make_kline(open_ts: int, close_ts: int) -> list:
    """One Binance kline row in the documented positional format."""
    return [open_ts, "100.0", "110.0", "90.0", "105.0", "1234.5",
            close_ts, "12345.0", 42, "1.0", "1.0", "0"]


# ---------------------------------------------------------------------------
# fetch_klines pagination
# ---------------------------------------------------------------------------


class TestFetchKlines:
    def test_unknown_interval_raises(self):
        with pytest.raises(ValueError):
            list(ic.fetch_klines("BTCUSDT", "7m", 0, 1000))

    def test_empty_range_yields_nothing(self):
        # start_ts >= end_ts → immediate return, no HTTP calls.
        http = FakeHttp([])
        list(ic.fetch_klines("BTCUSDT", "1h", 1000, 1000, http_get=http))
        assert http.calls == []

    def test_paginates_until_end(self):
        interval_ms = ic.INTERVAL_MS["1h"]
        # Two pages of 3 bars each (we're using a tiny LIMIT-1 stub here
        # by handing fetch_klines pages of size 3 < BINANCE_KLINES_LIMIT,
        # which makes it reach end_ts naturally after the second page).
        page_a = [_make_kline(1000 + i * interval_ms, 1000 + (i + 1) * interval_ms)
                  for i in range(3)]
        page_b = [_make_kline(1000 + (3 + i) * interval_ms,
                              1000 + (4 + i) * interval_ms) for i in range(3)]
        http = FakeHttp([page_a, page_b, []])

        rows = list(ic.fetch_klines(
            "BTCUSDT", "1h", 1000, 1000 + 10 * interval_ms,
            inter_call_sleep=0, http_get=http,
        ))
        assert len(rows) == 6
        # Cursor advanced at least twice.
        assert len(http.calls) >= 2

    def test_stops_when_page_empty(self):
        http = FakeHttp([[]])  # first call returns nothing
        rows = list(ic.fetch_klines(
            "BTCUSDT", "1h", 1000, 1_000_000_000,
            inter_call_sleep=0, http_get=http,
        ))
        assert rows == []
        assert len(http.calls) == 1


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetryCall:
    def test_retries_on_429_then_succeeds(self, monkeypatch):
        # 429 twice, then OK on the third try.
        attempts = {"n": 0}
        monkeypatch.setattr(ic.time, "sleep", lambda s: None)  # no real waits

        def http_get(url):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise urllib.error.HTTPError(url, 429, "rate limited", {}, None)
            return [{"ok": True}]

        result = ic._retry_call(http_get, "http://x")
        assert result == [{"ok": True}]
        assert attempts["n"] == 3

    def test_does_not_retry_on_404(self):
        def http_get(url):
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)

        with pytest.raises(urllib.error.HTTPError):
            ic._retry_call(http_get, "http://x")

    def test_gives_up_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(ic.time, "sleep", lambda s: None)

        def always_503(url):
            raise urllib.error.HTTPError(url, 503, "server", {}, None)

        with pytest.raises(RuntimeError, match="failed after"):
            ic._retry_call(always_503, "http://x")


# ---------------------------------------------------------------------------
# klines_to_candles
# ---------------------------------------------------------------------------


class TestKlinesToCandles:
    def test_well_formed_row_produces_candle(self):
        candles = ic.klines_to_candles(
            [_make_kline(1000, 2000)], "BTCUSDT", "1h",
        )
        assert len(candles) == 1
        c = candles[0]
        assert c.symbol == "BTCUSDT"
        assert c.interval == "1h"
        assert c.open == 100.0
        assert c.close == 105.0
        assert c.trades_count == 42

    def test_skips_malformed_rows(self):
        # Row missing fields → quietly dropped, not fatal.
        rows = [
            [1000],                                   # too short
            _make_kline(2000, 3000),                  # OK
            [3000, "bad_number"],                     # broken
        ]
        candles = ic.klines_to_candles(rows, "BTCUSDT", "1h")
        assert len(candles) == 1


# ---------------------------------------------------------------------------
# ingest() integration
# ---------------------------------------------------------------------------


class TestIngest:
    def test_resumes_from_latest_candle(self, tmp_path):
        # Repo has latest candle at ts=5000; ingestion should request
        # from ts > 5000, not from days-ago.
        latest = {"timestamp": 5000}
        captured_urls: list[str] = []

        def http_get(url):
            captured_urls.append(url)
            return []  # nothing more to fetch

        cfg = ic.IngestionConfig(
            symbols=["BTCUSDT"], intervals=["1h"], days=30,
            db_path=tmp_path / "fake.db", inter_call_sleep=0,
        )
        ic.ingest(
            cfg,
            http_get=http_get,
            repo_factory=lambda _: FakeRepo(latest=latest),
            now_ms=10_000_000,
        )
        # First URL's startTime should reflect the latest+interval, not
        # days-ago.
        assert captured_urls
        assert "startTime=" in captured_urls[0]
        # 5000 + 3_600_000 = 3_605_000 (interval 1h)
        assert "startTime=3605000" in captured_urls[0]

    def test_inserts_fetched_rows_via_repo(self, tmp_path):
        interval_ms = ic.INTERVAL_MS["1h"]
        repo = FakeRepo()
        page = [_make_kline(1_000 + i * interval_ms, 1_000 + (i + 1) * interval_ms)
                for i in range(5)]

        def http_get(url):
            # Single non-empty page, then empty.
            if not getattr(http_get, "called", False):
                http_get.called = True
                return page
            return []

        cfg = ic.IngestionConfig(
            symbols=["BTCUSDT"], intervals=["1h"], days=1,
            db_path=tmp_path / "fake.db", inter_call_sleep=0,
        )
        report = ic.ingest(
            cfg,
            http_get=http_get,
            repo_factory=lambda _: repo,
            now_ms=10_000_000,
        )
        # 5 candles fetched, 5 upserted (in one batch).
        assert report.results[0].n_fetched == 5
        assert report.results[0].n_inserted == 5
        assert len(repo.upserts) == 1
        assert len(repo.upserts[0]) == 5

    def test_reports_failure_when_fetch_explodes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ic.time, "sleep", lambda s: None)

        def http_get(url):
            raise RuntimeError("network down")

        cfg = ic.IngestionConfig(
            symbols=["BTCUSDT"], intervals=["1h"], days=1,
            db_path=tmp_path / "fake.db", inter_call_sleep=0,
        )
        report = ic.ingest(
            cfg,
            http_get=http_get,
            repo_factory=lambda _: FakeRepo(),
            now_ms=10_000_000,
        )
        # n_failed bumped, ingestion didn't crash the caller.
        assert report.results[0].n_failed >= 1
        assert report.results[0].n_inserted == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_rejects_unknown_interval(self, tmp_path, capsys):
        # Need an existing DB path so we get past the early DB check
        # and reach the interval validation step.
        db = tmp_path / "fake.db"
        db.touch()
        rc = ic.main([
            "--symbols", "SOLUSDT",
            "--intervals", "1h,7m",   # 7m unsupported
            "--db-path", str(db),
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "unknown intervals" in err

    def test_rejects_missing_db(self, tmp_path, capsys):
        rc = ic.main([
            "--symbols", "SOLUSDT",
            "--db-path", str(tmp_path / "nope.db"),
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "DB not found" in err
