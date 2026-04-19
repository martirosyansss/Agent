"""Tests for the universe-expansion audit script.

The script is read-only and produces a recommendation report — its
contracts are:

* Returns 0 even on a degenerate universe (it's an analysis tool, not a
  gate; a CI run should never fail because of it).
* Loads symbols from .env via JSON parse, falls back to defaults on
  missing/malformed input.
* Recommends additions in tier order (large-cap before mid-cap before
  high-beta) and never re-suggests an already-configured symbol.
* Estimate function is monotone in symbol count (more symbols ⇒ ≥ corpus
  size) and respects the project_missing flag.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import analyze_universe as audit_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Build a tiny SQLite DB with a candles table for two symbols."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE candles (symbol TEXT, timestamp INTEGER, close REAL)")
    # 100 candles each for BTC + ETH, 10 days apart.
    base_ts = 1_700_000_000_000
    for sym in ("BTCUSDT", "ETHUSDT"):
        for i in range(100):
            conn.execute(
                "INSERT INTO candles VALUES (?, ?, ?)",
                (sym, base_ts + i * 86_400_000, 100.0 + i),
            )
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def tmp_env(tmp_path: Path) -> Path:
    env = tmp_path / ".env"
    env.write_text('TRADING_SYMBOLS=["BTCUSDT","ETHUSDT"]\n', encoding="utf-8")
    return env


# ---------------------------------------------------------------------------
# load_configured_symbols
# ---------------------------------------------------------------------------


class TestLoadConfiguredSymbols:
    def test_parses_valid_json_list(self, tmp_env):
        assert audit_mod.load_configured_symbols(tmp_env) == ["BTCUSDT", "ETHUSDT"]

    def test_falls_back_when_env_missing(self, tmp_path):
        out = audit_mod.load_configured_symbols(tmp_path / "nonexistent.env")
        assert out == ["BTCUSDT", "ETHUSDT"]

    def test_falls_back_on_malformed_json(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("TRADING_SYMBOLS=not json\n", encoding="utf-8")
        out = audit_mod.load_configured_symbols(env)
        assert out == ["BTCUSDT", "ETHUSDT"]


# ---------------------------------------------------------------------------
# query_symbol_status
# ---------------------------------------------------------------------------


class TestQuerySymbolStatus:
    def test_marks_db_present_symbols(self, tmp_db):
        statuses = audit_mod.query_symbol_status(tmp_db, ["BTCUSDT", "ETHUSDT"])
        for sym in ("BTCUSDT", "ETHUSDT"):
            assert statuses[sym].has_candles
            assert statuses[sym].candle_count == 100
            assert statuses[sym].days_of_data > 0

    def test_marks_missing_symbols(self, tmp_db):
        statuses = audit_mod.query_symbol_status(tmp_db, ["BTCUSDT", "SOLUSDT"])
        assert not statuses["SOLUSDT"].has_candles
        assert statuses["SOLUSDT"].candle_count == 0


# ---------------------------------------------------------------------------
# recommend_additions
# ---------------------------------------------------------------------------


class TestRecommendAdditions:
    def test_returns_empty_when_target_already_met(self):
        assert audit_mod.recommend_additions(["A", "B", "C", "D", "E"], target=3) == []

    def test_picks_tier1_first(self):
        proposed = audit_mod.recommend_additions(["BTCUSDT", "ETHUSDT"], target=4)
        # All proposals must be tier-1 first.
        for sym in proposed[:2]:
            assert sym in audit_mod.TIER_1_LARGE_CAP

    def test_never_suggests_already_configured_symbol(self):
        proposed = audit_mod.recommend_additions(["BTCUSDT", "SOLUSDT"], target=5)
        assert "BTCUSDT" not in proposed
        assert "SOLUSDT" not in proposed

    def test_falls_through_tiers_when_tier1_exhausted(self):
        # All tier-1 already configured → must dip into tier-2.
        proposed = audit_mod.recommend_additions(audit_mod.TIER_1_LARGE_CAP, target=7)
        assert all(sym in audit_mod.TIER_2_MID_CAP for sym in proposed)


# ---------------------------------------------------------------------------
# estimate_corpus_size
# ---------------------------------------------------------------------------


def _make_status(has_candles: bool, days: int = 100) -> audit_mod.SymbolStatus:
    return audit_mod.SymbolStatus(
        symbol="X", has_candles=has_candles,
        candle_count=days * 24 if has_candles else 0,
        earliest_ts=0 if has_candles else None,
        latest_ts=days * 86_400_000 if has_candles else None,
        days_of_data=days if has_candles else 0,
    )


class TestEstimateCorpusSize:
    def test_zero_for_empty_input(self):
        assert audit_mod.estimate_corpus_size({}, []) == 0

    def test_monotone_in_symbol_count(self):
        statuses = {
            "A": _make_status(True, 100),
            "B": _make_status(True, 100),
            "C": _make_status(True, 100),
        }
        n1 = audit_mod.estimate_corpus_size(statuses, ["A"])
        n2 = audit_mod.estimate_corpus_size(statuses, ["A", "B"])
        n3 = audit_mod.estimate_corpus_size(statuses, ["A", "B", "C"])
        assert n1 < n2 < n3

    def test_missing_symbols_zero_unless_projected(self):
        statuses = {
            "A": _make_status(True, 200),
            "B": _make_status(False),
        }
        # Without projection, B contributes nothing.
        n_no_proj = audit_mod.estimate_corpus_size(statuses, ["A", "B"])
        n_a_only = audit_mod.estimate_corpus_size(statuses, ["A"])
        assert n_no_proj == n_a_only
        # With projection, B is imputed at the median day-coverage of A.
        n_proj = audit_mod.estimate_corpus_size(
            statuses, ["A", "B"], project_missing=True,
        )
        assert n_proj > n_no_proj


# ---------------------------------------------------------------------------
# End-to-end: audit + render
# ---------------------------------------------------------------------------


class TestAuditEndToEnd:
    def test_audit_produces_well_formed_report(self, tmp_db, tmp_env):
        report = audit_mod.audit(tmp_db, tmp_env, target_symbols=5)
        assert report.configured == ["BTCUSDT", "ETHUSDT"]
        assert "BTCUSDT" in report.available
        assert "ETHUSDT" in report.available
        # diversity score must be < 0.5 — only 2 configured symbols.
        assert report.diversity_score() < 0.5

    def test_render_includes_warning_for_low_diversity(self, tmp_db, tmp_env):
        report = audit_mod.audit(tmp_db, tmp_env, target_symbols=5)
        text = audit_mod.render_report(report, target=5)
        assert "WARNING" in text
        assert "single-asset risk" in text
        assert "Migration plan" in text

    def test_main_returns_zero_on_normal_run(self, tmp_db, tmp_env, capsys):
        rc = audit_mod.main([
            "--db-path", str(tmp_db),
            "--env-path", str(tmp_env),
            "--target-symbols", "5",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ML UNIVERSE AUDIT" in out

    def test_main_returns_one_on_missing_db(self, tmp_path, tmp_env, capsys):
        rc = audit_mod.main([
            "--db-path", str(tmp_path / "nope.db"),
            "--env-path", str(tmp_env),
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "DB not found" in err
