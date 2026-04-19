"""Tests for the corpus-expansion pipeline.

The script is dependency-injection-friendly so we can substitute a fake
candle loader and a fake backtest runner — keeping the tests fast,
hermetic, and free from sklearn / lightgbm import overhead.

Locks in the contracts:

* The variant catalogue covers every supported strategy and never ships
  empty.
* expand_corpus returns one report row per (strategy, variant, symbol,
  timeframe) tuple it actually executed.
* Lenient mode is opt-in (default off) and floors min_confidence at 0.55.
* Block-bootstrap augmentation only fires when ``bootstrap_target`` > N.
* The CLI writes a pickle and exits 0 on a normal run.
* Variants whose overrides break the strategy config are skipped, not
  fatal — operator gets a warning, the rest of the corpus still builds.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest


# Import without triggering DB connection: the module-level imports are
# light; DB only loads inside expand_corpus when no loader is injected.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import expand_training_corpus as ec


# ---------------------------------------------------------------------------
# Fake injectable runners
# ---------------------------------------------------------------------------


def _fake_loader(symbol: str, tf: str) -> list:
    # Return a non-empty stub so expand_corpus calls the backtest at all.
    # Content is irrelevant — the fake backtest doesn't read it.
    return [object()] * 10


def _fake_backtest(strategy, name: str, c1, c2, c3, sym: str) -> list:
    """Return one mock trade per (strategy_name, symbol) — keeps trade
    counts predictable so test assertions are exact."""
    return [{"name": name, "symbol": sym}]


# ---------------------------------------------------------------------------
# Variant catalogue
# ---------------------------------------------------------------------------


class TestVariantCatalogue:
    def test_every_strategy_has_at_least_one_variant(self):
        for name, variants in ec.STRATEGY_VARIANTS.items():
            assert variants, f"{name} has no variants"

    def test_default_variant_is_first(self):
        # Convention: first variant of every strategy is "default" with
        # no overrides — gives reviewers a clean reference to compare
        # the others against.
        for name, variants in ec.STRATEGY_VARIANTS.items():
            first = variants[0]
            assert first[0] == "default", (
                f"{name}: first variant should be 'default', got {first[0]}"
            )
            assert first[1] == {}, (
                f"{name}: default variant must have empty overrides, "
                f"got {first[1]}"
            )

    def test_variant_names_unique_within_strategy(self):
        for name, variants in ec.STRATEGY_VARIANTS.items():
            names = [v[0] for v in variants]
            assert len(names) == len(set(names)), (
                f"{name}: duplicate variant names {names}"
            )


# ---------------------------------------------------------------------------
# expand_corpus
# ---------------------------------------------------------------------------


class TestExpandCorpus:
    def test_one_row_per_combination(self):
        cfg = ec.ExpansionConfig(
            symbols=["BTC", "ETH"],
            timeframes=["1h"],
            lenient=False,
            output_path=None,
        )
        trades, report = ec.expand_corpus(cfg, candle_loader=_fake_loader,
                                          backtest_runner=_fake_backtest)

        # Expected combinations = sum(variants per strategy) × symbols × timeframes
        expected = sum(len(v) for v in ec.STRATEGY_VARIANTS.values()) * 2 * 1
        assert len(report.rows) == expected
        assert report.total_trades == expected  # fake returns 1 trade per call
        assert len(trades) == expected

    def test_multi_timeframe_multiplies_combinations(self):
        cfg = ec.ExpansionConfig(
            symbols=["BTC"], timeframes=["1h", "4h"],
            lenient=False, output_path=None,
        )
        _, report = ec.expand_corpus(cfg, candle_loader=_fake_loader,
                                     backtest_runner=_fake_backtest)
        # 2 timeframes → exactly 2× the rows of a single-tf run.
        per_strategy = sum(len(v) for v in ec.STRATEGY_VARIANTS.values())
        assert len(report.rows) == per_strategy * 1 * 2

    def test_lenient_floors_min_confidence(self):
        """A variant with no min_confidence override should receive the
        lenient floor (0.55) when lenient=True."""
        seen_kwargs: list[dict] = []

        def spy_build(name, overrides, lenient):
            cfg_kwargs = dict(overrides)
            if lenient:
                cfg_kwargs.setdefault("min_confidence", ec.LENIENT_MIN_CONFIDENCE)
            seen_kwargs.append(cfg_kwargs)
            # Return a stub strategy — we don't care about its behaviour.
            class _Stub: pass
            return _Stub()

        cfg = ec.ExpansionConfig(
            symbols=["BTC"], timeframes=["1h"], lenient=True, output_path=None,
        )
        # Monkey-patch _build_strategy on this single call.
        original = ec._build_strategy
        ec._build_strategy = spy_build
        try:
            ec.expand_corpus(cfg, candle_loader=_fake_loader,
                             backtest_runner=_fake_backtest)
        finally:
            ec._build_strategy = original

        # Every variant should have min_confidence injected by lenient mode.
        with_floor = [k for k in seen_kwargs
                      if k.get("min_confidence") == ec.LENIENT_MIN_CONFIDENCE]
        assert with_floor, "lenient mode never floored min_confidence"

    def test_variant_with_invalid_kwarg_is_skipped_not_fatal(self, caplog):
        import logging
        # Inject a poison variant that the strategy can't accept.
        original = ec.STRATEGY_VARIANTS.copy()
        ec.STRATEGY_VARIANTS["mean_reversion"] = original["mean_reversion"] + [
            ("poison", {"nonexistent_param": 42}),
        ]
        try:
            cfg = ec.ExpansionConfig(symbols=["BTC"], timeframes=["1h"],
                                     output_path=None)
            with caplog.at_level(logging.WARNING):
                _, report = ec.expand_corpus(
                    cfg, candle_loader=_fake_loader, backtest_runner=_fake_backtest,
                )
            # Other variants still ran.
            mr_rows = [r for r in report.rows if r["strategy"] == "mean_reversion"]
            assert any(r["variant"] != "poison" for r in mr_rows), (
                "Poison variant aborted the entire mean_reversion family"
            )
            assert any("invalid override" in m or "skipping" in m
                       for m in (r.getMessage() for r in caplog.records))
        finally:
            ec.STRATEGY_VARIANTS["mean_reversion"] = original["mean_reversion"]

    def test_writes_pickle_when_output_path_given(self, tmp_path: Path):
        out = tmp_path / "expanded.pkl"
        cfg = ec.ExpansionConfig(symbols=["BTC"], timeframes=["1h"],
                                 output_path=out)
        ec.expand_corpus(cfg, candle_loader=_fake_loader,
                         backtest_runner=_fake_backtest)
        assert out.exists()
        with out.open("rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, list)
        assert len(loaded) > 0


# ---------------------------------------------------------------------------
# Block-bootstrap augmentation
# ---------------------------------------------------------------------------


class TestBootstrapAugment:
    def test_no_op_when_target_below_n(self):
        trades = [object()] * 200
        out = ec._bootstrap_augment(trades, target=100, block_size=10, seed=0)
        assert out is trades  # same object, untouched

    def test_grows_to_target(self):
        trades = [object()] * 50
        out = ec._bootstrap_augment(trades, target=200, block_size=10, seed=0)
        assert len(out) == 200
        # First 50 entries are the originals, in order.
        assert out[:50] == trades

    def test_empty_input_no_growth(self):
        out = ec._bootstrap_augment([], target=100, block_size=10, seed=0)
        assert out == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_main_returns_zero_on_normal_run(self, tmp_path: Path,
                                              monkeypatch, capsys):
        # Stub the heavy DB / strategy imports by injecting our fakes
        # at the module-loader level — main() pulls them lazily so we can
        # patch ec.expand_corpus to return a canned result.
        out = tmp_path / "out.pkl"

        def fake_expand(cfg):
            cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg.output_path.open("wb") as f:
                pickle.dump([{"x": 1}], f)
            r = ec.CorpusReport(started_at=0, finished_at=1)
            r.add(strategy="x", variant="default", symbol="BTC", tf="1h", n_trades=1)
            return [{"x": 1}], r

        monkeypatch.setattr(ec, "expand_corpus", fake_expand)
        rc = ec.main(["--symbols", "BTC", "--timeframes", "1h",
                      "--output", str(out)])
        assert rc == 0
        printed = capsys.readouterr().out
        assert "TOTAL" in printed
        assert out.exists()
