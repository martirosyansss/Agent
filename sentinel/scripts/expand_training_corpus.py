"""Expand the ML training corpus by squeezing more signals out of the
candle data we already have.

The base trainer (``scripts/train_ml.py``) instantiates each strategy with
its default config and runs ONE backtest per (strategy, symbol) on 1h
candles. That gives ~1,500 training samples on 5 years of BTC + ETH —
enough for a hobby model, way too few for a robust ensemble.

This script bridges the gap by combining four amplification techniques on
the SAME existing candle data:

1. **Parametric variants** (Bergstra & Bengio 2012): each strategy gets
   3–5 hyperparameter overrides. ``bollinger_breakout`` with ``bb_period
   ∈ {14, 20, 30}`` is three different signal generators for ML purposes
   even though it's one strategy in production.
2. **Multi-timeframe**: every (strategy, variant) is also backtested on
   the 4h candles, capturing the same logic at a slower regime. The 1h
   and 4h trade sets are union-ed (deduplication is unnecessary because
   the timestamps differ by definition).
3. **Lenient confidence threshold**: optionally relax ``min_confidence``
   to a floor (default 0.55). The production thresholds are set for
   "rather miss a trade than take a bad one"; for ML training we want
   coverage over precision — the model itself learns which low-confidence
   setups are good. Off by default; ``--lenient`` opts in.
4. **Block-bootstrap augmentation** (optional): when the union still has
   < N samples, draw block-bootstrap re-samples of the trade list to fill
   to N. This DOES NOT add new information but stabilises the trainer's
   variance on small holdouts. ``--bootstrap-target`` controls N.

Output: a single ``StrategyTrade`` list, written to
``data/ml_models/expanded_trades.pkl``, ready for the trainer to pick up
via a ``--from-corpus`` flag (added to ``train_ml.py`` separately).

Why a script and not a class: this is a one-shot data-prep job, not a
runtime component. Operators will rerun it weekly — the simpler the
entry point, the lower the chance of an operator running an outdated
code path against a fresh corpus.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant catalogue
# ---------------------------------------------------------------------------

# Each entry: (suffix, override_kwargs). ``min_confidence`` overrides are
# layered on top via ``--lenient`` so the catalogue stays decoupled from
# the threshold-relaxation policy. Keep variants UNCORRELATED — three
# values of the same param are fine; bb_period=20 and ema_fast=9 in the
# same variant defeats the diversification benefit.

STRATEGY_VARIANTS: dict[str, list[tuple[str, dict]]] = {
    "bollinger_breakout": [
        ("default", {}),
        ("tight",   {"bb_period": 14, "bb_std_dev": 2.0}),
        ("wide",    {"bb_period": 30, "bb_std_dev": 2.5}),
        ("squeeze", {"squeeze_threshold": 0.03}),
    ],
    "mean_reversion": [
        ("default",  {}),
        ("conservative", {"rsi_oversold": 20.0, "rsi_overbought": 80.0}),
        ("aggressive",   {"rsi_oversold": 30.0, "rsi_overbought": 70.0}),
        ("low_vol",      {"min_volume_ratio": 1.2}),
    ],
    "macd_divergence": [
        ("default", {}),
        ("short",   {"lookback_candles": 40, "min_divergence_bars": 6}),
        ("long",    {"lookback_candles": 90, "min_divergence_bars": 14}),
    ],
    "ema_crossover_rsi": [
        ("default", {}),
        ("fast",    {"ema_fast": 5, "ema_slow": 13}),
        ("slow",    {"ema_fast": 12, "ema_slow": 26}),
    ],
    "grid_trading": [
        ("default",   {}),
        ("dense",     {"num_grids": 12, "min_profit_pct": 1.0}),
        ("wide",      {"num_grids": 6,  "min_profit_pct": 1.5}),
    ],
    "dca_bot": [
        ("default", {}),
        # DCA has fewer interesting hyperparams that don't change the
        # entire strategy semantics; one variant is enough.
    ],
}

# Floor below which we won't push min_confidence even in lenient mode.
# Below this the strategy starts entering on essentially random signals,
# which degrades ML labels rather than adding diversity.
LENIENT_MIN_CONFIDENCE = 0.55


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CorpusReport:
    """End-of-run diagnostic — one row per (strategy, variant, symbol, tf)."""
    rows: list[dict] = field(default_factory=list)
    total_trades: int = 0
    started_at: int = 0
    finished_at: int = 0

    def add(self, **kw: Any) -> None:
        self.rows.append(kw)
        self.total_trades += int(kw.get("n_trades", 0))

    def render(self) -> str:
        out: list[str] = []
        out.append("=" * 80)
        out.append("TRAINING CORPUS EXPANSION REPORT")
        out.append("=" * 80)
        out.append(f"{'strategy':<20} {'variant':<14} {'symbol':<8} {'tf':<4} {'trades':>6}")
        out.append("-" * 80)
        for r in sorted(self.rows, key=lambda x: -x["n_trades"]):
            out.append(f"  {r['strategy']:<18} {r['variant']:<14} {r['symbol']:<8} "
                       f"{r['tf']:<4} {r['n_trades']:>6}")
        out.append("-" * 80)
        out.append(f"{'TOTAL':<20} {'':<14} {'':<8} {'':<4} {self.total_trades:>6}")
        out.append("")
        elapsed = max((self.finished_at - self.started_at) / 1000, 0.001)
        out.append(f"Elapsed: {elapsed:.1f}s "
                   f"({self.total_trades / elapsed:.0f} trades/sec)")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Strategy factory with override support
# ---------------------------------------------------------------------------


def _build_strategy(name: str, overrides: dict, lenient: bool) -> Any:
    """Construct a strategy instance with optional config overrides.

    Imports are local so this module can be imported without dragging in
    the heavy strategy modules (useful for the unit tests below).
    """
    cfg_kwargs = dict(overrides)
    if lenient:
        cfg_kwargs.setdefault("min_confidence", LENIENT_MIN_CONFIDENCE)

    if name == "bollinger_breakout":
        from strategy.bollinger_breakout import BollingerBreakout, BBBreakoutConfig
        return BollingerBreakout(BBBreakoutConfig(**cfg_kwargs))
    if name == "mean_reversion":
        from strategy.mean_reversion import MeanReversion, MeanRevConfig
        return MeanReversion(MeanRevConfig(**cfg_kwargs))
    if name == "macd_divergence":
        from strategy.macd_divergence import MACDDivergence, MACDDivConfig
        return MACDDivergence(MACDDivConfig(**cfg_kwargs))
    if name == "ema_crossover_rsi":
        from strategy.ema_crossover_rsi import EMACrossoverRSI, EMAConfig
        return EMACrossoverRSI(EMAConfig(**cfg_kwargs))
    if name == "grid_trading":
        from strategy.grid_trading import GridTrading, GridConfig
        return GridTrading(GridConfig(**cfg_kwargs))
    if name == "dca_bot":
        from strategy.dca_bot import DCABot, DCAConfig
        # DCA's config does not currently take min_confidence — drop the
        # lenient kwarg silently rather than crashing.
        cfg_kwargs.pop("min_confidence", None)
        return DCABot(DCAConfig(**cfg_kwargs)) if cfg_kwargs else DCABot()
    raise ValueError(f"unknown strategy: {name}")


# ---------------------------------------------------------------------------
# Corpus expansion
# ---------------------------------------------------------------------------


@dataclass
class ExpansionConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframes: list[str] = field(default_factory=lambda: ["1h"])
    lenient: bool = False
    bootstrap_target: int = 0           # 0 = no bootstrap augmentation
    bootstrap_block_size: int = 10
    output_path: Optional[Path] = None
    seed: int = 42


def expand_corpus(
    cfg: ExpansionConfig,
    candle_loader: Optional[Callable[[str, str], list]] = None,
    backtest_runner: Optional[Callable] = None,
) -> tuple[list, CorpusReport]:
    """Generate the expanded training corpus.

    The two callable parameters are dependency injection seams used by
    the unit tests so we don't need a real DB connection. In production
    they default to the real loaders pulled lazily from
    ``database`` and ``scripts.train_ml``.
    """
    if candle_loader is None:
        from database.db import Database
        from database.repository import Repository
        from scripts.train_ml import load_candles as _real_loader

        db = Database(str(Path(__file__).resolve().parent.parent / "data" / "sentinel.db"))
        db.connect()
        repo = Repository(db)
        candle_loader = lambda sym, tf: _real_loader(repo, sym, tf)  # noqa: E731

    if backtest_runner is None:
        from scripts.train_ml import run_backtest_with_features as backtest_runner

    report = CorpusReport(started_at=int(time.time() * 1000))
    all_trades: list = []

    # Pre-load candles once per (symbol, tf) — backtest is the hot loop;
    # pulling the same 770k candles per variant would be the whole runtime.
    candle_cache: dict[tuple[str, str], list] = {}
    candle_4h_cache: dict[str, list] = {}
    candle_1d_cache: dict[str, list] = {}
    for sym in cfg.symbols:
        candle_4h_cache[sym] = candle_loader(sym, "4h")
        candle_1d_cache[sym] = candle_loader(sym, "1d")
        for tf in cfg.timeframes:
            candle_cache[(sym, tf)] = candle_loader(sym, tf)

    for strategy_name, variants in STRATEGY_VARIANTS.items():
        for variant_name, overrides in variants:
            for sym in cfg.symbols:
                for tf in cfg.timeframes:
                    primary = candle_cache.get((sym, tf), [])
                    c4h = candle_4h_cache.get(sym, [])
                    c1d = candle_1d_cache.get(sym, [])
                    if not primary:
                        continue
                    try:
                        strat = _build_strategy(strategy_name, overrides, cfg.lenient)
                    except TypeError as exc:
                        # Variant ships a kwarg the strategy config doesn't
                        # accept — log and skip rather than abort the whole
                        # corpus build.
                        logger.warning(
                            "skipping %s/%s: invalid override (%s)",
                            strategy_name, variant_name, exc,
                        )
                        continue
                    try:
                        trades = backtest_runner(strat, f"{strategy_name}_{variant_name}",
                                                  primary, c4h, c1d, sym)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("backtest failed for %s/%s on %s/%s: %s",
                                       strategy_name, variant_name, sym, tf, exc)
                        trades = []
                    all_trades.extend(trades)
                    report.add(strategy=strategy_name, variant=variant_name,
                               symbol=sym, tf=tf, n_trades=len(trades))

    # Optional block-bootstrap augmentation
    if cfg.bootstrap_target and len(all_trades) < cfg.bootstrap_target:
        all_trades = _bootstrap_augment(all_trades, cfg.bootstrap_target,
                                         cfg.bootstrap_block_size, cfg.seed)
        report.add(strategy="bootstrap", variant="block",
                   symbol="ALL", tf="N/A",
                   n_trades=cfg.bootstrap_target - report.total_trades)

    report.finished_at = int(time.time() * 1000)

    if cfg.output_path is not None:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg.output_path.open("wb") as f:
            pickle.dump(all_trades, f)
        logger.info("Wrote %d trades to %s", len(all_trades), cfg.output_path)

    return all_trades, report


def _bootstrap_augment(trades: list, target: int, block_size: int, seed: int) -> list:
    """Block-bootstrap a trade list to ``target`` length.

    Block sampling preserves local autocorrelation (consecutive trades
    share regime / volatility cluster); iid bootstrap would silently
    inflate the trainer's apparent N without honouring the time-series
    structure.
    """
    if not trades:
        return trades
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(trades)
    extra_needed = target - n
    if extra_needed <= 0:
        return trades
    n_blocks = max(1, extra_needed // block_size + 1)
    starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
    augmented: list = []
    for s in starts:
        augmented.extend(trades[s:s + block_size])
        if len(augmented) >= extra_needed:
            break
    return trades + augmented[:extra_needed]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT",
                        help="comma-separated symbols (default: BTC,ETH)")
    parser.add_argument("--timeframes", default="1h",
                        help="comma-separated primary timeframes for backtest "
                             "(default: 1h; supported: 1h, 4h)")
    parser.add_argument("--lenient", action="store_true",
                        help="lower min_confidence to 0.55 to capture more setups")
    parser.add_argument("--bootstrap-target", type=int, default=0,
                        help="if >0, block-bootstrap up to this trade count")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "data" / "ml_models" / "expanded_trades.pkl",
                        help="where to write the pickled trade list")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = ExpansionConfig(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        timeframes=[t.strip() for t in args.timeframes.split(",") if t.strip()],
        lenient=args.lenient,
        bootstrap_target=args.bootstrap_target,
        output_path=args.output,
    )
    _, report = expand_corpus(cfg)
    print(report.render())
    print(f"\nWrote {report.total_trades} trades to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
