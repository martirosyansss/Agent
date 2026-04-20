"""Tune all strategies on train slice, validate honestly on OOS slice.

Applies skill `backtesting-frameworks`:
  - 70/30 train/test split (no peeking)
  - Optuna TPE sampling on TRAIN ONLY
  - Report IS metrics + OOS metrics with PSR
  - Verdict: edge if OOS PSR >= 0.95 AND OOS PF >= 1.3 AND OOS Sharpe > 0
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import logging
logging.getLogger().setLevel(logging.ERROR)
try:
    from loguru import logger as _lg
    _lg.remove()
    _lg.add(sys.stderr, level="ERROR")
except Exception:
    pass

from core.models import Candle
from backtest.engine import BacktestConfig, BacktestEngine
from analyzer.strategy_tuner import StrategyTuner, TunerConfig


def load(db, sym, iv, since):
    rows = sqlite3.connect(db).cursor().execute(
        "SELECT timestamp, open, high, low, close, volume FROM candles "
        "WHERE symbol=? AND interval=? AND timestamp>=? ORDER BY timestamp",
        (sym, iv, since),
    ).fetchall()
    return [Candle(symbol=sym, interval=iv, timestamp=t, open=o, high=h, low=l, close=c, volume=v)
            for t, o, h, l, c, v in rows]


def split_70_30(candles, split_ratio=0.7):
    """Time-ordered split — no shuffling."""
    n = len(candles)
    cut = int(n * split_ratio)
    return candles[:cut], candles[cut:]


def build_strategy_with_params(name: str, params: dict):
    """Reconstruct strategy with tuned params."""
    if name == "grid_trading":
        from strategy.grid_trading import GridTrading, GridConfig
        return GridTrading(config=GridConfig(**params))
    if name == "dca_bot":
        from strategy.dca_bot import DCABot, DCAConfig
        return DCABot(config=DCAConfig(**params))
    if name == "bollinger_breakout":
        from strategy.bollinger_breakout import BollingerBreakout, BBBreakoutConfig
        return BollingerBreakout(config=BBBreakoutConfig(**params))
    if name == "mean_reversion":
        from strategy.mean_reversion import MeanReversion, MeanRevConfig
        return MeanReversion(config=MeanRevConfig(**params))
    if name == "macd_divergence":
        from strategy.macd_divergence import MACDDivergence, MACDDivConfig
        return MACDDivergence(config=MACDDivConfig(**params))
    if name == "ema_crossover_rsi":
        from strategy.ema_crossover_rsi import EMACrossoverRSI, EMAConfig
        return EMACrossoverRSI(config=EMAConfig(**params))
    raise ValueError(name)


def run_oos(strat, c1h, c4h, c1d, symbol):
    cfg = BacktestConfig(initial_balance=500.0, commission_pct=0.1,
                         realistic_execution=True, apply_risk_guards=False)
    return BacktestEngine(cfg).run(strat, c1h, c4h, symbol, c1d)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--db", default="data/sentinel.db")
    p.add_argument("--trials", type=int, default=15,
                   help="Optuna trials per strategy (default 15 = ~3 min/strategy)")
    p.add_argument("--strategies", nargs="+",
                   default=["grid_trading", "dca_bot", "bollinger_breakout"],
                   help="Subset to tune (default: 3 active ones)")
    args = p.parse_args()

    since = int((time.time() - args.days * 86400) * 1000)
    print(f"Loading {args.symbol} (last {args.days}d)…")
    c1h = load(args.db, args.symbol, "1h", since)
    c4h = load(args.db, args.symbol, "4h", since)
    c1d = load(args.db, args.symbol, "1d", since)
    print(f"  1h: {len(c1h)}  4h: {len(c4h)}  1d: {len(c1d)}")

    train_1h, test_1h = split_70_30(c1h)
    train_4h, test_4h = split_70_30(c4h)
    train_1d, test_1d = split_70_30(c1d)
    print(f"  TRAIN: {len(train_1h)} 1h bars  ({train_1h[0].timestamp} → {train_1h[-1].timestamp})")
    print(f"  TEST:  {len(test_1h)} 1h bars  ({test_1h[0].timestamp} → {test_1h[-1].timestamp})\n")

    tuner = StrategyTuner(TunerConfig(
        n_trials=args.trials,
        min_trades=15,
        max_drawdown_pct=30,
        max_candles_tune=len(train_1h),
    ))

    tune_methods = {
        "grid_trading": tuner.tune_grid,
        "dca_bot": tuner.tune_dca,
        "bollinger_breakout": tuner.tune_bollinger,
        "mean_reversion": tuner.tune_mean_reversion,
        "macd_divergence": tuner.tune_macd_divergence,
        "ema_crossover_rsi": tuner.tune_ema_crossover,
    }

    results = []
    for name in args.strategies:
        if name not in tune_methods:
            print(f"!! unknown strategy: {name}")
            continue
        print(f"\n{'='*70}\nTuning {name} (Optuna TPE, {args.trials} trials, train slice only)\n{'='*70}")
        t0 = time.time()
        try:
            tr = tune_methods[name](train_1h, train_4h, train_1d, args.symbol)
        except Exception as e:
            print(f"!! tune failed: {e}")
            continue
        elapsed = time.time() - t0
        print(f"\n  best score: {tr.best_score:.3f}  ({elapsed:.0f}s)")
        print(f"  best params: {json.dumps(tr.best_params, indent=2)}")

        # Honest OOS validation
        try:
            strat = build_strategy_with_params(name, tr.best_params)
            oos = run_oos(strat, test_1h, test_4h, test_1d, args.symbol)
        except Exception as e:
            print(f"  !! OOS run failed: {e}")
            continue

        results.append((name, tr, oos))
        wr = (oos.wins / oos.total_trades * 100) if oos.total_trades else 0
        print(f"\n  OOS METRICS (last 30% — never seen during tuning):")
        print(f"    trades:    {oos.total_trades}  (W/L: {oos.wins}/{oos.losses})  WR={wr:.1f}%")
        print(f"    PnL:       {oos.total_pnl_pct:+.2f}%  (${oos.total_pnl:+.2f})")
        print(f"    PF:        {oos.profit_factor:.2f}")
        print(f"    Sharpe:    {oos.sharpe_ratio:+.2f}")
        print(f"    PSR:       {oos.psr:.1%}")
        print(f"    Max DD:    {oos.max_drawdown_pct:.1f}%")

    # Summary table
    print(f"\n\n{'='*90}\nFINAL SUMMARY (OOS — out-of-sample, walk-forward honest)\n{'='*90}")
    print(f"{'strategy':22s}  {'trades':>6}  {'WR%':>5}  {'PnL%':>7}  {'PF':>5}  {'Sh':>6}  {'PSR%':>5}  verdict")
    print("─" * 90)
    for name, tr, oos in results:
        wr = (oos.wins / oos.total_trades * 100) if oos.total_trades else 0
        edge = (oos.psr >= 0.95 and oos.profit_factor >= 1.3 and oos.sharpe_ratio > 0)
        verdict = "EDGE ✓" if edge else "no edge"
        print(f"{name:22s}  {oos.total_trades:>6}  {wr:>5.1f}  {oos.total_pnl_pct:>+7.2f}  "
              f"{oos.profit_factor:>5.2f}  {oos.sharpe_ratio:>+6.2f}  {oos.psr*100:>5.1f}  {verdict}")

    # Save tuned params for reference
    out = Path("data/tuned_params.json")
    out.parent.mkdir(exist_ok=True)
    payload = {
        name: {
            "params": tr.best_params,
            "is_score": tr.best_score,
            "oos_psr": oos.psr,
            "oos_pf": oos.profit_factor,
            "oos_sharpe": oos.sharpe_ratio,
            "oos_pnl_pct": oos.total_pnl_pct,
            "oos_trades": oos.total_trades,
            "tuned_at": int(time.time()),
        }
        for name, tr, oos in results
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
