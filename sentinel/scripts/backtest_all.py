"""Walk-forward backtest for ALL strategies on ETH 1h.

Applies skill `backtesting-frameworks`: walk-forward (rolling), realistic
execution, PSR / degradation / profitable-folds decision criteria.

Usage:
    cd sentinel && python scripts/backtest_all.py [--symbol ETHUSDT] [--days 365]
"""

from __future__ import annotations

import argparse
import io
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
from backtest.walk_forward import WalkForwardAnalyser

from strategy.grid_trading import GridTrading
from strategy.mean_reversion import MeanReversion
from strategy.bollinger_breakout import BollingerBreakout
from strategy.macd_divergence import MACDDivergence
from strategy.dca_bot import DCABot
from strategy.ema_crossover_rsi import EMACrossoverRSI


STRATEGIES = {
    "ema_crossover_rsi": EMACrossoverRSI,
    "grid_trading":      GridTrading,
    "mean_reversion":    MeanReversion,
    "bollinger_breakout": BollingerBreakout,
    "macd_divergence":   MACDDivergence,
    "dca_bot":           DCABot,
}


def load(db, sym, iv, since):
    rows = sqlite3.connect(db).cursor().execute(
        "SELECT timestamp, open, high, low, close, volume FROM candles "
        "WHERE symbol=? AND interval=? AND timestamp>=? ORDER BY timestamp",
        (sym, iv, since),
    ).fetchall()
    return [Candle(symbol=sym, interval=iv, timestamp=t, open=o, high=h, low=l, close=c, volume=v)
            for t, o, h, l, c, v in rows]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--db", default="data/sentinel.db")
    p.add_argument("--is-bars", type=int, default=2160)
    p.add_argument("--oos-bars", type=int, default=720)
    args = p.parse_args()

    since = int((time.time() - args.days * 86400) * 1000)
    print(f"Loading {args.symbol} (last {args.days}d)…")
    c1h = load(args.db, args.symbol, "1h", since)
    c4h = load(args.db, args.symbol, "4h", since)
    c1d = load(args.db, args.symbol, "1d", since)
    print(f"  1h: {len(c1h)}  4h: {len(c4h)}  1d: {len(c1d)}\n")

    cfg = BacktestConfig(initial_balance=500.0, commission_pct=0.1,
                         realistic_execution=True, apply_risk_guards=False)
    engine = BacktestEngine(cfg)

    rows = []
    print(f"{'strategy':22s}  {'trades':>6}  {'WR%':>5}  {'PnL%':>7}  {'PF':>5}  "
          f"{'Sharpe':>7}  {'PSR%':>6}  {'MaxDD%':>7}  {'AvgW/L':>10}")
    print("─" * 102)

    single_results = {}
    for name, cls in STRATEGIES.items():
        strat = cls()
        try:
            res = engine.run(strat, c1h, c4h, args.symbol, candles_1d=c1d)
        except Exception as e:
            print(f"{name:22s}  ERROR: {e}")
            continue
        single_results[name] = res
        wr = (res.wins / res.total_trades * 100) if res.total_trades else 0
        avg_wl = f"+{res.avg_win:.1f}/{res.avg_loss:.1f}"
        print(f"{name:22s}  {res.total_trades:>6}  {wr:>5.1f}  "
              f"{res.total_pnl_pct:>+7.2f}  {res.profit_factor:>5.2f}  "
              f"{res.sharpe_ratio:>+7.2f}  {res.psr*100:>6.1f}  "
              f"{res.max_drawdown_pct:>7.1f}  {avg_wl:>10s}")

    # Walk-forward for top-3 by Sharpe
    print("\n=== Walk-forward (rolling, IS=90d, OOS=30d) ===\n")
    print(f"{'strategy':22s}  {'folds':>5}  {'IS_Sh':>6}  {'OOS_Sh':>7}  "
          f"{'OOS PSR':>7}  {'profit folds':>12}  {'stability':>9}")
    print("─" * 90)
    wfa = WalkForwardAnalyser(engine)
    for name, cls in STRATEGIES.items():
        if name not in single_results:
            continue
        try:
            r = wfa.run(candles_1h=c1h, candles_4h=c4h, candles_1d=c1d,
                        symbol=args.symbol, is_window_bars=args.is_bars,
                        oos_window_bars=args.oos_bars, mode="rolling",
                        strategy=cls())
        except Exception as e:
            print(f"{name:22s}  WF error: {e}")
            continue
        if not r.folds:
            print(f"{name:22s}  no folds")
            continue
        print(f"{name:22s}  {len(r.folds):>5}  {r.avg_is_sharpe:>+6.2f}  "
              f"{r.avg_oos_sharpe:>+7.2f}  {r.aggregate_oos_psr:>7.2%}  "
              f"{r.profitable_oos_folds}/{len(r.folds):<10}  "
              f"{r.parameter_stability_score:>9.2f}")

    # Verdict per strategy
    print("\n=== Verdict (per skill `backtesting-frameworks`) ===\n")
    for name, res in single_results.items():
        if res.total_trades == 0:
            print(f"  {name:22s}  N/A — no trades fired")
            continue
        flags = []
        if res.psr >= 0.95: flags.append("PSR✓")
        else: flags.append(f"PSR={res.psr:.0%}")
        if res.profit_factor >= 1.3: flags.append("PF✓")
        else: flags.append(f"PF={res.profit_factor:.2f}")
        if res.sharpe_ratio >= 1.0: flags.append("Sh✓")
        else: flags.append(f"Sh={res.sharpe_ratio:.2f}")
        verdict = "EDGE" if (res.psr >= 0.95 and res.profit_factor >= 1.3 and res.sharpe_ratio >= 1.0) else "no edge"
        print(f"  {name:22s}  {verdict:8s}  [{', '.join(flags)}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
