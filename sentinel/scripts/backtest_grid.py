"""Walk-forward backtest dlya grid_trading na ETH 1h.

Loads candles from local DB, runs strategy through BacktestEngine and
WalkForwardAnalyser, prints PSR, degradation ratio and stability.

Applies skill `backtesting-frameworks`: walk-forward (rolling), realistic
execution (spread + slippage), PSR >= 0.95 = real edge.

Usage:
    cd sentinel && python scripts/backtest_grid.py [--symbol ETHUSDT] [--days 180]
"""

from __future__ import annotations

import argparse
import io
import sqlite3
import sys
import time
from pathlib import Path

# Force stdout to UTF-8 on Windows so any unicode chars render.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Silence loguru/std-logging noise so the report is readable.
import logging
logging.getLogger().setLevel(logging.WARNING)
try:
    from loguru import logger as _lg
    _lg.remove()
    _lg.add(sys.stderr, level="WARNING")
except Exception:
    pass

from core.models import Candle
from backtest.engine import BacktestConfig, BacktestEngine
from backtest.walk_forward import WalkForwardAnalyser
from strategy.grid_trading import GridTrading


def load_candles(db_path: str, symbol: str, interval: str, since_ms: int) -> list[Candle]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM candles WHERE symbol=? AND interval=? AND timestamp>=? "
        "ORDER BY timestamp ASC",
        (symbol, interval, since_ms),
    ).fetchall()
    con.close()
    return [
        Candle(symbol=symbol, interval=interval,
               timestamp=ts, open=o, high=h, low=l, close=c, volume=v)
        for ts, o, h, l, c, v in rows
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--db", default="data/sentinel.db")
    p.add_argument("--is-bars", type=int, default=2160,  # ~90 days of 1h
                   help="In-sample window (bars)")
    p.add_argument("--oos-bars", type=int, default=720,  # ~30 days
                   help="Out-of-sample window (bars)")
    p.add_argument("--guards", action="store_true",
                   help="Apply production risk guards (multi-TF, regime, drawdown)")
    args = p.parse_args()

    since_ms = int((time.time() - args.days * 86400) * 1000)

    print(f"Loading {args.symbol} candles (last {args.days}d)…")
    c1h = load_candles(args.db, args.symbol, "1h", since_ms)
    c4h = load_candles(args.db, args.symbol, "4h", since_ms)
    c1d = load_candles(args.db, args.symbol, "1d", since_ms)
    print(f"  1h: {len(c1h)} candles, 4h: {len(c4h)}, 1d: {len(c1d)}")

    if len(c1h) < args.is_bars + args.oos_bars:
        print(f"❌ Need ≥{args.is_bars + args.oos_bars} 1h bars, have {len(c1h)}. "
              f"Run with --days {(args.is_bars + args.oos_bars) // 24 + 30}")
        return 1

    cfg = BacktestConfig(
        initial_balance=500.0,
        commission_pct=0.1,
        realistic_execution=True,
        apply_risk_guards=args.guards,
    )
    engine = BacktestEngine(cfg)

    # ── 1. Single-shot бэктест на всём окне ──
    print(f"\n=== Single-shot backtest ({len(c1h)} bars) ===")
    strat = GridTrading()
    res = engine.run(strat, c1h, c4h, args.symbol, candles_1d=c1d)
    print(f"  trades:           {res.total_trades}  (W/L: {res.wins}/{res.losses})")
    print(f"  win rate:         {res.win_rate:.1%}")
    print(f"  total PnL:        {res.total_pnl_pct:+.2f}%  (${res.total_pnl:+.2f})")
    print(f"  Sharpe:           {res.sharpe_ratio:.2f}")
    print(f"  PSR:              {res.psr:.2%}  (≥95% = stat. significant edge)")
    print(f"  profit factor:    {res.profit_factor:.2f}")
    print(f"  max drawdown:     {res.max_drawdown_pct:.1f}%")
    print(f"  avg win/loss:     +{res.avg_win:.2f}% / {res.avg_loss:.2f}%")
    print(f"  guards rejected:  {res.guards_rejected}")
    if res.returns_normal is not None:
        print(f"  returns normal:   {res.returns_normal} (Shapiro p={res.returns_shapiro_p:.3f})")

    # ── 2. Walk-forward ──
    print(f"\n=== Walk-forward (rolling, IS={args.is_bars}h, OOS={args.oos_bars}h) ===")
    wfa = WalkForwardAnalyser(engine)
    report = wfa.run(
        candles_1h=c1h, candles_4h=c4h, candles_1d=c1d,
        symbol=args.symbol,
        is_window_bars=args.is_bars,
        oos_window_bars=args.oos_bars,
        mode="rolling",
        strategy=GridTrading(),
    )
    print(WalkForwardAnalyser.format_report(report))

    # ── Verdict ──
    print("\n=== Verdict ===")
    edge_signals = []
    if res.psr >= 0.95:
        edge_signals.append("✓ PSR ≥ 0.95 — статистически значимый edge")
    else:
        edge_signals.append(f"✗ PSR={res.psr:.2%} < 0.95 — edge не доказан")
    if report.degradation_ratio >= 0.7:
        edge_signals.append(f"✓ degradation {report.degradation_ratio:.2f} — out-of-sample держится")
    else:
        edge_signals.append(f"✗ degradation {report.degradation_ratio:.2f} — переподогнано")
    if report.folds and report.profitable_oos_folds / len(report.folds) >= 0.6:
        edge_signals.append(f"✓ {report.profitable_oos_folds}/{len(report.folds)} прибыльных OOS-фолдов")
    elif report.folds:
        edge_signals.append(f"✗ только {report.profitable_oos_folds}/{len(report.folds)} прибыльных OOS")
    if res.profit_factor < 1.2:
        edge_signals.append(f"✗ PF={res.profit_factor:.2f} < 1.2 — после костов в минусе")
    for s in edge_signals:
        print(f"  {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
