"""Trace how many bars survive each filter in grid_trading on real ETH 1h data."""

from __future__ import annotations

import io
import sqlite3
import sys
import time
from collections import Counter
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
except Exception:
    pass

from core.models import Candle
from features.feature_builder import FeatureBuilder


def load(db, sym, iv, since):
    rows = sqlite3.connect(db).cursor().execute(
        "SELECT timestamp, open, high, low, close, volume FROM candles "
        "WHERE symbol=? AND interval=? AND timestamp>=? ORDER BY timestamp",
        (sym, iv, since),
    ).fetchall()
    return [Candle(symbol=sym, interval=iv, timestamp=t, open=o, high=h, low=l, close=c, volume=v)
            for t, o, h, l, c, v in rows]


def main():
    sym = "ETHUSDT"
    since = int((time.time() - 365 * 86400) * 1000)
    c1h = load("data/sentinel.db", sym, "1h", since)
    c4h = load("data/sentinel.db", sym, "4h", since)
    c1d = load("data/sentinel.db", sym, "1d", since)
    print(f"Loaded {len(c1h)} 1h, {len(c4h)} 4h, {len(c1d)} 1d candles")

    fb = FeatureBuilder()
    cnt = Counter()
    sample_features = []
    min_h = 55

    for i in range(min_h, len(c1h)):
        cnt["total_bars"] += 1
        cur = c1h[i]
        w1h = c1h[max(0, i - min_h):i]
        w4h = [c for c in c4h if c.timestamp <= cur.timestamp][-min_h:]
        w1d = [c for c in c1d if c.timestamp <= cur.timestamp][-min_h:] if c1d else None
        f = fb.build(sym, w1h, w4h, w1d)
        if f is None:
            cnt["features_none"] += 1
            continue
        cnt["features_built"] += 1

        # Replicate grid filters
        if f.bb_lower <= 0 or f.bb_upper <= 0:
            cnt["bb_invalid"] += 1
            continue
        # Build grid
        if f.bb_upper <= f.bb_lower:
            cnt["bb_inverted"] += 1
            continue
        levels_low = f.bb_lower
        levels_high = f.bb_upper
        price = f.close
        if price < levels_low * 0.98 or price > levels_high * 1.02:
            cnt["outside_grid_2pct"] += 1
            continue
        # Need price <= some level (i.e., price <= bb_upper basically)
        if price > levels_high:
            cnt["price_above_grid_top"] += 1
            continue
        # bb_lower breach guard
        if price < f.bb_lower:
            cnt["below_bb_lower"] += 1
            continue
        # bearish ema stack
        if (f.ema_9 > 0 and f.ema_21 > 0 and f.ema_50 > 0
                and f.ema_9 < f.ema_21 < f.ema_50 and price < f.ema_50):
            cnt["bearish_ema_stack"] += 1
            continue
        # falling knife
        if f.rsi_14 > 0 and f.rsi_14 < 30 and f.dmi_spread < -5:
            cnt["falling_knife"] += 1
            continue
        # volume filter
        if f.volume_ratio < 0.8:
            cnt["low_volume"] += 1
            continue
        # spread filter
        sp = f.spread / price * 100 if price > 0 and f.spread > 0 else 0
        if sp > 0.15:
            cnt["wide_spread"] += 1
            continue

        cnt["passed_all_filters"] += 1
        if len(sample_features) < 5:
            sample_features.append((cur.timestamp, price, f.bb_lower, f.bb_upper,
                                    f.rsi_14, f.dmi_spread, f.volume_ratio, f.spread,
                                    f.market_regime))

    print("\nFilter funnel:")
    total = cnt["total_bars"]
    for k, v in cnt.most_common():
        pct = v / total * 100 if total else 0
        print(f"  {k:30s} {v:>6d}  ({pct:5.1f}%)")

    if sample_features:
        print("\nSample bars passing all filters:")
        for ts, p, bl, bu, rsi, dmi, vr, sp, reg in sample_features:
            print(f"  ts={ts}  price={p:.2f}  bb=[{bl:.0f}..{bu:.0f}]  rsi={rsi:.1f}  "
                  f"dmi={dmi:+.1f}  vol={vr:.2f}  spread={sp:.4f}  regime={reg}")


if __name__ == "__main__":
    main()
