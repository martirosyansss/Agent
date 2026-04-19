"""Tests for the triple-barrier dataset builder (ML10).

Contracts:

* **Timestamp-based lookup works** — when ``timestamp_by_symbol`` is
  provided, ``np.searchsorted`` maps trade open time → correct
  entry_idx bar.
* **Fallback entry_idx attribute** — if the trade carries a pre-computed
  ``entry_idx``, the builder uses it when timestamps aren't available.
* **Direction normalisation** — "LONG" / "SHORT" strings get mapped to
  +1 / -1 before labelling.
* **Meta-label consistency** — ``meta_label_arrays`` output matches
  ``build_meta_labels`` from the pure module.
* **Bad trades are dropped, not crashed** — missing symbol, out-of-range
  timestamp, zero volatility → trade is skipped.
* **Dataset size matches fed-in trades minus drops** — operator can
  audit how many were retained.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml.domain.triple_barrier import build_meta_labels
from analyzer.ml.labeling.triple_barrier_dataset import (
    LabeledTrade,
    build_triple_barrier_dataset,
    meta_label_arrays,
)


@dataclass
class FakeTrade:
    """Duck-typed stand-in for StrategyTrade."""
    symbol: str
    timestamp_open: Any = 0
    direction: Any = 1
    entry_idx: Optional[int] = None


def _flat_prices_with_rally(n: int = 200, rally_at: int = 100) -> np.ndarray:
    """A noisy price series that rallies from ``rally_at``.

    The pre-rally section has small Gaussian noise so the rolling-std
    volatility estimate used by the dataset builder is non-zero (the
    builder drops entries with zero σ, which is what we want in
    production but breaks tests that use strictly constant prices).
    """
    rng = np.random.default_rng(123)
    p = 100.0 + rng.normal(0, 0.3, size=n)
    for t in range(rally_at, n):
        p[t] = p[t] + (t - rally_at) * 0.5
    return p


class TestTimestampLookup:
    def test_ms_timestamp_maps_to_entry_idx(self):
        prices = _flat_prices_with_rally(200, rally_at=100)
        ts = np.arange(200) * 3_600_000  # hourly epoch-ms
        trades = [FakeTrade(symbol="BTC", timestamp_open=int(ts[80]))]
        out = build_triple_barrier_dataset(
            trades,
            price_history_by_symbol={"BTC": prices},
            timestamp_by_symbol={"BTC": ts},
            pt_mult=2.0, sl_mult=1.0, horizon=50, vol_window=10,
        )
        assert len(out) == 1
        assert out[0].entry_idx == 80

    def test_iso8601_timestamp(self):
        from datetime import datetime, timezone

        prices = _flat_prices_with_rally(200, rally_at=100)
        base_ms = 1_700_000_000_000
        ts = (np.arange(200) * 3_600_000) + base_ms
        # Format the exact ts[80] as an ISO-8601 string so the
        # round-trip through the parser is bit-exact.
        target_ms = int(ts[80])
        iso = datetime.fromtimestamp(target_ms / 1000, tz=timezone.utc).isoformat()
        # Replace "+00:00" → "Z" to exercise the normalisation branch too.
        iso = iso.replace("+00:00", "Z")
        trade = FakeTrade(symbol="BTC", timestamp_open=iso)
        out = build_triple_barrier_dataset(
            [trade],
            price_history_by_symbol={"BTC": prices},
            timestamp_by_symbol={"BTC": ts},
            horizon=20, vol_window=10,
        )
        assert len(out) == 1
        assert out[0].entry_idx == 80


class TestEntryIdxFallback:
    def test_uses_attribute_when_no_timestamp_mapping(self):
        prices = _flat_prices_with_rally(200, rally_at=100)
        trades = [FakeTrade(symbol="BTC", entry_idx=80, timestamp_open=None)]
        out = build_triple_barrier_dataset(
            trades,
            price_history_by_symbol={"BTC": prices},
            timestamp_by_symbol=None,
            pt_mult=1.5, sl_mult=1.0, horizon=50, vol_window=10,
        )
        assert len(out) == 1
        assert out[0].entry_idx == 80


class TestDirectionNormalisation:
    def test_long_short_strings_mapped(self):
        prices = _flat_prices_with_rally(200, rally_at=100)
        trades = [
            FakeTrade(symbol="BTC", entry_idx=80, direction="LONG"),
            FakeTrade(symbol="BTC", entry_idx=80, direction="SHORT"),
        ]
        out = build_triple_barrier_dataset(
            trades,
            price_history_by_symbol={"BTC": prices},
            pt_mult=1.5, sl_mult=1.0, horizon=50, vol_window=10,
        )
        assert out[0].primary_direction == 1
        assert out[1].primary_direction == -1


class TestMetaLabelConsistency:
    def test_meta_labels_match_pure_helper(self):
        prices = _flat_prices_with_rally(200, rally_at=100)
        trades = [
            FakeTrade(symbol="BTC", entry_idx=80, direction=1),
            FakeTrade(symbol="BTC", entry_idx=80, direction=-1),
            FakeTrade(symbol="BTC", entry_idx=80, direction=1),
        ]
        out = build_triple_barrier_dataset(
            trades,
            price_history_by_symbol={"BTC": prices},
            pt_mult=1.5, sl_mult=1.0, horizon=50, vol_window=10,
        )
        primary, meta = meta_label_arrays(out)
        # Pure-module equivalent
        realised = np.array([lt.barrier.label for lt in out], dtype=np.int64)
        expected = build_meta_labels(primary, realised)
        assert np.array_equal(meta, expected)


class TestDropping:
    def test_missing_symbol_dropped(self):
        prices = _flat_prices_with_rally(200, rally_at=100)
        trades = [
            FakeTrade(symbol="UNKNOWN", entry_idx=80),
            FakeTrade(symbol="BTC", entry_idx=80),
        ]
        out = build_triple_barrier_dataset(
            trades,
            price_history_by_symbol={"BTC": prices},
            horizon=20, vol_window=5,
        )
        assert len(out) == 1
        assert out[0].symbol == "BTC"

    def test_zero_vol_dropped(self):
        # Constant prices → rolling std = 0 → entry rejected.
        prices = np.full(200, 100.0)
        trades = [FakeTrade(symbol="BTC", entry_idx=80)]
        out = build_triple_barrier_dataset(
            trades,
            price_history_by_symbol={"BTC": prices},
            horizon=20, vol_window=5,
        )
        assert out == []

    def test_entry_outside_price_range_dropped(self):
        prices = _flat_prices_with_rally(50, rally_at=25)
        trades = [FakeTrade(symbol="BTC", entry_idx=500)]
        out = build_triple_barrier_dataset(
            trades,
            price_history_by_symbol={"BTC": prices},
            horizon=5, vol_window=5,
        )
        assert out == []
