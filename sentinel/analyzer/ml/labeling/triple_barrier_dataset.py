"""
Triple-Barrier Dataset Builder.

Bridges ``core.models.StrategyTrade`` (what the backtest / live trading
records) to the pure triple-barrier primitives in
``analyzer.ml.domain.triple_barrier``. Operators rarely have raw
pre-event price slices indexed the way the pure functions want them —
they have a list of trades and OHLCV history per symbol. This helper
does the bookkeeping.

The builder produces two training-ready outputs:

1. **Triple-barrier labels** — for each trade, which of the upper /
   lower / vertical barrier hit first, and the realised label ∈
   {+1, 0, −1}. Use as a *replacement* target for the classical
   binary ``is_win`` — the signed magnitude distinguishes a small
   win from a large one and a small loss from a catastrophic one.

2. **Meta-labels** — a primary signal fired (the strategy opened the
   trade); the meta-label is 1 when the barrier outcome matches the
   primary's direction, 0 otherwise. Use as the target of a
   meta-classifier that predicts "should I trust the primary here?".

Inputs required:

* ``trades``: an iterable of objects with at minimum ``symbol``,
  ``timestamp_open`` (ISO8601 or milliseconds), and ``pnl_pct`` (for
  the fallback-volatility-from-realised path).
* ``price_history_by_symbol``: dict of ``{symbol: np.ndarray}`` —
  close prices per symbol, ordered oldest→newest.
* ``timestamp_by_symbol`` (optional): dict of ``{symbol: np.ndarray[int64]}``
  — epoch milliseconds aligned with the price arrays. Used to look up
  ``entry_idx`` for each trade. When omitted, the builder falls back to
  ``trade_idx_by_symbol`` if provided.

Volatility estimate for each entry defaults to the rolling std of the
close series over a configurable window; operators with ATR at hand
can pass their own ``volatility_by_symbol`` array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from analyzer.ml.domain.triple_barrier import (
    BarrierResult,
    build_meta_labels,
    triple_barrier_label,
)


@dataclass(slots=True)
class LabeledTrade:
    """One row of the triple-barrier training dataset."""
    trade: Any              # original ``StrategyTrade`` (or any duck-typed object)
    symbol: str
    entry_idx: int          # index into price_history_by_symbol[symbol]
    barrier: BarrierResult  # outcome (label + exit metadata)
    primary_direction: int  # +1 long / -1 short; 0 if unknown
    meta_label: int         # 1 if primary's sign matched barrier label, else 0


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling std aligned right. First ``window-1`` values are NaN.

    Not vectorised for speed — this runs once per training build,
    per symbol; clarity trumps micro-optimisation here.
    """
    n = x.size
    out = np.full(n, np.nan)
    if n < window or window < 2:
        return out
    for t in range(window - 1, n):
        out[t] = float(np.std(x[t - window + 1 : t + 1], ddof=1))
    return out


def _parse_timestamp(ts: Any) -> Optional[int]:
    """Best-effort conversion of a timestamp field to epoch milliseconds.

    Accepts: ``int`` / ``float`` (assumed ms), ``str`` ISO-8601 via
    ``datetime.fromisoformat``. Returns ``None`` on failure so the
    caller can fall back to another matching key.
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str):
        try:
            from datetime import datetime, timezone
            # Python's fromisoformat accepts "2026-04-19T12:34:56+00:00"
            # and similar; normalise trailing "Z" to "+00:00".
            norm = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(norm)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None
    return None


def build_triple_barrier_dataset(
    trades: Iterable[Any],
    *,
    price_history_by_symbol: dict[str, np.ndarray],
    timestamp_by_symbol: Optional[dict[str, np.ndarray]] = None,
    volatility_by_symbol: Optional[dict[str, np.ndarray]] = None,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    horizon: int = 100,
    vol_window: int = 20,
    primary_direction_attr: str = "direction",
) -> list[LabeledTrade]:
    """Turn a list of trades into triple-barrier labels + meta-labels.

    Args:
        trades: Iterable of objects; each one must expose ``symbol`` and
            either ``timestamp_open`` (ms or ISO string) or be placed by
            ``timestamp_by_symbol`` lookup.
        price_history_by_symbol: Close-price arrays per symbol.
        timestamp_by_symbol: Epoch-ms arrays aligned with prices. Used
            for entry_idx lookup via ``np.searchsorted``. When omitted
            the builder treats the trade sequence as pre-indexed (each
            trade must have an ``entry_idx`` attribute).
        volatility_by_symbol: Optional pre-computed σ array per symbol
            (e.g. ATR). When omitted, a rolling-std of ``vol_window``
            bars is used.
        pt_mult / sl_mult / horizon: Forwarded to ``triple_barrier_label``.
        vol_window: Window size for fallback rolling-std when no
            ``volatility_by_symbol`` is supplied.
        primary_direction_attr: Attribute name to read the primary
            signal's direction from (+1 / -1 / 0). Defaults to
            ``"direction"``; operators with an older schema can override.

    Returns:
        List of ``LabeledTrade``, one per input trade. Trades that can't
        be located in the price history (missing symbol or out-of-range
        timestamp) are silently dropped — log upstream.
    """
    # Pre-compute fallback volatility once per symbol
    if volatility_by_symbol is None:
        volatility_by_symbol = {
            sym: _rolling_std(np.asarray(prices, dtype=np.float64), vol_window)
            for sym, prices in price_history_by_symbol.items()
        }

    labeled: list[LabeledTrade] = []
    for trade in trades:
        symbol = getattr(trade, "symbol", None)
        if symbol is None or symbol not in price_history_by_symbol:
            continue
        prices = np.asarray(price_history_by_symbol[symbol], dtype=np.float64)

        # Determine entry_idx
        entry_idx: Optional[int] = None
        if timestamp_by_symbol is not None and symbol in timestamp_by_symbol:
            ts_ms = _parse_timestamp(getattr(trade, "timestamp_open", None))
            if ts_ms is not None:
                ts_arr = np.asarray(timestamp_by_symbol[symbol], dtype=np.int64)
                # searchsorted gives the first index >= ts_ms; back off one
                # so entry_idx points to the bar that contains (or
                # precedes) the signal — avoids future-leak.
                idx = int(np.searchsorted(ts_arr, ts_ms, side="right")) - 1
                if 0 <= idx < prices.size:
                    entry_idx = idx
        if entry_idx is None:
            # Fallback to an ``entry_idx`` attribute on the trade itself.
            idx_attr = getattr(trade, "entry_idx", None)
            if isinstance(idx_attr, (int, np.integer)):
                entry_idx = int(idx_attr)
        if entry_idx is None or not (0 <= entry_idx < prices.size):
            continue

        vol_arr = np.asarray(
            volatility_by_symbol.get(symbol, np.array([])), dtype=np.float64,
        )
        sigma = float(vol_arr[entry_idx]) if entry_idx < vol_arr.size else 0.0
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue

        raw_dir = getattr(trade, primary_direction_attr, 1)
        # Normalise to +1 / 0 / -1 — strategies may store as
        # "LONG" / "SHORT" strings or as ints. Do the string branch FIRST
        # so int() doesn't try to parse "LONG".
        if isinstance(raw_dir, str):
            upper = raw_dir.upper()
            if upper == "LONG":
                direction = 1
            elif upper == "SHORT":
                direction = -1
            else:
                direction = 0
        else:
            try:
                direction = int(raw_dir) if raw_dir is not None else 1
            except (TypeError, ValueError):
                direction = 1

        barrier = triple_barrier_label(
            prices, entry_idx,
            pt_abs=sigma * pt_mult, sl_abs=sigma * sl_mult,
            horizon=horizon, direction=direction if direction != 0 else 1,
        )
        meta = 1 if (direction != 0 and np.sign(direction) == np.sign(barrier.label)) else 0
        labeled.append(LabeledTrade(
            trade=trade, symbol=symbol, entry_idx=entry_idx,
            barrier=barrier, primary_direction=direction, meta_label=meta,
        ))

    return labeled


def meta_label_arrays(labeled: Sequence[LabeledTrade]) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(primary_decisions, meta_labels)`` as int64 NumPy arrays.

    Convenience for feeding an existing classifier; ``meta_labels`` is
    the training target, ``primary_decisions`` the gating mask (train
    on rows where ``primary_decisions != 0``).
    """
    n = len(labeled)
    primary = np.zeros(n, dtype=np.int64)
    meta = np.zeros(n, dtype=np.int64)
    for i, lt in enumerate(labeled):
        primary[i] = int(lt.primary_direction)
        meta[i] = int(lt.meta_label)
    # Consistency check with the pure helper — should match exactly.
    _expected = build_meta_labels(
        primary, np.array([lt.barrier.label for lt in labeled], dtype=np.int64),
    )
    assert np.array_equal(meta, _expected)
    return primary, meta
