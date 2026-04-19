"""
Triple-Barrier Labels & Meta-Labeling (López de Prado, Ch. 3).

Binary ``is_win = pnl > 0`` labels throw away the information that
actually matters for a trader: a 0.1% gain and a 5% gain both get +1,
while a −0.1% loss and a catastrophic −5% both get 0. A model trained
on this label is optimising "how often is the sign right?", not
"how much money do we make?".

The **triple-barrier method** labels each event by which of three
barriers it hits first:

* **Upper (profit-take)** — return ≥ ``+pt_mult × σ`` ⇒ label ``+1``
* **Lower (stop-loss)**   — return ≤ ``−sl_mult × σ`` ⇒ label ``−1``
* **Vertical (time)**     — neither barrier hit within ``horizon`` bars
  ⇒ label = sign of the terminal return (can be 0)

``σ`` is typically a rolling-volatility estimate (ATR or return-std), so
each entry is labelled relative to *its* market conditions, not a
globally-hard-coded %. The output is signed (+1 / 0 / −1) rather than
binary, so classification losses can weight them asymmetrically.

**Meta-labeling** is the second half: train a secondary classifier on
*only the entries flagged by a primary signal*, with target ``|label| = 1``
(did the primary succeed?). The meta-classifier's job is to suppress
false positives of the primary — it doesn't generate trades, it filters
them. This is what Sentinel's ``rollout_mode="block"`` architecturally
is; the helpers here formalise the construction so a training-script
can produce meta-labels from the primary's decisions and the realised
outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(slots=True, frozen=True)
class BarrierResult:
    """Outcome of a single triple-barrier evaluation."""
    label: int                   # +1 up-barrier, -1 down-barrier, 0 time-out flat
    exit_idx: int                # index (relative to price array) of barrier hit
    exit_price: float            # price at the hit
    exit_reason: str             # "upper" | "lower" | "vertical"
    duration: int                # bars between entry and exit


def triple_barrier_label(
    prices: Sequence[float],
    entry_idx: int,
    *,
    pt_abs: float,
    sl_abs: float,
    horizon: int,
    direction: int = 1,
) -> BarrierResult:
    """Label a single event using absolute price barriers.

    The ``pt_abs`` / ``sl_abs`` arguments are the *absolute* profit-take
    and stop-loss distances (in price units — e.g. dollars for USDT
    pairs). They're meant to be pre-computed from a volatility estimate
    at the entry bar: ``pt_abs = pt_mult × σ_entry × price_entry``.

    Args:
        prices: Close-price series (or typical price) covering at least
            ``entry_idx + horizon``.
        entry_idx: Index of the entry bar within ``prices``.
        pt_abs: Upside barrier distance. For a long, the upper barrier
            fires when ``price ≥ entry + pt_abs``.
        sl_abs: Downside barrier distance. For a long, lower fires when
            ``price ≤ entry − sl_abs``.
        horizon: Maximum holding period in bars. Vertical barrier fires
            at ``entry_idx + horizon`` if neither of the others did.
        direction: ``+1`` for a long position (default), ``-1`` for a
            short — the barriers swap semantics.

    Returns:
        ``BarrierResult`` with label ∈ {+1, 0, −1} and metadata.
    """
    p = np.asarray(prices, dtype=np.float64)
    n = p.size
    if n == 0 or entry_idx < 0 or entry_idx >= n:
        return BarrierResult(0, entry_idx, 0.0, "invalid", 0)
    if pt_abs <= 0 or sl_abs <= 0 or horizon <= 0:
        return BarrierResult(0, entry_idx, float(p[entry_idx]), "invalid", 0)

    entry_price = float(p[entry_idx])
    end_idx = min(entry_idx + horizon, n - 1)

    if direction > 0:
        upper = entry_price + pt_abs
        lower = entry_price - sl_abs
        for t in range(entry_idx + 1, end_idx + 1):
            if p[t] >= upper:
                return BarrierResult(+1, t, float(p[t]), "upper", t - entry_idx)
            if p[t] <= lower:
                return BarrierResult(-1, t, float(p[t]), "lower", t - entry_idx)
    else:
        # Short — profit is price dropping below (entry - pt_abs),
        # stop-loss is price rising above (entry + sl_abs).
        upper = entry_price - pt_abs
        lower = entry_price + sl_abs
        for t in range(entry_idx + 1, end_idx + 1):
            if p[t] <= upper:
                return BarrierResult(+1, t, float(p[t]), "upper", t - entry_idx)
            if p[t] >= lower:
                return BarrierResult(-1, t, float(p[t]), "lower", t - entry_idx)

    # Vertical barrier — label by sign of terminal return along direction.
    exit_price = float(p[end_idx])
    terminal_return = (exit_price - entry_price) * float(direction)
    if terminal_return > 0:
        terminal_label = +1
    elif terminal_return < 0:
        terminal_label = -1
    else:
        terminal_label = 0
    return BarrierResult(
        terminal_label, end_idx, exit_price, "vertical",
        end_idx - entry_idx,
    )


def triple_barrier_labels_batch(
    prices: Sequence[float],
    entry_indices: Sequence[int],
    *,
    volatility: Sequence[float],
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    horizon: int = 100,
    directions: Optional[Sequence[int]] = None,
) -> list[BarrierResult]:
    """Apply triple-barrier labelling to a batch of entries.

    Args:
        prices: Close-price series.
        entry_indices: Indices into ``prices`` where each event starts.
        volatility: Per-index volatility estimate (σ in price units —
            e.g. ATR). Barriers are set as ``pt_mult × σ`` and
            ``sl_mult × σ`` above/below entry price.
        pt_mult: Upside barrier multiple of σ. Default 2 ≈ R:R 2:1.
        sl_mult: Downside barrier multiple of σ. Default 1.
        horizon: Maximum holding period in bars.
        directions: Optional per-event ``+1`` / ``-1``; defaults to all
            longs.

    Returns:
        List of ``BarrierResult`` in the same order as ``entry_indices``.
    """
    vol = np.asarray(volatility, dtype=np.float64)
    dirs = (
        np.ones(len(entry_indices), dtype=np.int64) if directions is None
        else np.asarray(directions, dtype=np.int64)
    )
    results: list[BarrierResult] = []
    for k, idx in enumerate(entry_indices):
        sigma = float(vol[idx]) if 0 <= idx < vol.size else 0.0
        if sigma <= 0.0:
            results.append(BarrierResult(0, int(idx), 0.0, "invalid", 0))
            continue
        results.append(triple_barrier_label(
            prices, int(idx),
            pt_abs=sigma * pt_mult,
            sl_abs=sigma * sl_mult,
            horizon=horizon,
            direction=int(dirs[k]) if k < len(dirs) else 1,
        ))
    return results


# ────────────────────────────────────────────────────────────────
# Meta-labeling
# ────────────────────────────────────────────────────────────────


def build_meta_labels(
    primary_decisions: Sequence[int],
    realised_labels: Sequence[int],
) -> np.ndarray:
    """Construct meta-labels from primary decisions + realised outcomes.

    Given a primary signal (``+1`` long, ``-1`` short, ``0`` no trade)
    and the realised triple-barrier label (``+1`` / ``0`` / ``-1``), the
    meta-label is:

    * ``1`` — primary fired AND was correct (same sign as realised).
    * ``0`` — primary fired but was wrong, OR no primary signal.

    The meta-classifier is then trained on the subset where
    ``primary_decisions != 0`` with target ``meta_labels``. Its role is
    to predict "should I trust the primary here?" — it doesn't invent
    new trades.

    Returns:
        ``np.ndarray[int64]`` of the same length as the inputs.
    """
    p = np.asarray(primary_decisions, dtype=np.int64)
    r = np.asarray(realised_labels, dtype=np.int64)
    if p.shape != r.shape:
        raise ValueError(
            f"meta-label inputs must have same shape, got {p.shape} vs {r.shape}"
        )
    meta = np.zeros(p.shape, dtype=np.int64)
    fired = p != 0
    meta[fired] = (np.sign(p[fired]) == np.sign(r[fired])).astype(np.int64)
    return meta


def meta_filter(
    primary_decisions: Sequence[int],
    secondary_probs: Sequence[float],
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply a secondary confidence filter to primary trading decisions.

    ``primary_decisions[i]`` is the sign of the primary strategy signal
    (``+1`` long, ``-1`` short, ``0`` no trade). ``secondary_probs[i]``
    is the meta-classifier's probability that the primary will succeed
    on that row. The filtered output is the primary's sign where the
    secondary clears the threshold, ``0`` elsewhere.

    Returns:
        ``np.ndarray[int64]`` of filtered decisions.
    """
    p = np.asarray(primary_decisions, dtype=np.int64)
    q = np.asarray(secondary_probs, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(
            f"meta-filter inputs must have same shape, got {p.shape} vs {q.shape}"
        )
    keep = q >= threshold
    return np.where(keep, p, 0).astype(np.int64)
