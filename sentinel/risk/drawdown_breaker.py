"""
Drawdown Circuit Breaker — equity-peak-aware kill switch for new entries.

Triggers when realized drawdown from rolling peak equity breaches a threshold
on any of three windows (daily, weekly, monthly). When tripped, blocks NEW
entries until the cooldown clock expires (or until equity recovers above a
hysteresis band — whichever comes first).

This is intentionally separate from the eight Circuit Breakers in
``circuit_breakers.py``: those react to per-tick anomalies (price spikes,
spread, latency). The drawdown breaker reacts to compounded P&L and is the
last line of defence for an account-level blow-up.

Design notes:
- Peak equity is tracked per-window. The "weekly peak" is the highest equity
  observed within the current ISO week; monthly within the current calendar
  month. Daily uses UTC calendar day boundaries.
- We never block SELL orders — exiting a losing position is always safer than
  holding it. Callers must check ``allows_new_entry`` for BUY only.
- Hysteresis: once tripped, equity must recover to (peak * (1 - threshold *
  HYSTERESIS_RATIO)) before the breaker auto-resets. This prevents chatter
  when equity oscillates around the trigger.
- All clocks are injected via ``time_provider`` so unit tests can advance
  time deterministically.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from monitoring.event_log import emit_guard_tripped

logger = logging.getLogger(__name__)


# Hysteresis: equity must recover to (peak * (1 - threshold * 0.5)) to auto-reset.
# E.g. with 5% trigger, recovery to within 2.5% of peak unlocks trading again.
HYSTERESIS_RATIO = 0.5


@dataclass
class DrawdownThresholds:
    """Per-window drawdown limits expressed as fraction of peak equity (e.g. 0.05 = 5%).

    Defaults are conservative for a small retail account ($500-$10k). For
    larger accounts these can be tightened (1-2% daily is typical at funds).
    """
    daily_pct: float = 0.05      # 5% from daily peak → block until next UTC day
    weekly_pct: float = 0.10     # 10% from weekly peak → block until next ISO week
    monthly_pct: float = 0.15    # 15% from monthly peak → block until next month

    def __post_init__(self) -> None:
        for name in ("daily_pct", "weekly_pct", "monthly_pct"):
            v = getattr(self, name)
            if not (0.0 < v < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {v}")


@dataclass
class _WindowState:
    """Mutable state for one drawdown window."""
    name: str
    period_id: str = ""           # "2026-04-17" / "2026-W16" / "2026-04"
    peak_equity: float = 0.0
    current_equity: float = 0.0
    is_tripped: bool = False
    tripped_at_equity: float = 0.0
    tripped_at_period: str = ""

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)


def _utc_period_ids(ts: float) -> tuple[str, str, str]:
    """Return (daily_id, weekly_id, monthly_id) for a Unix timestamp.

    Uses ISO week (Monday start) and UTC calendar day/month boundaries.
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    iso_year, iso_week, _ = dt.isocalendar()
    return (
        dt.strftime("%Y-%m-%d"),
        f"{iso_year}-W{iso_week:02d}",
        dt.strftime("%Y-%m"),
    )


class DrawdownBreaker:
    """Three-window drawdown circuit breaker.

    Usage:
        breaker = DrawdownBreaker(DrawdownThresholds())
        breaker.update(current_equity=1050.0)  # call on every equity tick
        if not breaker.allows_new_entry():
            # block BUY signals; SELL still allowed
            ...

    The breaker is intentionally conservative: any one window tripping is
    enough to block new entries, and resets only happen at calendar period
    rollover OR via hysteresis recovery.
    """

    def __init__(
        self,
        thresholds: Optional[DrawdownThresholds] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self._thresholds = thresholds or DrawdownThresholds()
        self._time = time_provider or (lambda: __import__("time").time())
        self._daily = _WindowState(name="daily")
        self._weekly = _WindowState(name="weekly")
        self._monthly = _WindowState(name="monthly")
        self._trip_history: list[dict] = []  # for postmortem / dashboard
        # update() is invoked from candle-close (under _trade_decision_lock)
        # AND from the per-trade tick handler (without it). Without a mutex
        # the two paths race on peak_equity / is_tripped and can lose a
        # trip flag or drop a peak update. Guard the whole state block.
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def update(self, current_equity: float) -> Optional[str]:
        """Feed the latest equity reading. Returns trip reason if a window
        just transitioned from OK → tripped, else None.

        Call this on every position change OR on a periodic equity tick
        (e.g. every 30 seconds). Cheap to call repeatedly.
        """
        if current_equity <= 0 or current_equity != current_equity:  # NaN guard
            return None

        now = self._time()
        d_id, w_id, m_id = _utc_period_ids(now)

        trip_reason: Optional[str] = None
        # Queue trip emissions so we don't hold the lock while calling into
        # the event bus (which takes its own locks — nested locking risk).
        pending_trips: list[dict] = []
        with self._lock:
            for state, period_id, threshold in (
                (self._daily, d_id, self._thresholds.daily_pct),
                (self._weekly, w_id, self._thresholds.weekly_pct),
                (self._monthly, m_id, self._thresholds.monthly_pct),
            ):
                if state.period_id != period_id:
                    state.period_id = period_id
                    state.peak_equity = current_equity
                    state.is_tripped = False
                    state.tripped_at_equity = 0.0
                    state.tripped_at_period = ""

                state.current_equity = current_equity
                if current_equity > state.peak_equity:
                    state.peak_equity = current_equity

                if state.is_tripped:
                    recovery_band = state.peak_equity * (1.0 - threshold * HYSTERESIS_RATIO)
                    if current_equity >= recovery_band:
                        logger.info(
                            "DD-BREAKER %s reset via hysteresis: equity=%.2f recovered above %.2f (peak=%.2f)",
                            state.name, current_equity, recovery_band, state.peak_equity,
                        )
                        state.is_tripped = False
                    continue

                if state.drawdown_pct >= threshold:
                    state.is_tripped = True
                    state.tripped_at_equity = current_equity
                    state.tripped_at_period = period_id
                    reason = (
                        f"{state.name} DD {state.drawdown_pct * 100:.2f}% >= "
                        f"{threshold * 100:.1f}% (peak {state.peak_equity:.2f} → {current_equity:.2f})"
                    )
                    self._trip_history.append({
                        "ts": now,
                        "window": state.name,
                        "period": period_id,
                        "peak": state.peak_equity,
                        "equity": current_equity,
                        "dd_pct": state.drawdown_pct * 100,
                    })
                    pending_trips.append({
                        "name": state.name,
                        "period": period_id,
                        "peak": state.peak_equity,
                        "equity": current_equity,
                        "dd_pct": state.drawdown_pct * 100,
                        "threshold_pct": threshold * 100,
                        "reason": reason,
                    })
                    if trip_reason is None:
                        trip_reason = reason

        for t in pending_trips:
            logger.critical("DD-BREAKER TRIPPED: %s", t["reason"])
            try:
                emit_guard_tripped(
                    guard="drawdown_breaker",
                    name=t["name"],
                    period=t["period"],
                    peak_equity=round(t["peak"], 2),
                    current_equity=round(t["equity"], 2),
                    dd_pct=round(t["dd_pct"], 2),
                    threshold_pct=round(t["threshold_pct"], 2),
                )
            except Exception:
                pass
        return trip_reason

    def allows_new_entry(self) -> bool:
        """True if no window is tripped. Call before approving any BUY signal."""
        with self._lock:
            return not (self._daily.is_tripped or self._weekly.is_tripped or self._monthly.is_tripped)

    def active_trips(self) -> list[str]:
        """Names of currently-tripped windows (for diagnostics)."""
        with self._lock:
            return [s.name for s in (self._daily, self._weekly, self._monthly) if s.is_tripped]

    def force_reset(self, window: Optional[str] = None) -> None:
        """Manual reset (operator action). Without arg resets all windows."""
        with self._lock:
            targets = (
                (self._daily, self._weekly, self._monthly)
                if window is None
                else tuple(s for s in (self._daily, self._weekly, self._monthly) if s.name == window)
            )
            for state in targets:
                state.is_tripped = False
                state.tripped_at_equity = 0.0
                state.tripped_at_period = ""
        logger.warning("DD-BREAKER manual reset: %s", window or "all windows")

    def export_state(self) -> dict:
        """Full state blob suitable for round-trip persistence.

        Distinct from ``snapshot`` (which is a human-readable diagnostic
        view): this includes the raw fields needed to restore tripped
        flags after a restart. Without persistence, a bot restarting
        mid-drawdown forgets it was tripped and permits new BUYs against
        the real drawdown.
        """
        def _w(s: _WindowState) -> dict:
            return {
                "period_id": s.period_id,
                "peak_equity": s.peak_equity,
                "current_equity": s.current_equity,
                "is_tripped": s.is_tripped,
                "tripped_at_equity": s.tripped_at_equity,
                "tripped_at_period": s.tripped_at_period,
            }
        with self._lock:
            return {
                "daily": _w(self._daily),
                "weekly": _w(self._weekly),
                "monthly": _w(self._monthly),
                "trip_history": list(self._trip_history),
            }

    def restore_state(self, blob: dict) -> None:
        """Restore state previously produced by ``export_state``.

        Drops silently if the blob is malformed — better to start fresh
        than crash startup. Period-rollover logic in ``update()`` will
        reset any window whose period_id no longer matches current time.
        """
        try:
            with self._lock:
                for name, state in (
                    ("daily", self._daily),
                    ("weekly", self._weekly),
                    ("monthly", self._monthly),
                ):
                    w = blob.get(name) or {}
                    state.period_id = str(w.get("period_id", ""))
                    state.peak_equity = float(w.get("peak_equity", 0.0))
                    state.current_equity = float(w.get("current_equity", 0.0))
                    state.is_tripped = bool(w.get("is_tripped", False))
                    state.tripped_at_equity = float(w.get("tripped_at_equity", 0.0))
                    state.tripped_at_period = str(w.get("tripped_at_period", ""))
                hist = blob.get("trip_history")
                if isinstance(hist, list):
                    self._trip_history = list(hist)
                _d_t, _w_t, _m_t = self._daily.is_tripped, self._weekly.is_tripped, self._monthly.is_tripped
            logger.info(
                "DD-BREAKER state restored: daily_tripped=%s weekly_tripped=%s monthly_tripped=%s",
                _d_t, _w_t, _m_t,
            )
        except Exception as exc:
            logger.warning("DD-BREAKER restore_state failed (%s) — starting fresh", exc)

    def snapshot(self) -> dict:
        """Diagnostic snapshot for dashboard / Telegram."""
        with self._lock:
            allows = not (self._daily.is_tripped or self._weekly.is_tripped or self._monthly.is_tripped)
            active = [s.name for s in (self._daily, self._weekly, self._monthly) if s.is_tripped]
            return {
                "allows_entry": allows,
                "active_trips": active,
                "windows": {
                    s.name: {
                        "period": s.period_id,
                        "peak": round(s.peak_equity, 2),
                        "current": round(s.current_equity, 2),
                        "dd_pct": round(s.drawdown_pct * 100, 2),
                        "is_tripped": s.is_tripped,
                    }
                    for s in (self._daily, self._weekly, self._monthly)
                },
                "thresholds": {
                    "daily_pct": self._thresholds.daily_pct * 100,
                    "weekly_pct": self._thresholds.weekly_pct * 100,
                    "monthly_pct": self._thresholds.monthly_pct * 100,
                },
                "trip_history_last_5": self._trip_history[-5:],
            }
