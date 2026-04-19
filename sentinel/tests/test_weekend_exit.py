"""Tests for the weekend-exit guard (Phase 4).

Contracts:

* **Off by default** — when ``enabled=False``, the guard never fires
  regardless of the wall clock. Opt-in is explicit.
* **Window wraps midnight** — the Fri 20:00 → Mon 00:00 UTC window
  crosses Sunday midnight; the minutes-from-Monday arithmetic must
  treat it as one contiguous interval, not two disjoint pieces.
* **Boundary semantics** — cutoff is inclusive, reopen is exclusive.
  At Mon 00:00 exactly we're back in-market; at Fri 20:00 exactly we
  close.
* **Custom windows** — callers can pick any (day, hour) pair; the
  logic applies uniformly (useful for exotic holidays or testing).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.weekend_exit import is_in_weekend_exit_window, should_exit_before_weekend


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# 2026-04-13 is a Monday — anchor all test times to that week so the
# weekday math is readable.
MON = _utc(2026, 4, 13)           # Monday 00:00
TUE_NOON = _utc(2026, 4, 14, 12)  # Tuesday 12:00 — deep weekday
FRI_1930 = _utc(2026, 4, 17, 19, 30)  # just before cutoff
FRI_2000 = _utc(2026, 4, 17, 20, 0)   # cutoff moment
FRI_2100 = _utc(2026, 4, 17, 21, 0)
SAT_NOON = _utc(2026, 4, 18, 12)
SUN_2330 = _utc(2026, 4, 19, 23, 30)
MON_0000 = _utc(2026, 4, 20, 0, 0)    # reopen moment
MON_0030 = _utc(2026, 4, 20, 0, 30)   # just after reopen


class TestDefaultWindow:
    @pytest.mark.parametrize("when,expected", [
        (MON, False),
        (TUE_NOON, False),
        (FRI_1930, False),
        (FRI_2000, True),
        (FRI_2100, True),
        (SAT_NOON, True),
        (SUN_2330, True),
        (MON_0000, False),   # reopen is exclusive start of in-market
        (MON_0030, False),
    ])
    def test_in_window(self, when, expected):
        assert is_in_weekend_exit_window(when) is expected


class TestPolicyWrapper:
    def test_disabled_short_circuits_even_in_window(self):
        ok, reason = should_exit_before_weekend(SAT_NOON, enabled=False)
        assert ok is False
        assert reason == "weekend_exit_disabled"

    def test_enabled_fires_in_window(self):
        ok, reason = should_exit_before_weekend(SAT_NOON, enabled=True)
        assert ok is True
        assert reason.startswith("weekend_exit")
        # Reason string includes a readable timestamp — it ends up in
        # close_reason and the events.jsonl payload, so human operators
        # can see at a glance when the guard fired.
        assert "Sat" in reason

    def test_enabled_noop_outside_window(self):
        ok, reason = should_exit_before_weekend(TUE_NOON, enabled=True)
        assert ok is False
        assert reason == "outside_weekend_window"


class TestCustomWindow:
    def test_non_wrapping_window(self):
        # Thu 12:00 → Thu 18:00 — entirely within a single weekday.
        thu_1100 = _utc(2026, 4, 16, 11, 0)
        thu_1500 = _utc(2026, 4, 16, 15, 0)
        thu_1900 = _utc(2026, 4, 16, 19, 0)
        kwargs = dict(
            cutoff_day_of_week=3, cutoff_hour_utc=12,
            reopen_day_of_week=3, reopen_hour_utc=18,
        )
        assert is_in_weekend_exit_window(thu_1100, **kwargs) is False
        assert is_in_weekend_exit_window(thu_1500, **kwargs) is True
        assert is_in_weekend_exit_window(thu_1900, **kwargs) is False

    def test_wrapping_longer_window(self):
        # Thu 22:00 → Tue 06:00 (very long window — covers most of week).
        # Only Tue 06:00 … Thu 22:00 is out-of-window.
        kwargs = dict(
            cutoff_day_of_week=3, cutoff_hour_utc=22,
            reopen_day_of_week=1, reopen_hour_utc=6,
        )
        # Tue 07:00 — freshly out of window
        tue_0700 = _utc(2026, 4, 14, 7, 0)
        assert is_in_weekend_exit_window(tue_0700, **kwargs) is False
        # Thu 23:00 — just inside window
        thu_2300 = _utc(2026, 4, 16, 23, 0)
        assert is_in_weekend_exit_window(thu_2300, **kwargs) is True
        # Sunday — still in the long window
        sun_1200 = _utc(2026, 4, 19, 12, 0)
        assert is_in_weekend_exit_window(sun_1200, **kwargs) is True
