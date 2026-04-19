"""
Weekend Exit — opt-in guard that closes positions inside a defined
"thin-liquidity" window (by default Fri 20:00 UTC → Mon 00:00 UTC).

Crypto markets technically trade 24/7, but the order book on Binance
spot and the perps is materially thinner on weekends: documented
flash-crash incidents cluster around Sunday evening UTC, when two-week
highs can round-trip in five minutes and tightly-stopped positions get
wicked out for zero fundamental reason. For an unattended bot holding
over the weekend, sitting in cash is usually the right risk decision.

This is opt-in (``Settings.weekend_exit_enabled``) because some operators
genuinely want weekend exposure — e.g. DCA accumulators or very large
R-multiple targets where wick-out is rare. Leaving it default-off keeps
existing deployments bit-for-bit compatible.

The guard is pure: it takes a UTC datetime and the config window, and
returns ``(should_exit, reason)``. The main loop uses the same
``_force_exit_position`` helper that Phase 3 introduced.
"""

from __future__ import annotations

from datetime import datetime


def _minutes_from_week_start(dow: int, hour: int, minute: int = 0) -> int:
    """Minutes since the start of the ISO week (Monday 00:00).

    Uses ``datetime.weekday()`` convention: Monday=0, Sunday=6.
    """
    return dow * 24 * 60 + hour * 60 + minute


def is_in_weekend_exit_window(
    now_utc: datetime,
    *,
    cutoff_day_of_week: int = 4,   # Friday
    cutoff_hour_utc: int = 20,
    reopen_day_of_week: int = 0,   # Monday
    reopen_hour_utc: int = 0,
) -> bool:
    """Return True when ``now_utc`` sits inside the weekend exit window.

    The window wraps around Sunday midnight when ``cutoff > reopen``
    (the normal Fri-20 → Mon-00 case). When it doesn't wrap (e.g. a
    config that puts the whole window inside a single weekday for
    testing), the simple inclusive-start / exclusive-end check applies.
    """
    now_min = _minutes_from_week_start(
        now_utc.weekday(), now_utc.hour, now_utc.minute,
    )
    cutoff_min = _minutes_from_week_start(cutoff_day_of_week, cutoff_hour_utc)
    reopen_min = _minutes_from_week_start(reopen_day_of_week, reopen_hour_utc)

    if cutoff_min < reopen_min:
        # Window does not wrap the week boundary
        return cutoff_min <= now_min < reopen_min
    # Window wraps through Sunday→Monday
    return now_min >= cutoff_min or now_min < reopen_min


def should_exit_before_weekend(
    now_utc: datetime,
    *,
    enabled: bool,
    cutoff_day_of_week: int = 4,
    cutoff_hour_utc: int = 20,
    reopen_day_of_week: int = 0,
    reopen_hour_utc: int = 0,
) -> tuple[bool, str]:
    """Policy wrapper — returns ``(should_exit, reason)``.

    ``enabled=False`` short-circuits to ``(False, "disabled")`` so
    callers don't need a separate feature-flag check.
    """
    if not enabled:
        return (False, "weekend_exit_disabled")

    if is_in_weekend_exit_window(
        now_utc,
        cutoff_day_of_week=cutoff_day_of_week,
        cutoff_hour_utc=cutoff_hour_utc,
        reopen_day_of_week=reopen_day_of_week,
        reopen_hour_utc=reopen_hour_utc,
    ):
        return (
            True,
            f"weekend_exit:window_active "
            f"({now_utc.strftime('%a %H:%M')} UTC)",
        )
    return (False, "outside_weekend_window")
