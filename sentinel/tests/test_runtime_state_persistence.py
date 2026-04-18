"""Round-trip persistence tests for runtime state that survives restarts.

Covers:
- ``RiskSentinel.export_state`` / ``restore_state``
- ``RiskStateMachine.export_state`` / ``restore_state``
- ``AlertMonitor.export_state`` / ``restore_state``

Without these round-trips, a mid-day restart forgets daily trade caps,
cooldowns, and the current risk state — so the bot can re-enter trades
against real, still-binding limits.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import EventBus
from core.models import RiskState
from monitoring.alerts import AlertMonitor
from risk.sentinel import RiskLimits, RiskSentinel
from risk.state_machine import RiskStateMachine


def _make_sentinel(cooldown_sec: int = 1800) -> RiskSentinel:
    sm = RiskStateMachine(event_bus=EventBus(), max_daily_loss=50.0)
    limits = RiskLimits(
        max_daily_loss_usd=50.0,
        max_daily_trades=6,
        max_trades_per_hour=2,
        min_trade_interval_sec=cooldown_sec,
        min_order_usd=10.0,
        max_order_usd=100.0,
    )
    return RiskSentinel(limits=limits, state_machine=sm)


class TestRiskSentinelPersistence:
    def test_roundtrip_preserves_counters(self):
        s1 = _make_sentinel()
        s1.record_trade(commission=0.50)
        s1.record_trade(commission=0.75)
        s1.record_trade(commission=1.25)

        blob = json.dumps(s1.export_state())

        s2 = _make_sentinel()
        s2.restore_state(json.loads(blob))

        assert s2.daily_trades == 3
        assert s2.daily_commission == pytest.approx(2.50)
        assert s2.trades_last_hour == 3
        # Cooldown should be non-zero because last_trade_ts was just now.
        assert s2.cooldown_remaining_sec > 0

    def test_roundtrip_across_utc_day_drops_daily(self):
        """A restart across the UTC rollover must not resurrect yesterday's
        daily counters — those limits are reset at midnight regardless."""
        s1 = _make_sentinel()
        s1.record_trade(commission=5.0)
        s1.record_trade(commission=3.0)

        # Forge a blob dated yesterday.
        blob = s1.export_state()
        blob["utc_date"] = "2000-01-01"

        s2 = _make_sentinel()
        s2.restore_state(blob)

        assert s2.daily_trades == 0
        assert s2.daily_commission == 0.0
        # Rolling-hour window still self-cleans via trades_timestamps cutoff,
        # and last_trade_ts carries over (needed for cooldown math).
        assert s2._last_trade_ts > 0

    def test_restore_drops_stale_timestamps_beyond_24h(self):
        s1 = _make_sentinel()
        # Inject a fake ancient timestamp.
        s1._trades_timestamps = [time.time() - 100000, time.time() - 60]
        s1._last_trade_ts = time.time() - 60
        s1._daily_trades = 2
        s1._daily_commission = 1.0

        blob = s1.export_state()
        s2 = _make_sentinel()
        s2.restore_state(blob)

        # 100000s > 24h → pruned; only the -60s timestamp survives.
        assert len(s2._trades_timestamps) == 1

    def test_restore_malformed_blob_does_not_raise(self):
        s = _make_sentinel()
        # Should swallow the error and leave counters at default.
        s.restore_state({"trades_timestamps": "not-a-list", "daily_trades": "bad"})
        assert s.daily_trades == 0
        assert s.daily_commission == 0.0


class TestRiskStateMachinePersistence:
    def test_roundtrip_preserves_state(self):
        sm1 = RiskStateMachine(event_bus=EventBus(), max_daily_loss=50.0)
        sm1._state = RiskState.SAFE
        sm1._last_change_ts = 1234567890

        blob = sm1.export_state()

        sm2 = RiskStateMachine(event_bus=EventBus(), max_daily_loss=50.0)
        sm2.restore_state(blob)

        assert sm2.state == RiskState.SAFE
        assert sm2._last_change_ts == 1234567890

    def test_restore_unknown_state_falls_back_to_normal(self):
        sm = RiskStateMachine(event_bus=EventBus(), max_daily_loss=50.0)
        sm.restore_state({"state": "BOGUS", "last_change_ts": 0})
        assert sm.state == RiskState.NORMAL

    def test_restore_stop_state_blocks_until_reset(self):
        """Restoring STOP must actually keep the bot blocked — this is the
        whole point of persistence: a crash during kill-switch must not
        silently re-enable trading on restart."""
        sm = RiskStateMachine(event_bus=EventBus(), max_daily_loss=50.0)
        sm.restore_state({"state": "STOP", "last_change_ts": int(time.time())})
        assert sm.state == RiskState.STOP
        sm.reset()
        assert sm.state == RiskState.NORMAL


class TestAlertMonitorPersistence:
    def test_roundtrip_preserves_alerts_and_counters(self):
        m1 = AlertMonitor()
        m1.record_signal_rejection("Order too small: $7.00 < $10.0")
        m1.record_signal_rejection("Order too small: $8.00 < $10.0")
        m1.record_trade_result(is_win=False)
        m1.record_trade_result(is_win=False)
        m1._last_price = 123.45

        blob = json.dumps(m1.export_state())

        m2 = AlertMonitor()
        m2.restore_state(json.loads(blob))

        assert m2._rejection_count == 2
        assert m2._loss_streak == 2
        assert m2._last_price == pytest.approx(123.45)
        # Both rejections normalize to the same key, so the Counter is 1-deep.
        assert sum(m2._rejection_reasons.values()) == 2

    def test_roundtrip_preserves_alert_list(self):
        m1 = AlertMonitor(price_gap_pct=0.5)
        # First call seeds baseline; second triggers the alert.
        m1.check_price_gap(100.0)
        m1.check_price_gap(101.0)  # 1% gap > 0.5% threshold

        assert len(m1._alerts) == 1

        blob = json.dumps(m1.export_state())
        m2 = AlertMonitor()
        m2.restore_state(json.loads(blob))

        recent = m2.get_recent_alerts()
        assert len(recent) == 1
        assert recent[0]["category"] == "price_gap"

    def test_restore_malformed_blob_does_not_raise(self):
        m = AlertMonitor()
        m.restore_state({"alerts": "not-a-list", "rejection_count": "bad"})
        # Monitor should remain usable.
        assert m._rejection_count == 0
