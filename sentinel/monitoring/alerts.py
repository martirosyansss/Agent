"""
Monitoring & Alerts — real-time system health monitoring.

Tracks:
1. Price gaps (sudden moves > threshold)
2. Execution latency (order fill time)
3. Signal rejection rate
4. Data staleness
5. Consecutive losses

Emits alert events for Telegram notifications.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Alert:
    """Single alert instance."""
    timestamp: int
    severity: str       # 'info', 'warning', 'critical'
    category: str       # 'price_gap', 'latency', 'rejection', 'data_stale', 'loss_streak'
    message: str


class AlertMonitor:
    """Real-time alert monitor for SENTINEL."""

    def __init__(
        self,
        price_gap_pct: float = 2.0,
        max_latency_sec: float = 5.0,
        rejection_threshold: int = 5,
        stale_data_sec: float = 120.0,
        loss_streak_threshold: int = 3,
    ) -> None:
        self._price_gap_pct = price_gap_pct
        self._max_latency_sec = max_latency_sec
        self._rejection_threshold = rejection_threshold
        self._stale_data_sec = stale_data_sec
        self._loss_streak_threshold = loss_streak_threshold

        self._alerts: deque[Alert] = deque(maxlen=100)
        self._last_price: float = 0.0
        self._rejection_count: int = 0
        self._loss_streak: int = 0
        self._last_data_ts: float = 0.0

        # Callbacks
        self._on_alert = None

    def set_alert_callback(self, callback) -> None:
        """Set async callback for alert notifications."""
        self._on_alert = callback

    def check_price_gap(self, price: float) -> Alert | None:
        """Check for sudden price gap."""
        if self._last_price > 0 and price > 0:
            gap_pct = abs(price - self._last_price) / self._last_price * 100
            if gap_pct >= self._price_gap_pct:
                direction = "UP" if price > self._last_price else "DOWN"
                alert = Alert(
                    timestamp=int(time.time() * 1000),
                    severity="warning",
                    category="price_gap",
                    message=f"Price gap {direction} {gap_pct:.1f}%: "
                            f"${self._last_price:.2f} → ${price:.2f}",
                )
                self._alerts.append(alert)
                self._last_price = price
                return alert
        self._last_price = price
        return None

    def check_execution_latency(self, signal_ts: int, fill_ts: int) -> Alert | None:
        """Check order execution latency."""
        latency_sec = (fill_ts - signal_ts) / 1000
        if latency_sec > self._max_latency_sec:
            alert = Alert(
                timestamp=fill_ts,
                severity="warning",
                category="latency",
                message=f"High execution latency: {latency_sec:.1f}s "
                        f"(threshold: {self._max_latency_sec}s)",
            )
            self._alerts.append(alert)
            return alert
        return None

    def record_signal_rejection(self, reason: str) -> Alert | None:
        """Record a signal rejection. Alert if too many consecutive."""
        self._rejection_count += 1
        if self._rejection_count >= self._rejection_threshold:
            alert = Alert(
                timestamp=int(time.time() * 1000),
                severity="warning",
                category="rejection",
                message=f"{self._rejection_count} consecutive signal rejections. "
                        f"Last: {reason}",
            )
            self._alerts.append(alert)
            return alert
        return None

    def record_signal_accepted(self) -> None:
        """Reset rejection counter on successful signal."""
        self._rejection_count = 0

    def check_data_staleness(self, last_data_ts: float) -> Alert | None:
        """Check if market data is stale."""
        age = time.time() - last_data_ts
        if age > self._stale_data_sec:
            alert = Alert(
                timestamp=int(time.time() * 1000),
                severity="critical",
                category="data_stale",
                message=f"Market data stale: {age:.0f}s since last update "
                        f"(threshold: {self._stale_data_sec}s)",
            )
            self._alerts.append(alert)
            return alert
        return None

    def record_trade_result(self, is_win: bool) -> Alert | None:
        """Track consecutive losses."""
        if is_win:
            self._loss_streak = 0
            return None

        self._loss_streak += 1
        if self._loss_streak >= self._loss_streak_threshold:
            alert = Alert(
                timestamp=int(time.time() * 1000),
                severity="warning",
                category="loss_streak",
                message=f"{self._loss_streak} consecutive losing trades",
            )
            self._alerts.append(alert)
            return alert
        return None

    def get_recent_alerts(self, limit: int = 20) -> list[dict]:
        """Get recent alerts for dashboard."""
        alerts = list(self._alerts)[-limit:]
        return [
            {
                "timestamp": a.timestamp,
                "severity": a.severity,
                "category": a.category,
                "message": a.message,
            }
            for a in alerts
        ]

    @property
    def stats(self) -> dict:
        """Current monitoring stats."""
        return {
            "total_alerts": len(self._alerts),
            "rejection_count": self._rejection_count,
            "loss_streak": self._loss_streak,
            "last_price": self._last_price,
        }
