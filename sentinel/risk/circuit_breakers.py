"""
Circuit Breakers — 8 автоматических предохранителей.

Работают НЕЗАВИСИМО от Risk Sentinel — двойная линия защиты.
При срабатывании замораживают торговлю на cooldown_sec.
3+ срабатывания одного CB за день → ПОЛНАЯ ОСТАНОВКА.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """Состояние одного Circuit Breaker."""
    name: str
    cooldown_sec: int = 300
    is_tripped: bool = False
    tripped_at: float = 0.0
    trip_count_today: int = 0
    permanent_stop: bool = False  # 3+ срабатываний → permanent

    def trip(self) -> None:
        self.is_tripped = True
        self.tripped_at = time.time()
        self.trip_count_today += 1
        if self.trip_count_today >= 3:
            self.permanent_stop = True
        logger.warning(
            "CB TRIPPED: %s (count: %d, cooldown: %ds, permanent: %s)",
            self.name, self.trip_count_today, self.cooldown_sec, self.permanent_stop,
        )

    def check_cooldown(self) -> bool:
        """Вернуть True если CB всё ещё активен."""
        if self.permanent_stop:
            return True
        if not self.is_tripped:
            return False
        elapsed = time.time() - self.tripped_at
        if elapsed >= self.cooldown_sec:
            self.is_tripped = False
            logger.info("CB RESET: %s (after %ds)", self.name, int(elapsed))
            return False
        return True

    def reset_daily(self) -> None:
        self.is_tripped = False
        self.tripped_at = 0.0
        self.trip_count_today = 0
        self.permanent_stop = False


class CircuitBreakers:
    """Менеджер 8 Circuit Breakers."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreakerState] = {
            "CB-1": CircuitBreakerState(name="CB-1: Price Anomaly", cooldown_sec=300),
            "CB-2": CircuitBreakerState(name="CB-2: Consecutive Loss", cooldown_sec=1800),
            "CB-3": CircuitBreakerState(name="CB-3: Spread Anomaly", cooldown_sec=300),
            "CB-4": CircuitBreakerState(name="CB-4: Volume Anomaly", cooldown_sec=600),
            "CB-5": CircuitBreakerState(name="CB-5: API Error Rate", cooldown_sec=900),
            "CB-6": CircuitBreakerState(name="CB-6: Latency", cooldown_sec=300),
            "CB-7": CircuitBreakerState(name="CB-7: Balance Mismatch", cooldown_sec=0),  # manual only
            "CB-8": CircuitBreakerState(name="CB-8: Commission Spike", cooldown_sec=600),
        }
        self._consecutive_losses: int = 0
        self._api_errors: list[float] = []
        self._latency_violations: int = 0

    # ──────────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────────

    @property
    def any_tripped(self) -> bool:
        return any(cb.check_cooldown() for cb in self._breakers.values())

    @property
    def any_permanent(self) -> bool:
        return any(cb.permanent_stop for cb in self._breakers.values())

    def get_active_breakers(self) -> list[str]:
        return [name for name, cb in self._breakers.items() if cb.check_cooldown()]

    def is_trading_allowed(self) -> bool:
        """Можно ли торговать?"""
        return not self.any_tripped

    # ──────────────────────────────────────────────
    # CB-1: Price Anomaly
    # ──────────────────────────────────────────────

    def check_price_anomaly(self, price_change_pct_1m: float) -> Optional[str]:
        """CB-1: цена изменилась > 5% за 1 мин."""
        if abs(price_change_pct_1m) > 5.0:
            self._breakers["CB-1"].trip()
            return f"Price anomaly: {price_change_pct_1m:+.1f}% in 1 min"
        return None

    # ──────────────────────────────────────────────
    # CB-2: Consecutive Losses
    # ──────────────────────────────────────────────

    def record_trade_result(self, is_win: bool) -> Optional[str]:
        """CB-2: 3 убыточных сделки подряд."""
        if is_win:
            self._consecutive_losses = 0
            return None

        self._consecutive_losses += 1
        if self._consecutive_losses >= 3:
            self._breakers["CB-2"].trip()
            self._consecutive_losses = 0
            return f"3 consecutive losses"
        return None

    # ──────────────────────────────────────────────
    # CB-3: Spread Anomaly
    # ──────────────────────────────────────────────

    def check_spread(self, spread_pct: float) -> Optional[str]:
        """CB-3: спред > 0.5%."""
        if spread_pct > 0.5:
            self._breakers["CB-3"].trip()
            return f"Spread anomaly: {spread_pct:.2f}%"
        return None

    # ──────────────────────────────────────────────
    # CB-4: Volume Anomaly
    # ──────────────────────────────────────────────

    def check_volume_anomaly(self, volume_ratio: float) -> Optional[str]:
        """CB-4: объём > 10x или < 0.1x среднего."""
        if volume_ratio > 10.0 or volume_ratio < 0.1:
            self._breakers["CB-4"].trip()
            return f"Volume anomaly: ratio={volume_ratio:.1f}x"
        return None

    # ──────────────────────────────────────────────
    # CB-5: API Error Rate
    # ──────────────────────────────────────────────

    def record_api_error(self) -> Optional[str]:
        """CB-5: > 5 ошибок API за 5 минут."""
        now = time.time()
        self._api_errors.append(now)
        # Очистка старых
        cutoff = now - 300
        self._api_errors = [ts for ts in self._api_errors if ts > cutoff]

        if len(self._api_errors) > 5:
            self._breakers["CB-5"].trip()
            return f"API error rate: {len(self._api_errors)} errors in 5 min"
        return None

    # ──────────────────────────────────────────────
    # CB-6: Latency
    # ──────────────────────────────────────────────

    def check_latency(self, latency_sec: float) -> Optional[str]:
        """CB-6: задержка > 5 сек (3 раза подряд)."""
        if latency_sec > 5.0:
            self._latency_violations += 1
        else:
            self._latency_violations = 0

        if self._latency_violations >= 3:
            self._breakers["CB-6"].trip()
            self._latency_violations = 0
            return f"High latency: {latency_sec:.1f}s (3 consecutive)"
        return None

    # ──────────────────────────────────────────────
    # CB-7: Balance Mismatch
    # ──────────────────────────────────────────────

    def check_balance_mismatch(self, expected: float, actual: float) -> Optional[str]:
        """CB-7: расхождение баланса > 1%."""
        if expected <= 0:
            return None
        diff_pct = abs(expected - actual) / expected * 100
        if diff_pct > 1.0:
            self._breakers["CB-7"].trip()
            return f"Balance mismatch: expected=${expected:.2f}, actual=${actual:.2f} ({diff_pct:.1f}%)"
        return None

    # ──────────────────────────────────────────────
    # CB-8: Commission Spike
    # ──────────────────────────────────────────────

    def check_commission_spike(self, daily_commission: float, balance: float) -> Optional[str]:
        """CB-8: комиссии за день > 1% от капитала."""
        if balance <= 0:
            return None
        comm_pct = daily_commission / balance * 100
        if comm_pct > 1.0:
            self._breakers["CB-8"].trip()
            return f"Commission spike: ${daily_commission:.2f} ({comm_pct:.1f}% of balance)"
        return None

    # ──────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────

    def reset_daily(self) -> None:
        """Ежедневный сброс всех CB."""
        for cb in self._breakers.values():
            cb.reset_daily()
        self._consecutive_losses = 0
        self._api_errors.clear()
        self._latency_violations = 0
        logger.info("Circuit Breakers daily reset")
