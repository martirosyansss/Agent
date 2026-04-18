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

from monitoring.event_log import EventType, get_event_log

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
        try:
            get_event_log().emit(
                EventType.GUARD_TRIPPED,
                guard="circuit_breaker",
                name=self.name,
                trip_count_today=self.trip_count_today,
                cooldown_sec=self.cooldown_sec,
                permanent_stop=self.permanent_stop,
            )
        except Exception:
            pass  # telemetry must never break the breaker

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

    def __init__(
        self,
        consecutive_loss_threshold: int = 3,
        strategy_cooldown_sec: int = 14400,
        strategy_cooldown_overrides: Optional[dict[str, int]] = None,
    ) -> None:
        if consecutive_loss_threshold < 1:
            raise ValueError(
                f"consecutive_loss_threshold must be >= 1, got {consecutive_loss_threshold}"
            )
        if strategy_cooldown_sec < 0:
            raise ValueError(
                f"strategy_cooldown_sec must be >= 0, got {strategy_cooldown_sec}"
            )
        self._loss_threshold = consecutive_loss_threshold
        self._default_strategy_cooldown_sec = strategy_cooldown_sec
        self._strategy_cooldown_overrides = dict(strategy_cooldown_overrides or {})
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
        self._consecutive_losses: dict[str, int] = {}
        self._blocked_strategies: dict[str, float] = {}  # strategy -> blocked_until_ts
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
        """Можно ли торговать (глобально)?"""
        return not self.any_tripped and not self.any_permanent

    def is_strategy_allowed(self, strategy_name: str) -> bool:
        """Можно ли торговать этой конкретной стратегии?"""
        if not self.is_trading_allowed():
            return False
        
        # Check per-strategy block
        blocked_until = self._blocked_strategies.get(strategy_name, 0)
        if blocked_until > time.time():
            return False
            
        return True

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

    def record_trade_result(self, is_win: bool, strategy_name: str = "") -> Optional[str]:
        """CB-2: N убыточных сделок подряд для одной стратегии (N — _loss_threshold)."""
        if is_win:
            if strategy_name:
                self._consecutive_losses[strategy_name] = 0
            else:
                self._consecutive_losses.clear()
            return None

        key = strategy_name or "__global__"
        self._consecutive_losses[key] = self._consecutive_losses.get(key, 0) + 1
        count = self._consecutive_losses[key]

        if count >= self._loss_threshold:
            self._consecutive_losses[key] = 0
            if strategy_name:
                # Per-strategy trip: block for cooldown but DON'T trip global CB-2 unless global
                cooldown = self._strategy_cooldown_overrides.get(
                    strategy_name, self._default_strategy_cooldown_sec,
                )
                self._blocked_strategies[strategy_name] = time.time() + cooldown
                logger.warning(
                    "Strategy BLOCKED: %s for %ds due to %d consecutive losses",
                    strategy_name, cooldown, self._loss_threshold,
                )
                return f"{self._loss_threshold} consecutive losses ({strategy_name}) — strategy isolated"
            else:
                self._breakers["CB-2"].trip()
                return f"{self._loss_threshold} consecutive losses (global)"
        return None

    def get_strategy_block_remaining_sec(self, strategy_name: str) -> int:
        """Сколько секунд до конца блока стратегии. 0 = не заблокирована."""
        blocked_until = self._blocked_strategies.get(strategy_name, 0.0)
        remaining = blocked_until - time.time()
        return max(0, int(remaining))

    def get_blocked_strategies(self) -> dict[str, dict[str, int]]:
        """Снимок текущих per-strategy блокировок: {name: {remaining_sec, total_sec}}.

        Стратегии с истёкшим блоком автоматически удаляются из снимка
        (но остаются в _blocked_strategies до reset_daily — get_strategy_block_remaining_sec
        корректно возвращает 0 для них).
        """
        now = time.time()
        snapshot: dict[str, dict[str, int]] = {}
        for name, blocked_until in self._blocked_strategies.items():
            remaining = blocked_until - now
            if remaining <= 0:
                continue
            total = self._strategy_cooldown_overrides.get(
                name, self._default_strategy_cooldown_sec,
            )
            snapshot[name] = {
                "remaining_sec": int(remaining),
                "total_sec": int(total),
            }
        return snapshot

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
        self._consecutive_losses.clear()
        self._blocked_strategies.clear()
        self._api_errors.clear()
        self._latency_violations = 0
        logger.info("Circuit Breakers daily reset")
