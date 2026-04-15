"""
Risk Sentinel — абсолютный защитный слой.

Ни один ордер не исполняется без одобрения Risk Sentinel.
Pipeline: Signal → 7 проверок → APPROVED / REJECTED.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from core.models import Direction, RiskCheckResult, RiskState, Signal
from .state_machine import RiskStateMachine

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Настраиваемые лимиты (зажаты absolute_limits в config)."""
    max_daily_loss_usd: float = 50.0
    max_daily_loss_pct: float = 10.0
    max_daily_trades: int = 6
    max_position_pct: float = 20.0
    max_total_exposure_pct: float = 60.0
    max_open_positions: int = 5
    max_trades_per_hour: int = 2
    min_trade_interval_sec: int = 1800
    min_order_usd: float = 10.0
    max_order_usd: float = 100.0
    max_loss_per_trade_pct: float = 3.0
    mandatory_stop_loss: bool = True
    max_daily_commission_pct: float = 1.0


class RiskSentinel:
    """Главный модуль проверки рисков.

    Каждый сигнал проходит 7 проверок:
    1. Daily Loss
    2. Position Limit
    3. Exposure
    4. Frequency
    5. Order Size
    6. Stop-Loss
    7. Sanity Check
    """

    def __init__(
        self,
        limits: RiskLimits,
        state_machine: RiskStateMachine,
    ) -> None:
        self._limits = limits
        self._sm = state_machine
        self._trades_timestamps: list[float] = []  # timestamps of recent trades
        self._last_trade_ts: float = 0.0
        self._daily_trades: int = 0
        self._daily_commission: float = 0.0

    @property
    def state(self) -> RiskState:
        return self._sm.state

    @property
    def daily_trades(self) -> int:
        return self._daily_trades

    @property
    def daily_commission(self) -> float:
        return self._daily_commission

    @property
    def trades_last_hour(self) -> int:
        hour_ago = time.time() - 3600
        return sum(1 for ts in self._trades_timestamps if ts > hour_ago)

    @property
    def cooldown_remaining_sec(self) -> int:
        if self._last_trade_ts <= 0:
            return 0
        elapsed = time.time() - self._last_trade_ts
        remaining = self._limits.min_trade_interval_sec - elapsed
        return max(0, int(remaining))

    def get_runtime_metrics(self, balance: float = 0.0) -> dict[str, float | int | str]:
        commission_pct = self._daily_commission / balance * 100 if balance > 0 else 0.0
        return {
            "state": self.state.value,
            "daily_trades": self._daily_trades,
            "trades_last_hour": self.trades_last_hour,
            "daily_commission": self._daily_commission,
            "commission_pct": commission_pct,
            "cooldown_remaining_sec": self.cooldown_remaining_sec,
        }

    # ──────────────────────────────────────────────
    # Main check
    # ──────────────────────────────────────────────

    def check_signal(
        self,
        signal: Signal,
        daily_pnl: float,
        open_positions_count: int,
        total_exposure_pct: float,
        balance: float,
        current_market_price: float,
    ) -> RiskCheckResult:
        """Проверить сигнал по 7 правилам.

        Returns:
            RiskCheckResult(approved=True/False, reason=...)
        """
        # SELL (closing positions) must NEVER be blocked — holding a losing
        # position is always riskier than closing it.
        is_sell = signal.direction == Direction.SELL

        # [0] State check
        if self._sm.state == RiskState.STOP and not is_sell:
            return RiskCheckResult(approved=False, reason="Trading stopped: STOP state")

        if self._sm.state == RiskState.SAFE and signal.direction == Direction.BUY:
            return RiskCheckResult(approved=False, reason="SAFE state: only SELL allowed")

        if self._sm.state == RiskState.REDUCED and signal.confidence < 0.8 and not is_sell:
            return RiskCheckResult(
                approved=False,
                reason=f"REDUCED state: requires confidence >= 0.8, got {signal.confidence:.2f}",
            )

        # [1] Daily Loss (never block SELL — must be able to close losing positions)
        if daily_pnl <= -self._limits.max_daily_loss_usd and not is_sell:
            return RiskCheckResult(
                approved=False,
                reason=f"Daily loss limit exhausted: ${daily_pnl:.2f}",
            )

        # [2] Position Limit (only for BUY)
        if signal.direction == Direction.BUY:
            if open_positions_count >= self._limits.max_open_positions:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Max open positions reached: {open_positions_count}",
                )

        # [3] Exposure (only for BUY)
        if signal.direction == Direction.BUY:
            if balance <= 0:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Invalid balance: ${balance:.2f}",
                )
            order_value = signal.suggested_quantity * current_market_price
            new_exposure_pct = total_exposure_pct + (order_value / balance * 100)
            if new_exposure_pct > self._limits.max_total_exposure_pct:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Exposure limit exceeded: {new_exposure_pct:.1f}% > {self._limits.max_total_exposure_pct}%",
                )

        # [4] Frequency
        now = time.time()
        if signal.direction == Direction.BUY:
            if self._daily_trades >= self._limits.max_daily_trades:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Daily trade limit reached: {self._daily_trades}",
                )

            # Trades this hour
            hour_ago = now - 3600
            trades_this_hour = sum(1 for ts in self._trades_timestamps if ts > hour_ago)
            if trades_this_hour >= self._limits.max_trades_per_hour:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Hourly trade limit reached: {trades_this_hour}",
                )

            # Min interval
            if self._last_trade_ts > 0:
                elapsed = now - self._last_trade_ts
                if elapsed < self._limits.min_trade_interval_sec:
                    remaining = int(self._limits.min_trade_interval_sec - elapsed)
                    return RiskCheckResult(
                        approved=False,
                        reason=f"Min trade interval: wait {remaining}s more",
                    )

        # [5] Order Size
        if signal.direction == Direction.BUY:
            order_usd = signal.suggested_quantity * current_market_price
            if order_usd < self._limits.min_order_usd:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Order too small: ${order_usd:.2f} < ${self._limits.min_order_usd}",
                )
            # Max check — reduce instead of reject
            if order_usd > self._limits.max_order_usd:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Order too large: ${order_usd:.2f} > ${self._limits.max_order_usd}",
                )

        # [6] Stop-Loss
        if self._limits.mandatory_stop_loss and signal.direction == Direction.BUY:
            if signal.stop_loss_price <= 0:
                return RiskCheckResult(
                    approved=False,
                    reason="Stop-loss is mandatory but not set",
                )
            sl_pct = abs(current_market_price - signal.stop_loss_price) / current_market_price * 100
            if sl_pct > self._limits.max_loss_per_trade_pct:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Stop-loss too wide: {sl_pct:.1f}% > {self._limits.max_loss_per_trade_pct}%",
                )

        # [7] Sanity Check
        if signal.direction == Direction.BUY and current_market_price > 0:
            # Если цена сигнала сильно отличается от текущей рыночной
            if signal.suggested_quantity <= 0:
                return RiskCheckResult(
                    approved=False,
                    reason="Invalid order quantity <= 0",
                )

        # ✅ All checks passed
        logger.info(
            "Risk APPROVED: %s %s conf=%.2f",
            signal.direction.value, signal.symbol, signal.confidence,
        )
        return RiskCheckResult(approved=True, reason="All risk checks passed")

    # ──────────────────────────────────────────────
    # Trade recording
    # ──────────────────────────────────────────────

    def record_trade(self, commission: float = 0.0, *, increment_trade: bool = True) -> None:
        """Зарегистрировать совершённую сделку."""
        now = time.time()
        if increment_trade:
            self._trades_timestamps.append(now)
            self._last_trade_ts = now
            self._daily_trades += 1
        self._daily_commission += commission

        # Очистка старых timestamps (>24h)
        cutoff = now - 86400
        self._trades_timestamps = [ts for ts in self._trades_timestamps if ts > cutoff]

    def reset_daily(self) -> None:
        """Сброс дневных счётчиков."""
        self._daily_trades = 0
        self._daily_commission = 0.0
        self._sm.reset()
        logger.info("Risk Sentinel daily reset")
