"""
Paper Executor — виртуальное исполнение ордеров.

Имитирует исполнение без реальных денег:
- Проскальзывание: ±0.05%
- Комиссия: 0.1% (как Binance Spot)
- Всё записывается с is_paper=True
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional

from core.constants import EVENT_ORDER_FILLED
from core.events import EventBus
from core.models import Direction, Order, OrderStatus, OrderType, Signal

from .base_executor import BaseExecutor

logger = logging.getLogger(__name__)


class PaperExecutor(BaseExecutor):
    """Виртуальный исполнитель ордеров для Paper Trading."""

    SLIPPAGE_PCT = 0.05  # Проскальзывание ±0.05%
    MIN_ORDER_USD = 10.0  # Минимальный ордер (как на Binance)

    def __init__(
        self,
        event_bus: EventBus,
        commission_pct: float = 0.1,
    ) -> None:
        super().__init__(commission_pct=commission_pct)
        self._event_bus = event_bus

    async def execute_order(
        self,
        signal: Signal,
        quantity: float,
        current_price: float,
    ) -> Optional[Order]:
        """Виртуальное исполнение ордера.

        Returns:
            Order с заполненными fill_price, fill_quantity; None если ордер невалиден.
        """
        # Pre-flight checks
        order_value = quantity * current_price
        if order_value < self.MIN_ORDER_USD:
            logger.warning(
                "Paper order too small: $%.2f < $%.2f min",
                order_value, self.MIN_ORDER_USD,
            )
            return None

        if quantity <= 0 or current_price <= 0:
            logger.warning("Invalid order params: qty=%s price=%s", quantity, current_price)
            return None

        # Симулировать проскальзывание
        slippage_factor = 1 + random.uniform(-self.SLIPPAGE_PCT, self.SLIPPAGE_PCT) / 100
        if signal.direction == Direction.BUY:
            # При покупке проскальзывание обычно вверх
            slippage_factor = 1 + abs(random.uniform(0, self.SLIPPAGE_PCT)) / 100
        else:
            # При продаже проскальзывание обычно вниз
            slippage_factor = 1 - abs(random.uniform(0, self.SLIPPAGE_PCT)) / 100

        fill_price = current_price * slippage_factor

        # Комиссия
        commission = self.calculate_commission(quantity, fill_price)

        # Создать ордер
        order = Order(
            timestamp=int(time.time() * 1000),
            symbol=signal.symbol,
            side=signal.direction,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=current_price,
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            fill_quantity=quantity,
            commission=commission,
            is_paper=True,
            signal_id=signal.signal_id,
            strategy_name=signal.strategy_name,
            signal_reason=signal.reason,
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
        )

        logger.info(
            "Paper %s %s: qty=%.6f fill_price=%.2f commission=%.4f",
            signal.direction.value,
            signal.symbol,
            quantity,
            fill_price,
            commission,
        )

        # Оповестить систему
        await self._event_bus.emit(EVENT_ORDER_FILLED, order)

        return order
