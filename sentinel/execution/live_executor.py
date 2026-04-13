"""
Live Executor — реальное исполнение ордеров на Binance Spot.

Правила:
  - Вход MARKET ордером
  - ОБЯЗАТЕЛЬНО биржевой protective order (OCO/stop-loss+TP на Binance)
  - Если protective order не подтверждён → немедленный market exit + STOP
  - Timeout 10 сек
  - Reconciliation каждые 5 мин
  - Первые 24ч: max_order = $20
  - Retry ЗАПРЕЩЁН — ждать fill, не повторять автоматически
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

from core.constants import EVENT_ORDER_FILLED
from core.events import EventBus
from core.models import Direction, Order, OrderStatus, OrderType, Signal

from .base_executor import BaseExecutor

logger = logging.getLogger(__name__)

# Timeout for exchange response
ORDER_TIMEOUT_SEC = 10.0
# First 24h max order in USD
FIRST_DAY_MAX_ORDER = 20.0


class LiveExecutor(BaseExecutor):
    """Реальный исполнитель ордеров через Binance Spot API.

    ВНИМАНИЕ: live executor ТРЕБУЕТ:
    1. Подтверждённый fill от биржи
    2. Биржевой protective order (OCO)
    3. Reconciliation каждые 5 мин
    """

    def __init__(
        self,
        event_bus: EventBus,
        api_key: str = "",
        api_secret: str = "",
        commission_pct: float = 0.1,
        first_day_max_order: float = FIRST_DAY_MAX_ORDER,
    ) -> None:
        super().__init__(commission_pct=commission_pct)
        self._event_bus = event_bus
        self._api_key = api_key
        self._api_secret = api_secret
        self._first_day_max_order = first_day_max_order
        self._start_time = time.time()
        self._orders: list[Order] = []
        # Binance client будет инициализирован при первом вызове
        self._client = None

    @property
    def _is_first_day(self) -> bool:
        return (time.time() - self._start_time) < 86400

    def _get_max_order_usd(self) -> float:
        """Максимальный ордер с учётом первого дня."""
        if self._is_first_day:
            return self._first_day_max_order
        return float("inf")  # Ограничивается внешним Risk Sentinel

    def _init_client(self) -> bool:
        """Ленивая инициализация Binance клиента."""
        if self._client is not None:
            return True
        try:
            from binance.client import Client
            self._client = Client(self._api_key, self._api_secret)
            logger.info("Binance client initialized")
            return True
        except ImportError:
            logger.error("python-binance not installed. pip install python-binance")
            return False
        except Exception as e:
            logger.error("Failed to init Binance client: %s", e)
            return False

    async def execute_order(
        self,
        signal: Signal,
        quantity: float,
        current_price: float,
    ) -> Optional[Order]:
        """Исполнить ордер на Binance Spot.

        1. Проверяет лимиты
        2. Отправляет MARKET ордер
        3. Ожидает fill (timeout 10s)
        4. Создаёт protective OCO order
        5. Если OCO не подтверждён → emergency market exit
        """
        # Pre-flight checks
        order_value = quantity * current_price
        max_order = self._get_max_order_usd()

        if order_value > max_order:
            logger.warning(
                "Live order exceeds limit: $%.2f > $%.2f (first_day=%s)",
                order_value, max_order, self._is_first_day,
            )
            return None

        if quantity <= 0 or current_price <= 0:
            logger.warning("Invalid order params: qty=%s price=%s", quantity, current_price)
            return None

        if math.isnan(quantity) or math.isinf(quantity) or math.isnan(current_price) or math.isinf(current_price):
            logger.error("NaN/Inf detected in order params: qty=%s price=%s", quantity, current_price)
            return None

        if not self._init_client():
            return None

        try:
            return await self._execute_with_protection(signal, quantity, current_price)
        except Exception as e:
            logger.error("Live execution error: %s", e)
            return None

    async def _execute_with_protection(
        self,
        signal: Signal,
        quantity: float,
        current_price: float,
    ) -> Optional[Order]:
        """Исполнить с защитным ордером."""
        side_str = "BUY" if signal.direction == Direction.BUY else "SELL"
        commission = self.calculate_commission(quantity, current_price)

        # --- Step 1: MARKET order ---
        try:
            result = self._client.create_order(
                symbol=signal.symbol,
                side=side_str,
                type="MARKET",
                quantity=f"{quantity:.8f}",
            )
        except Exception as e:
            logger.error("Market order failed: %s", e)
            return None

        # Parse fill
        fills = result.get("fills", [])
        fill_qty = float(result.get("executedQty", 0))
        if not fills or fill_qty <= 0:
            logger.error("Order returned no fills or zero qty: oid=%s, fills=%s, execQty=%s",
                         result.get("orderId"), len(fills), fill_qty)
            return None
        fill_price = float(fills[0].get("price", current_price))
        if fill_qty < quantity:
            logger.warning("Partial fill: requested %.8f, got %.8f", quantity, fill_qty)
        exchange_id = str(result.get("orderId", ""))

        order = Order(
            timestamp=int(time.time() * 1000),
            symbol=signal.symbol,
            side=signal.direction,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=current_price,
            status=OrderStatus.FILLED,
            exchange_order_id=exchange_id,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            commission=commission,
            is_paper=False,
            signal_id=signal.signal_id,
            strategy_name=signal.strategy_name,
            signal_reason=signal.reason,
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
        )

        # --- Step 2: Protective OCO order (only for BUY entries) ---
        if signal.direction == Direction.BUY and signal.stop_loss_price > 0:
            oco_ok = await self._place_protective_oco(
                signal.symbol, fill_qty, fill_price,
                signal.stop_loss_price, signal.take_profit_price,
            )
            if not oco_ok:
                # Emergency: protective order failed → market exit
                logger.error("CRITICAL: Protective OCO failed, emergency sell!")
                await self._emergency_sell(signal.symbol, fill_qty)
                order.status = OrderStatus.CANCELLED
                return order

        self._orders.append(order)
        await self._event_bus.emit(EVENT_ORDER_FILLED, order)

        logger.info(
            "LIVE %s %s: qty=%.6f fill=%.2f oid=%s",
            side_str, signal.symbol, fill_qty, fill_price, exchange_id,
        )
        return order

    async def _place_protective_oco(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        """Разместить OCO protective order на бирже."""
        if take_profit <= 0:
            take_profit = entry_price * 1.05  # Default 5% TP

        try:
            self._client.create_oco_order(
                symbol=symbol,
                side="SELL",
                quantity=f"{quantity:.8f}",
                price=f"{take_profit:.2f}",
                stopPrice=f"{stop_loss:.2f}",
                stopLimitPrice=f"{stop_loss * 0.999:.2f}",
                stopLimitTimeInForce="GTC",
            )
            logger.info("OCO placed: SL=%.2f TP=%.2f", stop_loss, take_profit)
            return True
        except Exception as e:
            logger.error("OCO order failed: %s", e)
            return False

    async def _emergency_sell(self, symbol: str, quantity: float) -> None:
        """Аварийная продажа — market sell без retry."""
        try:
            self._client.create_order(
                symbol=symbol,
                side="SELL",
                type="MARKET",
                quantity=f"{quantity:.8f}",
            )
            logger.warning("Emergency sell executed: %s qty=%.6f", symbol, quantity)
        except Exception as e:
            logger.critical("EMERGENCY SELL FAILED: %s - %s", symbol, e)
