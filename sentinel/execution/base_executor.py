"""
Базовый класс исполнения ордеров.

Определяет интерфейс для PaperExecutor и LiveExecutor.
Каждый executor обрабатывает Signal → Order → заполненный Order.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from core.models import Order, Signal

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """Абстрактный базовый класс для исполнителей ордеров."""

    def __init__(self, commission_pct: float = 0.1) -> None:
        self.commission_pct = commission_pct

    @abstractmethod
    async def execute_order(self, signal: Signal, quantity: float, current_price: float) -> Optional[Order]:
        """Создать и исполнить ордер по сигналу.

        Args:
            signal: Торговый сигнал.
            quantity: Количество актива.
            current_price: Текущая рыночная цена.

        Returns:
            Заполненный Order при успехе, None при ошибке.
        """

    def calculate_commission(self, quantity: float, price: float) -> float:
        """Расчёт комиссии."""
        import math
        if math.isnan(quantity) or math.isnan(price) or math.isinf(quantity) or math.isinf(price):
            logger.error("NaN/Inf in commission calc: qty=%s, price=%s", quantity, price)
            return 0.0
        if quantity <= 0 or price <= 0:
            return 0.0
        return quantity * price * self.commission_pct / 100
