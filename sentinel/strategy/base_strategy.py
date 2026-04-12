"""
Базовый класс стратегии.

Все торговые стратегии SENTINEL наследуют BaseStrategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from core.models import FeatureVector, Signal


class BaseStrategy(ABC):
    """Абстрактный базовый класс для торговых стратегий."""

    NAME: str = "base"

    @abstractmethod
    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        """
        Проанализировать индикаторы и вернуть сигнал.

        Args:
            features: Текущий снимок индикаторов.
            has_open_position: True если уже есть открытая позиция для этого символа.
            entry_price: Цена входа в текущую позицию (для SL/TP).

        Returns:
            Signal или None (≡ HOLD).
        """
        ...
