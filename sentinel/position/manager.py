"""
Position Manager — управление позициями и расчёт PnL.

Функции:
- Открытие/закрытие позиций
- Расчёт unrealized/realized PnL
- Автоматическая проверка Stop-Loss / Take-Profit
- Обновление цен из маркет-данных
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from core.constants import (
    EVENT_POSITION_CLOSED,
    EVENT_POSITION_OPENED,
)
from core.events import EventBus
from core.models import (
    Direction,
    Order,
    Position,
    PositionStatus,
    Signal,
)

logger = logging.getLogger(__name__)


@dataclass
class PaperWallet:
    """Виртуальный кошелёк для Paper Trading."""
    initial_balance: float = 500.0
    usdt_balance: float = 500.0
    btc_balance: float = 0.0
    eth_balance: float = 0.0

    def available_usd(self) -> float:
        return self.usdt_balance


class PositionManager:
    """Менеджер позиций SENTINEL."""

    def __init__(
        self,
        event_bus: EventBus,
        initial_balance: float = 500.0,
        max_open_positions: int = 4,
    ) -> None:
        self._event_bus = event_bus
        self.wallet = PaperWallet(
            initial_balance=initial_balance,
            usdt_balance=initial_balance,
        )
        self._max_open_positions = max_open_positions
        self._lock = asyncio.Lock()  # Protects position open/close operations
        self._positions: dict[str, Position] = {}  # symbol → Position
        self._sl_tp: dict[str, tuple[float, float]] = {}  # symbol → (stop_loss, take_profit)
        self._closed_positions: list[Position] = []
        self._total_realized_pnl: float = 0.0
        self._trades_today: int = 0
        self._wins_today: int = 0
        self._losses_today: int = 0
        self._peak_balance: float = initial_balance
        self._max_drawdown_pct: float = 0.0
        self._equity_history: list[dict[str, float | str]] = []
        self._last_snapshot_time: float = 0.0
        self._record_equity_snapshot(force=True)

    # ──────────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────────

    @property
    def open_positions(self) -> list[Position]:
        return list(self._positions.values())

    @property
    def open_positions_count(self) -> int:
        return len(self._positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_realized_pnl(self) -> float:
        return self._total_realized_pnl

    @property
    def total_exposure_usd(self) -> float:
        return sum(p.quantity * p.current_price for p in self._positions.values())

    @property
    def balance(self) -> float:
        return self.wallet.usdt_balance + self.total_exposure_usd

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_balance <= 0:
            return 0.0
        return max(0.0, (self._peak_balance - self.balance) / self._peak_balance * 100)

    @property
    def max_drawdown_pct(self) -> float:
        return self._max_drawdown_pct

    @property
    def equity_history(self) -> list[dict[str, float | str]]:
        return list(self._equity_history)

    def _record_equity_snapshot(self, force: bool = False) -> None:
        balance = self.balance
        if balance > self._peak_balance:
            self._peak_balance = balance

        drawdown_pct = 0.0
        if self._peak_balance > 0:
            drawdown_pct = max(0.0, (self._peak_balance - balance) / self._peak_balance * 100)
        self._max_drawdown_pct = max(self._max_drawdown_pct, drawdown_pct)

        # Throttle chart history: max 1 snapshot per 30s unless forced
        now = time.time()
        if not force and (now - self._last_snapshot_time) < 30:
            return

        snapshot = {
            "date": str(int(time.time() * 1000)),
            "label": time.strftime("%H:%M:%S", time.localtime()),
            "pnl": round(balance - self.wallet.initial_balance, 4),
            "balance": round(balance, 4),
        }
        self._equity_history.append(snapshot)
        self._equity_history = self._equity_history[-100:]
        self._last_snapshot_time = now

    # ──────────────────────────────────────────────
    # Open position
    # ──────────────────────────────────────────────

    async def open_position(
        self,
        order: Order,
        stop_loss_price: float = 0.0,
        take_profit_price: float = 0.0,
    ) -> Optional[Position]:
        """Открыть позицию по исполненному BUY ордеру."""
        async with self._lock:
            if order.side != Direction.BUY:
                logger.warning("Cannot open position from SELL order")
                return None

            if self.open_positions_count >= self._max_open_positions:
                logger.warning("Max open positions reached (%d)", self._max_open_positions)
                return None

            if self.has_position(order.symbol):
                logger.warning("Position already open for %s", order.symbol)
                return None

            fill_price = order.fill_price or order.price or 0
            fill_qty = order.fill_quantity or order.quantity

            if fill_price <= 0:
                logger.error("Invalid fill_price for %s: %s — rejecting open", order.symbol, fill_price)
                return None
            if fill_qty <= 0:
                logger.error("Invalid fill_qty for %s: %s — rejecting open", order.symbol, fill_qty)
                return None

            cost = fill_qty * fill_price + order.commission
            effective_stop_loss = stop_loss_price or order.stop_loss_price
            effective_take_profit = take_profit_price or order.take_profit_price

            if cost > self.wallet.usdt_balance:
                logger.warning(
                    "Insufficient balance: need $%.2f, have $%.2f",
                    cost, self.wallet.usdt_balance,
                )
                return None

            # Списать средства
            self.wallet.usdt_balance -= cost
            if self.wallet.usdt_balance < 0:
                logger.error("Balance went negative after open: $%.4f — reverting", self.wallet.usdt_balance)
                self.wallet.usdt_balance += cost
                return None

            position = Position(
                symbol=order.symbol,
                side="LONG",
                entry_price=fill_price,
                quantity=fill_qty,
                current_price=fill_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                stop_loss_price=effective_stop_loss,
                take_profit_price=effective_take_profit,
                strategy_name=order.strategy_name,
                signal_id=order.signal_id,
                signal_reason=order.signal_reason,
                status=PositionStatus.OPEN,
                opened_at=str(int(time.time() * 1000)),
                is_paper=order.is_paper,
            )

            # Сохранить SL/TP
            self._sl_tp[order.symbol] = (effective_stop_loss, effective_take_profit)

            self._positions[order.symbol] = position
            self._record_equity_snapshot()
            logger.info(
                "Position opened: %s @ %.2f qty=%.6f SL=%.2f TP=%.2f",
                order.symbol, fill_price, fill_qty, effective_stop_loss, effective_take_profit,
            )

            await self._event_bus.emit(EVENT_POSITION_OPENED, position)
            return position

    # ──────────────────────────────────────────────
    # Close position
    # ──────────────────────────────────────────────

    async def close_position(
        self,
        order: Order,
    ) -> Optional[Position]:
        """Закрыть позицию по исполненному SELL ордеру."""
        async with self._lock:
            if order.side != Direction.SELL:
                logger.warning("Cannot close position from BUY order")
                return None

            position = self._positions.get(order.symbol)
            if not position:
                logger.warning("No open position for %s", order.symbol)
                return None

            fill_price = order.fill_price or order.price or 0
            fill_qty = order.fill_quantity or order.quantity

            if fill_price <= 0:
                logger.error("Invalid fill_price on close for %s: %s", order.symbol, fill_price)
                return None

            # Расчёт PnL
            realized_pnl = (fill_price - position.entry_price) * fill_qty - order.commission
            position.realized_pnl = realized_pnl
            position.current_price = fill_price
            position.status = PositionStatus.CLOSED
            position.closed_at = str(int(time.time() * 1000))

            # Вернуть средства
            self.wallet.usdt_balance += fill_qty * fill_price - order.commission
            self._total_realized_pnl += realized_pnl

            # Статистика
            self._trades_today += 1
            if realized_pnl > 0:
                self._wins_today += 1
            else:
                self._losses_today += 1

            # Переместить в закрытые
            del self._positions[order.symbol]
            self._sl_tp.pop(order.symbol, None)
            self._closed_positions.append(position)
            self._record_equity_snapshot()

            logger.info(
                "Position closed: %s @ %.2f PnL=%.2f",
                order.symbol, fill_price, realized_pnl,
            )

            await self._event_bus.emit(EVENT_POSITION_CLOSED, position)
            return position

    # ──────────────────────────────────────────────
    # Price updates & SL/TP checker
    # ──────────────────────────────────────────────

    def update_price(self, symbol: str, price: float) -> None:
        """Обновить текущую цену позиции."""
        pos = self._positions.get(symbol)
        if pos:
            pos.current_price = price
            pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            self._record_equity_snapshot()

    def check_stop_loss_take_profit(self, symbol: str) -> Optional[str]:
        """Проверить SL/TP для позиции.

        Returns:
            'stop_loss', 'take_profit' или None.
        """
        pos = self._positions.get(symbol)
        if not pos:
            return None

        sl, tp = self._sl_tp.get(symbol, (0.0, 0.0))

        if sl > 0 and pos.current_price <= sl:
            return "stop_loss"
        if tp > 0 and pos.current_price >= tp:
            return "take_profit"
        return None

    # ──────────────────────────────────────────────
    # Daily stats
    # ──────────────────────────────────────────────

    def get_daily_stats(self) -> dict:
        """Получить дневную статистику."""
        total_trades = self._trades_today
        win_rate = (self._wins_today / total_trades * 100) if total_trades > 0 else 0.0
        return {
            "trades_today": total_trades,
            "wins": self._wins_today,
            "losses": self._losses_today,
            "win_rate": win_rate,
            "pnl_today": self._total_realized_pnl,
            "balance": self.balance,
            "open_positions": self.open_positions_count,
        }

    def reset_daily_stats(self) -> None:
        """Сбросить ежедневные счётчики (вызывать в полночь UTC)."""
        self._trades_today = 0
        self._wins_today = 0
        self._losses_today = 0

    # ──────────────────────────────────────────────
    # State for Telegram / Dashboard
    # ──────────────────────────────────────────────

    def get_state(self) -> dict:
        """Состояние для Telegram/Dashboard."""
        stats = self.get_daily_stats()
        exposure_pct = self.total_exposure_usd / self.balance * 100 if self.balance > 0 else 0.0
        return {
            "balance": self.balance,
            "pnl_today": stats["pnl_today"],
            "pnl_total": self._total_realized_pnl,
            "open_positions": stats["open_positions"],
            "trades_today": stats["trades_today"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": stats["win_rate"],
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "current_drawdown_pct": self.current_drawdown_pct,
            "exposure_pct": exposure_pct,
            "positions": self.open_positions,
            "recent_trades": [
                {
                    "symbol": p.symbol,
                    "side": "SELL",
                    "price": p.current_price,
                    "pnl": p.realized_pnl,
                    "time": p.closed_at,
                    "strategy_name": p.strategy_name,
                    "signal_id": p.signal_id,
                    "signal_reason": p.signal_reason,
                }
                for p in self._closed_positions[-10:]
            ],
            "pnl_history": self.equity_history,
        }
