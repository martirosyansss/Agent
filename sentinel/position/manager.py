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
        self._trailing: dict[str, tuple[float, float, float]] = {}  # symbol → (activate_pct, trail_pct, max_price)
        self._closed_positions: list[Position] = []
        # Multi-stage TP: symbol → (tp1_price, tp2_price, tp3_price)
        self._tp_levels: dict[str, tuple[float, float, float]] = {}
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
                initial_quantity=fill_qty,
                open_commission=order.commission,
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

            # Расчёт PnL: close-side commission + pro-rata open-side commission.
            # open_commission was paid at entry; we allocate the unused remainder
            # (proportional to the qty being closed) so total realized PnL equals
            # real cash flow: qty*(exit-entry) - open_c - close_c.
            alloc_open_commission = position.open_commission
            if position.initial_quantity > 0:
                alloc_open_commission *= fill_qty / position.initial_quantity
            realized_pnl = (
                (fill_price - position.entry_price) * fill_qty
                - order.commission
                - alloc_open_commission
            )
            position.realized_pnl = realized_pnl + position.partial_realized_pnl
            position.current_price = fill_price
            position.status = PositionStatus.CLOSED
            position.closed_at = str(int(time.time() * 1000))
            position.close_reason = order.signal_reason or ""

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
            self._trailing.pop(order.symbol, None)
            self._tp_levels.pop(order.symbol, None)
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

    async def set_trailing_stop(self, symbol: str, activate_pct: float, trail_pct: float) -> None:
        """Настроить trailing stop для позиции (async-safe).

        Args:
            activate_pct: PnL% для активации (e.g. 2.5 = +2.5%)
            trail_pct: процент отката от максимума для срабатывания (e.g. 1.5)
        """
        async with self._lock:
            pos = self._positions.get(symbol)
            if pos:
                self._trailing[symbol] = (activate_pct, trail_pct, pos.entry_price)

    async def setup_tp_levels(self, symbol: str) -> None:
        """Setup multi-stage TP levels based on SL distance (async-safe).

        TP1 = entry + 1.0× risk (1R) → close 50%, move SL to breakeven
        TP2 = entry + 2.0× risk (2R) → close 30%, tighten trailing
        TP3 = original TP (full R:R) → remaining 20% rides trailing
        """
        async with self._lock:
            pos = self._positions.get(symbol)
            sl, tp = self._sl_tp.get(symbol, (0.0, 0.0))
            if not pos or pos.entry_price <= 0 or sl <= 0:
                return

            risk = pos.entry_price - sl  # dollar risk per unit
            if risk <= 0:
                return

            tp1 = pos.entry_price + risk * 1.0   # 1R
            tp2 = pos.entry_price + risk * 2.0   # 2R
            tp3 = tp if tp > tp2 else pos.entry_price + risk * 3.0  # 3R or original TP

            self._tp_levels[symbol] = (tp1, tp2, tp3)
            # initial_quantity set at open; keep as authoritative reference
            if pos.initial_quantity <= 0:
                pos.initial_quantity = pos.quantity
            pos.original_stop_loss = sl
            pos.tp_stage = 0

    async def move_sl_to_breakeven(self, symbol: str) -> None:
        """Move stop-loss to entry price (breakeven) after TP1 hit (async-safe)."""
        async with self._lock:
            pos = self._positions.get(symbol)
            if not pos:
                return
            sl, tp = self._sl_tp.get(symbol, (0.0, 0.0))
            # Breakeven = entry price + small buffer (0.1% to cover commission)
            breakeven = pos.entry_price * 1.001
            if breakeven > sl:
                self._sl_tp[symbol] = (breakeven, tp)
                logger.info("SL moved to breakeven for %s: %.2f → %.2f", symbol, sl, breakeven)

    async def apply_tp_stage_transition(
        self,
        symbol: str,
        *,
        stage: int,
        move_to_breakeven: bool = False,
        trailing: Optional[tuple[float, float]] = None,
    ) -> None:
        """Atomic transition to a new TP stage.

        Combines tp_stage update, optional breakeven SL move, and trailing
        re-configuration in a single locked critical section — prevents
        races between partial close and state mutation.

        Args:
            stage: new tp_stage value
            move_to_breakeven: if True, raise SL to entry + 0.1%
            trailing: (activate_pct, trail_pct) tuple, or None to skip
        """
        async with self._lock:
            pos = self._positions.get(symbol)
            if not pos:
                return
            pos.tp_stage = stage

            if move_to_breakeven:
                sl, tp = self._sl_tp.get(symbol, (0.0, 0.0))
                breakeven = pos.entry_price * 1.001
                if breakeven > sl:
                    self._sl_tp[symbol] = (breakeven, tp)
                    logger.info("SL → breakeven for %s: %.2f → %.2f", symbol, sl, breakeven)

            if trailing is not None:
                act_pct, trail_pct = trailing
                self._trailing[symbol] = (act_pct, trail_pct, pos.entry_price)

    async def partial_close_position(
        self,
        order: Order,
        close_pct: float,
    ) -> Optional[Position]:
        """Частичное закрытие позиции.

        Args:
            order: SELL order for partial close
            close_pct: % of current position to close (e.g., 50.0)

        Returns:
            Position with updated quantity, or None on failure.
        """
        async with self._lock:
            if order.side != Direction.SELL:
                return None

            position = self._positions.get(order.symbol)
            if not position:
                return None

            fill_price = order.fill_price or order.price or 0
            if fill_price <= 0:
                return None

            close_fraction = min(close_pct, 100.0) / 100.0
            close_qty = position.quantity * close_fraction
            remaining_qty = position.quantity - close_qty

            if close_qty <= 0:
                return None

            # Calculate partial PnL — include pro-rata open-side commission
            # so partial_realized_pnl reconciles with real cash flow.
            alloc_open_commission = 0.0
            if position.initial_quantity > 0:
                alloc_open_commission = position.open_commission * (close_qty / position.initial_quantity)
            partial_pnl = (
                (fill_price - position.entry_price) * close_qty
                - order.commission
                - alloc_open_commission
            )
            position.partial_realized_pnl += partial_pnl
            self._total_realized_pnl += partial_pnl

            # Return funds from partial close
            self.wallet.usdt_balance += close_qty * fill_price - order.commission

            # Update position quantity
            position.quantity = remaining_qty
            position.current_price = fill_price
            position.unrealized_pnl = (fill_price - position.entry_price) * remaining_qty

            # Statistics
            self._trades_today += 1
            if partial_pnl > 0:
                self._wins_today += 1
            else:
                self._losses_today += 1

            self._record_equity_snapshot()
            logger.info(
                "Partial close %s: %.0f%% @ %.2f, partial_pnl=%.2f, remaining_qty=%.6f",
                order.symbol, close_pct, fill_price, partial_pnl, remaining_qty,
            )

            # If remaining quantity is negligible, close fully
            if remaining_qty * fill_price < 1.0:  # less than $1 remaining
                return await self._force_full_close(position, fill_price, order)

            return position

    async def _force_full_close(
        self, position: Position, fill_price: float, order: Order,
    ) -> Position:
        """Force-close negligible remaining quantity (commission-aware)."""
        dust_value = position.quantity * fill_price
        dust_commission = dust_value * 0.001  # 0.1% Binance spot commission
        # Allocate the remaining share of entry-side commission to this dust close.
        alloc_open_commission = 0.0
        if position.initial_quantity > 0:
            alloc_open_commission = position.open_commission * (position.quantity / position.initial_quantity)
        dust_pnl = (
            (fill_price - position.entry_price) * position.quantity
            - dust_commission
            - alloc_open_commission
        )
        self.wallet.usdt_balance += dust_value - dust_commission
        self._total_realized_pnl += dust_pnl
        position.partial_realized_pnl += dust_pnl
        position.realized_pnl = position.partial_realized_pnl
        position.quantity = 0
        position.status = PositionStatus.CLOSED
        position.closed_at = str(int(time.time() * 1000))
        position.close_reason = order.signal_reason or "partial_close_dust"
        del self._positions[position.symbol]
        self._sl_tp.pop(position.symbol, None)
        self._trailing.pop(position.symbol, None)
        self._tp_levels.pop(position.symbol, None)
        self._closed_positions.append(position)
        self._record_equity_snapshot()
        await self._event_bus.emit(EVENT_POSITION_CLOSED, position)
        return position

    async def update_price(self, symbol: str, price: float) -> None:
        """Обновить текущую цену позиции (async-safe)."""
        async with self._lock:
            pos = self._positions.get(symbol)
            if pos:
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
                # Update trailing stop max price
                if symbol in self._trailing:
                    act, trail, max_p = self._trailing[symbol]
                    if price > max_p:
                        self._trailing[symbol] = (act, trail, price)
                self._record_equity_snapshot()

    def check_stop_loss_take_profit(self, symbol: str) -> Optional[str]:
        """Проверить SL/TP, multi-stage TP, и trailing stop.

        Returns:
            'stop_loss', 'take_profit', 'tp1_partial', 'tp2_partial',
            'trailing_stop' или None.
        """
        pos = self._positions.get(symbol)
        if not pos:
            return None

        sl, tp = self._sl_tp.get(symbol, (0.0, 0.0))

        # Stop-loss always checked first (highest priority)
        if sl > 0 and pos.current_price <= sl:
            return "stop_loss"

        # Multi-stage TP check (TP1 → TP2 → full TP)
        if symbol in self._tp_levels and pos.entry_price > 0:
            tp1, tp2, tp3 = self._tp_levels[symbol]

            if pos.tp_stage == 0 and tp1 > 0 and pos.current_price >= tp1:
                return "tp1_partial"   # close 50%, move SL to breakeven

            if pos.tp_stage == 1 and tp2 > 0 and pos.current_price >= tp2:
                return "tp2_partial"   # close 30% of remaining

            # Stage 2+: trailing stop takes over for the remaining position
            # Fall through to trailing check below

        # Full take-profit (fallback when TP levels not set)
        if tp > 0 and pos.current_price >= tp:
            return "take_profit"

        # Trailing stop (tick-level)
        if symbol in self._trailing and pos.entry_price > 0:
            activate_pct, trail_pct, max_price = self._trailing[symbol]
            pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100
            if pnl_pct >= activate_pct and max_price > 0:
                drawdown_from_max = (max_price - pos.current_price) / max_price * 100
                if drawdown_from_max >= trail_pct:
                    return "trailing_stop"

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

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss. >1.0 = profitable system."""
        gross_profit = sum(p.realized_pnl for p in self._closed_positions if p.realized_pnl > 0) or 0.0
        gross_loss = abs(sum(p.realized_pnl for p in self._closed_positions if p.realized_pnl < 0)) or 0.001
        return round(gross_profit / gross_loss, 2)

    @property
    def avg_rr_ratio(self) -> float:
        """Average risk:reward ratio from closed positions."""
        ratios = []
        for p in self._closed_positions:
            if p.realized_pnl > 0:
                ratios.append(abs(p.realized_pnl))
        avg_win = sum(ratios) / len(ratios) if ratios else 0.0
        losses = [abs(p.realized_pnl) for p in self._closed_positions if p.realized_pnl < 0]
        avg_loss = sum(losses) / len(losses) if losses else 1.0
        return round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

    @property
    def total_wins(self) -> int:
        return sum(1 for p in self._closed_positions if p.realized_pnl > 0)

    @property
    def total_losses(self) -> int:
        return sum(1 for p in self._closed_positions if p.realized_pnl <= 0)

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
            "profit_factor": self.profit_factor,
            "avg_rr_ratio": self.avg_rr_ratio,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "peak_balance": self._peak_balance,
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
