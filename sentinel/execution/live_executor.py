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

from core.constants import EVENT_ORDER_FILLED, EVENT_EXECUTION_DEGRADED
from core.events import EventBus
from core.models import Direction, Order, OrderStatus, OrderType, Signal
from monitoring.event_log import emit_component_error

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
            logger.error("Live execution error for %s: %s — verifying order status", signal.symbol, e)
            # Critical: verify if order was partially or fully filled on exchange
            orphan = await self._check_recent_order(signal.symbol, signal.direction)
            if not orphan:
                emit_component_error(
                    "live_executor.execute_order",
                    f"execution failed for {signal.symbol}: {e}",
                    exc=e,
                    severity="error",
                    symbol=signal.symbol,
                    direction=signal.direction.value,
                    orphan_detected=False,
                )
                return None
            # Orphan fill exists on exchange — reconstruct Order so the in-memory
            # position tracks reality, then try to attach protective OCO. If a
            # matching protective order is already live (e.g. the original OCO
            # placement succeeded before the caller's exception), skip re-placing
            # to avoid duplicates.
            logger.critical(
                "ORPHAN ORDER DETECTED: %s %s filled on exchange despite error — recovering. Order: %s",
                signal.direction.value, signal.symbol, orphan,
            )
            recovered = self._order_from_orphan(signal, orphan, current_price)
            oco_live = False
            if signal.direction == Direction.BUY and signal.stop_loss_price > 0:
                oco_live = self._has_live_protective_oco(signal.symbol, signal.signal_id)
                if not oco_live:
                    oco_live = await self._place_protective_oco(
                        signal.symbol, recovered.fill_quantity or quantity, recovered.fill_price or current_price,
                        signal.stop_loss_price, signal.take_profit_price,
                        signal_id=signal.signal_id,
                    )
                if not oco_live:
                    logger.critical("Orphan recovery: OCO failed, emergency sell for %s", signal.symbol)
                    emg_ok = await self._emergency_sell(
                        signal.symbol, recovered.fill_quantity or quantity, signal_id=signal.signal_id,
                    )
                    emit_component_error(
                        "live_executor.execute_order",
                        f"orphan recovery: OCO missing, emergency_sell={'ok' if emg_ok else 'failed'} for {signal.symbol}",
                        severity="critical",
                        symbol=signal.symbol,
                        direction=signal.direction.value,
                        orphan_detected=True,
                        orphan_recovered=False,
                        emergency_sell_ok=emg_ok,
                        exchange_order_id=recovered.exchange_order_id,
                    )
                    await self._event_bus.emit(
                        EVENT_EXECUTION_DEGRADED,
                        {"symbol": signal.symbol,
                         "reason": "orphan_unrecoverable" if not emg_ok else "orphan_flattened",
                         "exchange_order_id": recovered.exchange_order_id},
                    )
                    if emg_ok:
                        recovered.status = OrderStatus.CANCELLED
                        return recovered
                    # Position still live and unprotected — surface via FILLED
                    # so in-memory state matches exchange; main halts trading.
                    await self._event_bus.emit(EVENT_ORDER_FILLED, recovered)
                    return recovered
            emit_component_error(
                "live_executor.execute_order",
                f"orphan recovered for {signal.symbol}: fill={recovered.fill_quantity} @ {recovered.fill_price}, oco={'existing' if oco_live else 'placed'}",
                exc=e,
                severity="critical",
                symbol=signal.symbol,
                direction=signal.direction.value,
                orphan_detected=True,
                orphan_recovered=True,
                exchange_order_id=recovered.exchange_order_id,
            )
            # Emit to EventBus so _on_order_filled persists + opens the position.
            await self._event_bus.emit(EVENT_ORDER_FILLED, recovered)
            return recovered

    async def _check_recent_order(self, symbol: str, direction: Direction) -> Optional[dict]:
        """Check if a recent order was filled on exchange despite local error.

        Scans the last 20 orders (covers bursts during retries / reconnects)
        within a 2-minute window and matches by side, so we catch orphans even
        when multiple orders followed in quick succession.
        """
        if not self._client:
            return None
        try:
            orders = self._client.get_all_orders(symbol=symbol, limit=20)
            if not orders:
                return None
            now_ms = time.time() * 1000
            want_side = "BUY" if direction == Direction.BUY else "SELL"
            # Iterate newest → oldest; stop at first match within window.
            for entry in reversed(orders):
                status = entry.get("status")
                order_side = entry.get("side")
                ts = float(entry.get("time", 0))
                if (
                    status in ("FILLED", "PARTIALLY_FILLED")
                    and order_side == want_side
                    and abs(now_ms - ts) < 120_000  # 2-minute window
                ):
                    return entry
        except Exception as check_err:
            logger.error("Failed to verify order status: %s", check_err)
        return None

    def _order_from_orphan(
        self,
        signal: Signal,
        orphan: dict,
        fallback_price: float,
    ) -> Order:
        """Reconstruct an Order from a Binance ``get_all_orders`` entry.

        ``get_all_orders`` omits ``fills``, so VWAP = cummulativeQuoteQty /
        executedQty. Falls back to ``fallback_price`` if either field is zero.
        """
        executed_qty = float(orphan.get("executedQty", 0) or 0)
        cq_qty = float(orphan.get("cummulativeQuoteQty", 0) or 0)
        fill_price = (cq_qty / executed_qty) if executed_qty > 0 and cq_qty > 0 else float(orphan.get("price") or fallback_price)
        exchange_id = str(orphan.get("orderId", ""))
        commission = self.calculate_commission(executed_qty, fill_price)
        return Order(
            timestamp=int(orphan.get("time") or time.time() * 1000),
            symbol=signal.symbol,
            side=signal.direction,
            order_type=OrderType.MARKET,
            quantity=float(orphan.get("origQty") or executed_qty),
            price=fallback_price,
            status=OrderStatus.FILLED,
            exchange_order_id=exchange_id,
            fill_price=fill_price,
            fill_quantity=executed_qty,
            commission=commission,
            is_paper=False,
            signal_id=signal.signal_id,
            strategy_name=signal.strategy_name,
            signal_reason=signal.reason,
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
            features=signal.features,
        )

    def _has_live_protective_oco(self, symbol: str, signal_id: str) -> bool:
        """Return True if the deterministic SL/TP leg for this signal is open.

        Prevents duplicate OCO placement during orphan recovery. Matches the
        client order IDs produced by ``_place_protective_oco``.
        """
        if not self._client or not signal_id:
            return False
        want_sl = f"s{signal_id}-sl"[:36]
        want_tp = f"s{signal_id}-tp"[:36]
        try:
            open_orders = self._client.get_open_orders(symbol=symbol)
            for o in open_orders or []:
                cid = o.get("clientOrderId", "")
                if cid in (want_sl, want_tp):
                    return True
        except Exception as check_err:
            logger.error("Failed to list open orders for %s: %s", symbol, check_err)
        return False

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
        # newClientOrderId: deterministic per-signal ID so retries on timeout
        # don't create duplicate fills on Binance.
        entry_client_id = f"s{signal.signal_id}-ent"[:36]
        try:
            result = self._client.create_order(
                symbol=signal.symbol,
                side=side_str,
                type="MARKET",
                quantity=f"{quantity:.8f}",
                newClientOrderId=entry_client_id,
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
            features=signal.features,
        )

        # --- Step 2: Protective OCO order (only for BUY entries) ---
        if signal.direction == Direction.BUY and signal.stop_loss_price > 0:
            oco_ok = await self._place_protective_oco(
                signal.symbol, fill_qty, fill_price,
                signal.stop_loss_price, signal.take_profit_price,
                signal_id=signal.signal_id,
            )
            if not oco_ok:
                # Entry already filled on exchange. Try to exit. If that fails
                # too, the position is LIVE without SL/TP — we must surface
                # this (return the filled order + critical event) so callers
                # halt trading instead of quietly forgetting the exposure.
                logger.error("CRITICAL: Protective OCO failed, emergency sell!")
                emg_ok = await self._emergency_sell(signal.symbol, fill_qty, signal_id=signal.signal_id)
                if emg_ok:
                    emit_component_error(
                        "live_executor.execute_order",
                        f"OCO failed, emergency_sell OK for {signal.symbol} (position flattened)",
                        severity="critical",
                        symbol=signal.symbol,
                        exchange_order_id=exchange_id,
                        oco_ok=False,
                        emergency_sell_ok=True,
                    )
                    order.status = OrderStatus.CANCELLED
                    return order
                # Position is still live on the exchange, unprotected.
                emit_component_error(
                    "live_executor.execute_order",
                    f"OCO failed AND emergency_sell failed for {signal.symbol} — UNPROTECTED LIVE POSITION",
                    severity="critical",
                    symbol=signal.symbol,
                    exchange_order_id=exchange_id,
                    oco_ok=False,
                    emergency_sell_ok=False,
                    unprotected=True,
                )
                await self._event_bus.emit(
                    EVENT_EXECUTION_DEGRADED,
                    {"symbol": signal.symbol, "reason": "unprotected_fill",
                     "exchange_order_id": exchange_id},
                )
                # Fall through so the event bus fires and the position is
                # tracked in memory; caller is expected to halt trading.

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
        signal_id: str = "",
    ) -> bool:
        """Разместить OCO protective order на бирже."""
        if take_profit <= 0:
            take_profit = entry_price * 1.05  # Default 5% TP

        # Deterministic client IDs for the two legs of the OCO — prevents
        # duplicate protective orders on retry.
        limit_client_id = f"s{signal_id}-tp"[:36] if signal_id else None
        stop_client_id = f"s{signal_id}-sl"[:36] if signal_id else None

        try:
            kwargs: dict = dict(
                symbol=symbol,
                side="SELL",
                quantity=f"{quantity:.8f}",
                price=f"{take_profit:.2f}",
                stopPrice=f"{stop_loss:.2f}",
                stopLimitPrice=f"{stop_loss * 0.999:.2f}",
                stopLimitTimeInForce="GTC",
            )
            if limit_client_id and stop_client_id:
                kwargs["limitClientOrderId"] = limit_client_id
                kwargs["stopClientOrderId"] = stop_client_id
            self._client.create_oco_order(**kwargs)
            logger.info("OCO placed: SL=%.2f TP=%.2f", stop_loss, take_profit)
            return True
        except Exception as e:
            logger.error("OCO order failed: %s", e)
            return False

    async def reconcile_with_exchange(self, symbols: list[str]) -> dict:
        """Compare exchange state with provided symbols on startup.

        For each symbol returns whether a protective order exists and what
        the current base-asset free balance is. Used by main.py at boot to
        detect positions that survived (or died) during downtime. Does NOT
        mutate state — detection only, logging + events handled by caller.
        """
        if not self._init_client():
            return {}
        summary: dict[str, dict] = {}
        for symbol in symbols:
            entry: dict = {"open_orders": [], "has_protective_oco": False}
            try:
                open_orders = self._client.get_open_orders(symbol=symbol) or []
                entry["open_orders"] = [
                    {"orderId": o.get("orderId"),
                     "clientOrderId": o.get("clientOrderId"),
                     "type": o.get("type"),
                     "side": o.get("side"),
                     "origQty": o.get("origQty")}
                    for o in open_orders
                ]
                entry["has_protective_oco"] = any(
                    (o.get("clientOrderId", "").endswith("-sl")
                     or o.get("clientOrderId", "").endswith("-tp"))
                    for o in open_orders
                )
            except Exception as e:
                logger.error("reconcile: get_open_orders %s failed: %s", symbol, e)
                entry["error"] = str(e)
                emit_component_error(
                    "live_executor.reconcile",
                    f"reconcile open orders failed for {symbol}: {e}",
                    exc=e, severity="warning", symbol=symbol,
                )
            summary[symbol] = entry
        return summary

    async def cancel_all_open_orders(self, symbols: list[str]) -> int:
        """Cancel every open order (entry/OCO/SL/TP legs) on each symbol.

        Best-effort: a failure on one symbol does not stop the sweep for
        the others. Used by the Kill Switch so orphan protective orders
        don't linger on the exchange after an emergency stop.
        """
        if not self._init_client():
            return 0
        cancelled = 0
        for symbol in symbols:
            try:
                open_orders = self._client.get_open_orders(symbol=symbol)
            except Exception as e:
                logger.error("cancel_all: list open orders failed for %s: %s", symbol, e)
                emit_component_error(
                    "live_executor.cancel_all_open_orders",
                    f"list open orders failed for {symbol}: {e}",
                    exc=e, severity="error", symbol=symbol,
                )
                continue
            for entry in open_orders or []:
                oid = entry.get("orderId")
                if oid is None:
                    continue
                try:
                    self._client.cancel_order(symbol=symbol, orderId=oid)
                    cancelled += 1
                except Exception as e:
                    logger.error("cancel_all: cancel %s#%s failed: %s", symbol, oid, e)
                    emit_component_error(
                        "live_executor.cancel_all_open_orders",
                        f"cancel {symbol}#{oid} failed: {e}",
                        exc=e, severity="error", symbol=symbol, order_id=oid,
                    )
        logger.warning("cancel_all_open_orders: cancelled=%d symbols=%d", cancelled, len(symbols))
        return cancelled

    async def _emergency_sell(self, symbol: str, quantity: float, signal_id: str = "") -> bool:
        """Аварийная продажа — market sell без retry.

        Returns True if the sell was accepted by the exchange, False otherwise.
        Callers MUST treat False as "entry position is still live on exchange".
        """
        try:
            kwargs: dict = dict(
                symbol=symbol,
                side="SELL",
                type="MARKET",
                quantity=f"{quantity:.8f}",
            )
            if signal_id:
                kwargs["newClientOrderId"] = f"s{signal_id}-emg"[:36]
            self._client.create_order(**kwargs)
            logger.warning("Emergency sell executed: %s qty=%.6f", symbol, quantity)
            return True
        except Exception as e:
            logger.critical("EMERGENCY SELL FAILED: %s - %s", symbol, e)
            emit_component_error(
                "live_executor.emergency_sell",
                f"emergency sell failed for {symbol}: {e}",
                exc=e,
                severity="critical",
                symbol=symbol,
                quantity=quantity,
                signal_id=signal_id,
            )
            return False
