"""
Telegram-бот SENTINEL — управление и уведомления.

Использует python-telegram-bot (v20+) async API.
Отвечает ТОЛЬКО на сообщения от авторизованного chat_id.
Все команды логируются.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

from core.constants import (
    EVENT_CIRCUIT_BREAKER_TRIGGERED,
    EVENT_EMERGENCY_STOP,
    EVENT_NEW_SIGNAL,
    EVENT_ORDER_FILLED,
    EVENT_POSITION_CLOSED,
    EVENT_RISK_STATE_CHANGED,
    VERSION,
)
from core.events import EventBus
from core.models import Direction, Order, Position, RiskState, Signal

from .formatters import (
    format_config_summary,
    format_daily_report,
    format_error,
    format_order_filled,
    format_positions,
    format_pnl,
    format_risk_state_changed,
    format_signal,
    format_status,
    format_stop_loss,
    format_take_profit,
    format_trades,
)

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram-бот SENTINEL.

    Основные функции:
    - Команды: /status, /pnl, /positions, /trades, /stop, /resume, /kill, /mode, /config, /help
    - Автоматические уведомления: сигналы, ордера, SL/TP, смена Risk State, ошибки
    - Авторизация по chat_id
    - Журналирование всех команд
    """

    def __init__(
        self,
        settings: Settings,
        event_bus: EventBus,
        state_provider: Optional[Callable] = None,
    ) -> None:
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._pin = settings.telegram_pin
        self._event_bus = event_bus
        self._state_provider = state_provider  # callback для получения текущего состояния
        self._app = None
        self._running = False

        # Callback-и для управляющих команд — устанавливаются из main.py
        self.on_stop: Optional[Callable[[], Coroutine]] = None
        self.on_resume: Optional[Callable[[], Coroutine]] = None
        self.on_kill: Optional[Callable[[], Coroutine]] = None
        self.on_mode_change: Optional[Callable[[str], Coroutine]] = None

    @property
    def enabled(self) -> bool:
        return bool(self._token and self._chat_id)

    # ──────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────

    async def start(self) -> None:
        """Запуск бота (polling)."""
        if not self.enabled:
            logger.warning("Telegram bot disabled — token or chat_id not set")
            return

        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            logger.error("python-telegram-bot not installed. Run: pip install python-telegram-bot")
            return

        self._app = Application.builder().token(self._token).build()

        # Регистрация команд
        handlers = {
            "start": self._cmd_start,
            "help": self._cmd_help,
            "status": self._cmd_status,
            "pnl": self._cmd_pnl,
            "positions": self._cmd_positions,
            "trades": self._cmd_trades,
            "stop": self._cmd_stop,
            "resume": self._cmd_resume,
            "kill": self._cmd_kill,
            "mode": self._cmd_mode,
            "config": self._cmd_config,
        }
        for name, handler in handlers.items():
            self._app.add_handler(CommandHandler(name, handler))

        # Подписка на события EventBus
        self._subscribe_events()

        # Запуск polling
        await self._app.initialize()
        await self._app.start()
        if self._app.updater:
            await self._app.updater.start_polling(drop_pending_updates=True)
        self._running = True
        logger.info("Telegram bot started")

    async def stop(self) -> None:
        """Остановка бота."""
        if self._app and self._running:
            self._running = False
            if self._app.updater:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram bot stopped")

    # ──────────────────────────────────────────────
    # Auth
    # ──────────────────────────────────────────────

    def _is_authorized(self, chat_id: int | str) -> bool:
        """Проверка chat_id авторизации."""
        return str(chat_id) == str(self._chat_id)

    # ──────────────────────────────────────────────
    # Send helpers
    # ──────────────────────────────────────────────

    async def send_message(self, text: str) -> None:
        """Отправить сообщение в авторизованный чат."""
        if not self._app or not self.enabled:
            return
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="HTML",
            )
        except Exception as exc:
            logger.error("Failed to send Telegram message: %s", exc)

    async def send_notification(self, text: str) -> None:
        """Алиас для send_message."""
        await self.send_message(text)

    # ──────────────────────────────────────────────
    # Command handlers
    # ──────────────────────────────────────────────

    async def _cmd_start(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /start from %s", update.effective_chat.id)
        await update.message.reply_text(
            f"🤖 <b>SENTINEL v{VERSION}</b>\n\n"
            f"Торговый бот запущен.\n"
            f"Введите /help для списка команд.",
            parse_mode="HTML",
        )

    async def _cmd_help(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /help from %s", update.effective_chat.id)
        text = (
            "📋 <b>Команды SENTINEL</b>\n\n"
            "/status — Статус системы\n"
            "/pnl — PnL за день/неделю/месяц\n"
            "/positions — Открытые позиции\n"
            "/trades — Последние 10 сделок\n"
            "/stop — Остановить торговлю\n"
            "/resume — Возобновить торговлю\n"
            "/kill — ⚠️ Аварийная остановка\n"
            "/mode — Текущий режим (paper/live)\n"
            "/config — Текущие настройки\n"
            "/help — Эта справка"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_status(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /status from %s", update.effective_chat.id)

        state = self._get_state()
        text = format_status(
            mode=state.get("mode", "paper"),
            risk_state=state.get("risk_state", "NORMAL"),
            uptime=state.get("uptime", "N/A"),
            pnl_today=state.get("pnl_today", 0.0),
            pnl_total=state.get("pnl_total", 0.0),
            open_positions=state.get("open_positions", 0),
            trades_today=state.get("trades_today", 0),
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_pnl(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /pnl from %s", update.effective_chat.id)

        state = self._get_state()
        text = format_pnl(
            pnl_day=state.get("pnl_today", 0.0),
            pnl_week=state.get("pnl_week", 0.0),
            pnl_month=state.get("pnl_month", 0.0),
            balance=state.get("balance", 0.0),
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_positions(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /positions from %s", update.effective_chat.id)

        state = self._get_state()
        positions = state.get("positions", [])
        text = format_positions(positions)
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_trades(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /trades from %s", update.effective_chat.id)

        state = self._get_state()
        trades = state.get("recent_trades", [])
        text = format_trades(trades)
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_stop(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /stop from %s", update.effective_chat.id)

        if self.on_stop:
            await self.on_stop()
            await update.message.reply_text("⏸ Торговля остановлена (graceful)", parse_mode="HTML")
        else:
            await update.message.reply_text("⚠️ Stop handler не настроен", parse_mode="HTML")

    async def _cmd_resume(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /resume from %s", update.effective_chat.id)

        if self.on_resume:
            await self.on_resume()
            await update.message.reply_text("▶️ Торговля возобновлена", parse_mode="HTML")
        else:
            await update.message.reply_text("⚠️ Resume handler не настроен", parse_mode="HTML")

    async def _cmd_kill(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /kill from %s", update.effective_chat.id)

        await update.message.reply_text(
            "☠️ <b>АВАРИЙНАЯ ОСТАНОВКА</b>\n\n"
            "Все позиции будут закрыты.\n"
            "Для подтверждения отправьте: /kill_confirm",
            parse_mode="HTML",
        )

    async def _cmd_mode(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /mode from %s", update.effective_chat.id)

        state = self._get_state()
        mode = state.get("mode", "paper")
        await update.message.reply_text(
            f"⚙️ Текущий режим: <b>{mode.upper()}</b>",
            parse_mode="HTML",
        )

    async def _cmd_config(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /config from %s", update.effective_chat.id)

        state = self._get_state()
        config_data = state.get("config_summary", {})
        if not config_data:
            config_data = {"mode": state.get("mode", "paper"), "symbols": "BTCUSDT, ETHUSDT"}
        text = format_config_summary(config_data)
        await update.message.reply_text(text, parse_mode="HTML")

    # ──────────────────────────────────────────────
    # State provider
    # ──────────────────────────────────────────────

    def _get_state(self) -> dict:
        """Получить текущее состояние через callback."""
        if self._state_provider:
            try:
                result = self._state_provider()
                return result if isinstance(result, dict) else {}
            except Exception as exc:
                logger.error("State provider error: %s", exc)
        return {}

    # ──────────────────────────────────────────────
    # EventBus subscriptions
    # ──────────────────────────────────────────────

    def _subscribe_events(self) -> None:
        """Подписать бота на события системы."""
        self._event_bus.subscribe(EVENT_NEW_SIGNAL, self._on_new_signal)
        self._event_bus.subscribe(EVENT_ORDER_FILLED, self._on_order_filled)
        self._event_bus.subscribe(EVENT_RISK_STATE_CHANGED, self._on_risk_state_changed)
        self._event_bus.subscribe(EVENT_EMERGENCY_STOP, self._on_emergency_stop)

    async def _on_new_signal(self, signal: Signal) -> None:
        """Обработка нового сигнала."""
        text = format_signal(signal)
        await self.send_message(text)

    async def _on_order_filled(self, order: Order) -> None:
        """Обработка исполненного ордера."""
        text = format_order_filled(order)
        await self.send_message(text)

    async def _on_risk_state_changed(self, old_state: RiskState, new_state: RiskState, reason: str = "") -> None:
        """Обработка смены Risk State."""
        text = format_risk_state_changed(old_state, new_state, reason)
        await self.send_message(text)

    async def _on_emergency_stop(self, reason: str = "Unknown") -> None:
        """Обработка аварийной остановки."""
        text = format_error(f"EMERGENCY STOP: {reason}")
        await self.send_message(text)
