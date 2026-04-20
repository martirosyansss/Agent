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
    format_diagnostics,
    format_error,
    format_events,
    format_health,
    format_order_filled,
    format_portfolio,
    format_positions,
    format_pnl,
    format_risk_state_changed,
    format_signal,
    format_status,
    format_stop_loss,
    format_take_profit,
    format_trades,
    format_why,
    fmt_pnl,
    fmt_price,
    _esc,
    _now_str,
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
        self.on_manual_close: Optional[Callable[[str], Coroutine]] = None
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
                CallbackQueryHandler,
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
            "start":        self._cmd_start,
            "help":         self._cmd_help,
            "panel":        self._cmd_panel,
            "status":       self._cmd_status,
            "pnl":          self._cmd_pnl,
            "positions":    self._cmd_positions,
            "trades":       self._cmd_trades,
            "portfolio":    self._cmd_portfolio,
            "stop":         self._cmd_stop,
            "resume":       self._cmd_resume,
            "close":        self._cmd_close,
            "kill":         self._cmd_kill,
            "kill_confirm": self._cmd_kill_confirm,
            "mode":         self._cmd_mode,
            "config":       self._cmd_config,
            # Диагностика
            "diag":         self._cmd_diag,
            "why":          self._cmd_why,
            "events":       self._cmd_events,
            "health":       self._cmd_health,
        }
        for name, handler in handlers.items():
            self._app.add_handler(CommandHandler(name, handler))

        # Обработчик нажатий на inline-кнопки
        self._app.add_handler(CallbackQueryHandler(self._on_button))

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
            await asyncio.wait_for(
                self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode="HTML",
                ),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.error("Telegram send_message timed out (10s)")
        except Exception as exc:
            logger.error("Failed to send Telegram message: %s", exc)

    async def send_notification(self, text: str) -> None:
        """Алиас для send_message."""
        await self.send_message(text)

    # ──────────────────────────────────────────────
    # Inline keyboard builder
    # ──────────────────────────────────────────────

    def _build_panel_keyboard(self, trading_paused: bool = False):
        """Построить inline-клавиатуру панели управления."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        trade_btn = (
            InlineKeyboardButton("▶️ Запустить торговлю", callback_data="btn_resume")
            if trading_paused
            else InlineKeyboardButton("⏸ Остановить торговлю", callback_data="btn_stop")
        )

        keyboard = [
            # Строка 1 — информация
            [
                InlineKeyboardButton("📊 Статус",    callback_data="btn_status"),
                InlineKeyboardButton("💰 P&L",        callback_data="btn_pnl"),
                InlineKeyboardButton("📋 Позиции",   callback_data="btn_positions"),
            ],
            # Строка 2 — статистика
            [
                InlineKeyboardButton("📈 Сделки",    callback_data="btn_trades"),
                InlineKeyboardButton("🏆 Portfolio", callback_data="btn_portfolio"),
                InlineKeyboardButton("⚙️ Режим",     callback_data="btn_mode"),
            ],
            # Строка 3 — диагностика
            [
                InlineKeyboardButton("🔍 Диагностика", callback_data="btn_diag"),
                InlineKeyboardButton("❓ Почему",       callback_data="btn_why"),
                InlineKeyboardButton("🩺 Health",       callback_data="btn_health"),
            ],
            # Строка 4 — управление
            [trade_btn],
            # Строка 5 — опасная зона
            [InlineKeyboardButton("☠️ АВАРИЙНАЯ ОСТАНОВКА", callback_data="btn_kill")],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _build_kill_confirm_keyboard(self):
        """Клавиатура подтверждения аварийной остановки."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ ДА, остановить всё", callback_data="btn_kill_confirm"),
                InlineKeyboardButton("❌ Отмена",             callback_data="btn_kill_cancel"),
            ]
        ])

    # ──────────────────────────────────────────────
    # Command handlers
    # ──────────────────────────────────────────────

    async def _cmd_start(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /start from %s", update.effective_chat.id)
        state = self._get_state()
        trading_paused = state.get("trading_paused", False)
        mode = state.get("mode", "paper").upper()
        risk_state = state.get("risk_state", "NORMAL")
        state_icon = {"NORMAL": "🟢", "REDUCED": "🟡", "SAFE": "🟠", "STOP": "🔴"}.get(risk_state, "⚪")

        text = (
            f"🤖 <b>SENTINEL v{VERSION}</b>\n\n"
            f"{state_icon} Режим: <b>{mode}</b>  |  Risk: <b>{risk_state}</b>\n\n"
            f"Нажми кнопку или введи /help для списка команд."
        )
        await update.message.reply_text(
            text,
            parse_mode="HTML",
            reply_markup=self._build_panel_keyboard(trading_paused),
        )

    async def _cmd_help(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /help from %s", update.effective_chat.id)
        text = (
            "📋 <b>Команды SENTINEL</b>\n\n"
            "<b>Панель управления</b>\n"
            "/panel — 🎛️ Панель с кнопками (старт/стоп/статус)\n\n"
            "<b>Информация</b>\n"
            "/status — Статус системы\n"
            "/pnl — PnL за день/неделю/месяц\n"
            "/portfolio — PnL по стратегиям\n"
            "/positions — Открытые позиции\n"
            "/trades — Последние 10 сделок\n"
            "/mode — Текущий режим (paper/live)\n"
            "/config — Текущие настройки\n\n"
            "<b>Диагностика</b>\n"
            "/diag — 🔍 Что бот делает прямо сейчас\n"
            "/why [SYMBOL] — ❓ Почему нет сделок\n"
            "/events [N] — 📜 Последние N событий (по умолчанию 15)\n"
            "/health — 🩺 Здоровье компонентов за последний час\n\n"
            "<b>Управление</b>\n"
            "/stop — ⏸ Остановить торговлю\n"
            "/resume — ▶️ Возобновить торговлю\n"
            "/close SYMBOL — ✕ Закрыть одну позицию вручную\n"
            "/kill — ☠️ Аварийная остановка\n\n"
            "/help — Эта справка"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_panel(self, update, context) -> None:
        """Панель управления с кнопками."""
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /panel from %s", update.effective_chat.id)

        state = self._get_state()
        trading_paused = state.get("trading_paused", False)
        mode = state.get("mode", "paper").upper()
        risk_state = state.get("risk_state", "NORMAL")
        pnl_today = state.get("pnl_today", 0.0)
        balance = state.get("balance", 0.0)
        open_positions = state.get("open_positions", 0)
        state_icon = {"NORMAL": "🟢", "REDUCED": "🟡", "SAFE": "🟠", "STOP": "🔴"}.get(risk_state, "⚪")
        trade_status = "⏸ Торговля остановлена" if trading_paused else "▶️ Торговля активна"
        pnl_icon = "📈" if pnl_today >= 0 else "📉"

        text = (
            f"🎛️ <b>Панель управления SENTINEL</b>\n"
            f"⏰ {_now_str()}\n\n"
            f"{state_icon} Risk State: <b>{risk_state}</b>\n"
            f"⚙️ Режим: <b>{mode}</b>\n"
            f"{trade_status}\n\n"
            f"💵 P&L сегодня: <b>{fmt_pnl(pnl_today)}</b>  {pnl_icon}\n"
            f"📋 Открытых позиций: <b>{open_positions}</b>\n"
        )
        if balance > 0:
            text += f"💰 Баланс: <b>${balance:,.2f}</b>\n"

        await update.message.reply_text(
            text,
            parse_mode="HTML",
            reply_markup=self._build_panel_keyboard(trading_paused),
        )

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
            balance=state.get("balance", 0.0),
            win_rate_today=state.get("win_rate", 0.0),
            wins_today=state.get("total_wins", 0),
            losses_today=state.get("total_losses", 0),
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
            pnl_total=state.get("pnl_total", 0.0),
            trades_day=state.get("trades_today", 0),
            win_rate_day=state.get("win_rate", 0.0),
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_positions(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /positions from %s", update.effective_chat.id)

        state = self._get_state()
        positions = state.get("positions", [])
        text = format_positions(positions)
        reply_markup = self._build_close_keyboard(positions) if positions else None
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=reply_markup)

    def _build_close_keyboard(self, positions):
        """Клавиатура со строкой на позицию: кнопка ✕ Закрыть <symbol>."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        rows = []
        for p in positions:
            sym = getattr(p, "symbol", None) or (p.get("symbol") if isinstance(p, dict) else None)
            if not sym:
                continue
            rows.append([
                InlineKeyboardButton(f"✕ Закрыть {sym}", callback_data=f"btn_close:{sym}"),
            ])
        return InlineKeyboardMarkup(rows) if rows else None

    async def _cmd_close(self, update, context) -> None:
        """Закрыть позицию по символу: /close BTCUSDT"""
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /close from %s args=%s", update.effective_chat.id, context.args)

        if not context.args:
            await update.message.reply_text(
                "Использование: <code>/close SYMBOL</code>\n"
                "Пример: <code>/close BTCUSDT</code>",
                parse_mode="HTML",
            )
            return
        symbol = context.args[0].upper().strip()
        await self._prompt_close_confirm(update.message.reply_text, symbol)

    async def _prompt_close_confirm(self, reply_fn, symbol: str) -> None:
        """Показать подтверждение закрытия позиции."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Да, закрыть", callback_data=f"btn_close_confirm:{symbol}"),
            InlineKeyboardButton("❌ Отмена",      callback_data="btn_close_cancel"),
        ]])
        await reply_fn(
            f"✕ <b>Закрыть позицию {_esc(symbol)}?</b>\n"
            f"Ордер уйдёт на исполнение немедленно по рыночной цене.",
            parse_mode="HTML",
            reply_markup=keyboard,
        )

    async def _cmd_trades(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /trades from %s", update.effective_chat.id)

        state = self._get_state()
        trades = state.get("recent_trades", [])
        text = format_trades(trades)
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_portfolio(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /portfolio from %s", update.effective_chat.id)

        state = self._get_state()
        perf = state.get("strategy_performance", [])
        balance = state.get("balance", 0.0)
        text = format_portfolio(perf, balance)
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

    async def _cmd_kill_confirm(self, update, context) -> None:
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /kill_confirm from %s", update.effective_chat.id)

        if self.on_kill:
            await update.message.reply_text("☠️ Выполняю аварийную остановку...", parse_mode="HTML")
            await self.on_kill()
            await update.message.reply_text("☠️ <b>SENTINEL ОСТАНОВЛЕН</b>\nВсе позиции закрыты.", parse_mode="HTML")
        else:
            await update.message.reply_text("⚠️ Kill handler не настроен", parse_mode="HTML")

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
    # Диагностические команды
    # ──────────────────────────────────────────────

    def _recent_events(self, limit: int = 200) -> list[dict]:
        """Pull recent events from the process-wide EventLog.

        Uses the buffered in-memory window (последние N) — не читает
        файл с диска, чтобы команда отвечала за <50ms даже когда
        ``events.jsonl`` на сотни мегабайт.
        """
        try:
            from monitoring.event_log import get_event_log
            return get_event_log().recent_events(limit=limit)
        except Exception as exc:
            logger.warning("Failed to pull recent events: %s", exc)
            return []

    async def _cmd_diag(self, update, context) -> None:
        """Полная диагностика: что бот делает прямо сейчас и почему."""
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /diag from %s", update.effective_chat.id)

        state = self._get_state()
        text = format_diagnostics(state)
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_why(self, update, context) -> None:
        """Почему нет сделок — агрегирует rejections и блокирующие чекеры."""
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /why from %s args=%s", update.effective_chat.id, context.args)

        symbol = None
        if context.args:
            symbol = context.args[0].upper().strip()

        state = self._get_state()
        events = self._recent_events(limit=500)
        text = format_why(state, events, symbol=symbol)
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_events(self, update, context) -> None:
        """Tail последних N событий (по умолчанию 15, макс 50)."""
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /events from %s args=%s", update.effective_chat.id, context.args)

        limit = 15
        if context.args:
            try:
                limit = max(1, min(50, int(context.args[0])))
            except (ValueError, TypeError):
                pass

        events = self._recent_events(limit=max(limit * 2, 100))
        text = format_events(events, limit=limit)
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_health(self, update, context) -> None:
        """Здоровье: ошибки компонентов и guards за последний час."""
        if not self._is_authorized(update.effective_chat.id):
            return
        logger.info("CMD /health from %s", update.effective_chat.id)

        state = self._get_state()
        events = self._recent_events(limit=1000)
        text = format_health(state, events)
        await update.message.reply_text(text, parse_mode="HTML")

    # ──────────────────────────────────────────────
    # Inline button handler
    # ──────────────────────────────────────────────

    async def _on_button(self, update, context) -> None:
        """Обработка нажатий на inline-кнопки."""
        query = update.callback_query
        if not self._is_authorized(query.message.chat_id):
            await query.answer("⛔ Нет доступа", show_alert=True)
            return

        data = query.data
        logger.info("BUTTON %s from %s", data, query.message.chat_id)

        # Подтверждаем нажатие (убираем часики у кнопки)
        await query.answer()

        if data == "btn_status":
            state = self._get_state()
            text = format_status(
                mode=state.get("mode", "paper"),
                risk_state=state.get("risk_state", "NORMAL"),
                uptime=state.get("uptime", "N/A"),
                pnl_today=state.get("pnl_today", 0.0),
                pnl_total=state.get("pnl_total", 0.0),
                open_positions=state.get("open_positions", 0),
                trades_today=state.get("trades_today", 0),
                balance=state.get("balance", 0.0),
                win_rate_today=state.get("win_rate", 0.0),
                wins_today=state.get("total_wins", 0),
                losses_today=state.get("total_losses", 0),
            )
            await query.message.reply_text(text, parse_mode="HTML")

        elif data == "btn_pnl":
            state = self._get_state()
            text = format_pnl(
                pnl_day=state.get("pnl_today", 0.0),
                pnl_week=state.get("pnl_week", 0.0),
                pnl_month=state.get("pnl_month", 0.0),
                balance=state.get("balance", 0.0),
                pnl_total=state.get("pnl_total", 0.0),
                trades_day=state.get("trades_today", 0),
                win_rate_day=state.get("win_rate", 0.0),
            )
            await query.message.reply_text(text, parse_mode="HTML")

        elif data == "btn_positions":
            state = self._get_state()
            text = format_positions(state.get("positions", []))
            await query.message.reply_text(text, parse_mode="HTML")

        elif data == "btn_trades":
            state = self._get_state()
            text = format_trades(state.get("recent_trades", []))
            await query.message.reply_text(text, parse_mode="HTML")

        elif data == "btn_portfolio":
            state = self._get_state()
            text = format_portfolio(state.get("strategy_performance", []), state.get("balance", 0.0))
            await query.message.reply_text(text, parse_mode="HTML")

        elif data == "btn_mode":
            state = self._get_state()
            mode = state.get("mode", "paper").upper()
            mode_icon = "📄" if mode == "PAPER" else "💰"
            await query.message.reply_text(
                f"⚙️ Текущий режим: {mode_icon} <b>{mode}</b>",
                parse_mode="HTML",
            )

        elif data == "btn_diag":
            state = self._get_state()
            await query.message.reply_text(
                format_diagnostics(state), parse_mode="HTML",
            )

        elif data == "btn_why":
            state = self._get_state()
            events = self._recent_events(limit=500)
            await query.message.reply_text(
                format_why(state, events), parse_mode="HTML",
            )

        elif data == "btn_health":
            state = self._get_state()
            events = self._recent_events(limit=1000)
            await query.message.reply_text(
                format_health(state, events), parse_mode="HTML",
            )

        elif data == "btn_stop":
            if self.on_stop:
                await self.on_stop()
                state = self._get_state()
                trading_paused = state.get("trading_paused", True)
                # Обновляем кнопку панели
                await query.edit_message_reply_markup(
                    reply_markup=self._build_panel_keyboard(trading_paused)
                )
                await query.message.reply_text(
                    "⏸ <b>Торговля остановлена</b>\n"
                    "Новые сделки не будут открываться.\n"
                    "Нажми ▶️ чтобы возобновить.",
                    parse_mode="HTML",
                )
            else:
                await query.message.reply_text("⚠️ Stop handler не настроен", parse_mode="HTML")

        elif data == "btn_resume":
            if self.on_resume:
                await self.on_resume()
                state = self._get_state()
                trading_paused = state.get("trading_paused", False)
                await query.edit_message_reply_markup(
                    reply_markup=self._build_panel_keyboard(trading_paused)
                )
                await query.message.reply_text(
                    "▶️ <b>Торговля возобновлена</b>\n"
                    "Бот снова ищет сигналы.",
                    parse_mode="HTML",
                )
            else:
                await query.message.reply_text("⚠️ Resume handler не настроен", parse_mode="HTML")

        elif data == "btn_kill":
            await query.message.reply_text(
                "☠️ <b>АВАРИЙНАЯ ОСТАНОВКА</b>\n\n"
                "Все открытые позиции будут <b>немедленно закрыты</b>.\n"
                "Это действие нельзя отменить!\n\n"
                "Вы уверены?",
                parse_mode="HTML",
                reply_markup=self._build_kill_confirm_keyboard(),
            )

        elif data == "btn_kill_confirm":
            if self.on_kill:
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text("☠️ Выполняю аварийную остановку...", parse_mode="HTML")
                await self.on_kill()
                await query.message.reply_text(
                    "☠️ <b>SENTINEL ОСТАНОВЛЕН</b>\nВсе позиции закрыты.",
                    parse_mode="HTML",
                )
            else:
                await query.message.reply_text("⚠️ Kill handler не настроен", parse_mode="HTML")

        elif data == "btn_kill_cancel":
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("✅ Отменено. Торговля продолжается.", parse_mode="HTML")

        elif data.startswith("btn_close:"):
            symbol = data.split(":", 1)[1]
            await self._prompt_close_confirm(query.message.reply_text, symbol)

        elif data.startswith("btn_close_confirm:"):
            symbol = data.split(":", 1)[1]
            await query.edit_message_reply_markup(reply_markup=None)
            if not self.on_manual_close:
                await query.message.reply_text(
                    "⚠️ Manual close handler не настроен", parse_mode="HTML",
                )
                return
            await query.message.reply_text(
                f"⏳ Закрываю позицию <b>{_esc(symbol)}</b>...",
                parse_mode="HTML",
            )
            try:
                result = await self.on_manual_close(symbol)
            except Exception as exc:
                logger.exception("manual close failed for %s", symbol)
                await query.message.reply_text(
                    f"❌ Ошибка закрытия {_esc(symbol)}: {_esc(str(exc))}",
                    parse_mode="HTML",
                )
                return
            if isinstance(result, dict) and not result.get("ok"):
                await query.message.reply_text(
                    f"⚠️ Не удалось закрыть {_esc(symbol)}: "
                    f"{_esc(result.get('error', 'unknown error'))}",
                    parse_mode="HTML",
                )
            # Успешное закрытие: подробное сообщение придёт через _on_position_closed

        elif data == "btn_close_cancel":
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("✅ Отменено. Позиция не закрыта.", parse_mode="HTML")

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
        self._event_bus.subscribe(EVENT_POSITION_CLOSED, self._on_position_closed)
        self._event_bus.subscribe("ml_drift_detected", self._on_ml_drift)

    async def _on_ml_drift(self, payload: dict) -> None:
        """Alert when ML precision drops significantly below training baseline."""
        try:
            model = payload.get("model", "?")
            train = payload.get("train_precision", 0.0)
            live = payload.get("live_precision", 0.0)
            n = payload.get("n_pred_win", 0)
            version = payload.get("version", "")
            text = (
                "⚠️ <b>ML drift detected</b>\n\n"
                f"Model: <code>{model}</code>\n"
                f"Version: <code>{version}</code>\n"
                f"Train precision: <b>{train:.3f}</b>\n"
                f"Live precision: <b>{live:.3f}</b> (n={n})\n\n"
                "Consider retraining via /retrain_ml or inspecting recent trades."
            )
            await self.send_message(text)
        except Exception as exc:
            logger.warning("ML drift alert formatting failed: %s", exc)

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

    async def _on_position_closed(self, position: Position) -> None:
        """Подробное уведомление о закрытии позиции."""
        pnl = position.realized_pnl
        pnl_icon = "💰" if pnl >= 0 else "💸"
        result_word = "ПРИБЫЛЬ" if pnl >= 0 else "УБЫТОК"
        mode = "📄 Paper" if position.is_paper else "💰 LIVE"

        pnl_pct = 0.0
        if position.entry_price > 0 and position.quantity > 0:
            pnl_pct = pnl / (position.entry_price * position.quantity) * 100

        # Определяем иконку причины закрытия
        close_reason = position.close_reason or ""
        if "stop_loss" in close_reason:
            close_icon = "🛑"
            close_label = "Stop-Loss сработал"
        elif "take_profit" in close_reason:
            close_icon = "🎯"
            close_label = "Take-Profit достигнут"
        elif "trailing_stop" in close_reason:
            close_icon = "📉"
            close_label = "Trailing Stop сработал"
        elif "kill" in close_reason.lower() or "emergency" in close_reason.lower():
            close_icon = "☠️"
            close_label = "Аварийная остановка"
        elif close_reason:
            close_icon = "📋"
            close_label = close_reason
        else:
            close_icon = "📋"
            close_label = "Сигнал на закрытие"

        lines = [
            f"{pnl_icon} <b>ПОЗИЦИЯ ЗАКРЫТА — {_esc(position.symbol)}</b>",
            f"⏰ {_now_str()}  |  {mode}",
            "",
            f"{close_icon} <b>Причина: {_esc(close_label)}</b>",
            "",
            f"📊 <b>Результат: {result_word}</b>",
            f"  Реализованный P&amp;L: <b>{fmt_pnl(pnl)}</b>  ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)",
            f"  Цена входа:   <b>{fmt_price(position.entry_price)}</b>",
            f"  Цена выхода:  <b>{fmt_price(position.current_price)}</b>",
            f"  Сторона:      <b>{_esc(position.side)}</b>",
            f"  Количество:   <b>{position.quantity:.6f}</b>",
        ]

        if position.strategy_name:
            lines += ["", f"⚙️ Стратегия: <b>{_esc(position.strategy_name)}</b>"]
        if position.signal_reason:
            lines.append(f"💬 Открыта по: <i>{_esc(position.signal_reason)}</i>")
        if position.opened_at:
            lines.append(f"⏱️ Открыта: {_esc(position.opened_at)}")

        # Получаем актуальный P&L за день из state_provider
        state = self._get_state()
        pnl_today = state.get("pnl_today", 0.0)
        trades_today = state.get("trades_today", 0)
        balance = state.get("balance", 0.0)
        wins = state.get("total_wins", 0)
        losses_count = state.get("total_losses", 0)

        if trades_today > 0 or balance > 0:
            pnl_today_icon = "📈" if pnl_today >= 0 else "📉"
            lines += [
                "",
                f"📅 <b>Итог за сегодня</b>",
                f"  P&amp;L дня:  <b>{fmt_pnl(pnl_today)}</b>  {pnl_today_icon}",
            ]
            if trades_today > 0:
                wr = wins / (wins + losses_count) * 100 if (wins + losses_count) > 0 else 0
                lines.append(f"  Сделок:   <b>{trades_today}</b>  (W:{wins} / L:{losses_count})  WR:{wr:.0f}%")
            if balance > 0:
                lines.append(f"  Баланс:   <b>${balance:,.2f}</b>")

        lines += ["", f"🔖 Position ID: <code>{position.position_id}</code>"]
        await self.send_message("\n".join(lines))
