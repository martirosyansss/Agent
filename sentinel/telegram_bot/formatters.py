"""
Форматирование сообщений для Telegram.

Каждый метод возвращает готовую строку с Markdown-разметкой (MarkdownV2 НЕ используем —
обычный Markdown для простоты и совместимости с parse_mode="HTML").
"""

from __future__ import annotations

from typing import Optional

from core.models import Direction, Order, Position, RiskState, Signal


def fmt_price(value: float) -> str:
    """Две десятичных если < $10, иначе до доллара."""
    if value < 10:
        return f"${value:,.4f}"
    if value < 1000:
        return f"${value:,.2f}"
    return f"${value:,.2f}"


def fmt_pct(value: float) -> str:
    """Процент со знаком."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def fmt_pnl(value: float) -> str:
    """PnL со знаком."""
    if value >= 0:
        return f"+${value:.2f}"
    return f"-${abs(value):.2f}"


# ──────────────────────────────────────────────
# Автоматические уведомления
# ──────────────────────────────────────────────


def format_signal(signal: Signal) -> str:
    """Уведомление о новом сигнале."""
    icon = "📈" if signal.direction == Direction.BUY else "📉"
    action = "BUY" if signal.direction == Direction.BUY else "SELL"

    lines = [
        f"{icon} <b>СИГНАЛ {action} {signal.symbol}</b>",
        "",
        f"Цена: {fmt_price(signal.stop_loss_price if signal.direction == Direction.SELL else signal.take_profit_price) if False else fmt_price(0)}",
        f"Confidence: <b>{signal.confidence:.2f}</b>",
        f"Стратегия: {signal.strategy_name}",
        "",
        f"Причина: {signal.reason}",
    ]

    if signal.stop_loss_price > 0:
        lines.append(f"Stop-Loss: {fmt_price(signal.stop_loss_price)}")
    if signal.take_profit_price > 0:
        lines.append(f"Take-Profit: {fmt_price(signal.take_profit_price)}")

    return "\n".join(lines)


def format_order_filled(order: Order) -> str:
    """Уведомление об исполненном ордере."""
    icon = "✅"
    action = "КУПЛЕНО" if order.side == Direction.BUY else "ПРОДАНО"
    mode = "Paper" if order.is_paper else "LIVE"
    price = fmt_price(order.fill_price or order.price or 0)

    return (
        f"{icon} <b>{action}</b> {order.fill_quantity or order.quantity:.6f} "
        f"{order.symbol} @ {price} ({mode})"
    )


def format_stop_loss(position: Position, loss: float) -> str:
    """Уведомление о срабатывании stop-loss."""
    return (
        f"🛑 <b>STOP-LOSS</b> {position.symbol} @ "
        f"{fmt_price(position.current_price)} "
        f"(потеря: {fmt_pnl(loss)})"
    )


def format_take_profit(position: Position, profit: float) -> str:
    """Уведомление о срабатывании take-profit."""
    return (
        f"🎯 <b>TAKE-PROFIT</b> {position.symbol} @ "
        f"{fmt_price(position.current_price)} "
        f"(прибыль: {fmt_pnl(profit)})"
    )


def format_risk_state_changed(old_state: RiskState, new_state: RiskState, reason: str) -> str:
    """Уведомление о смене Risk State."""
    return (
        f"⚠️ Risk State: <b>{old_state.value} → {new_state.value}</b>\n"
        f"Причина: {reason}"
    )


def format_error(message: str) -> str:
    """Уведомление об ошибке."""
    return f"🚨 <b>ОШИБКА:</b> {message}"


def format_daily_report(
    pnl: float,
    win_rate: float,
    trades_count: int,
    wins: int,
    losses: int,
) -> str:
    """Ежедневный отчёт."""
    return (
        f"📊 <b>Итоги дня</b>\n\n"
        f"PnL: {fmt_pnl(pnl)}\n"
        f"Win Rate: {win_rate:.0f}%\n"
        f"Сделок: {trades_count} (Win: {wins}, Loss: {losses})"
    )


# ──────────────────────────────────────────────
# Ответы на команды
# ──────────────────────────────────────────────


def format_status(
    mode: str,
    risk_state: str,
    uptime: str,
    pnl_today: float,
    pnl_total: float,
    open_positions: int,
    trades_today: int,
) -> str:
    """Ответ на /status."""
    state_icon = {
        "NORMAL": "🟢",
        "REDUCED": "🟡",
        "SAFE": "🟠",
        "STOP": "🔴",
    }.get(risk_state, "⚪")

    return (
        f"{state_icon} <b>SENTINEL — {mode.upper()}</b>\n\n"
        f"Risk State: {risk_state}\n"
        f"Uptime: {uptime}\n\n"
        f"PnL сегодня: {fmt_pnl(pnl_today)}\n"
        f"PnL всего: {fmt_pnl(pnl_total)}\n"
        f"Открытых позиций: {open_positions}\n"
        f"Сделок сегодня: {trades_today}"
    )


def format_pnl(
    pnl_day: float,
    pnl_week: float,
    pnl_month: float,
    balance: float,
) -> str:
    """Ответ на /pnl."""
    return (
        f"💰 <b>PnL Report</b>\n\n"
        f"Сегодня: {fmt_pnl(pnl_day)}\n"
        f"За неделю: {fmt_pnl(pnl_week)}\n"
        f"За месяц: {fmt_pnl(pnl_month)}\n\n"
        f"Баланс: {fmt_price(balance)}"
    )


def format_positions(positions: list[Position]) -> str:
    """Ответ на /positions."""
    if not positions:
        return "📋 Нет открытых позиций"

    lines = ["📋 <b>Открытые позиции</b>\n"]
    for p in positions:
        pnl = fmt_pnl(p.unrealized_pnl)
        pct = fmt_pct((p.current_price - p.entry_price) / p.entry_price * 100) if p.entry_price else "+0.00%"
        lines.append(
            f"  {p.symbol}  {p.side}  {fmt_price(p.entry_price)}  {pnl} ({pct})"
        )
    return "\n".join(lines)


def format_trades(trades: list[dict]) -> str:
    """Ответ на /trades — последние 10 сделок."""
    if not trades:
        return "📋 Нет сделок"

    lines = ["📋 <b>Последние сделки</b>\n"]
    for t in trades[:10]:
        side = t.get("side", "?")
        symbol = t.get("symbol", "?")
        price = fmt_price(t.get("price", 0))
        pnl = ""
        if "pnl" in t and t["pnl"] is not None:
            pnl = f"  {fmt_pnl(t['pnl'])}"
        lines.append(f"  {side} {symbol} @ {price}{pnl}")
    return "\n".join(lines)


def format_config_summary(settings: dict) -> str:
    """Ответ на /config — основные настройки."""
    lines = ["⚙️ <b>Конфигурация</b>\n"]
    for key, val in settings.items():
        lines.append(f"  {key}: {val}")
    return "\n".join(lines)
