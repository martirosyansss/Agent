"""
Форматирование сообщений для Telegram.

Каждый метод возвращает готовую строку с HTML-разметкой (parse_mode="HTML").
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from core.models import Direction, Order, Position, RiskState, Signal


def _esc(text: str) -> str:
    """Escape HTML special characters for Telegram messages."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def fmt_price(value: float) -> str:
    """Форматирование цены."""
    if value == 0:
        return "$0.00"
    if value < 0.01:
        return f"${value:,.6f}"
    if value < 10:
        return f"${value:,.4f}"
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


def _now_str() -> str:
    """Текущее время UTC в читаемом формате."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


def _risk_reward(entry: float, sl: float, tp: float, side: str) -> str:
    """Вычислить Risk/Reward ratio."""
    try:
        if side == "LONG" or side == "BUY":
            risk = abs(entry - sl)
            reward = abs(tp - entry)
        else:
            risk = abs(sl - entry)
            reward = abs(entry - tp)
        if risk > 0:
            return f"{reward / risk:.2f}"
    except Exception:
        pass
    return "N/A"


def _confidence_bar(confidence: float) -> str:
    """Визуальная полоска уверенности (0..1)."""
    filled = int(confidence * 10)
    empty = 10 - filled
    return "█" * filled + "░" * empty


# ──────────────────────────────────────────────
# Автоматические уведомления
# ──────────────────────────────────────────────


def format_signal(signal: Signal) -> str:
    """Уведомление о новом сигнале — максимально подробное."""
    is_buy = signal.direction == Direction.BUY
    icon = "📈" if is_buy else "📉"
    action = "LONG / BUY" if is_buy else "SHORT / SELL"
    direction_color = "🟢" if is_buy else "🔴"

    conf_bar = _confidence_bar(signal.confidence)
    conf_pct = signal.confidence * 100

    lines = [
        f"{icon} <b>НОВЫЙ СИГНАЛ — {_esc(signal.symbol)}</b>",
        f"{direction_color} Направление: <b>{action}</b>",
        f"⏰ Время: {_now_str()}",
        "",
        f"📊 <b>Параметры входа</b>",
    ]

    if signal.suggested_quantity > 0:
        lines.append(f"  Кол-во: <b>{signal.suggested_quantity:.6f}</b> {_esc(signal.symbol.replace('USDT', ''))}")

    if signal.stop_loss_price > 0:
        lines.append(f"  🛑 Stop-Loss:   <b>{fmt_price(signal.stop_loss_price)}</b>")
    if signal.take_profit_price > 0:
        lines.append(f"  🎯 Take-Profit: <b>{fmt_price(signal.take_profit_price)}</b>")

    if signal.stop_loss_price > 0 and signal.take_profit_price > 0 and signal.suggested_quantity > 0:
        # Примерный потенциальный P&L
        if is_buy:
            potential_profit = (signal.take_profit_price - signal.stop_loss_price) * signal.suggested_quantity
            potential_loss = signal.suggested_quantity  # placeholder
        lines.append(f"  ⚖️ R/R Ratio: <b>{_risk_reward(signal.stop_loss_price, signal.stop_loss_price, signal.take_profit_price, 'BUY' if is_buy else 'SELL')}</b>")

    lines += [
        "",
        f"🧠 <b>Уверенность: {conf_pct:.0f}%</b>",
        f"  [{conf_bar}]",
        "",
        f"⚙️ Стратегия: <b>{_esc(signal.strategy_name)}</b>",
        f"💬 Причина входа:",
        f"  <i>{_esc(signal.reason)}</i>",
        "",
        f"🔖 Signal ID: <code>{signal.signal_id}</code>",
    ]

    return "\n".join(lines)


def format_order_filled(order: Order) -> str:
    """Уведомление об исполненном ордере — с полным описанием."""
    is_buy = order.side == Direction.BUY
    icon = "✅"
    action = "КУПЛЕНО" if is_buy else "ПРОДАНО"
    mode = "📄 Paper" if order.is_paper else "💰 LIVE"
    direction_color = "🟢" if is_buy else "🔴"

    fill_price = order.fill_price or order.price or 0
    fill_qty = order.fill_quantity or order.quantity

    total_value = fill_price * fill_qty

    lines = [
        f"{icon} <b>ОРДЕР ИСПОЛНЕН — {_esc(order.symbol)}</b>",
        f"{direction_color} Действие: <b>{action}</b>  |  {mode}",
        f"⏰ Время: {_now_str()}",
        "",
        f"💵 <b>Детали сделки</b>",
        f"  Количество: <b>{fill_qty:.6f}</b> {_esc(order.symbol.replace('USDT', ''))}",
        f"  Цена входа: <b>{fmt_price(fill_price)}</b>",
        f"  Сумма сделки: <b>${total_value:,.2f}</b>",
    ]

    if order.commission > 0:
        lines.append(f"  Комиссия: <b>-${order.commission:.4f}</b>")

    lines.append("")
    lines.append(f"🔒 <b>Уровни защиты</b>")

    if order.stop_loss_price > 0:
        sl_diff_pct = abs(fill_price - order.stop_loss_price) / fill_price * 100 if fill_price > 0 else 0
        sl_loss = abs(fill_price - order.stop_loss_price) * fill_qty
        lines.append(f"  🛑 Stop-Loss:   <b>{fmt_price(order.stop_loss_price)}</b>  ({sl_diff_pct:.2f}% / риск ${sl_loss:.2f})")
    else:
        lines.append(f"  🛑 Stop-Loss:   не задан")

    if order.take_profit_price > 0:
        tp_diff_pct = abs(order.take_profit_price - fill_price) / fill_price * 100 if fill_price > 0 else 0
        tp_profit = abs(order.take_profit_price - fill_price) * fill_qty
        lines.append(f"  🎯 Take-Profit: <b>{fmt_price(order.take_profit_price)}</b>  ({tp_diff_pct:.2f}% / цель ${tp_profit:.2f})")
    else:
        lines.append(f"  🎯 Take-Profit: не задан")

    if order.stop_loss_price > 0 and order.take_profit_price > 0:
        rr = _risk_reward(fill_price, order.stop_loss_price, order.take_profit_price, "BUY" if is_buy else "SELL")
        lines.append(f"  ⚖️ R/R Ratio: <b>{rr}</b>")

    if order.strategy_name:
        lines += ["", f"⚙️ Стратегия: <b>{_esc(order.strategy_name)}</b>"]
    if order.signal_reason:
        lines.append(f"💬 Причина: <i>{_esc(order.signal_reason)}</i>")

    lines += ["", f"🔖 Order ID: <code>{order.order_id}</code>"]

    return "\n".join(lines)


def format_stop_loss(position: Position, loss: float) -> str:
    """Уведомление о срабатывании stop-loss."""
    loss_pct = abs(loss) / (position.entry_price * position.quantity) * 100 if position.entry_price > 0 and position.quantity > 0 else 0
    entry_to_sl = abs(position.current_price - position.entry_price)
    mode = "📄 Paper" if position.is_paper else "💰 LIVE"

    lines = [
        f"🛑 <b>STOP-LOSS СРАБОТАЛ — {_esc(position.symbol)}</b>",
        f"⏰ Время: {_now_str()}  |  {mode}",
        "",
        f"💸 <b>Результат сделки</b>",
        f"  Потеря: <b>{fmt_pnl(loss)}</b>  ({fmt_pct(-loss_pct)})",
        f"  Цена входа:   <b>{fmt_price(position.entry_price)}</b>",
        f"  Цена выхода:  <b>{fmt_price(position.current_price)}</b>",
        f"  Движение цены: <b>{fmt_price(entry_to_sl)}</b> против позиции",
        f"  Количество: <b>{position.quantity:.6f}</b>",
    ]

    if position.strategy_name:
        lines += ["", f"⚙️ Стратегия: <b>{_esc(position.strategy_name)}</b>"]
    if position.signal_reason:
        lines.append(f"💬 Сигнал был: <i>{_esc(position.signal_reason)}</i>")

    lines += ["", f"🔖 Position ID: <code>{position.position_id}</code>"]
    return "\n".join(lines)


def format_take_profit(position: Position, profit: float) -> str:
    """Уведомление о срабатывании take-profit."""
    profit_pct = profit / (position.entry_price * position.quantity) * 100 if position.entry_price > 0 and position.quantity > 0 else 0
    entry_to_tp = abs(position.current_price - position.entry_price)
    mode = "📄 Paper" if position.is_paper else "💰 LIVE"

    lines = [
        f"🎯 <b>TAKE-PROFIT СРАБОТАЛ — {_esc(position.symbol)}</b>",
        f"⏰ Время: {_now_str()}  |  {mode}",
        "",
        f"💰 <b>Результат сделки</b>",
        f"  Прибыль: <b>{fmt_pnl(profit)}</b>  ({fmt_pct(profit_pct)})",
        f"  Цена входа:   <b>{fmt_price(position.entry_price)}</b>",
        f"  Цена выхода:  <b>{fmt_price(position.current_price)}</b>",
        f"  Движение цены: <b>{fmt_price(entry_to_tp)}</b> в пользу позиции",
        f"  Количество: <b>{position.quantity:.6f}</b>",
    ]

    if position.strategy_name:
        lines += ["", f"⚙️ Стратегия: <b>{_esc(position.strategy_name)}</b>"]
    if position.signal_reason:
        lines.append(f"💬 Сигнал был: <i>{_esc(position.signal_reason)}</i>")

    lines += ["", f"🔖 Position ID: <code>{position.position_id}</code>"]
    return "\n".join(lines)


def format_risk_state_changed(old_state: RiskState, new_state: RiskState, reason: str) -> str:
    """Уведомление о смене Risk State — с описанием каждого состояния."""
    state_icons = {
        "NORMAL":  "🟢",
        "REDUCED": "🟡",
        "SAFE":    "🟠",
        "STOP":    "🔴",
    }
    state_descriptions = {
        "NORMAL":  "Торговля в штатном режиме, все стратегии активны",
        "REDUCED": "Уменьшен размер позиций, повышен порог сигналов",
        "SAFE":    "Только выход из существующих позиций, новые не открываются",
        "STOP":    "Полная остановка — все позиции закрыты, торговля запрещена",
    }

    old_icon = state_icons.get(old_state.value, "⚪")
    new_icon = state_icons.get(new_state.value, "⚪")
    new_desc = state_descriptions.get(new_state.value, "")

    lines = [
        f"⚠️ <b>СМЕНА РИСК-СОСТОЯНИЯ</b>",
        f"⏰ Время: {_now_str()}",
        "",
        f"  {old_icon} <b>{_esc(old_state.value)}</b>  →  {new_icon} <b>{_esc(new_state.value)}</b>",
        "",
        f"📋 Новый режим: <i>{new_desc}</i>",
        f"💬 Причина: <b>{_esc(reason)}</b>",
    ]
    return "\n".join(lines)


def format_error(message: str) -> str:
    """Уведомление об ошибке."""
    return (
        f"🚨 <b>ОШИБКА СИСТЕМЫ</b>\n"
        f"⏰ {_now_str()}\n\n"
        f"{_esc(message)}"
    )


def format_daily_report(
    pnl: float,
    win_rate: float,
    trades_count: int,
    wins: int,
    losses: int,
    best_trade: float = 0.0,
    worst_trade: float = 0.0,
    avg_trade: float = 0.0,
    max_drawdown: float = 0.0,
    balance: float = 0.0,
) -> str:
    """Ежедневный отчёт — подробный."""
    pnl_icon = "📈" if pnl >= 0 else "📉"
    pnl_bar = _confidence_bar(win_rate / 100)

    lines = [
        f"📊 <b>ИТОГИ ДНЯ</b>",
        f"⏰ {_now_str()}",
        "",
        f"💵 <b>P&amp;L за день: {fmt_pnl(pnl)}</b>  {pnl_icon}",
    ]
    if balance > 0:
        lines.append(f"  Баланс: <b>${balance:,.2f}</b>")
        if pnl != 0:
            pnl_pct = pnl / (balance - pnl) * 100
            lines.append(f"  Изменение: <b>{fmt_pct(pnl_pct)}</b>")

    lines += [
        "",
        f"📈 <b>Статистика сделок</b>",
        f"  Всего сделок:    <b>{trades_count}</b>",
        f"  Прибыльных:      <b>{wins}</b> 🟢",
        f"  Убыточных:       <b>{losses}</b> 🔴",
        f"  Win Rate:        <b>{win_rate:.1f}%</b>",
        f"  [{pnl_bar}]",
    ]

    if trades_count > 0:
        lines += [
            "",
            f"📊 <b>Детали по сделкам</b>",
        ]
        if best_trade != 0:
            lines.append(f"  Лучшая сделка:   <b>{fmt_pnl(best_trade)}</b>")
        if worst_trade != 0:
            lines.append(f"  Худшая сделка:   <b>{fmt_pnl(worst_trade)}</b>")
        if avg_trade != 0:
            lines.append(f"  Средняя сделка:  <b>{fmt_pnl(avg_trade)}</b>")
        if max_drawdown != 0:
            lines.append(f"  Макс. просадка:  <b>-${abs(max_drawdown):.2f}</b>")

    return "\n".join(lines)


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
    balance: float = 0.0,
    win_rate_today: float = 0.0,
    wins_today: int = 0,
    losses_today: int = 0,
) -> str:
    """Ответ на /status — подробный статус системы."""
    state_icon = {
        "NORMAL":  "🟢",
        "REDUCED": "🟡",
        "SAFE":    "🟠",
        "STOP":    "🔴",
    }.get(risk_state, "⚪")

    mode_icon = "📄" if mode.lower() == "paper" else "💰"
    pnl_today_icon = "📈" if pnl_today >= 0 else "📉"
    pnl_total_icon = "📈" if pnl_total >= 0 else "📉"

    lines = [
        f"{state_icon} <b>SENTINEL — {mode.upper()}</b>  {mode_icon}",
        f"⏰ {_now_str()}  |  Uptime: {uptime}",
        "",
        f"🔒 Risk State: <b>{risk_state}</b>",
        f"📊 Открытых позиций: <b>{open_positions}</b>",
        "",
        f"💵 <b>P&amp;L сегодня: {fmt_pnl(pnl_today)}</b>  {pnl_today_icon}",
    ]

    if trades_today > 0:
        lines.append(f"  Сделок сегодня: <b>{trades_today}</b>  (Win: {wins_today} 🟢 / Loss: {losses_today} 🔴)")
        if win_rate_today > 0:
            lines.append(f"  Win Rate: <b>{win_rate_today:.1f}%</b>")

    lines += [
        "",
        f"💰 <b>P&amp;L всего: {fmt_pnl(pnl_total)}</b>  {pnl_total_icon}",
    ]

    if balance > 0:
        lines.append(f"  Баланс: <b>${balance:,.2f}</b>")

    return "\n".join(lines)


def format_pnl(
    pnl_day: float,
    pnl_week: float,
    pnl_month: float,
    balance: float,
    pnl_total: float = 0.0,
    trades_day: int = 0,
    win_rate_day: float = 0.0,
) -> str:
    """Ответ на /pnl — подробный отчёт по прибыли."""
    day_icon   = "📈" if pnl_day   >= 0 else "📉"
    week_icon  = "📈" if pnl_week  >= 0 else "📉"
    month_icon = "📈" if pnl_month >= 0 else "📉"

    lines = [
        f"💰 <b>P&amp;L Report</b>",
        f"⏰ {_now_str()}",
        "",
        f"📅 <b>Сегодня:</b>       {fmt_pnl(pnl_day)}  {day_icon}",
    ]

    if trades_day > 0:
        lines.append(f"   Сделок: {trades_day}  |  Win Rate: {win_rate_day:.1f}%")

    lines += [
        f"📅 <b>За неделю:</b>     {fmt_pnl(pnl_week)}  {week_icon}",
        f"📅 <b>За месяц:</b>      {fmt_pnl(pnl_month)}  {month_icon}",
    ]

    if pnl_total != 0:
        total_icon = "📈" if pnl_total >= 0 else "📉"
        lines.append(f"📅 <b>За всё время:</b>  {fmt_pnl(pnl_total)}  {total_icon}")

    lines += [
        "",
        f"💵 <b>Баланс: ${balance:,.2f}</b>",
    ]

    if balance > 0 and pnl_month != 0:
        month_pct = pnl_month / (balance - pnl_month) * 100
        lines.append(f"   Месячная доходность: <b>{fmt_pct(month_pct)}</b>")

    return "\n".join(lines)


def format_positions(positions: list[Position]) -> str:
    """Ответ на /positions — подробный список открытых позиций."""
    if not positions:
        return "📋 <b>Открытых позиций нет</b>"

    lines = [f"📋 <b>Открытые позиции ({len(positions)})</b>\n"]

    total_unrealized = 0.0
    for i, p in enumerate(positions, 1):
        pnl_icon = "🟢" if p.unrealized_pnl >= 0 else "🔴"
        pct = (p.current_price - p.entry_price) / p.entry_price * 100 if p.entry_price else 0
        total_value = p.current_price * p.quantity
        total_unrealized += p.unrealized_pnl

        lines.append(f"{'─' * 28}")
        lines.append(f"<b>{i}. {_esc(p.symbol)}</b>  {p.side}")
        lines.append(f"  {pnl_icon} Unrealized P&amp;L: <b>{fmt_pnl(p.unrealized_pnl)}</b>  ({fmt_pct(pct)})")
        lines.append(f"  Цена входа:    <b>{fmt_price(p.entry_price)}</b>")
        lines.append(f"  Текущая цена:  <b>{fmt_price(p.current_price)}</b>")
        lines.append(f"  Количество:    <b>{p.quantity:.6f}</b>")
        lines.append(f"  Стоимость:     <b>${total_value:,.2f}</b>")
        if p.stop_loss_price > 0:
            lines.append(f"  🛑 Stop-Loss:  <b>{fmt_price(p.stop_loss_price)}</b>")
        if p.take_profit_price > 0:
            lines.append(f"  🎯 Take-Profit:<b>{fmt_price(p.take_profit_price)}</b>")
        if p.strategy_name:
            lines.append(f"  ⚙️ Стратегия:  {_esc(p.strategy_name)}")
        if p.opened_at:
            lines.append(f"  ⏰ Открыта:    {_esc(p.opened_at)}")

    lines.append(f"{'─' * 28}")
    pnl_total_icon = "🟢" if total_unrealized >= 0 else "🔴"
    lines.append(f"{pnl_total_icon} <b>Итого нереализованный P&amp;L: {fmt_pnl(total_unrealized)}</b>")

    return "\n".join(lines)


def format_trades(trades: list[dict]) -> str:
    """Ответ на /trades — последние сделки с подробностями."""
    if not trades:
        return "📋 <b>Нет завершённых сделок</b>"

    count = min(len(trades), 10)
    lines = [f"📋 <b>Последние сделки ({count})</b>\n"]

    total_pnl = 0.0
    wins = 0
    losses = 0

    for i, t in enumerate(trades[:10], 1):
        side = t.get("side", "?")
        symbol = t.get("symbol", "?")
        price = t.get("price", 0)
        pnl = t.get("pnl")
        qty = t.get("quantity", 0)
        strategy = t.get("strategy_name", "")
        entry_price = t.get("entry_price", 0)
        closed_at = t.get("closed_at", "")

        side_icon = "🟢" if side in ("BUY", "LONG") else "🔴"

        lines.append(f"{'─' * 28}")
        lines.append(f"<b>{i}. {side_icon} {side} {_esc(symbol)}</b>")

        if entry_price > 0:
            lines.append(f"  Вход:   <b>{fmt_price(entry_price)}</b>")
        lines.append(f"  Выход:  <b>{fmt_price(price)}</b>")

        if qty > 0:
            lines.append(f"  Кол-во: <b>{qty:.6f}</b>")

        if pnl is not None:
            pnl_icon = "💰" if pnl >= 0 else "💸"
            lines.append(f"  {pnl_icon} P&amp;L: <b>{fmt_pnl(pnl)}</b>")
            total_pnl += pnl
            if pnl >= 0:
                wins += 1
            else:
                losses += 1

        if strategy:
            lines.append(f"  ⚙️ {_esc(strategy)}")
        if closed_at:
            lines.append(f"  ⏰ {_esc(closed_at)}")

    lines.append(f"{'─' * 28}")

    if wins + losses > 0:
        wr = wins / (wins + losses) * 100
        pnl_icon = "💰" if total_pnl >= 0 else "💸"
        lines.append(f"{pnl_icon} <b>Итого P&amp;L: {fmt_pnl(total_pnl)}</b>  |  Win Rate: {wr:.1f}%  ({wins}W/{losses}L)")

    return "\n".join(lines)


def format_config_summary(settings: dict) -> str:
    """Ответ на /config — основные настройки."""
    lines = ["⚙️ <b>Конфигурация SENTINEL</b>\n"]
    for key, val in settings.items():
        lines.append(f"  <b>{_esc(str(key))}:</b> {_esc(str(val))}")
    return "\n".join(lines)


def format_portfolio(strategy_perf: list[dict], balance: float) -> str:
    """Ответ на /portfolio — подробный отчёт по стратегиям."""
    if not strategy_perf:
        return "📊 <b>Нет данных по стратегиям</b> (0 сделок)"

    lines = [
        f"📊 <b>Portfolio Report</b>",
        f"⏰ {_now_str()}",
        f"💵 Баланс: <b>${balance:,.2f}</b>",
        "",
    ]

    total_pnl = 0.0
    total_trades = 0

    for s in strategy_perf:
        name      = _esc(s.get("strategy_name", "?"))
        n_trades  = s.get("total_trades", 0)
        wr        = s.get("win_rate", 0.0)
        pnl       = s.get("total_pnl", 0.0)
        wins      = s.get("wins", 0)
        losses    = s.get("losses", 0)
        avg_trade = s.get("avg_trade_pnl", 0.0)
        best      = s.get("best_trade", 0.0)
        worst     = s.get("worst_trade", 0.0)

        pnl_icon = "🟢" if pnl >= 0 else "🔴"
        wr_bar = _confidence_bar(wr / 100)

        lines.append(f"{'─' * 28}")
        lines.append(f"{pnl_icon} <b>{name}</b>")
        lines.append(f"  P&amp;L:    <b>{fmt_pnl(pnl)}</b>")
        lines.append(f"  Сделок: <b>{n_trades}</b>  (Win: {wins} 🟢 / Loss: {losses} 🔴)")
        lines.append(f"  WR:     <b>{wr:.1f}%</b>  [{wr_bar}]")
        if avg_trade != 0:
            lines.append(f"  Ср. сделка: <b>{fmt_pnl(avg_trade)}</b>")
        if best != 0:
            lines.append(f"  Лучшая:     <b>{fmt_pnl(best)}</b>")
        if worst != 0:
            lines.append(f"  Худшая:     <b>{fmt_pnl(worst)}</b>")

        total_pnl += pnl
        total_trades += n_trades

    lines.append(f"{'─' * 28}")
    overall_icon = "🟢" if total_pnl >= 0 else "🔴"
    lines.append(f"{overall_icon} <b>ИТОГО: {fmt_pnl(total_pnl)}</b>  |  {total_trades} сделок")

    if balance > 0 and total_pnl != 0:
        roi = total_pnl / (balance - total_pnl) * 100
        lines.append(f"📈 ROI: <b>{fmt_pct(roi)}</b>")

    return "\n".join(lines)
