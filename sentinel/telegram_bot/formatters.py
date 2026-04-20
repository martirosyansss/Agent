"""
Форматирование сообщений для Telegram.

Каждый метод возвращает готовую строку с HTML-разметкой (parse_mode="HTML").
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

LOCAL_TZ = timezone(timedelta(hours=4))
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
    """Текущее локальное время (UTC+4) в читаемом формате."""
    return datetime.now(LOCAL_TZ).strftime("%H:%M:%S GMT+4")


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
    side = "BUY" if is_buy else "SELL"

    conf_bar = _confidence_bar(signal.confidence)
    conf_pct = signal.confidence * 100
    entry_price = signal.features.close if signal.features else 0.0

    lines = [
        f"{icon} <b>НОВЫЙ СИГНАЛ — {_esc(signal.symbol)}</b>",
        f"{direction_color} Направление: <b>{action}</b>",
        f"⏰ Время: {_now_str()}",
        "",
        f"📊 <b>Параметры входа</b>",
    ]

    if entry_price > 0:
        lines.append(f"  Цена входа:     <b>{fmt_price(entry_price)}</b>")

    if signal.suggested_quantity > 0:
        lines.append(f"  Кол-во: <b>{signal.suggested_quantity:.6f}</b> {_esc(signal.symbol.replace('USDT', ''))}")

    if signal.stop_loss_price > 0:
        lines.append(f"  🛑 Stop-Loss:   <b>{fmt_price(signal.stop_loss_price)}</b>")
    if signal.take_profit_price > 0:
        lines.append(f"  🎯 Take-Profit: <b>{fmt_price(signal.take_profit_price)}</b>")

    if entry_price > 0 and signal.stop_loss_price > 0 and signal.take_profit_price > 0:
        rr = _risk_reward(entry_price, signal.stop_loss_price, signal.take_profit_price, side)
        lines.append(f"  ⚖️ R/R Ratio: <b>{rr}</b>")
        if signal.suggested_quantity > 0:
            potential_profit = abs(signal.take_profit_price - entry_price) * signal.suggested_quantity
            potential_loss = abs(entry_price - signal.stop_loss_price) * signal.suggested_quantity
            lines.append(
                f"  💰 Потенциал:   <b>+${potential_profit:.2f}</b> / риск <b>-${potential_loss:.2f}</b>"
            )

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


def format_rejection_summary(
    total: int,
    top: list[tuple[str, int]],
    window_hours: float,
) -> str:
    """Periodic summary of risk-rejected signals — throttled, grouped by reason."""
    lines = [
        f"🧾 <b>Отклонённые сигналы за {window_hours:.1f}ч</b>",
        f"⏰ {_now_str()}",
        "",
        f"Всего отказов: <b>{total}</b>",
        "",
        "Топ причин:",
    ]
    for reason, count in top:
        lines.append(f"  • <b>{count}×</b> {_esc(reason)}")
    return "\n".join(lines)


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


def format_ml_calibration_line(ml_metrics: dict | None) -> str | None:
    """One-line ML calibration summary for status messages, or None if absent.

    Renders Brier, ECE, calibration method, and the proba-distribution centre
    so operators can spot at a glance whether the deployed filter is healthy.
    Highlights overfit calibration (📉 when ECE>0.10) and "always high"
    plateaus (⚠ when p10>0.70).
    """
    if not ml_metrics:
        return None
    method = ml_metrics.get("calibration_method", "none")
    ece = float(ml_metrics.get("ece", 0.0))
    brier = float(ml_metrics.get("brier_score", 0.0))
    p10 = float(ml_metrics.get("proba_p10", 0.0))
    p90 = float(ml_metrics.get("proba_p90", 1.0))
    mean_p = float(ml_metrics.get("mean_proba", 0.5))
    flag = ""
    if p10 > 0.70:
        flag = " ⚠ inflated"
    elif ece > 0.10:
        flag = " 📉 miscalibrated"
    return (
        f"🤖 <b>ML calibration:</b> ECE={ece:.3f} Brier={brier:.3f} "
        f"[{method}]  proba mean={mean_p:.2f} p10={p10:.2f} p90={p90:.2f}{flag}"
    )


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
    ml_metrics: dict | None = None,
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

    ml_line = format_ml_calibration_line(ml_metrics)
    if ml_line:
        lines += ["", ml_line]

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


# ──────────────────────────────────────────────
# Диагностические команды (/diag /why /events /health)
#
# Эти форматтеры объясняют операторы "что бот делает прямо сейчас и
# почему" — читают snapshot state из state_provider и последние записи
# из EventLog.recent_events(). HTML-экранирование обязательно: поля
# ``reason`` из ``signal_rejected`` могут содержать угловые скобки
# (например ``"EMA50<EMA200"``), которые Telegram интерпретирует как
# теги и режет сообщение.
# ──────────────────────────────────────────────


_SEVERITY_ICON = {
    "info":     "ℹ️",
    "warning":  "⚡",
    "error":    "⚠️",
    "critical": "🚨",
}


def _age_str(ts_ms: float | int | None, now_ms: float | None = None) -> str:
    """Human-readable age of an ms timestamp: ``"5s"`` / ``"3m"`` / ``"1h 2m"``.
    Returns ``"—"`` for missing/invalid."""
    if not ts_ms:
        return "—"
    try:
        ts = float(ts_ms)
    except (TypeError, ValueError):
        return "—"
    now = now_ms if now_ms is not None else datetime.now(timezone.utc).timestamp() * 1000
    delta_sec = max(0, int((now - ts) / 1000))
    if delta_sec < 60:
        return f"{delta_sec}s"
    if delta_sec < 3600:
        return f"{delta_sec // 60}m"
    hours = delta_sec // 3600
    mins = (delta_sec % 3600) // 60
    return f"{hours}h {mins}m" if mins else f"{hours}h"


def _ml_verdict_icon(decision: str) -> str:
    d = str(decision or "").lower()
    if d == "allow":
        return "🟢"
    if d == "reduce":
        return "🟡"
    if d == "block":
        return "🔴"
    return "⚪"


def format_diagnostics(state: dict) -> str:
    """Полная диагностика: что бот делает прямо сейчас.

    Разделы:
      1. Режим / пауза / Risk State
      2. Готовность модулей (WS, свечи, стратегии)
      3. По каждому символу: когда был последний цикл, ML-вердикт
      4. Risk-checks (какой гейт зелёный/красный) + circuit breakers
      5. ML подсистема (enabled / ready / версия модели)
    """
    mode = str(state.get("mode", "paper")).upper()
    risk_state = state.get("risk_state", "NORMAL")
    trading_paused = bool(state.get("trading_paused", False))
    uptime = state.get("uptime", "N/A")
    risk_icon = {"NORMAL": "🟢", "REDUCED": "🟡", "SAFE": "🟠", "STOP": "🔴"}.get(risk_state, "⚪")
    mode_icon = "📄" if mode == "PAPER" else "💰"
    trade_icon = "⏸" if trading_paused else "▶️"

    lines = [
        f"🔍 <b>ДИАГНОСТИКА SENTINEL</b>",
        f"⏰ {_now_str()}  |  Uptime: {_esc(str(uptime))}",
        "",
        f"{mode_icon} Режим: <b>{mode}</b>  |  {risk_icon} Risk: <b>{_esc(str(risk_state))}</b>  |  {trade_icon} "
        f"{'на паузе' if trading_paused else 'торгует'}",
    ]

    # 1) Готовность
    readiness = state.get("readiness") or {}
    if readiness:
        pct = readiness.get("pct", 0)
        ready = readiness.get("ready", False)
        ready_icon = "✅" if ready else "⏳"
        lines += [
            "",
            f"{ready_icon} <b>Готовность: {pct}%</b>",
        ]
        for step in (readiness.get("steps") or []):
            icon = "✅" if step.get("done") else "⏳"
            detail = step.get("detail", "")
            lines.append(f"  {icon} {_esc(step.get('name', '?'))}: {_esc(str(detail))}")

    # 2) Свежесть котировок
    risk_details = state.get("risk_details") or {}
    age = risk_details.get("market_data_age_sec", -1)
    try:
        age_f = float(age)
    except (TypeError, ValueError):
        age_f = -1.0
    if age_f >= 0:
        age_icon = "✅" if age_f < 30 else ("⚡" if age_f < 120 else "🚨")
        lines += ["", f"{age_icon} Котировки: последняя {age_f:.1f}s назад"]

    # 3) Активность по символам
    last_cycle = state.get("last_cycle_ts_per_symbol") or {}
    standing = state.get("standing_ml_per_symbol") or {}
    symbols = state.get("trading_symbols") or sorted(
        set(last_cycle.keys()) | set(standing.keys())
    )
    if symbols:
        lines += ["", "📈 <b>По символам</b>"]
        now_ms = datetime.now(timezone.utc).timestamp() * 1000
        for sym in symbols:
            sym_line = f"  <b>{_esc(sym)}</b>: цикл {_age_str(last_cycle.get(sym), now_ms)} назад"
            ml = standing.get(sym) or {}
            if ml:
                prob = ml.get("prob")
                decision = ml.get("decision", "")
                ref = ml.get("ref_strategy", "")
                ml_icon = _ml_verdict_icon(decision)
                try:
                    prob_s = f"{float(prob):.2f}" if prob is not None else "—"
                except (TypeError, ValueError):
                    prob_s = "—"
                sym_line += f"  |  {ml_icon} ML p={prob_s} → <b>{_esc(str(decision) or '—')}</b>"
                if ref:
                    sym_line += f" ({_esc(str(ref))})"
            lines.append(sym_line)

    # 4) Risk-чекеры
    checks = (state.get("risk_details") or {}).get("risk_checks") or {}
    if checks:
        lines += ["", "🔒 <b>Risk-чеки</b> (✅ пропускает / ⛔ блокирует)"]
        labels = {
            "state_ok":        "Risk State ≠ STOP",
            "daily_loss_ok":   "Дневной убыток в лимите",
            "positions_ok":    "Есть слот под позицию",
            "exposure_ok":     "Экспозиция в лимите",
            "daily_trades_ok": "Сделок за день в лимите",
            "hourly_trades_ok":"Сделок за час в лимите",
            "cooldown_ok":     "Кулдаун истёк",
        }
        for key, label in labels.items():
            if key not in checks:
                continue
            ok = bool(checks[key])
            lines.append(f"  {'✅' if ok else '⛔'} {label}")

    cooldown = int(risk_details.get("cooldown_remaining_sec", 0) or 0)
    if cooldown > 0:
        lines.append(f"  ⏳ Кулдаун ещё <b>{cooldown}s</b>")

    # 5) Circuit breakers / blocked strategies
    rd_breakers = (state.get("risk_details") or {})
    blocked = rd_breakers.get("blocked_strategies") or {}
    if blocked:
        lines += ["", "🚧 <b>Заблокированные стратегии</b>"]
        for strat, info in blocked.items():
            reason = ""
            if isinstance(info, dict):
                reason = info.get("reason") or info.get("until") or ""
            lines.append(f"  • <code>{_esc(strat)}</code>  {_esc(str(reason))}")

    # 6) ML подсистема
    ml_status = state.get("ml_status") or {}
    if ml_status:
        enabled = ml_status.get("enabled", False)
        ready = ml_status.get("is_ready", False)
        ml_mode = ml_status.get("mode", "off")
        version = ml_status.get("model_version", "")
        lines += [
            "",
            f"🤖 <b>ML</b>: "
            f"{'✅ enabled' if enabled else '⛔ disabled'}  |  "
            f"{'ready' if ready else 'not ready'}  |  mode=<code>{_esc(str(ml_mode))}</code>"
            + (f"  |  v={_esc(str(version))}" if version else ""),
        ]

    # 7) Активные стратегии
    activity = state.get("activity") or {}
    strategies = activity.get("strategies_loaded") or []
    if strategies:
        lines += ["", f"⚙️ Загружены: <code>{_esc(', '.join(strategies))}</code>"]

    regime = activity.get("current_regime")
    if regime:
        lines.append(f"🌐 Regime: <b>{_esc(str(regime))}</b>")

    return "\n".join(lines)


def _group_rejections(events: list[dict]) -> list[tuple[str, int, str]]:
    """Group signal_rejected events by (gate). Returns ``[(gate, count, sample_reason), ...]``."""
    buckets: dict[str, dict] = {}
    for ev in events:
        if ev.get("type") != "signal_rejected":
            continue
        gate = str(ev.get("gate") or "unknown")
        b = buckets.setdefault(gate, {"count": 0, "reason": ""})
        b["count"] += 1
        if not b["reason"] and ev.get("reason"):
            b["reason"] = str(ev["reason"])
    return sorted(
        [(g, b["count"], b["reason"]) for g, b in buckets.items()],
        key=lambda x: -x[1],
    )


def format_why(
    state: dict,
    events: list[dict],
    symbol: Optional[str] = None,
) -> str:
    """Объяснение: почему бот НЕ открывает сделки прямо сейчас.

    Последовательно проверяет блокирующие факторы от самых грубых
    (kill / pause) к более тонким (rejections по гейтам). Опциональный
    ``symbol`` фильтрует и события, и ML-вердикт.
    """
    header_sym = f" — {_esc(symbol)}" if symbol else ""
    lines = [
        f"❓ <b>Почему нет сделок{header_sym}</b>",
        f"⏰ {_now_str()}",
    ]

    # Грубые блокировки
    risk_state = state.get("risk_state", "NORMAL")
    trading_paused = bool(state.get("trading_paused", False))
    if risk_state == "STOP":
        lines += ["", "🔴 <b>Risk State = STOP</b> — торговля полностью остановлена."]
    elif trading_paused:
        lines += ["", "⏸ <b>Торговля на паузе</b> (ручная остановка)."]
    elif risk_state == "SAFE":
        lines += ["", "🟠 <b>Risk State = SAFE</b> — только выход, новые позиции не открываются."]

    # Risk-чеки: показать красные
    checks = (state.get("risk_details") or {}).get("risk_checks") or {}
    red = [k for k, v in checks.items() if not v]
    if red:
        labels = {
            "state_ok":        "Risk State блокирует",
            "daily_loss_ok":   "Превышен дневной убыток",
            "positions_ok":    "Все слоты под позиции заняты",
            "exposure_ok":     "Превышена экспозиция",
            "daily_trades_ok": "Исчерпан лимит сделок за день",
            "hourly_trades_ok":"Исчерпан лимит сделок за час",
            "cooldown_ok":     "Активен кулдаун между сделками",
        }
        lines += ["", "⛔ <b>Блокирующие risk-чеки</b>"]
        for k in red:
            lines.append(f"  • {labels.get(k, k)}")

    cooldown = int((state.get("risk_details") or {}).get("cooldown_remaining_sec", 0) or 0)
    if cooldown > 0:
        lines.append(f"  ⏳ Кулдаун ещё <b>{cooldown}s</b>")

    # ML вердикт для конкретного символа
    if symbol:
        standing = (state.get("standing_ml_per_symbol") or {}).get(symbol) or {}
        if standing:
            prob = standing.get("prob")
            decision = standing.get("decision", "")
            ref = standing.get("ref_strategy", "")
            try:
                prob_s = f"{float(prob):.2f}" if prob is not None else "—"
            except (TypeError, ValueError):
                prob_s = "—"
            ml_icon = _ml_verdict_icon(decision)
            if decision and decision.lower() != "allow":
                lines += [
                    "",
                    f"{ml_icon} <b>ML вердикт: {_esc(str(decision))}</b>  (p={prob_s}"
                    + (f", ref={_esc(str(ref))}" if ref else "") + ")",
                ]

    # Фильтр событий
    filtered = events
    if symbol:
        sym_u = symbol.upper()
        filtered = [e for e in events if str(e.get("symbol", "")).upper() == sym_u]

    rejections = _group_rejections(filtered)
    if rejections:
        lines += ["", "🚫 <b>Последние отказы по гейтам</b>"]
        for gate, count, reason in rejections[:8]:
            line = f"  • <b>{count}×</b> <code>{_esc(gate)}</code>"
            if reason:
                snippet = reason if len(reason) <= 120 else reason[:117] + "..."
                line += f" — <i>{_esc(snippet)}</i>"
            lines.append(line)
    elif not red and risk_state == "NORMAL" and not trading_paused:
        lines += [
            "",
            "✅ <b>Блокировок не найдено.</b>",
            "Бот ищет сигналы — возможно, ни одна стратегия пока не эмитит их",
            "(слабый тренд / низкая волатильность / ML-фильтр).",
        ]

    # Подсказка
    lines += ["", "<i>Используй /events для сырого лога или /diag для полной картины.</i>"]
    return "\n".join(lines)


def format_events(events: list[dict], limit: int = 15) -> str:
    """Tail последних событий с иконками — сырой, но читаемый лог."""
    if not events:
        return "📜 <b>Событий нет</b> (EventLog пустой)."

    recent = events[-limit:]
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    lines = [f"📜 <b>Последние события ({len(recent)})</b>", ""]

    for ev in reversed(recent):
        etype = str(ev.get("type", "?"))
        age = _age_str(ev.get("ts"), now_ms)
        icon = "•"
        body = ""
        if etype == "signal_rejected":
            icon = "🚫"
            gate = ev.get("gate", "?")
            sym = ev.get("symbol", "")
            reason = str(ev.get("reason", ""))[:100]
            body = f"<code>{_esc(gate)}</code>"
            if sym:
                body += f" {_esc(sym)}"
            if reason:
                body += f" — <i>{_esc(reason)}</i>"
        elif etype == "signal_approved":
            icon = "✅"
            body = f"{_esc(ev.get('symbol', '?'))}  {_esc(ev.get('direction', ''))}"
        elif etype == "signal_generated":
            icon = "💡"
            body = f"{_esc(ev.get('symbol', '?'))} <i>{_esc(str(ev.get('strategy', ''))[:40])}</i>"
        elif etype == "order_filled":
            icon = "💵"
            body = f"{_esc(ev.get('symbol', '?'))}  {_esc(ev.get('side', ''))}  qty={ev.get('quantity', '?')}"
        elif etype == "position_opened":
            icon = "📈"
            body = f"{_esc(ev.get('symbol', '?'))}  {_esc(ev.get('side', ''))}"
        elif etype == "position_closed":
            icon = "📉"
            pnl = ev.get("pnl", 0)
            body = f"{_esc(ev.get('symbol', '?'))}  pnl={fmt_pnl(float(pnl) if pnl else 0)}"
        elif etype == "guard_tripped":
            icon = "🛑"
            body = f"<code>{_esc(ev.get('guard', '?'))}</code>"
            if ev.get("name"):
                body += f"/{_esc(ev['name'])}"
            if ev.get("reason"):
                body += f" — <i>{_esc(str(ev['reason'])[:80])}</i>"
        elif etype == "component_error":
            sev = str(ev.get("severity", "error"))
            icon = _SEVERITY_ICON.get(sev, "⚠️")
            body = f"<code>{_esc(ev.get('component', '?'))}</code>"
            if ev.get("reason"):
                body += f" — <i>{_esc(str(ev['reason'])[:80])}</i>"
        elif etype == "regime_change":
            icon = "🌐"
            body = f"{_esc(ev.get('from', '?'))} → {_esc(ev.get('to', '?'))}"
        elif etype == "ml_prediction":
            icon = "🤖"
            body = f"{_esc(ev.get('symbol', '?'))} p={ev.get('prob', '?')} → {_esc(ev.get('decision', ''))}"
        elif etype == "news_critical":
            icon = "📰"
            body = _esc(str(ev.get("headline", ""))[:80])
        elif etype == "strategy_toggled":
            icon = "⚙️"
            body = f"{_esc(ev.get('strategy', '?'))} → {'ON' if ev.get('enabled') else 'OFF'}"
        else:
            body = _esc(etype)

        lines.append(f"{icon} <b>{age}</b>  {body}")

    return "\n".join(lines)


def format_health(state: dict, events: list[dict]) -> str:
    """Сводка «здоровья»: компоненты, guards, свежесть данных за последний час."""
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    hour_ago = now_ms - 3600 * 1000

    recent_errors = [
        e for e in events
        if e.get("type") == "component_error" and float(e.get("ts", 0)) >= hour_ago
    ]
    recent_guards = [
        e for e in events
        if e.get("type") == "guard_tripped" and float(e.get("ts", 0)) >= hour_ago
    ]

    # Уникальные компоненты в ошибках
    comp_counts: dict[str, int] = {}
    max_sev = "info"
    sev_rank = {"info": 0, "warning": 1, "error": 2, "critical": 3}
    for e in recent_errors:
        c = str(e.get("component", "?"))
        comp_counts[c] = comp_counts.get(c, 0) + 1
        sev = str(e.get("severity", "error"))
        if sev_rank.get(sev, 0) > sev_rank.get(max_sev, 0):
            max_sev = sev

    guard_counts: dict[str, int] = {}
    for e in recent_guards:
        g = str(e.get("guard", "?"))
        guard_counts[g] = guard_counts.get(g, 0) + 1

    # Общий статус
    health_icon = "✅"
    health_word = "HEALTHY"
    if recent_errors:
        if max_sev == "critical":
            health_icon, health_word = "🚨", "CRITICAL"
        elif max_sev == "error":
            health_icon, health_word = "⚠️", "DEGRADED"
        else:
            health_icon, health_word = "⚡", "WARN"

    risk_details = state.get("risk_details") or {}
    try:
        age = float(risk_details.get("market_data_age_sec", -1))
    except (TypeError, ValueError):
        age = -1.0

    lines = [
        f"🩺 <b>ЗДОРОВЬЕ СИСТЕМЫ — {health_icon} {health_word}</b>",
        f"⏰ {_now_str()}",
        "",
    ]

    # Котировки
    if age < 0:
        lines.append("📡 Котировки: <b>нет данных</b>")
    else:
        quote_icon = "✅" if age < 30 else ("⚡" if age < 120 else "🚨")
        lines.append(f"📡 Котировки: {quote_icon} <b>{age:.1f}s</b> назад")

    # Component errors
    if recent_errors:
        lines += ["", f"⚠️ <b>Ошибки компонентов за час: {len(recent_errors)}</b>"]
        for comp, cnt in sorted(comp_counts.items(), key=lambda x: -x[1])[:6]:
            lines.append(f"  • <b>{cnt}×</b> <code>{_esc(comp)}</code>")
    else:
        lines += ["", "✅ Ошибок компонентов за час: <b>0</b>"]

    # Guards
    if recent_guards:
        lines += ["", f"🛑 <b>Guards сработали за час: {len(recent_guards)}</b>"]
        for g, cnt in sorted(guard_counts.items(), key=lambda x: -x[1])[:6]:
            lines.append(f"  • <b>{cnt}×</b> <code>{_esc(g)}</code>")
    else:
        lines += ["", "✅ Guards за час: <b>0</b>"]

    # Blocked strategies из state
    blocked = risk_details.get("blocked_strategies") or {}
    if blocked:
        lines += ["", f"🚧 Заблокированных стратегий: <b>{len(blocked)}</b>"]
        for strat in list(blocked)[:6]:
            lines.append(f"  • <code>{_esc(strat)}</code>")

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
