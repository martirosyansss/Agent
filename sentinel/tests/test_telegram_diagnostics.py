"""
Tests for diagnostic Telegram formatters (/diag /why /events /health).

Эти команды — операторский инструмент: "что бот делает и почему". Если
они падают или выдают нечитаемое сообщение в момент инцидента, оператор
остаётся без видимости. Поэтому проверяем три вещи:

1. Форматтеры не падают на пустом/частичном state (реальный сценарий:
   бот только стартовал, большинства полей ещё нет).
2. HTML-экранирование работает — reasons типа ``"EMA50<EMA200"`` не
   рвут parse_mode="HTML".
3. Ключевые наблюдаемые сигналы реально попадают в сообщение
   (gate name из rejection, component name из component_error и т.п.).
"""
from __future__ import annotations

import time

import pytest

from telegram_bot.formatters import (
    format_diagnostics,
    format_events,
    format_health,
    format_why,
)


def _ts(offset_sec: float = 0) -> int:
    return int((time.time() + offset_sec) * 1000)


# ──────────────────────────────────────────────
# format_diagnostics
# ──────────────────────────────────────────────


def test_diagnostics_empty_state_does_not_raise():
    # Only the header is guaranteed — everything else is "missing data".
    text = format_diagnostics({})
    assert "ДИАГНОСТИКА" in text
    assert "PAPER" in text  # default mode
    assert "NORMAL" in text  # default risk_state


def test_diagnostics_shows_paused_state():
    text = format_diagnostics({"trading_paused": True, "mode": "live", "risk_state": "NORMAL"})
    assert "на паузе" in text
    assert "LIVE" in text


def test_diagnostics_includes_ml_verdict_per_symbol():
    state = {
        "trading_symbols": ["BTCUSDT"],
        "last_cycle_ts_per_symbol": {"BTCUSDT": _ts(-30)},
        "standing_ml_per_symbol": {
            "BTCUSDT": {
                "prob": 0.42,
                "decision": "block",
                "ref_strategy": "ema_crossover_rsi",
            },
        },
    }
    text = format_diagnostics(state)
    assert "BTCUSDT" in text
    assert "block" in text
    assert "0.42" in text
    assert "ema_crossover_rsi" in text


def test_diagnostics_renders_red_risk_checks():
    state = {
        "risk_details": {
            "market_data_age_sec": 5.2,
            "risk_checks": {
                "state_ok": True,
                "daily_loss_ok": True,
                "positions_ok": False,  # ← слоты заняты
                "exposure_ok": True,
                "daily_trades_ok": True,
                "hourly_trades_ok": True,
                "cooldown_ok": True,
            },
        },
    }
    text = format_diagnostics(state)
    assert "⛔" in text  # есть хотя бы одна красная отметка


def test_diagnostics_escapes_html_in_strategy_name():
    state = {
        "activity": {
            "strategies_loaded": ["<script>"],  # паранойя
            "current_regime": "trending<up>",
        },
    }
    text = format_diagnostics(state)
    assert "&lt;script&gt;" in text
    assert "trending&lt;up&gt;" in text


# ──────────────────────────────────────────────
# format_why
# ──────────────────────────────────────────────


def test_why_flags_trading_paused():
    state = {"trading_paused": True, "risk_state": "NORMAL"}
    text = format_why(state, events=[])
    assert "на паузе" in text


def test_why_flags_stop_state():
    state = {"trading_paused": False, "risk_state": "STOP"}
    text = format_why(state, events=[])
    assert "STOP" in text


def test_why_groups_rejections_by_gate():
    events = [
        {"type": "signal_rejected", "ts": _ts(-5),  "gate": "liquidity_gate", "reason": "low volume", "symbol": "BTCUSDT"},
        {"type": "signal_rejected", "ts": _ts(-10), "gate": "liquidity_gate", "reason": "low volume", "symbol": "BTCUSDT"},
        {"type": "signal_rejected", "ts": _ts(-20), "gate": "ml_filter",      "reason": "p=0.32",     "symbol": "BTCUSDT"},
    ]
    text = format_why({"risk_state": "NORMAL"}, events)
    # гейт с большим count должен быть выше
    assert text.index("liquidity_gate") < text.index("ml_filter")
    assert "2×" in text
    assert "1×" in text


def test_why_filters_by_symbol():
    events = [
        {"type": "signal_rejected", "ts": _ts(), "gate": "ml_filter", "reason": "x", "symbol": "BTCUSDT"},
        {"type": "signal_rejected", "ts": _ts(), "gate": "ml_filter", "reason": "x", "symbol": "ETHUSDT"},
        {"type": "signal_rejected", "ts": _ts(), "gate": "ml_filter", "reason": "x", "symbol": "ETHUSDT"},
    ]
    text = format_why({"risk_state": "NORMAL"}, events, symbol="ETHUSDT")
    # для ETHUSDT — 2 rejection, а BTCUSDT отфильтрован
    assert "2×" in text
    assert "ETHUSDT" in text


def test_why_escapes_angle_brackets_in_reason():
    events = [
        {"type": "signal_rejected", "ts": _ts(), "gate": "regime_filter",
         "reason": "EMA50<EMA200", "symbol": "BTCUSDT"},
    ]
    text = format_why({"risk_state": "NORMAL"}, events)
    # reason с "<EMA200>" не должен попасть в HTML как тег
    assert "EMA50&lt;EMA200" in text
    assert "<EMA200>" not in text


def test_why_reports_healthy_when_no_blockers():
    text = format_why({"risk_state": "NORMAL", "trading_paused": False}, events=[])
    assert "Блокировок не найдено" in text


# ──────────────────────────────────────────────
# format_events
# ──────────────────────────────────────────────


def test_events_empty():
    assert "пустой" in format_events([]).lower()


def test_events_renders_variety_of_types():
    events = [
        {"type": "signal_generated", "ts": _ts(-1), "symbol": "BTCUSDT", "strategy": "ema"},
        {"type": "signal_rejected",  "ts": _ts(-2), "gate": "ml_filter", "reason": "low p", "symbol": "BTCUSDT"},
        {"type": "guard_tripped",    "ts": _ts(-3), "guard": "drawdown_breaker", "reason": "-5%"},
        {"type": "component_error",  "ts": _ts(-4), "component": "ml_predictor", "severity": "critical", "reason": "boom"},
        {"type": "position_closed",  "ts": _ts(-5), "symbol": "BTCUSDT", "pnl": 12.5},
    ]
    text = format_events(events)
    assert "signal_generated" not in text  # type is rendered via icon, not raw name
    assert "ema" in text
    assert "ml_filter" in text
    assert "drawdown_breaker" in text
    assert "ml_predictor" in text
    assert "pnl=" in text


def test_events_respects_limit():
    events = [{"type": "signal_rejected", "ts": _ts(-i), "gate": f"g{i}", "reason": "x"} for i in range(30)]
    text = format_events(events, limit=5)
    # только 5 строк-событий (header не считает)
    assert text.count("🚫") == 5


# ──────────────────────────────────────────────
# format_health
# ──────────────────────────────────────────────


def test_health_healthy_when_no_errors():
    state = {"risk_details": {"market_data_age_sec": 2.0}}
    text = format_health(state, events=[])
    assert "HEALTHY" in text
    assert "✅" in text


def test_health_degraded_on_recent_errors():
    events = [
        {"type": "component_error", "ts": _ts(-10), "component": "ml_predictor", "severity": "error", "reason": "x"},
        {"type": "component_error", "ts": _ts(-20), "component": "ml_predictor", "severity": "error", "reason": "x"},
    ]
    state = {"risk_details": {"market_data_age_sec": 1.0}}
    text = format_health(state, events)
    assert "DEGRADED" in text
    assert "ml_predictor" in text
    assert "2×" in text


def test_health_critical_on_critical_severity():
    events = [
        {"type": "component_error", "ts": _ts(-10), "component": "collector", "severity": "critical", "reason": "x"},
    ]
    text = format_health({"risk_details": {"market_data_age_sec": 1.0}}, events)
    assert "CRITICAL" in text


def test_health_ignores_events_older_than_hour():
    old = _ts(-4000)  # ~67 min назад
    events = [
        {"type": "component_error", "ts": old, "component": "ml_predictor", "severity": "error", "reason": "x"},
    ]
    text = format_health({"risk_details": {"market_data_age_sec": 1.0}}, events)
    # Старая ошибка не должна влиять на статус
    assert "HEALTHY" in text
    assert "Ошибок компонентов за час: <b>0</b>" in text


def test_health_flags_stale_quotes():
    state = {"risk_details": {"market_data_age_sec": 500.0}}
    text = format_health(state, events=[])
    # 500s > 120s → красная метка
    assert "🚨" in text
    assert "500.0s" in text
