# 📘 ТЕХНИЧЕСКОЕ ЗАДАНИЕ V1.5 (SELF-LEARNING)
# AI QUANT TRADING SYSTEM — "SENTINEL"
# (Домашний ПК, Binance Spot, Начинающий трейдер)

**Дата:** 12 апреля 2026
**Обновлено:** 12 апреля 2026 — 6-Strategy Arsenal + Trade Analyzer (самообучение) + ML Skill Test на исторических данных
**Статус:** Утверждено
**Версия:** 1.5 (Self-Learning Edition)

> **История версий:**
> - V1.0: Базовая архитектура, 12 модулей
> - V1.1: Risk-Hardened — 30+ рисков, 4 новых модуля безопасности
> - V1.2: Financial-Optimized — переход на swing (1h/4h), SL 3%, TP 5%
> - V1.3: Multi-Strategy — Grid Trading + Mean Reversion + авто-выбор стратегии
> - V1.4: 6-Strategy Arsenal — Bollinger Breakout + DCA Bot + MACD Divergence
> - **V1.5: Self-Learning — Trade Analyzer (3 уровня обучения на своих ошибках)**
>
> **Цель V1.5:** дойти до $200/мес через 15-18 месяцев при внешнем пополнении капитала,
> самообучении бота и без ослабления risk-limits.

---

# 📋 СОДЕРЖАНИЕ

1. [Профиль пользователя и ограничения](#1-профиль-пользователя-и-ограничения)
2. [Общая концепция и философия](#2-общая-концепция-и-философия)
3. [Архитектура системы](#3-архитектура-системы)
4. [Этапы разработки (Roadmap)](#4-этапы-разработки-roadmap)
5. [Модуль 1: Data Collector](#5-модуль-1-data-collector)
6. [Модуль 2: Database Layer](#6-модуль-2-database-layer)
7. [Модуль 3: Feature Engine](#7-модуль-3-feature-engine)
8. [Модуль 4: Strategy Engine (6 стратегий)](#8-модуль-4-strategy-engine)
9. [Модуль 5: Risk Sentinel](#9-модуль-5-risk-sentinel)
10. [Модуль 6: Execution Engine](#10-модуль-6-execution-engine)
11. [Модуль 7: Position Manager](#11-модуль-7-position-manager)
12. [Модуль 8: Paper Trading](#12-модуль-8-paper-trading)
13. [Модуль 9: Backtest Engine](#13-модуль-9-backtest-engine)
14. [Модуль 10: Telegram Bot](#14-модуль-10-telegram-bot)
15. [Модуль 11: Web Dashboard](#15-модуль-11-web-dashboard)
16. [Модуль 12: Logging & Monitoring](#16-модуль-12-logging--monitoring)
17. [Безопасность](#17-безопасность)
18. [Модуль 13: Circuit Breakers](#18-модуль-13-circuit-breakers)
19. [Модуль 14: Watchdog](#19-модуль-14-watchdog)
20. [Модуль 15: Data Integrity Guard](#20-модуль-15-data-integrity-guard)
21. [Модуль 16: Anti-Corruption Layer](#21-модуль-16-anti-corruption-layer)
22. [**НОВОЕ V1.5:** Модуль 17: Trade Analyzer (Самообучение)](#22-модуль-17-trade-analyzer)
23. [Полный аудит рисков (30+ сценариев)](#23-полный-аудит-рисков-30-сценариев)
24. [Структура проекта](#24-структура-проекта)
25. [Конфигурация](#25-конфигурация)
26. [Запуск и остановка](#26-запуск-и-остановка)
27. [Критерии успеха](#27-критерии-успеха)
28. [Глоссарий](#28-глоссарий)
29. [План исправления рисков](#29-план-исправления-рисков)

---

# 1. ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ И ОГРАНИЧЕНИЯ

## 1.1 Профиль

| Параметр | Значение |
|---|---|
| Опыт трейдинга | Новичок |
| Опыт программирования | Начинающий (Python, работаем вместе с AI) |
| Стартовый капитал | $100–500 |
| Максимальная допустимая потеря | $50 (абсолютный лимит) |
| Время на систему | 1–2 часа/день |
| Биржа | Binance |
| Рынок | СПОТ (без плеча, без фьючерсов) |
| Торговые пары | BTC/USDT, ETH/USDT |
| ОС | Windows |
| ПК | Мощный (16GB+ RAM, SSD, быстрый интернет) |

## 1.2 Жёсткие ограничения

| Ограничение | Причина |
|---|---|
| ❌ НЕТ фьючерсов | Новичок → можно потерять больше, чем вложил |
| ❌ НЕТ плеча (leverage) | $100–500 → потеря всего за минуты при плече |
| ❌ НЕТ вывода через API | Безопасность → API key без права withdrawal |
| ❌ НЕТ маржинальной торговли | Та же причина что и фьючерсы |
| ✅ ТОЛЬКО спот BUY/SELL | Купил → продал, максимальная потеря = стоимость покупки |

## 1.3 Почему именно спот BTC + ETH

- **Ликвидность**: спред 0.01%, ордера исполняются мгновенно
- **Предсказуемость**: поведение цены хорошо изучено
- **Безопасность**: невозможно потерять больше, чем вложили
- **Комиссии**: 0.1% (maker/taker), при BNB оплате — 0.075%

---

# 2. ОБЩАЯ КОНЦЕПЦИЯ И ФИЛОСОФИЯ

## 2.1 Приоритеты системы (в порядке важности)

```
1. 🛡 БЕЗОПАСНОСТЬ    — система ограничивает открытие нового риска и стремится удерживать дневной убыток в пределах $50
2. 🔒 КОНТРОЛЬ        — пользователь видит ВСЁ и может остановить ВСЁ
3. 📊 СТАБИЛЬНОСТЬ    — система работает без сбоев 24/7
4. 🎯 ПРЕДСКАЗУЕМОСТЬ — поведение системы логично и объяснимо
5. 💰 ПРИБЫЛЬ         — последний приоритет, только после выполнения 1–4
```

## 2.2 Режимы работы системы

```
ЭТАП 1: SIGNAL MODE (сигналы)
  → Система анализирует рынок
  → Отправляет сигналы в Telegram
  → Пользователь решает сам: торговать или нет
  → Длительность: 2–4 недели

ЭТАП 2: PAPER TRADING (виртуальная торговля) ← СТАРТУЕМ ЗДЕСЬ
  → Реальные данные с биржи
  → Виртуальное исполнение (без реальных денег)
  → Полное логирование результатов
  → Длительность: 2–4 недели

ЭТАП 3: LIVE MICRO (реальная торговля, мини капитал)
  → Торговля на $50–100
  → Жёсткие лимиты
  → Длительность: 2–4 недели

ЭТАП 4: LIVE FULL (полный автомат)
  → Весь капитал $100–500
  → Автоматическая торговля 24/7
  → Telegram уведомления
```

## 2.3 Ключевое правило

> **Система НИКОГДА не переходит на следующий этап автоматически.**
> Только пользователь вручную меняет режим после анализа результатов.

## 2.4 Честные гарантии и границы системы

> **Важно:** на spot-рынке невозможно честно гарантировать абсолютный потолок фактического убытка
> при гэпах, flash crash, проскальзывании, делее API, потере интернета или отказе биржи.

ТЗ использует три разных уровня ограничений:

1. **Hard limits на открытие нового риска**
  → система не откроет позицию/ордер выше заданных лимитов.
2. **Soft limits на ожидаемый дневной убыток**
  → при достижении лимита система обязана прекратить новые входы и перейти в STOP.
3. **Биржевая защитная инфраструктура для live**
  → live-торговля разрешена только при наличии exchange-native protective orders
  (OCO или эквивалентных stop/take-profit ордеров на стороне биржи).

Следствие:
- Risk Sentinel ограничивает **новый** риск, но не отменяет рыночный гэп.
- Watchdog снижает время реакции на отказ, но не заменяет биржевой stop.
- Заявления вида "невозможно потерять больше X" трактуются как операционная цель,
  а не как математически гарантированный потолок фактического убытка.

---

# 3. АРХИТЕКТУРА СИСТЕМЫ

## 3.1 Общая схема

```
                    ┌─────────────────┐
                    │   CONFIG (.env)  │
                    └────────┬────────┘
                             │
┌──────────────┐    ┌────────▼────────┐    ┌──────────────────┐
│  BINANCE API │◄──►│  DATA COLLECTOR  │───►│  SQLite DATABASE │
│  (WebSocket) │    │  (real-time)     │    │  (trades, OHLCV) │
└──────────────┘    └─────────────────┘    └────────┬─────────┘
                                                     │
                                            ┌────────▼─────────┐
                                            │  FEATURE ENGINE   │
                                            │  (индикаторы)     │
                                            └────────┬─────────┘
                                                     │
                                            ┌────────▼─────────┐
                                            │  STRATEGY ENGINE  │
                                            │  (сигналы)        │
                                            └────────┬─────────┘
                                                     │
                              ┌───────────────────────┤
                              │                       │
                    ┌─────────▼──────────┐  ┌────────▼─────────┐
                    │   RISK SENTINEL    │  │  TELEGRAM BOT     │
                    │   (проверка)       │  │  (уведомления)    │
                    └─────────┬──────────┘  └──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  EXECUTION ENGINE  │
                    │  (Paper / Live)    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  POSITION MANAGER  │
                    │  (PnL, трекинг)    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  WEB DASHBOARD     │
                    │  (localhost:8080)   │
                    └────────────────────┘
```

## 3.2 Технологический стек

| Компонент | Технология | Почему |
|---|---|---|
| Язык | Python 3.11+ | Простой, огромная экосистема для трейдинга |
| База данных | SQLite | Не нужен сервер, файл на диске, достаточно для спота |
| WebSocket | `websockets` / `aiohttp` | Async, быстрый |
| Binance API | `python-binance` или `ccxt` | Готовые библиотеки |
| Telegram | `python-telegram-bot` | Официальная библиотека |
| Web Dashboard | FastAPI + простой HTML/JS | Лёгкий, быстрый |
| Индикаторы | `pandas` + `ta-lib` / `pandas-ta` | Стандарт для quant |
| Логирование | `loguru` | Простой и мощный |
| Конфиг | `.env` + `pydantic-settings` | Безопасное хранение ключей |
| Async | `asyncio` | Для параллельной работы без блокировок |

## 3.3 Принципы кода

```
1. Каждый модуль — отдельный файл/папка
2. Модули общаются через чёткие интерфейсы (dataclasses)
3. Все настройки — в config, НЕ в коде
4. Все действия — логируются
5. Все ошибки — перехватываются и обрабатываются
6. Код читаемый, с комментариями на русском
```

---

# 4. ЭТАПЫ РАЗРАБОТКИ (ROADMAP)

## Этап 0: Фундамент (Неделя 1)
- [ ] Структура проекта
- [ ] Конфигурация (.env, settings)
- [ ] Логирование
- [ ] Базовые dataclasses (Trade, Signal, Position)

## Этап 1: Сбор данных (Неделя 1–2)
- [ ] Подключение к Binance WebSocket
- [ ] Получение trades и свечей BTC/USDT, ETH/USDT
- [ ] Сохранение в SQLite
- [ ] Auto reconnect при разрыве

## Этап 2: Анализ и сигналы (Неделя 2–3)
- [ ] Feature Engine (индикаторы: EMA, RSI, MACD, Volume)
- [ ] Strategy Engine (простая стратегия EMA crossover + RSI)
- [ ] Генерация сигналов BUY / SELL / HOLD

## Этап 3: Telegram бот (Неделя 3)
- [ ] Отправка сигналов в Telegram
- [ ] Команды: /status, /pnl, /stop, /start
- [ ] Подтверждение сделок через Telegram (полуавтомат)

## Этап 4: Paper Trading (Неделя 3–4) ← ОСНОВНОЙ СТАРТ
- [ ] Виртуальный кошелёк ($500)
- [ ] Симуляция исполнения ордеров
- [ ] PnL трекинг
- [ ] Логирование всех виртуальных сделок

## Этап 5: Risk Sentinel (Неделя 4–5)
- [ ] Жёсткие лимиты ($50/день, 20 сделок/час)
- [ ] Kill-switch (аварийная остановка)
- [ ] State machine (NORMAL → REDUCED → STOP)

## Этап 6: Backtest Engine (Неделя 5–6)
- [ ] Тестирование стратегии на исторических данных
- [ ] Симуляция комиссий и проскальзывания
- [ ] Отчёт: Win Rate, Sharpe, Max Drawdown

## Этап 7: Web Dashboard (Неделя 6–7)
- [ ] Страница с PnL графиком
- [ ] Текущие позиции
- [ ] Список сделок
- [ ] Кнопки управления (Start/Stop/Kill)

## Этап 8: Live Trading (Неделя 8+)
- [ ] Подключение реального Execution Engine
- [ ] Старт с $50–100
- [ ] Мониторинг 24/7

---

# 5. МОДУЛЬ 1: DATA COLLECTOR

## 5.1 Назначение
Сбор рыночных данных в реальном времени с Binance.

## 5.2 Источники данных

> **V1.2:** Переход с 1m/5m на 1h/4h свечи для снижения частоты торговли и комиссий.
> При $500 капитала частая торговля убыточна из-за комиссий (см. Финансовый анализ).

| Тип данных | Источник | Частота | Символы |
|---|---|---|---|
| Trades | Binance WebSocket `@trade` | Real-time | BTCUSDT, ETHUSDT |
| Свечи (1h) | Binance WebSocket `@kline_1h` | 1 час | BTCUSDT, ETHUSDT |
| Свечи (4h) | Binance WebSocket `@kline_4h` | 4 часа | BTCUSDT, ETHUSDT |
| Свечи (1d) | Binance WebSocket `@kline_1d` или REST | 1 день | BTCUSDT, ETHUSDT |
| Свечи (1m) | Binance WebSocket `@kline_1m` | 1 мин | BTCUSDT, ETHUSDT |
| Order Book | Binance REST API (top 5) | Каждые 30 сек | BTCUSDT, ETHUSDT |

> **Примечание:** Свечи 1m используются только для мониторинга и stop-loss.
> Торговые решения принимаются на 1h и 4h таймфреймах.
> Mean Reversion и Strategy Selector НЕ включаются в live/paper, пока доступны 1d свечи,
> ADX и полный набор regime-features.

## 5.3 Формат данных

### Trade (сделка на бирже)
```python
@dataclass
class MarketTrade:
    timestamp: int        # Unix ms
    symbol: str           # "BTCUSDT"
    price: float          # 67234.50
    quantity: float       # 0.001
    is_buyer_maker: bool  # True = продажа, False = покупка
```

### Candle (свеча)
```python
@dataclass
class Candle:
    timestamp: int    # Unix ms начала свечи
    symbol: str       # "BTCUSDT"
    interval: str     # "1m", "1h" или "4h"
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades_count: int
```

## 5.4 Требования

| Требование | Описание |
|---|---|
| Auto Reconnect | При потере WebSocket — переподключение через 1, 3, 10, 30 секунд |
| Валидация данных | Проверка: price > 0, volume > 0, timestamp не из будущего |
| Дедупликация | Уникальность по (timestamp + symbol + price) для trades |
| Heartbeat | Ping каждые 30 секунд, если нет данных > 60 сек — reconnect |
| Логирование | Каждое подключение/отключение/ошибка записывается в лог |
| Graceful Shutdown | При остановке — корректное закрытие WebSocket |

## 5.5 Обработка ошибок

```
Ошибка соединения → retry с экспоненциальной задержкой (1s, 3s, 10s, 30s, 60s)
Невалидные данные → логируем, пропускаем, не сохраняем
Binance maintenance → ждём, логируем, уведомление в Telegram
Превышен rate limit → пауза 60 секунд, затем retry
```

---

# 6. МОДУЛЬ 2: DATABASE LAYER

## 6.1 Выбор: SQLite

**Почему SQLite, а не ClickHouse:**
- Не нужен отдельный сервер
- Для 2 торговых пар на спот — более чем достаточно
- Файл на диске — легко бэкапить
- Python имеет встроенную поддержку (`sqlite3`)

## 6.2 Схема таблиц

### trades — сырые сделки с биржи
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,        -- Unix ms
    symbol TEXT NOT NULL,              -- "BTCUSDT"
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    is_buyer_maker INTEGER NOT NULL,   -- 0 или 1
    UNIQUE(timestamp, symbol, price, quantity)
);
CREATE INDEX idx_trades_symbol_time ON trades(symbol, timestamp);
```

### candles — свечи (OHLCV)
```sql
CREATE TABLE candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,            -- "1m", "5m"
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    trades_count INTEGER DEFAULT 0,
    UNIQUE(timestamp, symbol, interval)
);
CREATE INDEX idx_candles_symbol_time ON candles(symbol, interval, timestamp);
```

### signals — сигналы стратегии
```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,           -- "BUY", "SELL", "HOLD"
    confidence REAL NOT NULL,          -- 0.0 — 1.0
    strategy TEXT NOT NULL,            -- "ema_crossover_rsi"
    features TEXT,                     -- JSON с значениями индикаторов
    created_at TEXT DEFAULT (datetime('now'))
);
```

### orders — ордера (реальные и виртуальные)
```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,                -- "BUY", "SELL"
    order_type TEXT NOT NULL,          -- "MARKET", "LIMIT"
    quantity REAL NOT NULL,
    price REAL,                        -- NULL для MARKET
    status TEXT NOT NULL,              -- "PENDING", "FILLED", "CANCELLED", "FAILED"
    exchange_order_id TEXT,            -- ID ордера на бирже (NULL для paper)
    fill_price REAL,                   -- Цена исполнения
    fill_quantity REAL,
    commission REAL DEFAULT 0,
    is_paper INTEGER DEFAULT 1,        -- 1 = paper trading, 0 = real
    created_at TEXT DEFAULT (datetime('now'))
);
```

### positions — текущие позиции
```sql
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,                -- "LONG" (купили, ждём роста)
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    status TEXT DEFAULT 'OPEN',        -- "OPEN", "CLOSED"
    opened_at TEXT DEFAULT (datetime('now')),
    closed_at TEXT,
    is_paper INTEGER DEFAULT 1
);
```

### daily_stats — дневная статистика
```sql
CREATE TABLE daily_stats (
    date TEXT PRIMARY KEY,             -- "2026-04-12"
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0,
    max_drawdown REAL DEFAULT 0,
    total_commission REAL DEFAULT 0,
    is_paper INTEGER DEFAULT 1
);
```

### strategy_trades — завершённые сделки стратегий
```sql
CREATE TABLE strategy_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL UNIQUE,       -- UUID сделки стратегии
    signal_id INTEGER,
    symbol TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    market_regime TEXT,
    timestamp_open TEXT NOT NULL,
    timestamp_close TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    quantity REAL NOT NULL,
    pnl_usd REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    is_win INTEGER NOT NULL,
    confidence REAL,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    rsi_at_entry REAL,
    adx_at_entry REAL,
    volume_ratio_at_entry REAL,
    exit_reason TEXT,
    hold_duration_hours REAL,
    max_drawdown_during_trade REAL,
    max_profit_during_trade REAL,
    commission_usd REAL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_strategy_trades_strategy_time ON strategy_trades(strategy_name, timestamp_close);
CREATE INDEX idx_strategy_trades_regime_time ON strategy_trades(market_regime, timestamp_close);
```

### ml_model_registry — история моделей ML
```sql
CREATE TABLE ml_model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    trained_from TEXT NOT NULL,
    trained_to TEXT NOT NULL,
    test_from TEXT NOT NULL,
    test_to TEXT NOT NULL,
    train_samples INTEGER NOT NULL,
    test_samples INTEGER NOT NULL,
    precision REAL,
    recall REAL,
    roc_auc REAL,
    uplift_profit_factor REAL,
    uplift_drawdown REAL,
    rollout_mode TEXT NOT NULL,         -- off / shadow / block
    is_active INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
  );
  CREATE INDEX idx_ml_model_registry_active ON ml_model_registry(is_active, created_at);
  ```

## 6.3 Политика хранения

| Данные | Срок хранения | Примечание |
|---|---|---|
| Trades | 7 дней | Архивируются в CSV, удаляются из БД |
| Candles | 90 дней | Нужны для бэктестов |
| Signals | Бессрочно | Малый объём |
| Orders | Бессрочно | Важно для аудита |
| Positions | Бессрочно | Важно для аудита |
  | Strategy Trades | Бессрочно | Основа для Trade Analyzer и ML |
  | ML Model Registry | Бессрочно | Нужен аудит качества и откат модели |

---

# 7. МОДУЛЬ 3: FEATURE ENGINE

## 7.1 Назначение
Вычисление технических индикаторов из сырых рыночных данных.

## 7.2 Список индикаторов

### Трендовые (определяют направление рынка)
| Индикатор | Параметры | Описание |
|---|---|---|
| EMA 9 | period=9 | Быстрая скользящая средняя |
| EMA 21 | period=21 | Медленная скользящая средняя |
| EMA 50 | period=50 | Среднесрочный тренд |
| ADX | period=14 | Сила тренда для Strategy Selector |
| MACD | fast=12, slow=26, signal=9 | Схождение/расхождение скользящих |

### Осцилляторы (определяют перекупленность/перепроданность)
| Индикатор | Параметры | Описание |
|---|---|---|
| RSI | period=14 | Индекс относительной силы (0–100) |
| Stochastic RSI | period=14 | Более чувствительный RSI |

### Волатильность
| Индикатор | Параметры | Описание |
|---|---|---|
| Bollinger Bands | period=20, std=2 | Каналы волатильности |
| ATR | period=14 | Средний истинный диапазон |

### Объём
| Индикатор | Параметры | Описание |
|---|---|---|
| Volume MA | period=20 | Средний объём за 20 свечей |
| Volume Ratio | - | Текущий объём / средний объём |
| OBV | - | On-Balance Volume (давление покупок/продаж) |

### Производные
| Индикатор | Описание |
|---|---|
| Price Change % | Изменение цены за 1, 5, 15, 60 мин |
| Spread | Разница bid/ask (из order book) |
| Momentum | Rate of Change за N свечей |

## 7.3 Вход и выход

### Вход
```python
# История свечей по таймфреймам, достаточная для стратегий V1/V2/V3
candles_by_interval: dict[str, list[Candle]]  # 1m, 1h, 4h, 1d
```

### Выход
```python
@dataclass
class FeatureVector:
    timestamp: int
    symbol: str
    
    # Трендовые
    ema_9: float
    ema_21: float
    ema_50: float
    adx: float
    macd: float
    macd_signal: float
    macd_histogram: float
    
    # Осцилляторы
    rsi: float
    stoch_rsi: float
    
    # Волатильность
    bb_upper: float
    bb_middle: float
    bb_lower: float
    atr: float
    
    # Объём
    volume_ratio: float
    obv: float
    
    # Производные
    price_change_1m: float
    price_change_5m: float
    price_change_15m: float
    momentum: float
    spread: float
```

## 7.4 Требования

- Пересчёт при каждой новой релевантной свече:
  1m для мониторинга, 1h для входов, 4h для фильтров, 1d для trend filter Mean Reversion
- Если недостаточно данных для активной стратегии — возвращать `None` и НЕ генерировать сигнал
- Все значения нормализованы и валидны (без NaN, без inf)
- Strategy Selector запрещён к запуску без ADX, ATR и Bollinger features на 4h

---

# 8. МОДУЛЬ 4: STRATEGY ENGINE

## 8.1 Назначение
Анализ feature vectors и генерация торговых сигналов.

## 8.2 Стратегия V1: EMA Crossover + RSI Filter (SWING, 1h/4h)

> **V1.2:** Стратегия адаптирована под SWING TRADING (1-3 сделки/день).
> Таймфрейм: 1h для входов, 4h для подтверждения тренда.
> Цель: ловить движения 3-8%, а не 0.5-1%.

**Это простая, проверенная стратегия для начинающих.**

### Логика BUY (покупка):
```
ЕСЛИ:
  1. EMA 9 пересекает EMA 21 СНИЗУ ВВЕРХ на 1h свече (golden cross)
  2. RSI (1h) < 70 (не перекуплен)
  3. Volume Ratio > 1.0 (объём выше среднего)
  4. Цена выше EMA 50 на 4h (общий тренд вверх — подтверждение)
  5. НОВОЕ V1.2: Confidence ≥ 0.75 (только сильные сигналы)
ТО:
  → Сигнал BUY
  → Confidence = нормализованная сумма факторов (0.0–1.0)
```

### Логика SELL (продажа):
```
ЕСЛИ:
  1. EMA 9 пересекает EMA 21 СВЕРХУ ВНИЗ на 1h свече (death cross)
  2. RSI > 30 (не перепродан)  
  3. Volume Ratio > 0.8
  ИЛИ:
  4. Цена упала на 3% от entry_price (stop-loss)     ← V1.2: было 2%
  5. Цена выросла на 5% от entry_price (take-profit)  ← V1.2: было 3%
ТО:
  → Сигнал SELL
```

### Почему 3% stop / 5% profit (V1.2):
```
Risk:Reward = 1:1.67 (на каждый $1 риска — $1.67 потенциала)
При Win Rate 50%: (50% × 5%) - (50% × 3%) = +1% на сделку
При Win Rate 45%: (45% × 5%) - (55% × 3%) = +0.6% на сделку
Даже при 45% Win Rate стратегия прибыльная!
```

### Логика HOLD (ничего не делать):
```
ЕСЛИ ни BUY, ни SELL условия не выполнены
ТО → HOLD
```

## 8.3 Параметры стратегии (настраиваемые)

```python
@dataclass
class StrategyConfig:
    # EMA
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    
    # Таймфреймы (V1.2)
    signal_timeframe: str = "1h"    # Таймфрейм для сигналов
    trend_timeframe: str = "4h"     # Таймфрейм для подтверждения тренда
    
    # RSI
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Volume
    min_volume_ratio: float = 1.0
    
    # Risk per trade (V1.2: расширены для swing trading)
    stop_loss_pct: float = 3.0      # -3% → продаём (было 2%)
    take_profit_pct: float = 5.0    # +5% → фиксируем прибыль (было 3%)
    
    # Confidence threshold (V1.2: повышен для качества)
    min_confidence: float = 0.75    # Сигналы ниже 0.75 игнорируются (было 0.6)
    
    # Position sizing
    max_position_pct: float = 20.0  # Макс 20% капитала на 1 сделку
    
    # Частота (V1.2: ограничена для экономии комиссий)
    max_trades_per_day: int = 6     # Макс 3 BUY + 3 SELL за день
```

## 8.4 Выход — SignalOutput

```python
@dataclass
class Signal:
    timestamp: int
    symbol: str
    direction: str          # "BUY", "SELL", "HOLD"
    confidence: float       # 0.0 — 1.0
    strategy_name: str      # "ema_crossover_rsi_v1"
    reason: str             # Человекочитаемое объяснение
    suggested_quantity: float
    stop_loss_price: float  # Цена для автоматического стоп-лосса
    take_profit_price: float
    features: FeatureVector # Снимок индикаторов в момент сигнала
```

## 8.5 Защита от overtrading (УСИЛЕНО V1.2)

```
- Максимум 1 BUY сигнал на символ, если уже есть открытая позиция
- Минимальный интервал между сигналами: 1 час (было 5 минут)
- Максимум 3 сделок в день (было 20/час) — экономия $1.40/день на комиссиях
- Confidence < 0.75 → сигнал логируется, но НЕ отправляется (было 0.6)
- Макс 6 ордеров/день (3 BUY + 3 SELL)
```

### Экономия на комиссиях (V1.2):
```
БЫЛО: 10 сделок/день × $100 × 0.2% = $2.00/день = $60/месяц (12% капитала!)
СТАЛО: 3 сделки/день × $100 × 0.2% = $0.60/день = $18/месяц (3.6% капитала)
ЭКОНОМИЯ: $42/месяц — это деньги, которые остаются у вас
```

## 8.6 Стратегия V2: Grid Trading (НОВОЕ V1.3)

> **Grid Trading** — автоматическая покупка/продажа в заданном ценовом диапазоне.
> Идеальна для **бокового рынка** (sideways), когда цена колеблется без чёткого тренда.
> Источник: Investopedia, Gate.io Academy — проверенная стратегия с 2010-х годов.

### Принцип работы:
```
1. Определяем ВЕРХНЮЮ и НИЖНЮЮ границу ценового диапазона
2. Делим диапазон на N уровней (грид-линий)
3. На каждом уровне НИЖЕ текущей цены — ордер на ПОКУПКУ
4. На каждом уровне ВЫШЕ текущей цены — ордер на ПРОДАЖУ
5. Когда цена падает → покупаем, растёт → продаём купленное

Пример для BTCUSDT ($66,000-$70,000), 8 уровней:
Уровень 8: $70,000 → SELL
Уровень 7: $69,500 → SELL
Уровень 6: $69,000 → SELL
Уровень 5: $68,500 → SELL
--- Текущая цена: $68,200 ---
Уровень 4: $68,000 → BUY
Уровень 3: $67,500 → BUY
Уровень 2: $67,000 → BUY
Уровень 1: $66,500 → BUY
```

### GridConfig:
```python
@dataclass
class GridConfig:
    # Диапазон
    upper_price: float = 0.0        # Верхняя граница (авто или ручная)
    lower_price: float = 0.0        # Нижняя граница (авто или ручная)
    num_grids: int = 8              # Количество уровней
    grid_type: str = "arithmetic"   # "arithmetic" или "geometric"
    
    # Авто-определение диапазона (на основе Bollinger Bands 4h)
    auto_range: bool = True         # Автоматический расчёт границ
    bb_period: int = 20             # Период Bollinger Bands
    bb_std: float = 2.0             # Количество стандартных отклонений
    
    # Размер ордера
    investment_per_grid: float = 0.0  # Авто: total_invest / num_grids
    total_investment_pct: float = 30.0  # 30% капитала на Grid
    
    # Безопасность
    min_profit_per_grid_pct: float = 0.3  # Мин. прибыль на грид (после комиссии)
    max_unrealized_loss_pct: float = 5.0  # Стоп если позиция теряет 5%
    auto_rebalance: bool = True     # Перестроить грид при выходе цены за границы
    
    # Условие активации
    activate_when: str = "sideways"  # Только при боковом рынке
```

### Математика Grid на $500 (30% = $150):
```
Капитал на Grid: $150
Уровней: 8
Инвестиция на уровень: $150 / 8 = $18.75
Спред между уровнями: ($70,000 - $66,000) / 8 = $500 (0.74%)
Прибыль на 1 цикл (buy+sell): 0.74% - 0.2% комиссия = 0.54% = $0.10
Если 3 цикла/день: $0.30/день = $9/месяц

ЭТО ПАССИВНЫЙ ДОХОД — работает 24/7 без сигналов
```

### Защита Grid:
```
- Grid ОСТАНАВЛИВАЕТСЯ если цена выходит за границы на >2%
- Макс 30% капитала на Grid (никогда больше)
- Обязательный stop-loss для всех Grid позиций
- Авто-перестройка не чаще 1 раз в 24ч
- При резком падении (>3% за час) — Grid отключается, Risk Sentinel берёт контроль
```

---

## 8.7 Стратегия V3: Mean Reversion (RSI Extreme) (НОВОЕ V1.3)

> **Mean Reversion** — покупка на экстремально низком RSI, продажа на высоком.
> Принцип: цена всегда возвращается к среднему значению.
> Источник: Investopedia — одна из классических торговых стратегий.

### Логика:
```
BUY (покупка) когда:
  1. RSI (4h) < 25 (экстремальная перепроданность)
  2. Цена ниже нижней Bollinger Band (4h)
  3. Volume Ratio > 1.5 (повышенный объём = капитуляция)
  4. EMA 50 (1d) всё ещё растёт (общий тренд не сломан)
  → Confidence = 0.8+ для таких экстремумов

SELL (продажа) когда:
  1. RSI (4h) > 75 (перекупленность) 
  2. Цена выше верхней Bollinger Band (4h)
  ИЛИ:
  3. Цена вернулась к EMA 21 (4h) — "возврат к среднему"
  4. Stop-loss: -4% от entry (шире, т.к. ловим разворот)
  5. Take-profit: +6% от entry
```

### MeanReversionConfig:
```python
@dataclass
class MeanReversionConfig:
    # RSI экстремумы
    rsi_oversold: float = 25.0      # BUY ниже этого уровня
    rsi_overbought: float = 75.0    # SELL выше этого уровня
    rsi_timeframe: str = "4h"       # Таймфрейм RSI
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Risk per trade
    stop_loss_pct: float = 4.0      # Шире чем EMA (ловим разворот)
    take_profit_pct: float = 6.0    # R:R = 1:1.5
    
    # Position sizing
    max_position_pct: float = 15.0  # Макс 15% капитала (более рискованная)
    
    # Частота
    max_trades_per_day: int = 2     # Макс 2 Mean Rev сделки/день
    min_confidence: float = 0.80    # Только сильные сигналы
    
    # Условие активации
    activate_when: str = "any"      # Работает в любом рынке (ловит экстремумы)
```

### Математика Mean Reversion на $500 (15% = $75):
```
Капитал на сделку: $75
Средняя прибыль: +6% × 50% побед - 4% × 50% проигрышей = +1%/сделку
Комиссия: 0.2%
Чистая прибыль: +0.8%/сделку = $0.60
Если 1 сделка/день × 20 дней: $12/месяц

Редкие сигналы (1-3 в неделю), но высокий Win Rate (~55-60%)
```

---

## 8.8 Strategy Selector — Авто-выбор стратегии (ОБНОВЛЕНО V1.4, 6 стратегий)

> **Ключевое нововведение V1.3:** система автоматически определяет тип рынка
> и активирует оптимальную стратегию (или комбинацию).
> **V1.4:** расширено до 6 стратегий — добавлены Bollinger Breakout, DCA Bot, MACD Divergence.

### Определение типа рынка:
```python
@dataclass
class MarketRegime:
    """Определяется каждые 4 часа по 4h свечам"""
    
    TRENDING_UP = "trending_up"     # EMA 9 > EMA 21 > EMA 50 + ADX > 25
    TRENDING_DOWN = "trending_down" # EMA 9 < EMA 21 < EMA 50 + ADX > 25
    SIDEWAYS = "sideways"           # ADX < 20 + цена в Bollinger Band
    VOLATILE = "volatile"           # ATR > 2× среднего
    UNKNOWN = "unknown"             # Неопределённый режим

class StrategySelector:
    """
    Выбирает активные стратегии на основе текущего рынка.
    Обновляет распределение капитала между стратегиями.
    V1.4: 6 стратегий + DCA всегда активен.
    """
    
    # ВАЖНО: это целевые бюджеты стратегий, а НЕ обещание одновременной экспозиции.
    # Фактическая загрузка капитала всегда ограничена:
    #   - max_total_exposure_pct <= 60%
    #   - max_open_positions <= 2 (направленных) + 1 grid + 1 DCA
    #   - max_position_pct <= 20% на направленную позицию
    ALLOCATION_TABLE = {
      # Режим рынка: {стратегия: % капитала}
      "trending_up": {
        "ema_crossover": 25,
        "grid": 5,
        "mean_reversion": 0,
        "bollinger_breakout": 15,
        "dca_bot": 5,
        "macd_divergence": 0,
        "reserve": 50,
      },
      "trending_down": {
        "ema_crossover": 0,
        "grid": 0,
        "mean_reversion": 5,
        "bollinger_breakout": 0,
        "dca_bot": 10,          # DCA покупает на падении!
        "macd_divergence": 5,   # Ищет разворот
        "reserve": 80,
      },
      "sideways": {
        "ema_crossover": 5,
        "grid": 25,
        "mean_reversion": 10,
        "bollinger_breakout": 5,
        "dca_bot": 5,
        "macd_divergence": 0,
        "reserve": 50,
      },
      "volatile": {
        "ema_crossover": 5,
        "grid": 0,
        "mean_reversion": 5,
        "bollinger_breakout": 10,
        "dca_bot": 5,
        "macd_divergence": 5,
        "reserve": 70,
      },
      "unknown": {
        "ema_crossover": 5,
        "grid": 0,
        "mean_reversion": 0,
        "bollinger_breakout": 0,
        "dca_bot": 5,
        "macd_divergence": 0,
        "reserve": 90,
      },
    }
```

  ### Консервативная ожидаемая доходность 6-Strategy Arsenal:
```
БЫЧИЙ РЫНОК (trending_up):
    EMA Crossover (25%): ~8%/мес = $10
    Bollinger Breakout (15%): ~6%/мес = $4.5
    Grid Trading (5%): ~3%/мес = $0.75
    DCA Bot (5%): покупает, не фиксирует
    ИТОГО: ~$15-16/мес (3.0-3.2%)

БОКОВОЙ РЫНОК (sideways):
    Grid Trading (25%): ~5%/мес = $6.25
    Mean Reversion (10%): ~4%/мес = $2
    EMA Crossover (5%): ~$0.5
    Bollinger Breakout (5%): ~$0.5
    DCA Bot (5%): покупает по плану
    ИТОГО: ~$9-10/мес (1.8-2.0%)

МЕДВЕЖИЙ РЫНОК (trending_down):
    DCA Bot (10%): накупает дёшево → будущая прибыль
    Mean Reversion (5%): ~$0.5 (ловит отскоки)
    MACD Divergence (5%): ~$0.5 (ловит развороты)
    Всё остальное выключено, 80% в кэше
    ИТОГО: ~$0.5-1/мес (0.1-0.2%) — СОХРАНЯЕМ КАПИТАЛ + накапливаем позицию

СРЕДНЕЕ ЗА ГОД (смешанный рынок):
    ~2.0-2.5%/мес после комиссий и с учётом reserve
    $500 × 2.0-2.5% = ~$10-12.5/мес на старте
  
  С реинвестированием через 12 месяцев:
    $500 → $500 × 1.022^12 ≈ $649
    $649 × 2.0-2.5% ≈ $13-16/мес
  
  С добавлением $100/мес из зарплаты:
    Через 12 мес: ~$1,849 → $37-46/мес
    Через 18 мес: ~$2,549 → $51-64/мес
    Цель $200/мес: при депозите ~$8,000-10,000 (месяц 24-30 с пополнениями)

  Эти оценки НЕ являются KPI безопасности и НЕ должны использоваться для ослабления лимитов.
```

---

## 8.9 Дорожная карта стратегий (ОБНОВЛЕНО V1.4)

| Фаза | Стратегия | Когда | Условие |
|---|---|---|---|
| Фаза 1 | EMA Crossover (V1) | Сразу | Paper trading 2 недели |
| Фаза 2 | + Grid Trading (V2) | Месяц 2 | После 1 месяца paper trading |
| Фаза 3 | + Mean Reversion (V3) | Месяц 3 | После 2 месяцев данных |
| Фаза 4 | + Strategy Selector | Месяц 3 | Когда все 3 стратегии работают |
| Фаза 5 | + Bollinger Breakout (V4) | Месяц 4 | После проверки волатильности |
| Фаза 6 | + DCA Bot (V5) | Месяц 1 | Запускается почти сразу (low risk) |
| Фаза 7 | + MACD Divergence (V6) | Месяц 5 | После 3+ месяцев данных для дивергенций |
| Фаза 8 | Trade Analyzer (Level 1) | С дня 1 | Статистика с первой сделки |
| Фаза 9 | Trade Analyzer (Level 2) | Месяц 6+ | Адаптивная оптимизация параметров |
| Фаза 10 | Trade Analyzer (Level 3) | Месяц 9+ | ML предсказания (scikit-learn) |

---

## 8.10 Стратегия V4: Bollinger Band Breakout (НОВОЕ V1.4)

> **Идея:** Торгует пробои Bollinger Bands с подтверждением объёмом.
> При сжатии полос (squeeze) предвещает сильное движение.
> **Режим рынка:** trending_up, volatile (работает при сильных движениях)
> **Таймфрейм:** 4h

### Индикаторы:
```python
@dataclass
class BollingerBreakoutConfig:
    """Конфигурация Bollinger Breakout Strategy"""
    
    bb_period: int = 20                 # Период Bollinger Bands
    bb_std_dev: float = 2.0             # Стандартные отклонения
    volume_confirm_mult: float = 1.5    # Объём > 1.5× средний = подтверждение
    squeeze_lookback: int = 20          # Период для определения squeeze
    squeeze_threshold: float = 0.05     # Если bandwidth < 5% → squeeze
    
    # Risk params
    stop_loss_pct: float = 3.0          # SL: -3% от входа
    take_profit_pct: float = 6.0        # TP: +6% (R:R = 1:2)
    trailing_stop_pct: float = 2.0      # Трейлинг стоп после +3%
    max_position_pct: float = 15.0      # Макс 15% капитала
    min_confidence: float = 0.70        # Мин. уверенность
```

### Логика входа (BUY):
```
СИГНАЛ BUY (Bollinger Breakout Long):
  1. Цена ЗАКРЫЛАСЬ выше верхней Bollinger Band (4h close > upper_bb)
  2. Объём текущей свечи > 1.5× среднего объёма за 20 свечей
  3. Предшествовал squeeze: bandwidth за последние 5 свечей < threshold
  4. RSI(14) < 80 (не в экстремальной перекупленности)
  5. ADX > 20 (есть тренд)
  
CONFIDENCE = 0.6 (base)
  + 0.10 если volume > 2× среднего
  + 0.10 если squeeze длился > 10 свечей (сильное сжатие)
  + 0.05 если ADX > 30 (сильный тренд)
  + 0.05 если EMA 9 > EMA 21 (подтверждение тренда)
```

### Логика выхода (SELL):
```
СИГНАЛ SELL:
  1. Цена вернулась внутрь полос (close < upper_bb) — ослабление импульса
  2. ИЛИ trailing stop сработал (после +3% от входа, trail -2%)
  3. ИЛИ stop-loss -3%
  4. ИЛИ take-profit +6%
  5. ИЛИ RSI > 85 (экстремальная перекупленность)
```

### Реализация:
```python
class BollingerBreakoutStrategy(BaseStrategy):
    """
    Стратегия V4: Торговля пробоев Bollinger Bands.
    
    Лучше всего работает:
    - При выходе из бокового движения в тренд
    - На волатильных рынках с объёмом
    
    Плохо работает:
    - На чистых боковиках (false breakouts)
    - При низком объёме
    """
    
    NAME = "bollinger_breakout"
    REQUIRED_FEATURES = ["bb_upper", "bb_lower", "bb_middle", "bb_bandwidth",
                         "volume", "volume_sma_20", "rsi_14", "adx", "ema_9", "ema_21"]
    
    def generate_signal(self, features: FeatureVector) -> Optional[Signal]:
        # 1. Check squeeze (was bandwidth compressed recently?)
        is_squeeze = self._detect_squeeze(features)
        
        # 2. Breakout detection
        if features.close > features.bb_upper:
            # Volume confirmation
            vol_ratio = features.volume / features.volume_sma_20
            if vol_ratio < self.config.volume_confirm_mult:
                return None  # Нет подтверждения объёмом
            
            if features.rsi_14 > 80:
                return None  # Перекупленность
                
            # Calculate confidence
            confidence = 0.60
            if vol_ratio > 2.0:
                confidence += 0.10
            if is_squeeze:
                confidence += 0.10
            if features.adx > 30:
                confidence += 0.05
            if features.ema_9 > features.ema_21:
                confidence += 0.05
            
            if confidence >= self.config.min_confidence:
                return Signal(
                    action="BUY",
                    strategy=self.NAME,
                    confidence=min(confidence, 0.95),
                    stop_loss_pct=self.config.stop_loss_pct,
                    take_profit_pct=self.config.take_profit_pct,
                    reason=f"BB Breakout: close={features.close:.2f} > upper={features.bb_upper:.2f}, "
                           f"vol_ratio={vol_ratio:.1f}x, squeeze={is_squeeze}"
                )
        
        return None
    
    def _detect_squeeze(self, features: FeatureVector) -> bool:
        """Определяет, было ли сжатие полос (предвестник сильного движения)"""
        return features.bb_bandwidth < self.config.squeeze_threshold
```

### Ожидаемые характеристики:
```
Win Rate: ~50-55% (пробои часто ложные, но профитные сделки крупнее)
Avg Win/Loss Ratio: 1.8-2.2 (R:R = 1:2 за счёт trailing stop)
Частота: 1-2 сигнала в неделю (пробои редки)
Лучший рынок: переход sideways → trending
Worst case: -3% per trade (strict stop-loss)
```

---

## 8.11 Стратегия V5: DCA Bot (НОВОЕ V1.4)

> **Идея:** Dollar-Cost Averaging — покупка фиксированной суммы через интервалы.
> Smart DCA: покупает БОЛЬШЕ при падении цены (buy the dip).
> **Режим рынка:** ALL (работает в любом режиме, но лучше всего в trending_down/sideways)
> **Таймфрейм:** Daily (1d)
> **Риск:** Минимальный (маленькие покупки, усреднение входа)

### Конфигурация:
```python
@dataclass
class DCABotConfig:
    """Конфигурация DCA Bot Strategy"""
    
    # Базовые параметры
    base_amount_usd: float = 10.0       # Базовая покупка: $10
    interval_hours: int = 24            # Интервал: каждые 24 часа
    max_daily_buys: int = 3             # Макс покупок в день (при dip)
    
    # Smart DCA: множители при падении
    dip_thresholds: list = None         # По умолчанию ниже
    # [(-3%, 1.5x), (-5%, 2.0x), (-10%, 3.0x)]
    # Если цена упала на 3% за 24ч → покупай 1.5x ($15)
    # Если на 5% → 2x ($20)
    # Если на 10% → 3x ($30)
    
    # Safety
    max_total_invested_pct: float = 40.0  # Макс 40% капитала через DCA
    stop_dca_drawdown_pct: float = 15.0   # Если drawdown > 15% → пауза DCA
    min_balance_reserve_usd: float = 100.0 # Всегда оставлять $100 в USDT
    
    # Take Profit
    take_profit_pct: float = 8.0        # TP: +8% от средней цены входа
    partial_tp_pct: float = 5.0         # Частичная фиксация при +5%
    partial_tp_sell_pct: float = 30.0    # Продать 30% позиции при partial TP
    
    def __post_init__(self):
        if self.dip_thresholds is None:
            self.dip_thresholds = [
                (-3.0, 1.5),
                (-5.0, 2.0),
                (-10.0, 3.0),
            ]
```

### Логика работы:
```
КАЖДЫЕ 24 ЧАСА (или interval_hours):
  
  1. Проверить: total_invested < max_total_invested_pct?
  2. Проверить: balance > min_balance_reserve_usd?
  3. Проверить: drawdown < stop_dca_drawdown_pct?
  
  Если всё ОК:
    price_change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
    
    multiplier = 1.0  # базовый
    for (threshold, mult) in dip_thresholds:
        if price_change_24h <= threshold:
            multiplier = mult  # берём максимальный mult
    
    buy_amount = base_amount_usd * multiplier
    
    → Signal(action="BUY", strategy="dca_bot", amount=buy_amount)
  
TAKE PROFIT:
  avg_entry_price = total_cost / total_quantity
  current_profit_pct = (current_price - avg_entry_price) / avg_entry_price * 100
  
  Если current_profit_pct >= partial_tp_pct (5%):
    → Продать 30% позиции (фиксация прибыли)
  
  Если current_profit_pct >= take_profit_pct (8%):
    → Продать оставшееся (полная фиксация)
```

### Реализация:
```python
class DCABotStrategy(BaseStrategy):
    """
    Стратегия V5: Smart Dollar-Cost Averaging.
    
    Уникальность: единственная стратегия, которая ПОКУПАЕТ при падении.
    Другие стратегии уходят в reserve при медвежьем рынке,
    а DCA наоборот — усредняет вход для будущего роста.
    
    ВАЖНО: DCA не пытается предсказать рынок.
    Она использует математику усреднения.
    """
    
    NAME = "dca_bot"
    REQUIRED_FEATURES = ["close", "price_change_24h"]
    
    def __init__(self, config: DCABotConfig):
        self.config = config
        self.last_buy_time = None
        self.total_invested = 0.0
        self.total_quantity = 0.0
        self.avg_entry_price = 0.0
    
    def generate_signal(self, features: FeatureVector) -> Optional[Signal]:
        now = datetime.utcnow()
        
        # Check interval
        if self.last_buy_time:
            hours_since = (now - self.last_buy_time).total_seconds() / 3600
            if hours_since < self.config.interval_hours:
                # Check for dip opportunity (extra buy)
                return self._check_dip_buy(features)
        
        # Regular DCA buy
        return self._generate_dca_signal(features)
    
    def _generate_dca_signal(self, features: FeatureVector) -> Optional[Signal]:
        # Safety checks
        if not self._safety_checks_pass(features):
            return None
        
        # Calculate multiplier based on price change
        multiplier = 1.0
        price_change = features.price_change_24h
        for threshold, mult in self.config.dip_thresholds:
            if price_change <= threshold:
                multiplier = max(multiplier, mult)
        
        buy_amount = self.config.base_amount_usd * multiplier
        
        return Signal(
            action="BUY",
            strategy=self.NAME,
            confidence=0.80,  # DCA always confident (it's math, not prediction)
            amount_usd=buy_amount,
            reason=f"DCA: regular buy ${buy_amount:.0f} "
                   f"(change_24h={price_change:+.1f}%, mult={multiplier}x)"
        )
    
    def check_take_profit(self, current_price: float) -> Optional[Signal]:
        """Проверяет условия фиксации прибыли"""
        if self.total_quantity <= 0:
            return None
        
        profit_pct = (current_price - self.avg_entry_price) / self.avg_entry_price * 100
        
        if profit_pct >= self.config.take_profit_pct:
            return Signal(action="SELL", strategy=self.NAME, confidence=0.90,
                         reason=f"DCA TP: +{profit_pct:.1f}% (full exit)")
        
        if profit_pct >= self.config.partial_tp_pct:
            return Signal(action="SELL_PARTIAL", strategy=self.NAME, confidence=0.85,
                         sell_pct=self.config.partial_tp_sell_pct,
                         reason=f"DCA partial TP: +{profit_pct:.1f}% (sell {self.config.partial_tp_sell_pct}%)")
        
        return None
```

### Ожидаемые характеристики:
```
Win Rate: ~65-70% (усреднение входа сглаживает волатильность)
Avg Profit per Cycle: +5-8% (полный цикл DCA → take profit)
Частота покупок: 1/день (base) + 1-2 extra при dip
Лучший рынок: медвежий → бычий переход (накупил дёшево → продал дорого)
Worst case: drawdown 15% → DCA пауза, ждём восстановления
Капитал: $10-30/день → макс 40% от депозита
```

---

## 8.12 Стратегия V6: MACD Divergence (НОВОЕ V1.4)

> **Идея:** Ищет расхождения (дивергенции) между ценой и MACD.
> Бычья дивергенция: цена делает lower low, а MACD — higher low → разворот вверх.
> **Режим рынка:** trending_down → trending_up (ловит развороты)
> **Таймфрейм:** 4h
> **Риск:** Средний (развороты непредсказуемы, но R:R хороший)

### Индикаторы:
```python
@dataclass
class MACDDivergenceConfig:
    """Конфигурация MACD Divergence Strategy"""
    
    # MACD params
    macd_fast: int = 12                 # Fast EMA period
    macd_slow: int = 26                 # Slow EMA period
    macd_signal: int = 9                # Signal line period
    
    # Divergence detection
    lookback_candles: int = 30          # Искать дивергенцию за 30 свечей
    min_divergence_bars: int = 5        # Мин. расстояние между пиками/впадинами
    
    # Confirmation
    require_rsi_confirm: bool = True    # RSI подтверждение
    rsi_oversold: float = 35.0          # RSI < 35 для бычьей дивергенции
    rsi_overbought: float = 65.0        # RSI > 65 для медвежьей
    require_volume_confirm: bool = True # Объём подтверждение
    
    # Risk params
    stop_loss_pct: float = 3.5          # SL: -3.5%
    take_profit_pct: float = 7.0        # TP: +7% (R:R = 1:2)
    max_position_pct: float = 15.0      # Макс 15% капитала
    min_confidence: float = 0.72        # Мин. уверенность
```

### Логика обнаружения дивергенции:
```
БЫЧЬЯ ДИВЕРГЕНЦИЯ (Bullish Divergence) → BUY SIGNAL:
  
  Шаг 1: Найти 2 последних минимума цены (swing lows)
    price_low_1 (раньше) > price_low_2 (позже)
    = цена сделала LOWER LOW
  
  Шаг 2: Найти MACD histogram в тех же точках
    macd_low_1 (раньше) < macd_low_2 (позже)
    = MACD сделал HIGHER LOW
  
  Шаг 3: Расчёт расхождения
    Цена: ↘ (падает) vs MACD: ↗ (растёт) = дивергенция
    → Моментум ослабевает, вероятен разворот вверх

МЕДВЕЖЬЯ ДИВЕРГЕНЦИЯ (Bearish Divergence) → SELL SIGNAL:
  
  price_high_1 < price_high_2 (цена: higher high)
  macd_high_1 > macd_high_2 (MACD: lower high)
  → Моментум ослабевает на вершине, вероятен разворот вниз
  
CONFIDENCE:
  base = 0.55
  + 0.10 если RSI подтверждает (oversold для бычьей / overbought для медвежьей)
  + 0.08 если объём растёт на точке дивергенции (объём подтверждает разворот)
  + 0.07 если дивергенция длинная (> 15 свечей между точками)
  + 0.05 если MACD histogram пересёк ноль
```

### Реализация:
```python
class MACDDivergenceStrategy(BaseStrategy):
    """
    Стратегия V6: Торговля дивергенций MACD.
    
    Самая сложная стратегия в арсенале.
    Ищет моменты, когда цена идёт в одну сторону,
    а моментум (MACD) — в другую. Это предвестник разворота.
    
    ВАЖНО: Дивергенции часто дают ложные сигналы.
    Подтверждение RSI + объёмом ОБЯЗАТЕЛЬНО.
    """
    
    NAME = "macd_divergence"
    REQUIRED_FEATURES = ["close", "macd_hist", "rsi_14", "volume", "volume_sma_20"]
    
    def generate_signal(self, features: FeatureVector) -> Optional[Signal]:
        # 1. Detect divergence
        divergence = self._detect_divergence(features)
        if divergence is None:
            return None
        
        div_type, confidence = divergence
        
        # 2. RSI confirmation (mandatory)
        if self.config.require_rsi_confirm:
            if div_type == "bullish" and features.rsi_14 > self.config.rsi_oversold:
                return None  # RSI не в зоне перепроданности
            if div_type == "bearish" and features.rsi_14 < self.config.rsi_overbought:
                return None  # RSI не в зоне перекупленности
            confidence += 0.10
        
        # 3. Volume confirmation
        if self.config.require_volume_confirm:
            vol_ratio = features.volume / features.volume_sma_20
            if vol_ratio > 1.3:
                confidence += 0.08
        
        # 4. Check minimum confidence
        if confidence < self.config.min_confidence:
            return None
        
        action = "BUY" if div_type == "bullish" else "SELL"
        
        return Signal(
            action=action,
            strategy=self.NAME,
            confidence=min(confidence, 0.95),
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
            reason=f"MACD {div_type} divergence: conf={confidence:.2f}, "
                   f"RSI={features.rsi_14:.1f}"
        )
    
    def _detect_divergence(self, features: FeatureVector) -> Optional[tuple]:
        """
        Анализирует историю цен и MACD за lookback_candles свечей.
        Возвращает (тип, базовая уверенность) или None.
        """
        price_lows = self._find_swing_lows(features.price_history)
        macd_lows = self._find_swing_lows(features.macd_hist_history)
        
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            # Бычья дивергенция: цена lower low, MACD higher low
            if (price_lows[-1].value < price_lows[-2].value and 
                macd_lows[-1].value > macd_lows[-2].value):
                bars_apart = price_lows[-1].index - price_lows[-2].index
                confidence = 0.55
                if bars_apart > 15:
                    confidence += 0.07
                return ("bullish", confidence)
        
        price_highs = self._find_swing_highs(features.price_history)
        macd_highs = self._find_swing_highs(features.macd_hist_history)
        
        if len(price_highs) >= 2 and len(macd_highs) >= 2:
            # Медвежья дивергенция: цена higher high, MACD lower high
            if (price_highs[-1].value > price_highs[-2].value and 
                macd_highs[-1].value < macd_highs[-2].value):
                bars_apart = price_highs[-1].index - price_highs[-2].index
                confidence = 0.55
                if bars_apart > 15:
                    confidence += 0.07
                return ("bearish", confidence)
        
        return None
```

### Ожидаемые характеристики:
```
Win Rate: ~45-50% (дивергенции часто ложные)
Avg Win/Loss Ratio: 2.0-2.5 (R:R = 1:2, профитные сделки крупнее)
Частота: 1-3 сигнала в месяц (дивергенции редки)
Лучший рынок: развороты трендов (bottom/top fishing)
Worst case: -3.5% per trade (strict stop-loss)
КРИТИЧЕСКАЯ ЗАМЕТКА: Не торговать без RSI + Volume подтверждения!
```

---

# 9. МОДУЛЬ 5: RISK SENTINEL (КРИТИЧЕСКИЙ МОДУЛЬ)

## 9.1 Назначение
> **Абсолютный защитный слой.** Ни один ордер не проходит без одобрения Risk Sentinel.
> Это единственный модуль, который имеет право ЗАБЛОКИРОВАТЬ торговлю.
> **V1.1:** Risk Sentinel работает в связке с Circuit Breakers (раздел 18) и Watchdog (раздел 19).

## 9.2 Жёсткие лимиты (HARD LIMITS)

```python
@dataclass
class RiskLimits:
    # === ДНЕВНЫЕ ЛИМИТЫ (V1.2: снижены для swing trading) ===
    max_daily_loss_usd: float = 50.0       # Макс потеря за день в $
    max_daily_loss_pct: float = 10.0       # Макс потеря за день в %
    max_daily_trades: int = 6              # V1.2: было 50, теперь 6 (swing)
    
    # === ПОЗИЦИОННЫЕ ЛИМИТЫ ===
    max_position_pct: float = 20.0         # Макс 20% капитала на 1 позицию
    max_total_exposure_pct: float = 60.0   # Макс 60% капитала в позициях
    max_open_positions: int = 2            # V1.2: было 3, теперь 2 (меньше risk)
    
    # === ЧАСТОТНЫЕ ЛИМИТЫ (V1.2: резко снижены) ===
    max_trades_per_hour: int = 2           # V1.2: было 20, теперь 2
    max_trades_per_minute: int = 1         # V1.2: было 3, теперь 1
    min_trade_interval_sec: int = 1800     # V1.2: было 30 сек, теперь 30 мин
    
    # === МИНИМАЛЬНЫЕ РАЗМЕРЫ ===
    min_order_usd: float = 10.0            # Binance минимум
    max_order_usd: float = 100.0           # Макс размер ордера
    
    # === STOP-LOSS (ОБЯЗАТЕЛЕН) (V1.2: расширен для swing) ===
    max_loss_per_trade_pct: float = 3.0    # V1.2: было 2%, теперь 3% (swing)
    mandatory_stop_loss: bool = True       # Stop-loss ОБЯЗАТЕЛЕН для каждой сделки
    
    # === КОМИССИЯ КОНТРОЛЬ (НОВОЕ V1.2) ===
    max_daily_commission_pct: float = 1.0  # Макс 1% капитала на комиссии/день
```

## 9.3 Risk Check Pipeline

Каждый сигнал проходит ВСЕ проверки последовательно:

```
Signal → [1] Daily Loss Check
       → [2] Position Limit Check
       → [3] Exposure Check
       → [4] Frequency Check
       → [5] Order Size Check
       → [6] Stop-Loss Check
       → [7] Sanity Check (цена адекватна рынку?)
       → ✅ APPROVED или ❌ REJECTED (с причиной)
```

### Детали каждой проверки:

**[1] Daily Loss Check:**
```
ЕСЛИ realized_pnl_today + unrealized_pnl_today < -max_daily_loss_usd:
  → REJECT("Дневной лимит потерь исчерпан: ${loss}")
  → Перевод в STATE = STOP
  → Уведомление в Telegram: "⚠️ ТОРГОВЛЯ ОСТАНОВЛЕНА: потеря $X за день"
```

**[2] Position Limit Check:**
```
ЕСЛИ open_positions_count >= max_open_positions:
  → REJECT("Достигнут лимит открытых позиций: {count}")
```

**[3] Exposure Check:**
```
ЕСЛИ total_exposure_pct + new_order_pct > max_total_exposure_pct:
  → REJECT("Превышение общей экспозиции: {total}%")
```

**[4] Frequency Check:**
```
ЕСЛИ trades_this_hour >= max_trades_per_hour:
  → REJECT("Слишком много сделок за час: {count}")
ЕСЛИ time_since_last_trade < min_trade_interval_sec:
  → REJECT("Слишком частые сделки")
```

**[5] Order Size Check:**
```
ЕСЛИ order_usd < min_order_usd:
  → REJECT("Ордер меньше минимума: ${amount}")
ЕСЛИ order_usd > max_order_usd:
  → REDUCE order to max_order_usd
```

**[6] Stop-Loss Check:**
```
ЕСЛИ mandatory_stop_loss AND signal.stop_loss_price == None:
  → REJECT("Стоп-лосс обязателен")
ЕСЛИ (entry_price - stop_loss_price) / entry_price > max_loss_per_trade_pct:
  → REJECT("Стоп-лосс слишком широкий: {pct}%")
```

**[7] Sanity Check:**
```
ЕСЛИ abs(signal_price - current_market_price) > 1%:
  → REJECT("Цена сигнала отличается от рыночной > 1%")
ЕСЛИ signal.quantity <= 0:
  → REJECT("Невалидный размер ордера")
```

## 9.4 State Machine (Режимы безопасности)

```
  ┌──────────┐    loss > 30% limit    ┌──────────┐
  │  NORMAL  │ ──────────────────────► │ REDUCED  │
  │          │                         │          │
  │ Полная   │ ◄────────────────────── │ Размер   │
  │ торговля │    manual reset         │ x0.5     │
  └──────────┘                         └────┬─────┘
                                            │
                                       loss > 70% limit
                                            │
                                       ┌────▼─────┐
                                       │   SAFE   │
                                       │          │
                                       │ Только   │
                                       │ SELL     │
                                       └────┬─────┘
                                            │
                                       loss > 100% limit
                                            │
                                       ┌────▼─────┐
                                       │   STOP   │
                                       │          │
                                       │ Всё      │
                                       │ остановл.│
                                       └──────────┘
```

| Состояние | Условие входа | Действия |
|---|---|---|
| NORMAL | По умолчанию | Полная торговля, все стратегии |
| REDUCED | Потеря > $15 за день (30% лимита) | Размер позиций × 0.5, только высокий confidence > 0.8 |
| SAFE | Потеря > $35 за день (70% лимита) | Только закрытие позиций (SELL), новые BUY запрещены |
| STOP | Потеря > $50 за день (100% лимита) | ВСЯ торговля остановлена, ордера отменены |

## 9.5 Kill Switch

```python
async def emergency_stop():
    """
    Аварийная остановка. Вызывается:
    1. Автоматически при STOP state
    2. Вручную через Telegram: /kill
    3. Вручную через Dashboard: кнопка EMERGENCY STOP
    """
    # 1. Отменить все открытые ордера на бирже
    # 2. Закрыть все позиции по рынку (market sell)
    # 3. Остановить Strategy Engine
    # 4. Остановить Execution Engine
    # 5. Отправить уведомление в Telegram
    # 6. Записать в лог: EMERGENCY STOP activated
    # 7. НЕ останавливать Data Collector (данные нужны)
```

## 9.6 Логирование Risk Sentinel

**Каждое решение Risk Sentinel записывается:**
```json
{
    "timestamp": "2026-04-12T14:30:05Z",
    "signal_id": 1234,
    "decision": "REJECTED",
    "reason": "Дневной лимит потерь исчерпан: $52.30",
    "state": "STOP",
    "daily_pnl": -52.30,
    "open_positions": 2,
    "exposure_pct": 35.2,
    "trades_today": 18
}
```

---

# 10. МОДУЛЬ 6: EXECUTION ENGINE

## 10.1 Назначение
Исполнение ордеров: виртуальное (Paper) или реальное (Binance API).

## 10.2 Два режима

### Paper Execution (виртуальное)
```python
class PaperExecutor:
    """
    Имитирует исполнение ордеров без реальных денег.
    - Использует текущую рыночную цену
    - Добавляет симуляцию проскальзывания: +/- 0.05%
    - Учитывает комиссию Binance: 0.1%
    - Записывает результат в БД с is_paper=1
    """
```

### Live Execution (реальное)
```python
class LiveExecutor:
    """
    Реальное исполнение через Binance Spot API.
    - Вход допускается MARKET ордером только для micro/live после pre-flight checks
    - ПОСЛЕ fill ОБЯЗАТЕЛЬНО ставится биржевой protective order
      (OCO или эквивалентный stop-loss/take-profit на стороне Binance)
    - Если protective order не подтверждён биржей → позиция немедленно закрывается, система уходит в STOP
    - ОБЯЗАТЕЛЬНО подтверждение fill через API
    - Все ордера логируются ПЕРЕД отправкой и ПОСЛЕ получения ответа
    - Timeout: если нет ответа за 10 сек → отмена + лог
    """
```

## 10.3 Типы ордеров

**Paper / Signal Mode:** Только MARKET ордера
| Ордер | Описание |
|---|---|
| MARKET BUY | Покупка по текущей рыночной цене |
| MARKET SELL | Продажа по текущей рыночной цене |

**Live Trading:** вход + обязательная биржевая защита
| Ордер | Описание |
|---|---|
| MARKET BUY | Вход только после одобрения Risk Sentinel |
| MARKET SELL | Выход по сигналу / emergency |
| OCO / STOP / TAKE-PROFIT | Биржевая защитная связка, обязательна для live |

## 10.4 Поток исполнения

```
Signal (APPROVED by Risk Sentinel)
    │
    ▼
[1] Pre-flight check
    - Достаточно ли баланса?
    - Ордер >= минимальный размер Binance?
    - Символ торгуется?
    │
    ▼
[2] Запись в лог: "Отправляем ордер..."
    │
    ▼
[3] Отправка ордера (Paper или Binance API)
    │
    ▼
[4] Ожидание подтверждения (timeout 10 сек)
    │
    ├── ✅ Fill получен
  │   → НОВОЕ: Для live немедленно выставить биржевой protective order
  │   → Если protective order не принят → market exit + STOP
    │   → Запись в orders (status=FILLED)
    │   → Обновление Position Manager
    │   → Уведомление в Telegram
    │
    ├── ⏱ Timeout
    │   → Проверка статуса ордера через REST API
    │   → Если pending → отмена
    │   → Лог: "Order timeout"
    │
    └── ❌ Ошибка
        → Лог ошибки
        → Уведомление в Telegram
        → НЕ повторять автоматически
```

## 10.5 Важные правила

```
1. НИКОГДА не доверять "order sent" — ждём fill confirmation
2. НИКОГДА не повторять неудачный ордер автоматически
3. ВСЕГДА сохранять exchange_order_id для сверки
4. ПРИ любой ошибке API → уведомление в Telegram
5. Reconciliation: раз в 5 мин сверяем наши позиции с Binance
6. LIVE ЗАПРЕЩЁН без подтверждённой биржевой stop-защиты
```

---

# 11. МОДУЛЬ 7: POSITION MANAGER

## 11.1 Назначение
Отслеживание текущих позиций и расчёт PnL.

## 11.2 Данные позиции

```python
@dataclass
class Position:
    symbol: str
    side: str               # "LONG" (для спота всегда LONG)
    entry_price: float      # Средняя цена входа
    quantity: float          # Количество монет
    current_price: float    # Текущая цена (обновляется)
    
    # PnL
    unrealized_pnl: float   # (current_price - entry_price) * quantity
    unrealized_pnl_pct: float
    realized_pnl: float     # Зафиксированная прибыль/убыток
    
    # Защита
    stop_loss_price: float
    take_profit_price: float
    
    # Мета
    opened_at: datetime
    is_paper: bool
```

## 11.3 Функции

| Функция | Описание |
|---|---|
| open_position() | Создать новую позицию при BUY |
| close_position() | Закрыть позицию при SELL |
| update_prices() | Обновление текущих цен (каждую секунду) |
| check_stop_loss() | Проверка: цена дошла до stop-loss? → auto SELL |
| check_take_profit() | Проверка: цена дошла до take-profit? → auto SELL |
| get_total_pnl() | Суммарный PnL по всем позициям |
| reconcile() | Сверка с балансом Binance (каждые 5 мин) |

## 11.4 Stop-Loss / Take-Profit (автоматически)

```
При открытии позиции ОБЯЗАТЕЛЬНО устанавливаются (V1.2: обновлено):
  - Stop-Loss: entry_price × (1 - stop_loss_pct/100) = entry - 3%
  - Take-Profit: entry_price × (1 + take_profit_pct/100) = entry + 5%

Пример для BTC купленного по $67,000:
  - Stop-Loss: $64,990 (-3%)      ← V1.2: было $65,660 (-2%)
  - Take-Profit: $70,350 (+5%)    ← V1.2: было $69,010 (+3%)
  - Risk:Reward = 1:1.67

Почему расширили (V1.2):
  На 1h/4h таймфрейме цена BTC может колебаться ±1.5% за час.
  SL 2% на 1h = будет выбивать слишком часто (шум).
  SL 3% на 1h = даёт пространство для нормального колебания.
  
Если цена достигает stop-loss → автоматическая продажа
Если цена достигает take-profit → автоматическая продажа
```

---

# 12. МОДУЛЬ 8: PAPER TRADING

## 12.1 Назначение
Полная симуляция торговли без реальных денег.

## 12.2 Виртуальный кошелёк

```python
@dataclass
class PaperWallet:
    initial_balance: float = 500.0   # Стартовый баланс (виртуальный)
    usdt_balance: float = 500.0      # Текущий баланс USDT
    btc_balance: float = 0.0         # Текущий баланс BTC
    eth_balance: float = 0.0         # Текущий баланс ETH
```

## 12.3 Симуляция реалистичности

| Параметр | Значение | Зачем |
|---|---|---|
| Комиссия | 0.1% от суммы | Как на Binance |
| Проскальзывание | 0.05% | Цена исполнения ≠ цена сигнала |
| Задержка исполнения | 100–500ms | Имитация реальной задержки |
| Минимальный ордер | $10 | Как на Binance |

## 12.4 Метрики Paper Trading

После каждого дня система считает:
```
- Общий PnL ($)
- Общий PnL (%)
- Количество сделок
- Win Rate (% прибыльных)
- Макс просадка (max drawdown)
- Sharpe Ratio (доходность / волатильность)
- Средняя прибыль на сделку
- Средний убыток на сделку
- Profit Factor (суммарная прибыль / суммарный убыток)
```

## 12.5 Условия перехода на Live

**Paper Trading → Live Trading ТОЛЬКО если:**
```
✅ Win Rate > 50% за последние 7 дней
✅ Общий PnL > 0 за последние 7 дней
✅ Max Drawdown < 5% за весь период
✅ Минимум 50 завершённых сделок
✅ Нет критических ошибок в логах
✅ Пользователь ВРУЧНУЮ подтвердил переход
```

---

# 13. МОДУЛЬ 9: BACKTEST ENGINE

## 13.1 Назначение
Тестирование стратегии на исторических данных.

## 13.2 Функции

```python
class BacktestEngine:
    """
    Входные данные: исторические свечи из БД
    Процесс:
      1. Прогоняем стратегию по каждой свече
      2. Генерируем виртуальные сигналы
      3. Симулируем исполнение
      4. Считаем PnL, метрики
      5. При необходимости тестируем ML Skill Score на старых сделках
    Выход: отчёт
    """
```

## 13.3 Отчёт бэктеста

```
═══════════════════════════════════════
       BACKTEST REPORT
═══════════════════════════════════════
 Стратегия:    ema_crossover_rsi_v1
 Период:       2026-01-01 — 2026-04-12
 Символы:      BTC/USDT, ETH/USDT
───────────────────────────────────────
 Начальный баланс:    $500.00
 Конечный баланс:     $XXX.XX
 Общий PnL:           $XX.XX (X.X%)
───────────────────────────────────────
 Всего сделок:        XXX
 Прибыльных:          XXX (XX.X%)
 Убыточных:           XXX (XX.X%)
───────────────────────────────────────
 Макс просадка:       X.X%
 Sharpe Ratio:        X.XX
 Profit Factor:       X.XX
 Средняя сделка:      $X.XX
───────────────────────────────────────
 ⚠️  ВНИМАНИЕ: Backtest ≠ реальность
 Применён коэф. безопасности: 0.7
 Ожидаемый реальный PnL: ~$XX.XX
═══════════════════════════════════════
```

## 13.4 Safety Discount

> **ПРАВИЛО: Реальный результат = Backtest × 0.7**
> Бэктест всегда показывает лучший результат, чем реальность.
> Мы применяем коэффициент 0.7 для реалистичной оценки.

## 13.5 Тест навыка стратегии на исторических данных

> Отдельная функция Backtest Engine должна уметь не только считать PnL,
> но и проверять, насколько стратегия "умеет" работать в текущем типе рынка
> на основе старых, уже закрытых сделок.

```python
def test_strategy_skill_on_history(
  strategy_name: str,
  historical_trades: list[TradeRecord],
  lookback_days: int = 180,
  train_ratio: float = 0.7,
) -> dict:
  """
  1. Взять только старые сделки выбранной стратегии
  2. Отсортировать их по времени
  3. Обучиться на ранней части истории
  4. Протестировать на более новой части истории
  5. Вернуть skill score, precision, recall, expected_pnl
  """
```

Минимальные требования:
- Разделение только по времени, без random shuffle
- В тест попадают только сделки, которых модель ещё не "видела"
- Skill test считается отдельно по каждой стратегии
- Skill test считается отдельно по каждому market regime, если данных достаточно
- Если данных мало, результат помечается как low_confidence и НЕ влияет на торговлю

---

# 14. МОДУЛЬ 10: TELEGRAM BOT

## 14.1 Назначение
Интерфейс управления и уведомлений через Telegram.

## 14.2 Команды

| Команда | Описание |
|---|---|
| `/start` | Запуск бота, приветствие |
| `/status` | Текущий статус системы (режим, PnL, позиции) |
| `/pnl` | Детальный PnL за день/неделю/месяц |
| `/positions` | Список открытых позиций |
| `/trades` | Последние 10 сделок |
| `/stop` | Остановить торговлю (graceful) |
| `/resume` | Возобновить торговлю |
| `/kill` | АВАРИЙНАЯ ОСТАНОВКА (kill switch) |
| `/mode` | Показать/изменить режим (paper/live) |
| `/config` | Показать текущие настройки |
| `/help` | Список команд |

## 14.3 Уведомления (автоматические)

| Событие | Формат |
|---|---|
| Сигнал BUY | `📈 СИГНАЛ BUY BTC/USDT @ $67,234 (confidence: 0.82)` |
| Сигнал SELL | `📉 СИГНАЛ SELL BTC/USDT @ $67,890 (PnL: +$3.28)` |
| Ордер исполнен | `✅ КУПЛЕНО 0.001 BTC @ $67,234 (Paper)` |
| Stop-Loss | `🛑 STOP-LOSS BTC/USDT @ $65,660 (потеря: -$1.57)` |
| Take-Profit | `🎯 TAKE-PROFIT BTC/USDT @ $69,010 (прибыль: +$1.78)` |
| Смена состояния | `⚠️ Risk State: NORMAL → REDUCED (потеря: -$16)` |
| Дневной отчёт | `📊 Итоги дня: PnL $5.23, Win Rate 58%, Сделок: 12` |
| Ошибка | `🚨 ОШИБКА: Потеряно соединение с Binance` |

## 14.4 Подтверждение сделок (полуавтомат)

В режиме Signal Mode бот присылает:
```
📈 СИГНАЛ BUY BTC/USDT

Цена: $67,234.50
Размер: $20.00 (0.000297 BTC)
Stop-Loss: $65,660 (-2.3%)
Take-Profit: $69,010 (+2.6%)
Confidence: 0.82

Причина: EMA9 пересёк EMA21 вверх, RSI=45, Volume×1.3

[✅ Подтвердить] [❌ Отклонить]
```

## 14.5 Безопасность Telegram

```
- Бот отвечает ТОЛЬКО на ваш chat_id (настраивается в .env)
- Все неизвестные сообщения игнорируются
- Команда /kill требует подтверждения
- Логирование всех команд
```

---

# 15. МОДУЛЬ 11: WEB DASHBOARD

## 15.1 Назначение
Визуальная панель мониторинга в браузере (localhost:8080).

## 15.2 Страницы

### Главная (Dashboard)
```
┌─────────────────────────────────────────────┐
│  🟢 СИСТЕМА РАБОТАЕТ    Mode: PAPER TRADING  │
│  State: NORMAL           Uptime: 2д 14ч      │
├─────────────────┬───────────────────────────┤
│  PnL Сегодня    │  PnL Всего                │
│  +$3.27 (+0.7%) │  +$18.42 (+3.7%)          │
├─────────────────┼───────────────────────────┤
│  Открытые       │  Сделок сегодня           │
│  позиции: 2     │  8 (Win: 5, Loss: 3)      │
├─────────────────┴───────────────────────────┤
│  [📈 PnL ГРАФИК ЗА 7 ДНЕЙ]                  │
│                                              │
│  ───────────────────────────────────         │
│                                              │
├──────────────────────────────────────────────┤
│  ОТКРЫТЫЕ ПОЗИЦИИ                            │
│  BTC/USDT  LONG  $67,234  +$1.45 (+0.21%)   │
│  ETH/USDT  LONG  $3,412   -$0.32 (-0.09%)   │
├──────────────────────────────────────────────┤
│  ПОСЛЕДНИЕ СДЕЛКИ                            │
│  14:30  BUY  BTC  $67,234  ✅ Filled         │
│  14:15  SELL ETH  $3,445   ✅ +$0.82         │
│  13:50  BUY  ETH  $3,412   ✅ Filled         │
├──────────────────────────────────────────────┤
│  [🟢 START] [🔴 STOP] [☠️ EMERGENCY STOP]    │
└──────────────────────────────────────────────┘
```

### Технические требования
| Параметр | Значение |
|---|---|
| Backend | FastAPI (Python) |
| Frontend | Простой HTML + Chart.js + vanilla JS |
| Обновление | WebSocket (реальное время) или polling 5 сек |
| Адрес | `http://localhost:8080` |
| Аутентификация | Простой пароль в .env (опционально) |

---

# 16. МОДУЛЬ 12: LOGGING & MONITORING

## 16.1 Уровни логирования

| Уровень | Использование | Пример |
|---|---|---|
| DEBUG | Детали работы модулей | "Feature calculated: RSI=45.2" |
| INFO | Нормальные события | "Signal BUY BTC confidence=0.82" |
| WARNING | Потенциальные проблемы | "WebSocket reconnecting..." |
| ERROR | Ошибки, не блокирующие работу | "Order failed: insufficient balance" |
| CRITICAL | Ошибки, требующие остановки | "Risk limit exceeded, STOP" |

## 16.2 Файлы логов

```
logs/
  sentinel.log          # Общий лог (ротация: 10MB, 5 файлов)
  trades.log            # Только сделки
  risk.log              # Решения Risk Sentinel
  errors.log            # Только ERROR и CRITICAL
```

## 16.3 Формат лога

```
2026-04-12 14:30:05.123 | INFO | strategy | Signal BUY BTCUSDT confidence=0.82 reason="EMA crossover + RSI filter"
2026-04-12 14:30:05.125 | INFO | risk     | APPROVED signal_id=1234 state=NORMAL daily_pnl=$3.27
2026-04-12 14:30:05.230 | INFO | executor | Order FILLED BUY 0.000297 BTC @ $67,234.50 (paper)
```

## 16.4 Healthcheck

Каждые 60 секунд система проверяет:
```
✅ WebSocket подключён к Binance
✅ Последние данные < 2 минут назад
✅ SQLite доступна
✅ Risk Sentinel активен
✅ Telegram бот отвечает
✅ Свободное место на диске > 1GB
```

Если любая проверка не прошла → WARNING в лог + Telegram.

---

# 17. БЕЗОПАСНОСТЬ

## 17.1 API Keys

```
РАСПОЛОЖЕНИЕ: .env файл в корне проекта
ПРАВА API KEY:
  ✅ Enable Reading — чтение данных
  ✅ Enable Spot & Margin Trading — торговля
  ❌ Enable Withdrawals — ЗАПРЕЩЕНО
  ❌ Enable Futures — ЗАПРЕЩЕНО
  
IP RESTRICTION: привязать к домашнему IP (опционально, рекомендуется)
```

## 17.2 Файл .env

```env
# Binance
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# System
TRADING_MODE=paper          # paper / live
LOG_LEVEL=INFO
DASHBOARD_PASSWORD=          # Опционально

# Risk Limits
MAX_DAILY_LOSS_USD=50
MAX_POSITION_PCT=20
MAX_ORDER_USD=100
```

## 17.3 Правила безопасности

```
1. .env НИКОГДА не попадает в git (.gitignore)
2. API ключи НИКОГДА не пишутся в код
3. API ключи НИКОГДА не логируются (маскируются: BNab...xyz)
4. Все HTTP запросы через HTTPS
5. Telegram chat_id проверяется для каждого сообщения
6. Dashboard пароль хешируется (если включён)
7. Регулярный аудит: проверка что withdrawal ВЫКЛЮЧЕН
8. НОВОЕ V1.1: Абсолютные лимиты зашиты в коде (absolute_limits.py)
9. НОВОЕ V1.1: Pre-flight check проверяет API permissions при каждом запуске
10. НОВОЕ V1.1: Критические Telegram команды требуют PIN
11. НОВОЕ V1.1: PID lock файл предотвращает двойной запуск
12. НОВОЕ V1.1: Watchdog контролирует основной процесс извне
```

---

# 18. МОДУЛЬ 13: CIRCUIT BREAKERS (НОВОЕ V1.1)

## 18.1 Назначение
> **Автоматические предохранители, которые останавливают систему при аномалиях.**
> Работают НЕЗАВИСИМО от Risk Sentinel — двойная линия защиты.

## 18.2 Типы Circuit Breakers

### CB-1: Price Anomaly Breaker
```
СРАБАТЫВАЕТ ЕСЛИ:
  Цена BTC/ETH изменилась > 5% за 1 минуту
  
ДЕЙСТВИЕ:
  → Заморозить ВСЕ новые ордера на 5 минут
  → Telegram: "⚡ CIRCUIT BREAKER: аномальное движение цены BTC -6.2% за 1 мин"
  → Логирование: CB_PRICE_ANOMALY
  
ОБЪЯСНЕНИЕ:
  Резкое движение цены = либо flash crash, либо ошибка данных.
  В обоих случаях торговать ОПАСНО.
```

### CB-2: Consecutive Loss Breaker
```
СРАБАТЫВАЕТ ЕСЛИ:
  3 убыточных сделки ПОДРЯД
  
ДЕЙСТВИЕ:
  → Пауза торговли на 30 минут
  → Telegram: "🔴 3 убытка подряд. Пауза 30 мин для анализа"
  → Автоматическое ужесточение confidence > 0.85
  
ОБЪЯСНЕНИЕ:
  Серия потерь = стратегия не работает в текущих условиях.
  Пауза предотвращает каскадные убытки.
```

### CB-3: Spread Anomaly Breaker
```
СРАБАТЫВАЕТ ЕСЛИ:
  Спред bid/ask > 0.5% (нормально: 0.01–0.05%)
  
ДЕЙСТВИЕ:
  → Запрет MARKET ордеров
  → Telegram: "⚠️ Аномальный спред BTC: 0.7%. Торговля приостановлена"
  
ОБЪЯСНЕНИЕ:
  Широкий спред = низкая ликвидность = высокое проскальзывание.
  Market ордер при спреде 0.5% сразу потеряет 0.5%.
```

### CB-4: Volume Anomaly Breaker
```
СРАБАТЫВАЕТ ЕСЛИ:
  Объём за 1 мин > 10× среднего объёма
  ИЛИ
  Объём за 1 мин < 0.1× среднего объёма
  
ДЕЙСТВИЕ:
  → Пауза новых BUY на 10 минут
  → Telegram: "⚠️ Аномальный объём. Возможна манипуляция"
  
ОБЪЯСНЕНИЕ:
  Аномальный объём = pump&dump или новость = высокий риск.
```

### CB-5: API Error Rate Breaker
```
СРАБАТЫВАЕТ ЕСЛИ:
  > 5 ошибок API за последние 5 минут
  
ДЕЙСТВИЕ:
  → ПОЛНАЯ остановка торговли
  → Telegram: "🚨 Множественные ошибки API. Система остановлена"
  → Автоматический retry через 15 минут
  
ОБЪЯСНЕНИЕ:
  Если API постоянно ошибается = проблемы на бирже.
  Продолжать торговлю = слепая торговля.
```

### CB-6: Latency Breaker
```
СРАБАТЫВАЕТ ЕСЛИ:
  Задержка ответа Binance API > 5 секунд (3 раза подряд)
  
ДЕЙСТВИЕ:
  → Пауза новых ордеров
  → Telegram: "⚠️ Высокая задержка API: 7.2 сек"
  
ОБЪЯСНЕНИЕ:
  Высокая задержка = цена может измениться пока ордер летит.
  Для $100 это может быть ±$5 проскальзывания.
```

### CB-7: Balance Mismatch Breaker
```
СРАБАТЫВАЕТ ЕСЛИ:
  Наш расчётный баланс отличается от реального на бирже > 1%
  
ДЕЙСТВИЕ:
  → ПОЛНАЯ остановка торговли
  → Telegram: "🚨 РАССИНХРОНИЗАЦИЯ БАЛАНСА! Ожидали: $487, На бирже: $460"
  → Требуется ручной разбор
  
ОБЪЯСНЕНИЕ:
  Рассинхронизация = потерянный ордер, баг, или чужой доступ к аккаунту.
  НЕЛЬЗЯ торговать пока не выяснена причина.
```

### CB-8: Commission Spike Breaker
```
СРАБАТЫВАЕТ ЕСЛИ:
  Суммарные комиссии за день > 1% от капитала
  
ДЕЙСТВИЕ:
  → Снижение частоты торговли в 2 раза
  → Telegram: "⚠️ Комиссии за день: $8.50 (1.7%). Снижаем частоту"
  
ОБЪЯСНЕНИЕ:
  При капитале $500 и комиссии 0.1% = 50 сделок × $100 = $5.
  Если комиссии > 2% = торговля просто кормит биржу.
```

## 18.3 Состояния Circuit Breakers

```python
@dataclass
class CircuitBreakerState:
    name: str                    # "CB-1: Price Anomaly"
    is_tripped: bool = False     # Сработал?
    tripped_at: datetime = None  # Когда сработал
    cooldown_sec: int = 300      # Через сколько сек авто-сброс
    trip_count_today: int = 0    # Сколько раз сработал сегодня
    
    # Если один CB сработал >= 3 раз за день → PERMANENT STOP
```

## 18.4 Правило эскалации

```
CB сработал 1 раз   → пауза (cooldown)
CB сработал 2 раза  → пауза × 2 + Telegram alert
CB сработал 3+ раз  → ПОЛНАЯ ОСТАНОВКА + "🚨 Система требует ручной проверки"
```

---

# 19. МОДУЛЬ 14: WATCHDOG (СТОРОЖЕВОЙ ПЁС) (НОВОЕ V1.1)

## 19.1 Назначение
> **Независимый процесс, который СЛЕДИТ за основной системой и может её убить.**
> Watchdog НЕ является частью торговой системы — это отдельный скрипт-надзиратель.

## 19.2 Зачем нужен отдельный Watchdog

```
ПРОБЛЕМА: Что если сама торговая система зависнет или сойдёт с ума?
  - Risk Sentinel не поможет — он внутри зависшей системы
  - Kill Switch не поможет — система не обрабатывает команды

РЕШЕНИЕ: Watchdog — отдельный процесс Python, который:
  1. Каждые 30 сек проверяет "жив ли main.py"
  2. Проверяет файл heartbeat.json
  3. Если система не отвечает > 2 мин → аварийное закрытие позиций через API
```

## 19.3 Механизм

### Heartbeat файл (пишет main.py каждые 10 сек):
```json
{
    "timestamp": "2026-04-12T14:30:05Z",
    "status": "running",
    "risk_state": "NORMAL",
    "open_positions": 2,
    "daily_pnl": -3.27,
    "last_trade": "2026-04-12T14:28:00Z",
    "errors_last_hour": 0
}
```

### Watchdog проверяет:
```
1. heartbeat.json обновлён < 2 минут назад?
   НЕТ → система зависла → EMERGENCY ACTION

2. risk_state == "STOP" > 5 минут, но система ещё торгует?
   ДА → Risk Sentinel игнорируется → EMERGENCY ACTION

3. daily_pnl < -max_daily_loss?
   ДА → лимит нарушен → EMERGENCY ACTION

4. Процесс main.py существует?
   НЕТ → система упала → уведомление + попытка перезапуска

5. open_positions > 0 И система не работает?
   ДА → EMERGENCY: закрыть все позиции через прямой API вызов
```

### Emergency Action (Watchdog):
```python
async def watchdog_emergency():
    """
    Watchdog получает доступ к тем же ключам из защищённого источника конфигурации
    и может:
    1. Отменить все ордера на Binance
    2. Продать ВСЕ позиции MARKET SELL
    3. Отправить Telegram: "☠️ WATCHDOG: Система не отвечает.
       Все позиции закрыты принудительно"
    4. Записать лог: watchdog_emergency.log
    5. НЕ перезапускать торговлю автоматически
    """
```

## 19.4 Запуск Watchdog

```powershell
# Watchdog запускается ОТДЕЛЬНО от main.py
python watchdog.py

# Он работает параллельно и НЕ зависит от основной системы
```

## 19.5 Watchdog тоже защищён

```
- Watchdog НЕ имеет доступа к Strategy Engine
- Watchdog НЕ может открыть новые позиции
- Watchdog МОЖЕТ только: отменить ордера + market sell + уведомить
- Watchdog сам логирует свои действия в отдельный файл
```

---

# 20. МОДУЛЬ 15: DATA INTEGRITY GUARD (НОВОЕ V1.1)

## 20.1 Назначение
> **Защита от испорченных, поддельных или аномальных данных.**
> Плохие данные → плохие сигналы → потеря денег.

## 20.2 Уровни валидации

### Уровень 1: Валидация формата (каждое сообщение с биржи)
```
✅ timestamp — число, > 0, не из будущего (не > now + 10 сек)
✅ price — число, > 0, конечное (не NaN, не Infinity)
✅ quantity — число, > 0, конечное
✅ symbol — из разрешённого списка ["BTCUSDT", "ETHUSDT"]
✅ JSON структура — все обязательные поля присутствуют
```

### Уровень 2: Валидация логики (санитарные проверки)
```
✅ Цена BTC в диапазоне $1,000 — $1,000,000 (защита от явных ошибок)
✅ Цена ETH в диапазоне $50 — $100,000
✅ Изменение цены за 1 секунду < 2% (защита от bad tick)
✅ Объём одной сделки < 1000 BTC (фильтр fat-finger errors)
✅ Timestamp монотонно возрастает (защита от replay attack)
```

### Уровень 3: Статистическая валидация (каждую минуту)
```
✅ Количество trades/мин в нормальном диапазоне (±5σ от среднего)
✅ Средняя цена за минуту ≈ close предыдущей свечи (±0.5%)
✅ OHLCV данные консистентны: Low ≤ Open,Close ≤ High
✅ Volume > 0 для каждой свечи
```

### Уровень 4: Cross-validation (каждые 5 минут)
```
✅ Наша цена совпадает с REST API (/api/v3/ticker/price) с точностью 0.1%
✅ Если есть расхождение → использовать REST API как "источник правды"
✅ При расхождении > 3 раз подряд → остановка + "Данные ненадёжны"
```

## 20.3 Действия при обнаружении плохих данных

```
Невалидный формат    → отбросить сообщение, логировать
Аномальная цена      → отбросить, проверить через REST API
Пропуск данных > 1м  → Telegram alert + использовать REST API
Все данные плохие    → ОСТАНОВКА торговли + "🚨 Data feed corrupted"
```

## 20.4 Защита от Stale Data (устаревшие данные)

```
ПРОБЛЕМА: WebSocket отключился, но система этого не заметила
  → Стратегия торгует на старых ценах → убыток

ЗАЩИТА:
  Каждый сигнал содержит data_age = now - last_data_timestamp
  ЕСЛИ data_age > 30 секунд:
    → REJECT сигнал
    → "Данные устарели на {data_age} сек"
  ЕСЛИ data_age > 60 секунд:
    → ПОЛНАЯ остановка торговли
    → Telegram: "🚨 Нет свежих данных > 60 сек!"
```

---

# 21. МОДУЛЬ 16: ANTI-CORRUPTION LAYER (НОВОЕ V1.1)

## 21.1 Назначение
> **Защита от внутренних ошибок системы: баги, race conditions, memory leaks.**

## 21.2 Защита от Double Execution (дублирование ордеров)

```python
class OrderDeduplicator:
    """
    ПРОБЛЕМА: Один сигнал может случайно отправить 2 ордера
      (из-за бага, retry, race condition)
    
    РЕШЕНИЕ:
    1. Каждый сигнал получает уникальный UUID
    2. Перед отправкой ордера проверяем:
       - Был ли ЭТОТ signal_id уже исполнен?
       - Был ли ордер на ЭТОТ символ в последние 30 секунд?
    3. Если ДА → REJECT с логом "DUPLICATE PREVENTED"
    """
    
    # Хранит signal_id последних 1000 ордеров
    recent_signals: set[str]
    # Хранит (symbol, timestamp) последних ордеров
    recent_orders: dict[str, datetime]
```

## 21.3 Защита от Race Conditions (гонка потоков)

```
ПРОБЛЕМА: В async коде два корутины могут одновременно:
  - Проверить баланс (оба видят $100)
  - Отправить ордер на $80 каждый
  - Итого: $160 > $100 → ошибка или двойная покупка

РЕШЕНИЕ:
  1. Все операции с балансом через asyncio.Lock()
  2. Все операции с позициями через asyncio.Lock()
  3. Execution Engine обрабатывает ордера ПОСЛЕДОВАТЕЛЬНО (очередь)
  4. НИКОГДА не параллельное исполнение двух ордеров
```

## 21.4 Защита от Partial Fill (частичное исполнение)

```
ПРОБЛЕМА: Отправили BUY 0.001 BTC, но исполнилось только 0.0007 BTC
  - Позиция меньше ожидаемой
  - Stop-loss рассчитан на 0.001, а реально 0.0007

РЕШЕНИЕ:
  1. ВСЕГДА читать fill_quantity из ответа биржи (не assumed)
  2. Если fill_quantity < ordered_quantity:
     → Обновить позицию по РЕАЛЬНОМУ fill
     → Пересчитать stop-loss / take-profit
     → Telegram: "⚠️ Частичное исполнение: 0.0007 из 0.001 BTC"
  3. Оставшийся unfilled → автоматическая отмена
```

## 21.5 Защита от Integer/Float Overflow

```
ПРОБЛЕМА: При расчёте PnL или размера ордера может получиться:
  - NaN (деление на ноль)
  - Infinity (переполнение)
  - Отрицательный размер ордера

РЕШЕНИЕ:
  Каждый числовой результат проверяется:
  
  def safe_value(value: float, min_val: float, max_val: float) -> float | None:
      if math.isnan(value) or math.isinf(value):
          log.error(f"Невалидное значение: {value}")
          return None
      if value < min_val or value > max_val:
          log.error(f"Значение вне диапазона: {value}")
          return None
      return value
```

## 21.6 Защита от Memory Leak (утечка памяти)

```
ПРОБЛЕМА: Система работает 24/7, данные копятся в RAM

РЕШЕНИЕ:
  1. Данные trades в RAM — максимум 10,000 записей (FIFO)
  2. Данные candles в RAM — максимум 1,000 записей за символ
  3. Лог-буферы — flush каждые 60 секунд
  4. Мониторинг: если RAM > 2GB → Telegram alert
  5. Если RAM > 4GB → перезапуск Data Collector
```

## 21.7 Защита от Timezone / Clock Drift

```
ПРОБЛЕМА: Часы на ПК отстают/спешат → timestamp сигналов неверный
  → Data age check блокирует все сигналы
  → Или хуже: stop-loss не сработает вовремя

РЕШЕНИЕ:
  1. При старте: проверить время с Binance server time API
  2. Если расхождение > 1 секунда → WARNING + коррекция
  3. Если расхождение > 10 секунд → НЕ СТАРТОВАТЬ + ошибка
  4. Каждый час: повторная синхронизация
```

---

# 22. МОДУЛЬ 17: TRADE ANALYZER — САМООБУЧЕНИЕ (НОВОЕ V1.5)

> **Ключевое нововведение V1.5:** Бот учится на своих ошибках.
> Trade Analyzer анализирует каждую сделку и постепенно улучшает параметры.
> **Принцип безопасности:** "Улучшай атаку, не трогай оборону" — анализатор может
> ТОЛЬКО ужесточать защиту (повышать confidence, добавлять временные блоки),
> НИКОГДА не ослаблять (снижать SL, увеличивать позицию).

## 22.1 Три уровня обучения

| Уровень | Название | Когда | Что делает | Сложность |
|---|---|---|---|---|
| **Level 1** | Trade Statistician | С дня 1 | Собирает статистику: Win Rate по стратегиям, часам, дням, символам | ⭐ Простой |
| **Level 2** | Parameter Optimizer | Месяц 6+ | Адаптирует параметры: SL/TP, confidence threshold, time filters | ⭐⭐ Средний |
| **Level 3** | ML Predictor | Месяц 9+ | Предсказывает вероятность успеха сделки (scikit-learn, lightgbm) | ⭐⭐⭐ Сложный |

```
ВАЖНО: Каждый уровень НЕ ЗАМЕНЯЕТ предыдущий, а ДОПОЛНЯЕТ.
Level 3 работает ПОВЕРХ Level 2, который работает ПОВЕРХ Level 1.

Timeline:
  День 1 ──────── Месяц 6 ──────── Месяц 9 ──────── ...
  [Level 1: Stats] [+Level 2: Optimize] [+Level 3: ML]
```

## 22.2 Level 1: Trade Statistician (с дня 1)

> Записывает каждую сделку и строит детальную статистику.
> Отправляет еженедельные и ежемесячные отчёты в Telegram.

### Структура записи:
```python
@dataclass
class TradeRecord:
    """Запись о каждой завершённой сделке"""
    
    trade_id: str                   # UUID
    timestamp_open: datetime        # Когда открыта
    timestamp_close: datetime       # Когда закрыта
    symbol: str                     # BTCUSDT / ETHUSDT
    strategy: str                   # ema_crossover / grid / mean_reversion / ...
    
    # Цены
    entry_price: float
    exit_price: float
    quantity: float
    
    # Результат
    pnl_usd: float                  # Прибыль/убыток в $
    pnl_pct: float                  # Прибыль/убыток в %
    is_win: bool                    # pnl_usd > 0
    
    # Контекст входа (для анализа)
    confidence: float               # Confidence сигнала
    market_regime: str              # trending_up / sideways / ...
    hour_of_day: int                # 0-23 UTC
    day_of_week: int                # 0=Monday, 6=Sunday
    rsi_at_entry: float
    adx_at_entry: float
    volume_ratio_at_entry: float    # vol / avg_vol
    
    # Контекст выхода
    exit_reason: str                # "take_profit" / "stop_loss" / "trailing_stop" / "signal"
    hold_duration_hours: float      # Сколько часов держали позицию
    max_drawdown_during_trade: float  # Макс просадка во время сделки
    max_profit_during_trade: float    # Макс прибыль (для анализа ранних выходов)
    
    # Комиссии
    commission_usd: float
```

### Класс статистика:
```python
class TradeStatistician:
    """
    Level 1: Собирает и анализирует статистику всех сделок.
    Работает с ПЕРВОЙ сделки, никаких минимальных порогов.
    """
    
    def record_trade(self, trade: TradeRecord):
        """Записать завершённую сделку в БД"""
        self.db.save_trade(trade)
        self._update_running_stats(trade)
    
    def get_stats(self, filters: StrategyFilters = None) -> TradeStats:
        """
        Получить статистику с фильтрами.
        
        Примеры:
          get_stats()  # Все сделки
          get_stats(strategy="ema_crossover")  # Только EMA
          get_stats(strategy="grid", symbol="BTCUSDT")  # Grid по BTC
          get_stats(hour_range=(0, 6))  # Ночные сделки UTC
          get_stats(day_of_week=[0, 1, 2, 3, 4])  # Будни
          get_stats(market_regime="sideways")  # Боковик
        """
        trades = self.db.get_trades(filters)
        
        return TradeStats(
            total_trades=len(trades),
            wins=sum(1 for t in trades if t.is_win),
            losses=sum(1 for t in trades if not t.is_win),
            win_rate=self._calc_win_rate(trades),
            total_pnl=sum(t.pnl_usd for t in trades),
            avg_pnl=self._calc_avg_pnl(trades),
            avg_win=self._calc_avg_win(trades),
            avg_loss=self._calc_avg_loss(trades),
            profit_factor=self._calc_profit_factor(trades),
            max_drawdown=self._calc_max_drawdown(trades),
            best_trade=max(trades, key=lambda t: t.pnl_usd, default=None),
            worst_trade=min(trades, key=lambda t: t.pnl_usd, default=None),
            avg_hold_hours=self._calc_avg_hold(trades),
            total_commission=sum(t.commission_usd for t in trades),
        )
    
    def get_best_hours(self) -> dict:
        """Определить лучшие и худшие часы для торговли"""
        stats_by_hour = {}
        for hour in range(24):
            stats = self.get_stats(StrategyFilters(hour_range=(hour, hour+1)))
            stats_by_hour[hour] = stats.win_rate
        return stats_by_hour
    
    def get_best_strategies_by_regime(self) -> dict:
        """Какая стратегия лучше в каком рыночном режиме"""
        result = {}
        for regime in ["trending_up", "trending_down", "sideways", "volatile"]:
            for strategy in STRATEGY_NAMES:
                key = f"{regime}/{strategy}"
                stats = self.get_stats(StrategyFilters(
                    market_regime=regime, strategy=strategy
                ))
                result[key] = {
                    "win_rate": stats.win_rate,
                    "avg_pnl": stats.avg_pnl,
                    "total_trades": stats.total_trades,
                }
        return result

@dataclass
class StrategyFilters:
    """Фильтры для запросов статистики"""
    strategy: str = None
    symbol: str = None
    market_regime: str = None
    hour_range: tuple = None        # (start_hour, end_hour)
    day_of_week: list = None        # [0,1,2,3,4] = будни
    date_from: datetime = None
    date_to: datetime = None
    min_confidence: float = None
```

## 22.3 Level 2: Parameter Optimizer (месяц 6+)

> Адаптирует параметры стратегий на основе реальных данных.
> Включается ТОЛЬКО после накопления 100+ сделок (минимальная выборка).
> **БЕЗОПАСНОСТЬ:** Оптимизатор МОЖЕТ ТОЛЬКО ужесточать, НИКОГДА не ослаблять.

### Параметры, которые можно менять (TUNABLE):
```python
TUNABLE_PARAMS = {
    # Параметры, которые Level 2 может оптимизировать
    
    "confidence_threshold": {
        "current": 0.75,
        "range": (0.70, 0.95),     # Может повысить до 0.95, понизить до 0.70
        "direction": "up_only",     # ТОЛЬКО повышать! (ужесточение)
    },
    
    "time_blocks": {
        "description": "Часы, в которые НЕ торговать",
        "current": [],
        "direction": "add_only",    # ТОЛЬКО добавлять блоки! (ужесточение)
        # Пример: если ночью (0-6 UTC) Win Rate < 35% → добавить блок
    },
    
    "strategy_weight_adjustments": {
        "description": "Корректировки весов в ALLOCATION_TABLE",
        "range": (-10, +10),        # ±10% от базового
        "direction": "reduce_only", # ТОЛЬКО уменьшать аллокацию! (ужесточение)
        # Если стратегия показывает плохо → уменьшить, но НИКОГДА не увеличивать
    },
    
    "min_volume_ratio": {
        "current": 1.0,
        "range": (1.0, 3.0),
        "direction": "up_only",     # ТОЛЬКО повышать! (ужесточение)
    },
}

# FROZEN — параметры, которые НЕЛЬЗЯ трогать автоматически
FROZEN_PARAMS = {
    "stop_loss_pct": "НЕЛЬЗЯ! Только человек может менять SL",
    "max_position_pct": "НЕЛЬЗЯ! Размер позиции — только вручную",
    "max_daily_loss": "НЕЛЬЗЯ! Дневной лимит — священный",
    "max_exposure": "НЕЛЬЗЯ! Общая экспозиция — только вручную",
    "take_profit_pct": "НЕЛЬЗЯ! TP влияет на R:R ratio",
    "trading_symbols": "НЕЛЬЗЯ! Символы — только вручную",
    "absolute_limits": "НЕЛЬЗЯ! Абсолютные лимиты — зашиты в код",
}
```

### Алгоритм оптимизации:
```python
class ParameterOptimizer:
    """
    Level 2: Оптимизирует TUNABLE параметры стратегий.
    
    ПРИНЦИП: A/B тестирование с walk-forward validation.
    1. Анализируем последние 100 сделок
    2. Находим слабые места (Low Win Rate по часам/стратегиям)
    3. Предлагаем изменение (только ужесточение!)
    4. Тестируем на paper trading 2 недели
    5. Если improvement > 5% win rate → применяем
    6. Если worse → откатываем
    
    ЗАЩИТА ОТ OVERFITTING:
    - Walk-forward: обучаемся на 70%, тестируем на 30%
    - Минимум 100 сделок для каждого изменения
    - Обязательный paper trading test перед live
    - Макс 1 изменение в неделю (нет batch-изменений!)
    - Каждое изменение логируется в Telegram
    """
    
    MIN_TRADES_FOR_OPTIMIZATION = 100
    MAX_CHANGES_PER_WEEK = 1
    PAPER_TEST_DURATION_DAYS = 14
    IMPROVEMENT_THRESHOLD_PCT = 5.0
    
    def analyze_and_suggest(self) -> Optional[ParameterChange]:
        """
        Анализировать статистику и предложить одно изменение.
        Вызывается раз в неделю.
        """
        stats = self.statistician.get_stats()
        
        if stats.total_trades < self.MIN_TRADES_FOR_OPTIMIZATION:
            return None  # Недостаточно данных
        
        # 1. Найти худшие часы
        worst_hours = self._find_bad_hours(threshold_win_rate=35.0)
        if worst_hours:
            return ParameterChange(
                param="time_blocks",
                action="add",
                value=worst_hours,
                reason=f"Win Rate в часы {worst_hours} = {stats}% < 35%",
            )
        
        # 2. Найти худшие стратегии по режимам
        bad_combos = self._find_bad_strategy_regimes(threshold_win_rate=40.0)
        if bad_combos:
            return ParameterChange(
                param="strategy_weight_adjustments",
                action="reduce",
                value=bad_combos[0],
                reason=f"Стратегия {bad_combos[0]} Win Rate < 40% в данном режиме",
            )
        
        # 3. Проверить, можно ли повысить confidence
        high_conf_stats = self.statistician.get_stats(
            StrategyFilters(min_confidence=0.85)
        )
        if (high_conf_stats.win_rate > stats.win_rate + 10 and
            high_conf_stats.total_trades > 30):
            return ParameterChange(
                param="confidence_threshold",
                action="raise",
                value=0.80,
                reason=f"High-confidence trades: {high_conf_stats.win_rate}% "
                       f"vs overall {stats.win_rate}%",
            )
        
        return None  # Всё хорошо, не трогаем

    def apply_change(self, change: ParameterChange):
        """Применить изменение (сначала в paper mode!)"""
        # 1. Log to DB and Telegram
        self._log_change(change)
        self._notify_telegram(
            f"🧠 Trade Analyzer предлагает:\n"
            f"Параметр: {change.param}\n"
            f"Действие: {change.action}\n"
            f"Причина: {change.reason}\n"
            f"→ Тестируем в paper mode {self.PAPER_TEST_DURATION_DAYS} дней"
        )
        
        # 2. Apply in paper mode first
        self._apply_to_paper(change)
        
        # 3. After PAPER_TEST_DURATION_DAYS, evaluate
        # (это делается в scheduled task)
```

## 22.4 Level 3: ML Predictor (месяц 9+)

> Машинное обучение предсказывает вероятность успеха сделки.
> Используется как ДОПОЛНИТЕЛЬНЫЙ фильтр, а не как источник сигналов.
> **БЕЗОПАСНОСТЬ:** ML может только ЗАБЛОКИРОВАТЬ сделку, никогда не инициировать.

### Rollout ML:
```
OFF MODE:
  - ML выключен полностью

SHADOW MODE:
  - ML считает вероятность и skill score
  - Все решения только логируются
  - На исполнение сигналов не влияет

BLOCK MODE:
  - ML может блокировать только после успешного shadow периода
  - Risk Sentinel всё равно остаётся последним gatekeeper
```

### Архитектура:
```python
@dataclass
class StrategySkillTest:
  strategy_name: str
  trades_count: int
  regime: str
  skill_score: float             # Нормализованный score 0.0 - 1.0
  precision: float
  recall: float
  roc_auc: float
  expected_pnl_pct: float
  confidence_level: str          # "low" / "medium" / "high"
  decision: str                  # "allow" / "reduce" / "block"


class MLPredictor:
  """
  Level 3: ML модель предсказывает P(win) для каждого сигнала.

  ВАЖНО:
    - ML НЕ генерирует сигналы
    - ML НЕ обходит Risk Sentinel
    - ML включается по ступеням: off -> shadow -> block

  Модель: LightGBM или RandomForest
  Обучение: walk-forward только на закрытых сделках из strategy_trades
  """

  MIN_TRADES_FOR_ML = 500
  RETRAIN_INTERVAL_DAYS = 30
  SHADOW_DAYS_MIN = 14
  ML_BLOCK_THRESHOLD = 0.40
  MIN_PRECISION = 0.55
  MIN_RECALL = 0.50
  MIN_ROC_AUC = 0.58
  MIN_SKILL_SCORE = 0.55

  FEATURES = [
    "rsi_14", "adx", "ema_9_vs_21", "bb_bandwidth",
    "volume_ratio", "macd_hist", "atr_ratio",
    "hour_of_day", "day_of_week", "market_regime_encoded",
    "strategy_encoded", "recent_win_rate_10",
    "hours_since_last_trade", "daily_pnl_so_far",
    "consecutive_losses",
  ]

  rollout_mode: str = "off"     # off / shadow / block

  def predict(self, signal: Signal, features: FeatureVector) -> MLPrediction:
    """Предсказать вероятность успеха сделки"""
    if self.model is None or self.total_trades < self.MIN_TRADES_FOR_ML:
      return MLPrediction(probability=None, action="pass")

    X = self._build_feature_vector(signal, features)
    probability = self.model.predict_proba(X)[0][1]

    if self.rollout_mode == "shadow":
      self._log_shadow_prediction(signal, probability)
      return MLPrediction(
        probability=probability,
        action="pass",
        reason=f"ML shadow: P(win) = {probability:.1%}",
      )

    if self.rollout_mode == "block" and probability < self.ML_BLOCK_THRESHOLD:
      return MLPrediction(
        probability=probability,
        action="block",
        reason=f"ML: P(win) = {probability:.1%} < {self.ML_BLOCK_THRESHOLD:.0%}",
      )

    return MLPrediction(probability=probability, action="pass")

  def retrain(self):
    """
    Переобучение модели (раз в месяц).
    Разделение строго по времени: train -> validation -> test.
    """
    trades = self.statistician.get_all_trades()
    if len(trades) < self.MIN_TRADES_FOR_ML:
      return

    train_trades, val_trades, test_trades = self._split_by_time(trades)
    X_train, y_train = self._prepare_data(train_trades)
    X_val, y_val = self._prepare_data(val_trades)
    X_test, y_test = self._prepare_data(test_trades)

    model = lgb.LGBMClassifier(
      n_estimators=100,
      max_depth=5,
      learning_rate=0.05,
      min_child_samples=20,
      subsample=0.8,
      colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)

    val_precision, val_recall, val_roc_auc = self._evaluate_classifier(model, X_val, y_val)
    test_precision, test_recall, test_roc_auc = self._evaluate_classifier(model, X_test, y_test)
    uplift_pf, uplift_dd = self._evaluate_economic_uplift(model, test_trades)

    if (
      test_precision >= self.MIN_PRECISION and
      test_recall >= self.MIN_RECALL and
      test_roc_auc >= self.MIN_ROC_AUC and
      (uplift_pf > 0 or uplift_dd > 0)
    ):
      self.model = model
      self.rollout_mode = "shadow"
      self._save_model_registry(
        precision=test_precision,
        recall=test_recall,
        roc_auc=test_roc_auc,
        uplift_profit_factor=uplift_pf,
        uplift_drawdown=uplift_dd,
        rollout_mode="shadow",
      )
    else:
      self._notify_telegram(
        "⚠️ ML retrain rejected: metrics below safety thresholds"
      )

  def test_strategy_skill_on_history(
    self,
    strategy_name: str,
    regime: str | None = None,
    lookback_days: int = 180,
  ) -> StrategySkillTest:
    """
    Обучает ML на старых данных и тестирует "навык" стратегии.

    Skill score считается только по историческим закрытым сделкам.
    Никаких признаков из будущего периода использовать нельзя.
    """
    trades = self.statistician.get_trades(
      strategy=strategy_name,
      regime=regime,
      lookback_days=lookback_days,
    )

    if len(trades) < 100:
      return StrategySkillTest(
        strategy_name=strategy_name,
        trades_count=len(trades),
        regime=regime or "all",
        skill_score=0.0,
        precision=0.0,
        recall=0.0,
        roc_auc=0.0,
        expected_pnl_pct=0.0,
        confidence_level="low",
        decision="allow",
      )

    train_trades, _, test_trades = self._split_by_time(trades)
    model = self._fit_model(train_trades)
    precision, recall, roc_auc = self._evaluate_trades(model, test_trades)
    expected_pnl_pct = self._expected_pnl_pct(model, test_trades)

    normalized_pnl = min(max(expected_pnl_pct / 5.0, 0.0), 1.0)
    skill_score = (
      0.40 * precision +
      0.25 * recall +
      0.25 * roc_auc +
      0.10 * normalized_pnl
    )

    if skill_score < self.MIN_SKILL_SCORE or precision < self.ML_BLOCK_THRESHOLD:
      decision = "block"
    elif precision < self.MIN_PRECISION:
      decision = "reduce"
    else:
      decision = "allow"

    return StrategySkillTest(
      strategy_name=strategy_name,
      trades_count=len(trades),
      regime=regime or "all",
      skill_score=skill_score,
      precision=precision,
      recall=recall,
      roc_auc=roc_auc,
      expected_pnl_pct=expected_pnl_pct,
      confidence_level="high" if len(trades) >= 300 else "medium",
      decision=decision,
    )
```

### Защита от overfitting:
```
1. Walk-forward validation (НЕ случайное разбиение!)
   - Train: старый отрезок истории
   - Validation: следующий отрезок
   - Test: самый новый out-of-sample отрезок

2. Экономический gate, а не только classifier metrics
   - ML должен либо повышать Profit Factor
   - либо снижать Max Drawdown

3. Minimum samples (500 trades)
   - Не обучаемся на маленьких данных

4. Shadow mode обязателен
   - Сначала модель только логирует решения
   - И только после успешного shadow периода разрешается block mode

5. ML только ФИЛЬТРУЕТ, не генерирует
   - Если ML ошибается, максимум пропустим сделку
   - Но не откроем новую сделку из-за ML
```

## 22.5 Отчёты в Telegram

### Еженедельный отчёт (каждое воскресенье 20:00 UTC):
```
📊 ЕЖЕНЕДЕЛЬНЫЙ ОТЧЁТ SENTINEL

📅 Период: 07.04 — 13.04.2026

💰 Результаты:
  Сделок: 12 (7 win / 5 loss)
  Win Rate: 58.3%
  PnL: +$18.42 (+3.7%)
  Комиссии: -$2.10
  Чистый PnL: +$16.32

📈 По стратегиям:
  EMA Crossover: 4 сделки, WR 75%, +$12.30
  Grid Trading: 5 сделок, WR 60%, +$6.20
  DCA Bot: 2 покупки, avg price $67,421
  Bollinger Breakout: 1 сделка, WR 0%, -$4.18

🕐 Лучшие часы: 14:00-18:00 UTC (WR 71%)
🕐 Худшие часы: 02:00-06:00 UTC (WR 33%)

🧠 Trade Analyzer:
  Level 1: ✅ Активен (128 сделок в базе)
  Level 2: ⏳ Через ~14 сделок (min 100 за strategy)
  Level 3: 🔒 Через ~372 сделки (min 500)

💡 Рекомендация: Рассмотреть блокировку торговли 02-06 UTC
```

### Ежемесячный отчёт:
```
📊 ЕЖЕМЕСЯЧНЫЙ ОТЧЁТ SENTINEL — Апрель 2026

💰 Итого за месяц:
  Сделок: 47 (28 win / 19 loss)
  Win Rate: 59.6%
  PnL: +$72.18 (+14.4%)
  Комиссии: -$8.40
  Чистый PnL: +$63.78
  Max Drawdown: -$23.50 (-4.7%)
  Profit Factor: 1.68

📈 TOP-3 Стратегии:
  1. 🥇 EMA Crossover: WR 67%, +$31.20
  2. 🥈 Grid Trading: WR 62%, +$22.40
  3. 🥉 DCA Bot: накопил 0.0015 BTC, avg $66,800

📉 WORST Стратегия:
  MACD Divergence: WR 40%, -$8.20
  → Trade Analyzer снизит вес на 5%

📊 Баланс: $500 → $563.78
📊 ROI за месяц: +12.8%
📊 Сравнение с BTC hold: BTC +8.2% → Мы лучше на +4.6%

🧠 Изменения Trade Analyzer:
  - Добавлен time block 02:00-06:00 UTC
  - EMA confidence повышен 0.75 → 0.78
  - Grid weight в volatile снижен 5% → 3%
```

## 22.6 Правила безопасности Trade Analyzer

```
╔══════════════════════════════════════════════════════════════╗
║          ЗОЛОТОЕ ПРАВИЛО TRADE ANALYZER                      ║
║                                                              ║
║  МОЖЕТ ТОЛЬКО УЖЕСТОЧАТЬ, НИКОГДА НЕ ОСЛАБЛЯТЬ             ║
║                                                              ║
║  ✅ Повысить confidence threshold (0.75 → 0.80)             ║
║  ✅ Добавить time block (не торговать 02-06 UTC)            ║
║  ✅ Снизить аллокацию стратегии (EMA 25% → 20%)            ║
║  ✅ Повысить min volume ratio (1.0 → 1.5)                   ║
║  ✅ Заблокировать сделку (ML: P(win) < 40%)                 ║
║                                                              ║
║  ❌ Снизить stop-loss (3% → 2%)                             ║
║  ❌ Увеличить размер позиции (20% → 25%)                    ║
║  ❌ Снизить confidence (0.75 → 0.60)                        ║
║  ❌ Увеличить аллокацию стратегии (EMA 25% → 35%)          ║
║  ❌ Убрать time block                                        ║
║  ❌ Увеличить max exposure                                   ║
║  ❌ Отключить любой Circuit Breaker                          ║
║  ❌ Открыть сделку самостоятельно (только фильтрация)       ║
╚══════════════════════════════════════════════════════════════╝
```

## 22.7 Конфигурация Trade Analyzer

```python
@dataclass
class TradeAnalyzerConfig:
    """Конфигурация модуля Trade Analyzer"""
    
    # Level 1: Statistics (всегда активен)
    stats_enabled: bool = True
    weekly_report_enabled: bool = True
    weekly_report_day: int = 6          # 0=Mon, 6=Sun
    weekly_report_hour: int = 20        # UTC
    monthly_report_enabled: bool = True
    
    # Level 2: Optimizer
    optimizer_enabled: bool = False     # Включить после 100+ trades
    min_trades_for_optimization: int = 100
    max_changes_per_week: int = 1
    paper_test_days: int = 14
    improvement_threshold_pct: float = 5.0
    
    # Level 3: ML
    ml_enabled: bool = False            # Включить после 500+ trades
    min_trades_for_ml: int = 500
    ml_retrain_interval_days: int = 30
    ml_block_threshold: float = 0.40
    ml_shadow_mode_enabled: bool = True
    ml_shadow_min_days: int = 14
    ml_history_days: int = 180
    ml_test_window_days: int = 60
    ml_min_precision: float = 0.55
    ml_min_recall: float = 0.50
    ml_min_roc_auc: float = 0.58
    ml_min_skill_score: float = 0.55
    
    # Safety
    max_confidence_increase: float = 0.10   # Макс +10% за одно изменение
    max_allocation_decrease: float = 10.0   # Макс -10% за одно изменение
    rollback_on_worse: bool = True          # Откатить если стало хуже
```

---

# 23. ПОЛНЫЙ АУДИТ РИСКОВ (30+ СЦЕНАРИЕВ) (НОВОЕ V1.1)

## 23.1 КАТЕГОРИЯ A: ФИНАНСОВЫЕ РИСКИ

| # | Риск | Вер. | Последствие | Защита (уровни) |
|---|---|---|---|---|
| A1 | Потеря > $50 за день (дневной лимит) | Средняя | ⚠️ Критическое | **L1:** Risk Sentinel прекращает НОВЫЕ входы при достижении лимита и переводит систему в STOP<br>**L2:** Circuit Breaker CB-2 (3 потери подряд → пауза)<br>**L3:** Watchdog проверяет daily_pnl<br>**L4:** State Machine NORMAL→REDUCED→SAFE→STOP<br>**Примечание:** это hard-limit на новый риск, а не гарантированный потолок фактического убытка при гэпе |
| A2 | Overtrading (комиссии съедают прибыль) | Высокая | 💰 Среднее | **L1:** max_trades_per_hour=2, min_interval=30 мин<br>**L2:** Hard limit: комиссии > 1% капитала/день → запрет новых BUY<br>**L3:** CB-8 (комиссии > 1% капитала → ужесточение и alert)<br>**L4:** Ежедневный отчёт комиссий в Telegram |
| A3 | Stop-loss не сработал (проскальзывание) | Низкая | ⚠️ Критическое | **L1:** Обязательный stop-loss при открытии<br>**L2:** Для live — обязательный exchange-native protective order<br>**L3:** Max position size 20% (ограничивает ущерб)<br>**L4:** При невозможности поставить защитный ордер live блокируется |
| A4 | Flash crash (-20% за минуту) | Низкая | ⚠️ Критическое | **L1:** CB-1 (цена ±5% за минуту → заморозка)<br>**L2:** Max exposure 60% (часть в USDT)<br>**L3:** Stop-loss + Market sell<br>**L4:** НЕТ плеча → максимум потеряем стоимость монет |
| A5 | Торговля на устаревших данных | Средняя | 💰 Среднее | **L1:** Data age check (> 30s → блок)<br>**L2:** Cross-validation с REST API<br>**L3:** Stale data breaker в Watchdog |
| A6 | Fees accumulation (скрытые комиссии) | Высокая | 💰 Среднее | **L1:** Реалистичная симуляция комиссий в paper trading<br>**L2:** Ежедневный подсчёт total_commissions<br>**L3:** В бэктесте — комиссия × 1.5 запас |
| A7 | Купил на пике (買い入れ at top) | Средняя | 💰 Среднее | **L1:** RSI filter (RSI > 70 → не покупать)<br>**L2:** Обязательный stop-loss -3% для EMA / -4% для Mean Reversion по стратегии<br>**L3:** Max position 20% капитала |

## 23.2 КАТЕГОРИЯ B: ТЕХНИЧЕСКИЕ РИСКИ

| # | Риск | Вер. | Последствие | Защита (уровни) |
|---|---|---|---|---|
| B1 | Баг в коде → неправильный ордер | Средняя | ⚠️ Критическое | **L1:** Paper trading 2–4 недели до live<br>**L2:** Unit тесты на Risk Sentinel (100% coverage)<br>**L3:** Sanity check в Risk check pipeline<br>**L4:** Max order size $100 (ограничивает любой баг) |
| B2 | Double execution (ордер дважды) | Низкая | ⚠️ Критическое | **L1:** OrderDeduplicator (UUID на каждый сигнал)<br>**L2:** Проверка recent_orders за 30 сек<br>**L3:** Exposure check поймает второй ордер<br>**L4:** Reconciliation с биржей каждые 5 мин |
| B3 | Race condition (параллельный доступ) | Средняя | 💰 Среднее | **L1:** asyncio.Lock() на все финансовые операции<br>**L2:** Последовательная очередь ордеров<br>**L3:** Atomic записи в SQLite |
| B4 | Partial fill (частичное исполнение) | Средняя | 💰 Низкое | **L1:** Всегда читать реальный fill из ответа API<br>**L2:** Пересчёт stop-loss по фактическому fill<br>**L3:** Отмена unfilled части |
| B5 | Memory leak при 24/7 работе | Средняя | 💰 Среднее | **L1:** FIFO буферы (max 10K trades)<br>**L2:** Мониторинг RAM (alert > 2GB)<br>**L3:** Auto restart Data Collector > 4GB |
| B6 | SQLite corruption (повреждение БД) | Низкая | ⚠️ Критическое | **L1:** WAL mode (Write-Ahead Logging)<br>**L2:** Автоматический backup каждые 6 часов<br>**L3:** Integrity check при старте (PRAGMA integrity_check)<br>**L4:** Если БД повреждена → сообщение + работа на новой БД |
| B7 | NaN/Infinity в расчётах | Средняя | ⚠️ Критическое | **L1:** safe_value() проверка каждого числа<br>**L2:** Feature Engine: если NaN → return None → нет сигнала<br>**L3:** Risk Sentinel проверяет все числа в Signal |
| B8 | Timezone / Clock drift | Низкая | 💰 Среднее | **L1:** Sync с Binance server time при старте<br>**L2:** Блок при drift > 10 сек<br>**L3:** Ежечасная ресинхронизация |

## 23.3 КАТЕГОРИЯ C: ИНФРАСТРУКТУРНЫЕ РИСКИ

| # | Риск | Вер. | Последствие | Защита (уровни) |
|---|---|---|---|---|
| C1 | Потеря интернета | Средняя | ⚠️ Критическое | **L1:** Auto reconnect с exp. backoff (1s→60s)<br>**L2:** Exchange-native protective orders на бирже (обязательны для live)<br>**L3:** Watchdog: если heartbeat>2мин → emergency close / escalation<br>**L4:** Telegram через мобильный (независимый канал) |
| C2 | Binance API недоступен | Низкая | 💰 Среднее | **L1:** CB-5 (>5 ошибок за 5 мин → стоп)<br>**L2:** Retry с exp. backoff<br>**L3:** Graceful degradation (только мониторинг без торговли) |
| C3 | Binance maintenance (плановое обслуживание) | Средняя | 💰 Низкое | **L1:** Проверка Binance System Status API при старте<br>**L2:** Если maintenance → не запускать торговлю<br>**L3:** Telegram: "Binance на обслуживании" |
| C4 | Переполнение диска | Низкая | 💰 Среднее | **L1:** Ротация логов (10MB × 5 файлов)<br>**L2:** Архивация trades > 7 дней<br>**L3:** Healthcheck: свободно > 1GB?<br>**L4:** Alert если < 500MB |
| C5 | Выключение ПК / BSOD / перезагрузка | Средняя | ⚠️ Критическое | **L1:** Биржевой protective order (stop-loss независимо от ПК)<br>**L2:** Watchdog как отдельный процесс<br>**L3:** State persistence: при старте загружаем последнее состояние<br>**L4:** Graceful shutdown handler (Ctrl+C, SIGTERM) |
| C6 | Электричество отключилось | Низкая | ⚠️ Критическое | **L1:** Биржевой protective order (критическая защита)<br>**L2:** UPS рекомендация (опционально)<br>**L3:** Telegram alert при восстановлении |
| C7 | Rate Limit Binance (429 Too Many Requests) | Средняя | 💰 Низкое | **L1:** Отслеживание X-MBX-USED-WEIGHT в headers<br>**L2:** При weight > 800/1200 → снижение частоты<br>**L3:** При 429 → пауза 60 сек<br>**L4:** Экономия requests: WebSocket вместо polling |

## 23.4 КАТЕГОРИЯ D: РЫНОЧНЫЕ РИСКИ

| # | Риск | Вер. | Последствие | Защита (уровни) |
|---|---|---|---|---|
| D1 | Боковой рынок (no trend) → стратегия теряет | Высокая | 💰 Среднее | **L1:** Confidence threshold 0.75 для EMA-стратегии<br>**L2:** Strategy Selector переводит капитал в Grid/Reserve только при доступных regime-features<br>**L3:** CB-2 (3 потери подряд → пауза)<br>**L4:** Дневной лимит на новый риск ограничивает суммарный ущерб |
| D2 | Высокая волатильность (неожиданные новости) | Средняя | ⚠️ Критическое | **L1:** CB-1 (цена ±5%/мин → заморозка)<br>**L2:** ATR фильтр (если ATR > 2× нормы → осторожность)<br>**L3:** Stop-loss обязателен |
| D3 | Pump & Dump манипуляция | Низкая | 💰 Среднее | **L1:** CB-4 (аномальный объём → пауза)<br>**L2:** Только BTC/ETH (высокая ликвидность, сложно манипулировать)<br>**L3:** Не торгуем альткоинами |
| D4 | Model Drift (стратегия устарела) | Высокая | 💰 Среднее | **L1:** Ежедневный мониторинг Win Rate в Telegram<br>**L2:** Если Win Rate < 40% за 7 дней → Telegram alert<br>**L3:** Регулярный бэктест на свежих данных<br>**L4:** Ручная оценка раз в неделю |
| D5 | Листинг/делистинг монеты (внезапный) | Очень низкая | 💰 Низкое | **L1:** Только BTC и ETH (невозможен делистинг)<br>**L2:** Символы захардкожены в конфиге |

## 23.5 КАТЕГОРИЯ E: БЕЗОПАСНОСТЬ И ЧЕЛОВЕЧЕСКИЙ ФАКТОР

| # | Риск | Вер. | Последствие | Защита (уровни) |
|---|---|---|---|---|
| E1 | Утечка API ключей (GitHub, скриншот) | Средняя | ⚠️ Критическое | **L1:** .env в .gitignore (НИКОГДА в git)<br>**L2:** API ключи НЕВИДИМЫ в логах (маскируются: `BNabc...xyz`)<br>**L3:** Права API: НЕТ withdrawal<br>**L4:** IP restriction на Binance<br>**L5:** При подозрении: revoke ключи за 30 сек через Binance |
| E2 | Кто-то получил доступ к ПК | Низкая | ⚠️ Критическое | **L1:** Dashboard пароль<br>**L2:** API без withdrawal → деньги не вывести<br>**L3:** Telegram 2FA<br>**L4:** .env можно зашифровать (опционально) |
| E3 | Пользователь случайно переключил на Live | Средняя | ⚠️ Критическое | **L1:** Смена режима ТРЕБУЕТ подтверждения (чек-лист)<br>**L2:** Telegram: "Вы уверены? Win Rate: XX%, PnL: $XX"<br>**L3:** Первые 24 часа live → max_order = $20<br>**L4:** При первом live запуске → торговля с $50, не $500 |
| E4 | Пользователь изменил лимиты на опасные | Средняя | ⚠️ Критическое | **L1:** HARD LIMITS зашиты в коде (не в .env)<br>**L2:** Даже если в .env max_daily_loss=10000 → код не позволит > $100<br>**L3:** Абсолютные потолки: |

```python
# АБСОЛЮТНЫЕ ЛИМИТЫ (НЕ МЕНЯЮТСЯ ЧЕРЕЗ CONFIG)
ABSOLUTE_MAX_DAILY_LOSS = 100.0        # $100 максимум НАВСЕГДА
ABSOLUTE_MAX_ORDER = 200.0             # $200 за ордер максимум
ABSOLUTE_MAX_LEVERAGE = 1              # Только спот, ВСЕГДА
ABSOLUTE_FORBIDDEN_PERMISSIONS = [     # API НЕ ИМЕЕТ ПРАВА:
    "withdraw",
    "futures",
    "margin"
]
```

| E5 | Эмоциональное решение (жадность/паника) | Высокая | 💰 Среднее | **L1:** Система автоматическая — эмоции не влияют<br>**L2:** Лимиты не обойти через Telegram<br>**L3:** Cooling period: после STOP нельзя resume 30 мин<br>**L4:** Дневной отчёт показывает факты, не эмоции |
| E6 | Telegram бот взломан | Очень низкая | 💰 Среднее | **L1:** Белый список chat_id (только ваш)<br>**L2:** Критические команды (/kill, /mode) требуют PIN<br>**L3:** Бот не выполняет произвольные команды<br>**L4:** Лимит: бот не может увеличить risk limits |
| E7 | Потеря .env файла | Низкая | 💰 Низкое | **L1:** Бэкап .env в защищённое место<br>**L2:** API ключи можно пересоздать на Binance<br>**L3:** Инструкция по восстановлению в README |

## 23.6 КАТЕГОРИЯ F: ОПЕРАЦИОННЫЕ РИСКИ

| # | Риск | Вер. | Последствие | Защита (уровни) |
|---|---|---|---|---|
| F1 | Система работает, но не торгует (silent failure) | Средняя | 💰 Среднее | **L1:** Watchdog проверяет heartbeat<br>**L2:** Telegram: если нет сделок > 4 часов → alert<br>**L3:** Dashboard показывает last_signal_time |
| F2 | Логи не пишутся (потеря аудита) | Низкая | 💰 Среднее | **L1:** Healthcheck: проверка записи в лог<br>**L2:** Дублирование критических логов в Telegram<br>**L3:** SQLite как второй лог (orders, signals таблицы) |
| F3 | Обновление Python / библиотеки сломало код | Средняя | 💰 Среднее | **L1:** requirements.txt с точными версиями (pinned)<br>**L2:** Виртуальное окружение (venv)<br>**L3:** НЕ обновлять без тестирования |
| F4 | Два инстанса системы одновременно | Средняя | ⚠️ Критическое | **L1:** PID lock file: если main.py уже запущен → не стартовать<br>**L2:** SQLite lock (только один writer) |
| F5 | Backtest показал прибыль → Live убыток | Высокая | 💰 Среднее | **L1:** Safety discount × 0.7 для бэктеста<br>**L2:** Paper trading 2–4 недели на реальных данных<br>**L3:** Live micro ($50) перед full ($500) |

## 23.7 МАТРИЦА ПРИОРИТЕТОВ

```
                    Вероятность
                    Низкая    Средняя    Высокая
                ┌──────────┬──────────┬──────────┐
    Критическое │ A3,A4,C5 │ A1,B1,E3 │          │
                │ C6,B6    │ E4,C1    │          │
Последствие     ├──────────┼──────────┼──────────┤
    Среднее     │ C2,E7    │ A5,B3,B5 │ A2,D1,D4 │
                │ E2       │ A7,B8,F4 │ E5,F5    │
                ├──────────┼──────────┼──────────┤
    Низкое      │ D5       │ B4,C3    │ A6       │
                │          │ C7,F1    │          │
                └──────────┴──────────┴──────────┘
```

**Фокус: верхняя строка (критическое последствие) — ВСЕ закрыты многоуровневой защитой.**

## 23.8 КАРТА ЗАЩИТ (резюме)

```
┌─────────────────────────────────────────────────────────────┐
│                    ЛИНИИ ОБОРОНЫ                             │
│                                                              │
│  L6: ЧЕЛОВЕК (ручная остановка через Telegram/Dashboard)     │
│  ─────────────────────────────────────────────────────────── │
│  L5: WATCHDOG (независимый процесс, emergency close)         │
│  ─────────────────────────────────────────────────────────── │
│  L4: CIRCUIT BREAKERS (8 автоматических предохранителей)      │
│  ─────────────────────────────────────────────────────────── │
│  L3: RISK SENTINEL (7 проверок + State Machine)              │
│  ─────────────────────────────────────────────────────────── │
│  L2: DATA INTEGRITY GUARD (4 уровня валидации данных)        │
│  ─────────────────────────────────────────────────────────── │
│  L1: ANTI-CORRUPTION LAYER (dedup, locks, safe math)         │
│  ─────────────────────────────────────────────────────────── │
│  L0: ABSOLUTE LIMITS (зашиты в коде, нельзя изменить)        │
│                                                              │
│  ДАЖЕ ЕСЛИ L1–L5 откажут → L0 всё ещё ограничит конфигурацию │
│  и не позволит повысить риск выше зашитых потолков:          │
│    • Открыть новый риск выше абсолютных лимитов              │
│    • Использовать плечо (только спот, x1)                    │
│    • Вывести деньги через API (право отключено)              │
│    • Торговать фьючерсами (право отключено)                  │
└─────────────────────────────────────────────────────────────┘
```

---

# 24. СТРУКТУРА ПРОЕКТА (ОБНОВЛЕНО V1.5)

```
sentinel/
│
├── .env                          # API ключи (НЕ в git!)
├── .env.example                  # Шаблон .env
├── .gitignore
├── requirements.txt              # Зависимости Python (pinned versions)
├── README.md                     # Инструкция по запуску
│
├── main.py                       # Точка входа, запуск всех модулей
├── watchdog.py                   # НОВОЕ: Независимый сторожевой процесс
├── config.py                     # Загрузка настроек из .env
│
├── core/                         # Ядро системы
│   ├── __init__.py
│   ├── models.py                 # Dataclasses: Trade, Signal, Position, Order
│   ├── events.py                 # Система событий (pub/sub между модулями)
│   ├── constants.py              # Константы
│   └── absolute_limits.py        # НОВОЕ: Хардкодные лимиты (нельзя обойти)
│
├── collector/                    # Модуль 1: Сбор данных
│   ├── __init__.py
│   ├── binance_ws.py             # WebSocket подключение к Binance
│   └── data_validator.py         # Валидация входящих данных
│
├── database/                     # Модуль 2: База данных
│   ├── __init__.py
│   ├── db.py                     # Подключение к SQLite (WAL mode)
│   ├── repository.py             # CRUD операции
│   └── backup.py                 # НОВОЕ: Автоматический backup каждые 6ч
│
├── features/                     # Модуль 3: Feature Engine
│   ├── __init__.py
│   ├── indicators.py             # Расчёт индикаторов
│   └── feature_builder.py        # Сборка FeatureVector
│
├── strategy/                     # Модуль 4: Strategy Engine
│   ├── __init__.py
│   ├── base_strategy.py          # Базовый класс стратегии
│   ├── ema_crossover_rsi.py      # Стратегия V1: EMA Crossover
│   ├── grid_trading.py           # НОВОЕ V1.3: Grid Trading
│   ├── mean_reversion.py         # НОВОЕ V1.3: Mean Reversion
│   ├── bollinger_breakout.py     # НОВОЕ V1.4: Bollinger Band Breakout
│   ├── dca_bot.py                # НОВОЕ V1.4: DCA Bot
│   ├── macd_divergence.py        # НОВОЕ V1.4: MACD Divergence
│   ├── strategy_selector.py      # НОВОЕ V1.3: Авто-выбор стратегии
│   └── market_regime.py          # НОВОЕ V1.3: Определение типа рынка
│
├── risk/                         # Модуль 5: Risk Sentinel
│   ├── __init__.py
│   ├── sentinel.py               # Основная логика проверок
│   ├── limits.py                 # Конфигурация лимитов
│   ├── state_machine.py          # State: NORMAL → REDUCED → SAFE → STOP
│   ├── circuit_breakers.py       # НОВОЕ: 8 Circuit Breakers
│   └── kill_switch.py            # НОВОЕ: Выделенный kill switch
│
├── execution/                    # Модуль 6: Execution Engine
│   ├── __init__.py
│   ├── paper_executor.py         # Виртуальное исполнение
│   ├── live_executor.py          # Реальное исполнение через Binance API
│   ├── base_executor.py          # Общий интерфейс
│   └── deduplicator.py           # НОВОЕ: Защита от дублей
│
├── position/                     # Модуль 7: Position Manager
│   ├── __init__.py
│   └── manager.py                # Трекинг позиций и PnL
│
├── guards/                       # НОВОЕ: Защитные модули
│   ├── __init__.py
│   ├── data_integrity.py         # Модуль 15: Data Integrity Guard
│   ├── anti_corruption.py        # Модуль 16: Anti-Corruption Layer
│   └── safe_math.py              # Безопасные вычисления (no NaN/Inf)
│
├── analyzer/                     # НОВОЕ V1.5: Trade Analyzer (Модуль 17)
│   ├── __init__.py
│   ├── trade_record.py           # TradeRecord dataclass
│   ├── statistician.py           # Level 1: Trade Statistician
│   ├── optimizer.py              # Level 2: Parameter Optimizer
│   ├── ml_predictor.py           # Level 3: ML Predictor
│   ├── skill_tests.py            # Исторические skill tests по стратегиям
│   └── reports.py                # Telegram отчёты (еженедельный/ежемесячный)
│
├── backtest/                     # Модуль 9: Backtest Engine
│   ├── __init__.py
│   └── engine.py                 # Бэктестирование стратегий
│
├── telegram_bot/                 # Модуль 10: Telegram
│   ├── __init__.py
│   ├── bot.py                    # Telegram бот
│   └── formatters.py             # Форматирование сообщений
│
├── dashboard/                    # Модуль 11: Web Dashboard
│   ├── __init__.py
│   ├── app.py                    # FastAPI приложение
│   ├── static/                   # HTML, CSS, JS
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   └── api_routes.py             # REST API endpoints
│
├── logs/                         # Логи (создаётся автоматически)
│   └── .gitkeep
│
├── data/                         # Файлы БД (создаётся автоматически)
│   ├── .gitkeep
│   └── backups/                  # НОВОЕ: Автоматические бэкапы БД
│
└── tests/                        # Тесты
    ├── __init__.py
    ├── test_risk_sentinel.py     # Тесты Risk Sentinel (ПРИОРИТЕТ!)
    ├── test_circuit_breakers.py  # НОВОЕ: Тесты Circuit Breakers
    ├── test_data_integrity.py    # НОВОЕ: Тесты валидации данных
    ├── test_deduplicator.py      # НОВОЕ: Тесты защиты от дублей
    ├── test_safe_math.py         # НОВОЕ: Тесты безопасных вычислений
    ├── test_strategy.py
    ├── test_feature_engine.py
    ├── test_paper_executor.py
    ├── test_bollinger_breakout.py
    ├── test_dca_bot.py
    ├── test_macd_divergence.py
    ├── test_trade_analyzer.py
    └── test_ml_skill_tests.py
```

---

# 25. КОНФИГУРАЦИЯ (ОБНОВЛЕНО V1.5)

## 25.1 Полный файл config.py (структура)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # === Binance ===
    binance_api_key: str
    binance_api_secret: str
    
    # === Telegram ===
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_pin: str = ""                  # НОВОЕ: PIN для опасных команд
    
    # === Trading ===
    trading_mode: str = "paper"             # "paper" или "live"
    trading_symbols: list = ["BTCUSDT", "ETHUSDT"]
    
    # === Risk Limits (настраиваемые, но ≤ absolute limits) ===
    max_daily_loss_usd: float = 50.0
    max_daily_loss_pct: float = 10.0
    max_position_pct: float = 20.0
    max_total_exposure_pct: float = 60.0
    max_open_positions: int = 2             # V1.2: было 3
    max_order_usd: float = 100.0
    max_trades_per_hour: int = 2            # V1.2: было 20
    
    # === Circuit Breakers ===
    cb_price_anomaly_pct: float = 5.0       # НОВОЕ: CB-1 порог
    cb_consecutive_losses: int = 3          # НОВОЕ: CB-2 порог
    cb_spread_anomaly_pct: float = 0.5      # НОВОЕ: CB-3 порог
    cb_volume_anomaly_mult: float = 10.0    # НОВОЕ: CB-4 порог
    cb_api_error_count: int = 5             # НОВОЕ: CB-5 порог
    cb_latency_threshold_sec: float = 5.0   # НОВОЕ: CB-6 порог
    cb_balance_mismatch_pct: float = 1.0    # НОВОЕ: CB-7 порог
    cb_commission_alert_pct: float = 1.0    # НОВОЕ: CB-8 порог
    
    # === Data Integrity ===
    max_data_age_sec: int = 30              # НОВОЕ: Макс возраст данных
    price_cross_validation_interval: int = 300  # НОВОЕ: секунд
    
    # === Watchdog ===
    watchdog_heartbeat_interval: int = 10   # НОВОЕ: секунд
    watchdog_timeout: int = 120             # НОВОЕ: секунд до emergency
    
    # === Strategy (V1.2: обновлено для swing trading) ===
    stop_loss_pct: float = 3.0              # V1.2: было 2.0
    take_profit_pct: float = 5.0            # V1.2: было 3.0
    min_confidence: float = 0.75            # V1.2: было 0.6
    signal_timeframe: str = "1h"            # V1.2: было 1m/5m
    trend_timeframe: str = "4h"             # V1.2: новое
    max_trades_per_day: int = 6             # V1.2: новое
    
    # === Grid Trading (НОВОЕ V1.3) ===
    grid_enabled: bool = False              # Включить Grid (Фаза 2)
    grid_num_levels: int = 8                # Кол-во уровней
    grid_capital_pct: float = 30.0          # % капитала на Grid
    grid_auto_range: bool = True            # Авто-диапазон (Bollinger)
    grid_min_profit_pct: float = 0.3        # Мин. прибыль на грид
    grid_max_loss_pct: float = 5.0          # Стоп Grid
    
    # === Mean Reversion (НОВОЕ V1.3) ===
    meanrev_enabled: bool = False           # Включить MR (Фаза 3)
    meanrev_rsi_oversold: float = 25.0      # RSI порог BUY
    meanrev_rsi_overbought: float = 75.0    # RSI порог SELL
    meanrev_stop_loss_pct: float = 4.0      # SL для MR
    meanrev_take_profit_pct: float = 6.0    # TP для MR
    meanrev_capital_pct: float = 15.0       # % капитала на MR
    
    # === Strategy Selector (НОВОЕ V1.3) ===
    auto_strategy_selection: bool = False    # Включить (Фаза 4)
    regime_check_interval_hours: int = 4    # Проверка рынка каждые N часов
    adx_trending_threshold: float = 25.0    # ADX > 25 = тренд
    adx_sideways_threshold: float = 20.0    # ADX < 20 = боковик
    
    # === Bollinger Breakout (НОВОЕ V1.4) ===
    bb_breakout_enabled: bool = False       # Включить (Фаза 5)
    bb_period: int = 20                     # Период BB
    bb_std_dev: float = 2.0                 # Стандартные отклонения
    bb_volume_confirm_mult: float = 1.5     # Множитель подтверждения объёма
    bb_squeeze_threshold: float = 0.05      # Порог сжатия (bandwidth < 5%)
    bb_stop_loss_pct: float = 3.0           # SL для Bollinger Breakout
    bb_take_profit_pct: float = 6.0         # TP для Bollinger Breakout
    bb_trailing_stop_pct: float = 2.0       # Trailing stop после +3%
    
    # === DCA Bot (НОВОЕ V1.4) ===
    dca_enabled: bool = False               # Включить (Фаза 6)
    dca_base_amount_usd: float = 10.0       # Базовая покупка $10
    dca_interval_hours: int = 24            # Интервал покупок
    dca_max_daily_buys: int = 3             # Макс покупок при dip
    dca_max_invested_pct: float = 40.0      # Макс % капитала через DCA
    dca_stop_drawdown_pct: float = 15.0     # Drawdown > 15% → пауза
    dca_take_profit_pct: float = 8.0        # TP от средней цены входа
    dca_partial_tp_pct: float = 5.0         # Частичная фиксация при +5%
    
    # === MACD Divergence (НОВОЕ V1.4) ===
    macd_div_enabled: bool = False          # Включить (Фаза 7)
    macd_fast: int = 12                     # Fast EMA period
    macd_slow: int = 26                     # Slow EMA period
    macd_signal_period: int = 9             # Signal line period
    macd_lookback_candles: int = 30         # Lookback для дивергенции
    macd_div_stop_loss_pct: float = 3.5     # SL для MACD Divergence
    macd_div_take_profit_pct: float = 7.0   # TP для MACD Divergence
    macd_require_rsi_confirm: bool = True   # RSI подтверждение
    macd_require_vol_confirm: bool = True   # Volume подтверждение
    
    # === Trade Analyzer (НОВОЕ V1.5) ===
    analyzer_stats_enabled: bool = True     # Level 1: статистика (всегда)
    analyzer_weekly_report: bool = True     # Еженедельный отчёт
    analyzer_monthly_report: bool = True    # Ежемесячный отчёт
    analyzer_optimizer_enabled: bool = False # Level 2: оптимизация (месяц 6+)
    analyzer_min_trades: int = 100          # Мин. сделок для Level 2
    analyzer_max_changes_week: int = 1      # Макс изменений в неделю
    analyzer_paper_test_days: int = 14      # Дни тестирования изменений
    analyzer_ml_enabled: bool = False       # Level 3: ML (месяц 9+)
    analyzer_ml_shadow_mode: bool = True    # Сначала только логирование
    analyzer_min_trades_ml: int = 500       # Мин. сделок для Level 3
    analyzer_ml_retrain_days: int = 30      # Переобучение раз в N дней
    analyzer_ml_block_threshold: float = 0.40  # P(win) < 40% → блок
    analyzer_ml_history_days: int = 180     # Глубина истории для обучения
    analyzer_ml_test_window_days: int = 60  # Окно out-of-sample теста
    analyzer_ml_min_skill_score: float = 0.55  # Мин. skill score для block mode
    analyzer_ml_min_precision: float = 0.55    # Мин. precision на test
    analyzer_ml_min_recall: float = 0.50       # Мин. recall на test
    analyzer_ml_min_roc_auc: float = 0.58      # Мин. ROC-AUC на test
    
    # === Paper Trading ===
    paper_initial_balance: float = 500.0
    paper_commission_pct: float = 0.1
    paper_slippage_pct: float = 0.05
    
    # === System ===
    log_level: str = "INFO"
    dashboard_port: int = 8080
    dashboard_password: str = ""
    db_path: str = "data/sentinel.db"
    db_backup_interval_hours: int = 6       # НОВОЕ
    max_ram_mb: int = 2048                  # НОВОЕ: Alert порог RAM
    
    # === Cooling Period ===
    resume_cooldown_min: int = 30           # НОВОЕ: мин. пауза после STOP
    live_first_day_max_order: float = 20.0  # НОВОЕ: лимит в первые 24ч live
    
    class Config:
        env_file = ".env"
```

## 25.2 Абсолютные лимиты (absolute_limits.py)

```python
"""
АБСОЛЮТНЫЕ ЛИМИТЫ — ЗАШИТЫ В КОД, НЕ МЕНЯЮТСЯ ЧЕРЕЗ ENV

Даже если пользователь поставит max_daily_loss=999999 в .env,
система всё равно ограничит до ABSOLUTE_MAX_DAILY_LOSS.
"""

# === ФИНАНСОВЫЕ АБСОЛЮТЫ ===
ABSOLUTE_MAX_DAILY_LOSS_USD = 100.0     # Макс потеря за день
ABSOLUTE_MAX_ORDER_USD = 200.0          # Макс размер одного ордера
ABSOLUTE_MAX_POSITION_PCT = 50.0        # Макс % капитала на позицию
ABSOLUTE_MAX_EXPOSURE_PCT = 80.0        # Макс % капитала в позициях
ABSOLUTE_MAX_LEVERAGE = 1               # ТОЛЬКО спот, x1

# === ЧАСТОТНЫЕ АБСОЛЮТЫ (V1.2: снижены) ===
ABSOLUTE_MAX_TRADES_PER_HOUR = 10       # V1.2: было 60
ABSOLUTE_MAX_TRADES_PER_DAY = 20        # V1.2: было 200

# === РАЗРЕШЁННЫЕ СИМВОЛЫ ===
ALLOWED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# === ЗАПРЕЩЁННЫЕ API ПРАВА ===
FORBIDDEN_API_PERMISSIONS = ["withdraw", "futures", "margin"]

# === ЦЕНА: АДЕКВАТНЫЕ ДИАПАЗОНЫ ===
PRICE_RANGES = {
    "BTCUSDT": (1_000.0, 1_000_000.0),
    "ETHUSDT": (50.0, 100_000.0),
}
```

---

# 26. ЗАПУСК И ОСТАНОВКА (ОБНОВЛЕНО V1.1)

## 26.1 Первый запуск

```powershell
# 1. Установить Python 3.11+
# 2. Создать виртуальное окружение
python -m venv venv
.\venv\Scripts\activate

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Скопировать и заполнить .env
copy .env.example .env
# → Вписать API ключи Binance и Telegram

# 5. Запустить основную систему
python main.py

# 6. НОВОЕ: В ОТДЕЛЬНОМ терминале запустить Watchdog
python watchdog.py
```

## 26.2 Нормальный запуск

```powershell
# Терминал 1: Основная система
.\venv\Scripts\activate
python main.py

# Терминал 2: Watchdog (ОБЯЗАТЕЛЬНО)
.\venv\Scripts\activate
python watchdog.py
```

**Последовательность инициализации (ОБНОВЛЕНО):**
```
 1. ✅ Загрузка конфигурации из .env
 2. ✅ НОВОЕ: Проверка абсолютных лимитов (config ≤ absolute_limits)
 3. ✅ НОВОЕ: Проверка PID lock (не запущен ли уже?)
 4. ✅ НОВОЕ: Синхронизация времени с Binance (drift < 10 сек?)
 5. ✅ НОВОЕ: Проверка Binance System Status (не maintenance?)
 6. ✅ Инициализация SQLite (WAL mode, integrity check)
 7. ✅ НОВОЕ: Загрузка последнего состояния (state recovery)
 8. ✅ Запуск логирования
 9. ✅ Запуск Data Collector (WebSocket → Binance)
10. ✅ НОВОЕ: Запуск Data Integrity Guard
11. ✅ Запуск Feature Engine
12. ✅ Запуск Strategy Engine
13. ✅ Запуск Risk Sentinel
14. ✅ НОВОЕ: Запуск Circuit Breakers (8 шт.)
15. ✅ Запуск Execution Engine (paper mode)
16. ✅ Запуск Position Manager
17. ✅ Запуск Telegram Bot
18. ✅ Запуск Web Dashboard (localhost:8080)
19. ✅ НОВОЕ: Запуск Heartbeat writer (каждые 10 сек)
20. ✅ → Отправка в Telegram: "🟢 Система запущена. Режим: PAPER TRADING"
21. ✅ → НОВОЕ: "🐕 Watchdog: подключён, контролирую"
```

## 26.3 Pre-flight Checklist (НОВОЕ V1.1)

**Перед КАЖДЫМ запуском система автоматически проверяет:**
```
[1/10] .env файл существует и читаем?                    ✅/❌
[2/10] API ключи формат валиден (длина, символы)?         ✅/❌
[3/10] Binance API доступен (ping)?                       ✅/❌
[4/10] API ключи правильные (test auth)?                  ✅/❌
[5/10] API НЕ имеет права withdrawal?                     ✅/❌ КРИТИЧНО!
[6/10] Время синхронизировано (drift < 10 сек)?           ✅/❌
[7/10] SQLite доступна и целая (integrity_check)?         ✅/❌
[8/10] Свободно диска > 1GB?                              ✅/❌
[9/10] Telegram бот отвечает?                             ✅/⚠️ (не блокирует)
[10/10] Нет другого запущенного инстанса (PID lock)?      ✅/❌

Если ЛЮБАЯ ❌ → СИСТЕМА НЕ ЗАПУСКАЕТСЯ + сообщение об ошибке
```

## 26.4 Остановка

### Graceful (нормальная):
```
Ctrl+C или /stop в Telegram
1. Остановить Strategy Engine (нет новых сигналов)
2. Дождаться исполнения текущих ордеров
3. НОВОЕ: Сохранить текущее состояние в state.json
4. Закрыть WebSocket
5. НОВОЕ: Сделать backup БД
6. Закрыть SQLite
7. НОВОЕ: Остановить Watchdog heartbeat
8. → Telegram: "🔴 Система остановлена. Состояние сохранено"
```

### Emergency (аварийная):
```
/kill в Telegram (+ PIN подтверждение) или кнопка на Dashboard
1. Отменить ВСЕ открытые ордера НА БИРЖЕ
2. Продать ВСЕ позиции по рынку (если live mode)
3. Остановить ВСЕ модули
4. НОВОЕ: Заблокировать resume на 30 минут (cooling period)
5. → Telegram: "☠️ АВАРИЙНАЯ ОСТАНОВКА ВЫПОЛНЕНА. Resume через 30 мин"
```

---

# 27. КРИТЕРИИ УСПЕХА (ОБНОВЛЕНО V1.5)

## 27.1 Технические критерии

| Критерий | Минимум | Цель |
|---|---|---|
| Аптайм системы | 95% | 99% |
| Потеря данных | 0 | 0 |
| Время реакции на сигнал | < 2 сек | < 500ms |
| Ошибки исполнения | < 5% | < 1% |
| Risk Sentinel accuracy | 100% | 100% (ни одного пропуска) |
| Circuit Breaker false positive | < 10% | < 5% |
| Watchdog availability | 99% | 99.9% |
| Data Integrity pass rate | > 99.9% | 100% |

## 27.2 Торговые критерии (после 1 месяца paper trading)

| Критерий | Минимум | Цель |
|---|---|---|
| Win Rate | > 50% | > 55% |
| Max Drawdown | < 10% | < 5% |
| Profit Factor | > 1.0 | > 1.3 |
| Sharpe Ratio | > 0.5 | > 1.0 |
| Ежемесячный PnL | > 0% | 1–5% |
| ML precision (out-of-sample) | > 0.55 | > 0.60 |
| ML recall (out-of-sample) | > 0.50 | > 0.55 |
| ML ROC-AUC (out-of-sample) | > 0.58 | > 0.65 |
| ML uplift vs baseline | не ухудшает | PF +5% или DD -10% |

## 27.3 Критерии безопасности (РАСШИРЕНО V1.1)

```
✅ Ни одного случая потери > $50/день ($100 абсолют)
✅ Ни одного выполненного ордера без одобрения Risk Sentinel
✅ Ни одного случая утечки API ключей
✅ Kill-switch срабатывает < 3 секунд
✅ Все сделки залогированы и объяснимы
✅ НОВОЕ: Ни одного дублированного ордера
✅ НОВОЕ: Ни одной торговли на устаревших данных
✅ НОВОЕ: Watchdog ВСЕГДА работает параллельно
✅ НОВОЕ: Баланс синхронизирован с биржей (расхождение < 1%)
✅ НОВОЕ: Все Circuit Breakers протестированы
✅ НОВОЕ: БД бэкапится автоматически
✅ НОВОЕ: Pre-flight check проходит на 100% при каждом старте
✅ НОВОЕ: ML проходит shadow mode перед block mode
✅ НОВОЕ: Ни одна ML-модель не включается без out-of-sample проверки
```

---

# 28. ГЛОССАРИЙ (ОБНОВЛЕНО V1.5)

| Термин | Значение |
|---|---|
| **Спот** | Покупка/продажа реального актива (монеты) |
| **Фьючерсы** | Контракт на цену актива (можно потерять > 100%) |
| **Плечо (leverage)** | Торговля заёмными средствами (умножает прибыль И убытки) |
| **PnL** | Profit and Loss — прибыль/убыток |
| **Drawdown** | Максимальное падение баланса от пика |
| **Win Rate** | % прибыльных сделок от общего количества |
| **Sharpe Ratio** | Доходность ÷ волатильность (чем выше, тем лучше) |
| **Stop-Loss** | Автоматическая продажа при достижении макс убытка |
| **Take-Profit** | Автоматическая продажа при достижении цели прибыли |
| **EMA** | Exponential Moving Average — скользящая средняя |
| **RSI** | Relative Strength Index — индекс перекупленности (0–100) |
| **MACD** | Moving Average Convergence Divergence — индикатор тренда |
| **Paper Trading** | Виртуальная торговля на реальных данных без денег |
| **Kill Switch** | Аварийная остановка всей торговли |
| **Exposure** | Процент капитала, вложенный в открытые позиции |
| **Fill** | Подтверждение исполнения ордера биржей |
| **Slippage** | Разница между ожидаемой и фактической ценой исполнения |
| **Confidence** | Уверенность стратегии в сигнале (0.0 — 1.0) |
| **OCO** | One-Cancels-Other — ордер с take-profit И stop-loss |
| **Bollinger Bands** | Канал волатильности: средняя ± N стандартных отклонений |
| **Squeeze** | Сжатие Bollinger Bands — предвестник сильного движения цены |
| **DCA** | Dollar-Cost Averaging — покупка фиксированной суммой через интервалы |
| **Divergence** | Расхождение между ценой и индикатором (предвестник разворота) |
| **Trade Analyzer** | Модуль самообучения: анализирует сделки и оптимизирует параметры |
| **Walk-forward** | Метод валидации: обучение на прошлых данных, тест на будущих |
| **Overfitting** | Переобучение ML модели: хорошо на истории, плохо на реале |
| **LightGBM** | Быстрая ML библиотека градиентного бустинга (дерево решений) |
| **Shadow Mode** | Режим, в котором ML только логирует решения и не влияет на торговлю |
| **Strategy Skill Score** | Нормализованная оценка качества стратегии на исторических данных |
| **Feature Vector** | Набор числовых признаков для принятия решения |
| **OHLCV** | Open-High-Low-Close-Volume — данные свечи |
| **Circuit Breaker** | Автоматический предохранитель, срабатывает при аномалии |
| **Watchdog** | Независимый процесс-надзиратель за основной системой |
| **Heartbeat** | Сигнал "я жив", отправляемый системой каждые 10 сек |
| **Data Age** | Возраст данных (сколько секунд назад получены) |
| **Deduplication** | Защита от повторного выполнения одного и того же ордера |
| **Race Condition** | Ошибка когда два процесса одновременно меняют одни данные |
| **Stale Data** | Устаревшие данные, на которых нельзя торговать |
| **Cooling Period** | Обязательная пауза после аварийной остановки |
| **Pre-flight Check** | Проверка всех систем перед запуском |
| **Absolute Limit** | Жёсткий лимит в коде, не меняется через конфиг |
| **WAL** | Write-Ahead Log — режим SQLite для защиты от повреждений |

---

# 29. ПЛАН ИСПРАВЛЕНИЯ РИСКОВ

## 29.1 Приоритетные исправления

| Риск | Что изменить в ТЗ/архитектуре | Приоритет | Минимальное исправление |
|---|---|---|---|
| Ложная гарантия "не потерять > $50/день" | Разделить hard-limit на новый риск и фактический убыток рынка | P0 | Формулировать STOP как блок новых входов, а не как абсолютную гарантию PnL |
| Live без биржевого stop | Сделать exchange-native protective order обязательным условием live | P0 | Если после fill не удалось поставить OCO/STOP, немедленно закрывать позицию и уводить систему в STOP |
| Смешение market trades и strategy trades | Развести таблицы `trades` и `strategy_trades` | P0 | Хранить завершённые сделки стратегии отдельно от сырых биржевых сделок |
| ML без безопасного rollout | Ввести режимы `off -> shadow -> block` | P0 | Block mode запрещён до завершения shadow периода и out-of-sample теста |
| ML gate только по accuracy | Заменить gate на precision/recall/ROC-AUC + economic uplift | P0 | Не активировать модель, если она не улучшает PF или DD |
| Skill score не влияет на решение | Связать `skill_score` с decision policy | P1 | Для block mode требовать `skill_score >= threshold` |
| Отсутствие реестра моделей | Хранить историю обученных моделей и их метрик | P1 | Добавить `ml_model_registry` с rollback и audit trail |
| Нечистая структура проекта | Привести дерево проекта к реальной модульной структуре | P1 | Убрать битые строки и разнести guards/tests/data по корректным каталогам |

## 29.2 Статус документа после аудита

```text
Версия: 1.5 (Self-Learning Edition)
Статус: Согласовано после внутреннего аудита структуры и ML-разделов
Дата обновления: 12 апреля 2026

Исправлено в этой редакции:
- Починены повреждённые хвостовые секции документа
- Разведены сущности market trades и strategy trades
- Добавлен реестр моделей ML
- Введён безопасный rollout ML: off -> shadow -> block
- Исправлены критерии допуска модели: precision/recall/ROC-AUC + uplift
- Очищена структура проекта и список тестов
```

**Следующий шаг:** Начать реализацию Этапа 0 — структура проекта, схема БД, конфиг, абсолютные лимиты, затем Level 1 Trade Analyzer без включения ML block mode.

---

> **ВАЖНОЕ ПРИМЕЧАНИЕ:** Данная система предназначена для образовательных целей и экспериментальной торговли с малыми суммами. Торговля криптовалютами сопряжена с риском потери капитала. Никакая система не гарантирует прибыль. Всегда торгуйте только теми средствами, потерю которых можете себе позволить.
