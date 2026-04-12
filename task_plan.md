# Task Plan: Поэтапная реализация SENTINEL по ТЗ V1.5

## Goal
Собрать точный поэтапный план реализации проекта SENTINEL по фазам из ТЗ V1.5 так, чтобы разработка шла от безопасного MVP на paper trading к контролируемому live и только затем к самообучению Level 2/3.

## Current Phase
Phase 10

## Phases

### Phase 1: Requirements & Discovery
- [x] Прочитать planning skill и правила ведения плана
- [x] Выделить из ТЗ обязательные модули, этапы, ограничения и критерии перехода
- [x] Зафиксировать выводы в findings.md
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Разбить проект на рабочие фазы реализации
- [x] Увязать фазы с модулями 1-17 из ТЗ
- [x] Определить зависимости между фазами и критерии выхода
- **Status:** complete

### Phase 3: Foundation / Этап 0
- [x] Создать каркас проекта по структуре из ТЗ
- [x] Подготовить `.env.example`, `config.py`, `requirements.txt`, `README.md`
- [x] Реализовать `core/models.py`, `core/events.py`, `core/constants.py`, `core/absolute_limits.py`
- [x] Настроить логирование и PID/state infrastructure
- [x] Подготовить базовые тесты загрузки конфигурации и absolute limits
- **Status:** complete

### Phase 4: Data Platform / Этап 1
- [x] Реализовать SQLite слой: `db.py`, `repository.py`, WAL mode, integrity checks
- [x] Создать схемы таблиц: candles, trades, signals, orders, positions, daily_stats, strategy_trades, ml_model_registry
- [x] Реализовать Binance WebSocket collector и validator
- [x] Добавить reconnect, heartbeat, deduplication и backup
- [x] Покрыть тестами Data Collector и Database Layer (21 тест — PASS)
- **Status:** complete

### Phase 5: Baseline Analytics / Этап 2
- [x] Реализовать Feature Engine: EMA, RSI, MACD, ADX, BB, ATR, OBV, StochRSI, volume features, FeatureVector
- [x] Реализовать FeatureBuilder: candles → FeatureVector pipeline
- [x] Реализовать базовую стратегию `ema_crossover_rsi` с BUY/SELL логикой
- [x] Добавить explainable reason и confidence scoring для каждого сигнала
- [x] Покрыть тестами индикаторы, FeatureBuilder и EMA стратегию (28 тестов — PASS)
- **Status:** complete

### Phase 6: Operator Interfaces / Этап 3
- [x] Реализовать Telegram bot с командами /status, /pnl, /positions, /trades, /stop, /resume, /kill, /mode, /config
- [x] Подготовить formatters для сигналов, ордеров, SL/TP, Risk State и аварий
- [x] Поднять Web Dashboard на FastAPI + HTML/Chart.js с WebSocket
- [x] Реализовать health, status, positions, trades, pnl-history, control endpoints
- [x] Авторизация по chat_id и логирование команд (29 тестов — PASS)
- **Status:** complete

### Phase 7: Paper Trading Loop / Этап 4
- [x] Реализовать base_executor.py, paper_executor.py с проскальзыванием и комиссией
- [x] Реализовать PositionManager с open/close, PnL, SL/TP, PaperWallet
- [x] Реализовать дневную статистику и state provider
- [x] Покрыть тестами PaperExecutor и PositionManager (24 теста — PASS)
- **Status:** complete

### Phase 8: Risk Hardening / Этап 5
- [x] Реализовать RiskStateMachine (NORMAL → REDUCED → SAFE → STOP)
- [x] Реализовать RiskSentinel с 7 проверками (daily loss, positions, exposure, frequency, size, SL, sanity)
- [x] Реализовать 8 Circuit Breakers (price, losses, spread, volume, API, latency, balance, commission)
- [x] Реализовать KillSwitch с callbacks
- [x] Покрыть тестами (45 тестов — PASS, итого 147 всего)
- **Status:** complete

### Phase 9: Backtesting & Quality Gates / Этап 6
- [x] Реализовать `backtest/engine.py` с комиссией, slippage и safety discount 0.7
- [x] Сформировать backtest report с Win Rate, Sharpe, Max Drawdown, Profit Factor
- [x] Реализовать исторический skill test по стратегиям с time-based split
- [x] Подготовить paper-to-live quality gates по критериям ТЗ (QualityGates: 5 ворот)
- [x] Покрыть тестами (37 тестов — PASS, итого 184 всего)
- **Status:** complete

### Phase 10: Dashboard Completion / Этап 7
- [x] Полный редизайн UI/UX по UI/UX Pro Max skill: Inter + JetBrains Mono fonts, SVG icons, CSS variables
- [x] Убрать все emojis из UI, заменить на Lucide/Heroicons SVG
- [x] 6-колоночная метрик-сетка: PnL Today, PnL Total, Balance, Positions, Trades, Win Rate
- [x] PnL chart с gradient fill, styled tooltips, smooth tension, point hover
- [x] Risk Overview panel с daily loss, max drawdown, exposure, trade frequency
- [x] WebSocket connection indicator (connected/disconnected) с exponential backoff reconnect
- [x] Toast notifications для control actions (start/stop/emergency)
- [x] Accessibility: aria-labels, focus-visible, prefers-reduced-motion, role attributes
- [x] Responsive: 4 breakpoints (480/768/1024/1280px), mobile-friendly controls
- [x] Uptime display в header, version badge
- [x] Skeleton loading animations, empty states для таблиц
- [x] Button states: disabled during async, 44px min height, hover transitions 200ms
- [x] HTML escaping (XSS protection) в dynamic content
- [x] API: /api/backtest-results endpoint, uptime в /api/health, win_rate в /api/status
- [x] Emergency Stop confirmation dialog с предупреждением
- [x] Покрыть тестами (46 тестов interface — PASS, 201 тест всего — PASS)
- **Status:** complete

### Phase 11: Strategy Arsenal Expansion / Этапы после MVP
- [x] Добавить `grid_trading.py` и `market_regime.py`
- [x] Добавить `mean_reversion.py` после готовности 1d features и regime-features
- [x] Добавить `strategy_selector.py` только после baseline-метрик по отдельным стратегиям
- [x] Добавить `bollinger_breakout.py`, `dca_bot.py`, `macd_divergence.py`
- [x] Для каждой стратегии отдельно подготовить тесты и backtest baseline
- [x] Исправлен баг сортировки дип-множителей в DCA Bot
- **Status:** complete (7 файлов, 30 тестов)

### Phase 12: Trade Analyzer Level 1 / V1.5 старт
- [x] Реализовать `analyzer/statistician.py` — TradeStats, фильтры, отчёты
- [x] Собирать закрытые сделки в `strategy_trades`
- [x] Win/loss паттерны, best hours/days, profit factor, max drawdown
- [x] Покрыть тестами (9 тестов)
- **Status:** complete

### Phase 13: Trade Analyzer Level 2
- [x] Реализовать `analyzer/optimizer.py` для осторожной оптимизации параметров
- [x] FROZEN_PARAMS / TUNABLE_PARAMS — разделение параметров
- [x] Walk-forward split 70/30, min 100 trades, improvement > 5%
- [x] Max 1 change/week, apply/rollback механизм
- [x] Покрыть тестами (6 тестов)
- **Status:** complete

### Phase 14: Trade Analyzer Level 3 / ML Shadow Mode
- [x] Реализовать `analyzer/ml_predictor.py` — RandomForest, 15 features
- [x] Rollout: off → shadow → block
- [x] Skill score = 0.40*precision + 0.25*recall + 0.25*roc_auc + 0.10*accuracy
- [x] Shadow mode: allow всегда, block mode: block/reduce/allow
- [x] Покрыть тестами (8 тестов включая train + predict)
- **Status:** complete

### Phase 15: Live Micro Rollout / Этап 8
- [x] Реализовать `execution/live_executor.py` — Binance Spot MARKET + OCO
- [x] First-day limit $20, ORDER_TIMEOUT 10s
- [x] Emergency sell если OCO не подтверждён
- [x] Retry запрещён — ждать fill
- [x] Покрыть тестами (4 теста)
- **Status:** complete

### Phase 16: Delivery
- [x] Все 16 фаз реализованы
- [x] 263 теста — все проходят
- [x] 11 новых модулей: 5 стратегий, market_regime, strategy_selector, statistician, optimizer, ml_predictor, live_executor
- [x] README.md актуален
- [x] task_plan.md обновлён
- **Status:** complete

## Key Questions
1. Какой минимальный вертикальный срез нужен для безопасного MVP? Ответ: Phase 3-8, без live и без ML.
2. Когда можно включать расширенные стратегии? Ответ: после стабильного baseline по первой стратегии и готового backtest pipeline.
3. Когда можно включать самообучение? Ответ: Level 1 после накопления сделок; Level 2 и 3 только после достаточной статистики и out-of-sample проверки.
4. Когда возможен переход в live? Ответ: только после выполнения paper criteria из ТЗ и ручного решения пользователя.

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Сначала строить безопасный paper MVP, потом расширять арсенал | ТЗ жёстко ставит безопасность выше прибыли |
| Разделить Risk Hardening в отдельную фазу до live | Circuit Breakers, Watchdog и Guards обязательны для production-path |
| Trade Analyzer вводить после накопления данных, а не в начале | Self-learning без истории будет шумом и переобучением |
| ML включать только через shadow mode | Это прямо следует из критериев безопасности V1.5 |
| Strategy Selector не делать раньше regime-features и baseline-тестов | Иначе selection layer будет принимать решения без валидной базы |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Нет planning-файлов в проекте | 1 | Создать `task_plan.md`, `findings.md`, `progress.md` в корне проекта |

## Notes
- Базовый roadmap ТЗ 0-8 сохранён, но дополнен модулями 13-17 и quality gates.
- Реализацию начинать с инфраструктуры и paper loop, а не со всех 6 стратегий сразу.
- Level 2 и Level 3 analyzer запрещено включать до накопления достаточного объёма сделок.