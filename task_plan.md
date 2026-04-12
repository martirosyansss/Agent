# Task Plan: Поэтапная реализация SENTINEL по ТЗ V1.5

## Goal
Собрать точный поэтапный план реализации проекта SENTINEL по фазам из ТЗ V1.5 так, чтобы разработка шла от безопасного MVP на paper trading к контролируемому live и только затем к самообучению Level 2/3.

## Current Phase
Phase 9

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
- [ ] Доделать Dashboard: PnL chart, open positions, recent trades, control buttons
- [ ] Добавить auto-refresh или websocket updates
- [ ] Вывести system mode, risk state, uptime и health indicators
- [ ] Проверить сценарии graceful stop и emergency stop из UI
- **Status:** pending

### Phase 11: Strategy Arsenal Expansion / Этапы после MVP
- [ ] Добавить `grid_trading.py` и `market_regime.py`
- [ ] Добавить `mean_reversion.py` после готовности 1d features и regime-features
- [ ] Добавить `strategy_selector.py` только после baseline-метрик по отдельным стратегиям
- [ ] Добавить `bollinger_breakout.py`, `dca_bot.py`, `macd_divergence.py`
- [ ] Для каждой стратегии отдельно подготовить тесты и backtest baseline
- **Status:** pending

### Phase 12: Trade Analyzer Level 1 / V1.5 старт
- [ ] Реализовать `analyzer/trade_record.py`, `statistician.py`, `reports.py`
- [ ] Собирать закрытые сделки в `strategy_trades`
- [ ] Запускать еженедельный и ежемесячный анализ ошибок, win/loss паттернов и market regimes
- [ ] Не менять параметры автоматически; только рекомендации и отчёты
- **Status:** pending

### Phase 13: Trade Analyzer Level 2
- [ ] Реализовать `optimizer.py` для осторожной оптимизации параметров
- [ ] Разрешать не более одного изменения в неделю
- [ ] Каждое изменение сначала тестировать 14 дней в paper mode
- [ ] Вести журнал гипотез и откатов
- **Status:** pending

### Phase 14: Trade Analyzer Level 3 / ML Shadow Mode
- [ ] Реализовать `ml_predictor.py` и `skill_tests.py`
- [ ] Обучать модели только после накопления достаточного числа сделок
- [ ] Сначала включить только shadow mode: ML ничего не блокирует, лишь логирует вероятности
- [ ] Проверять out-of-sample метрики: precision, recall, ROC-AUC, uplift vs baseline
- [ ] Не включать block mode без прохождения всех порогов ТЗ
- **Status:** pending

### Phase 15: Live Micro Rollout / Этап 8
- [ ] Подключить `live_executor.py` только после прохождения paper criteria
- [ ] Начать с `$50-100`, с `live_first_day_max_order` и exchange-native protection orders
- [ ] Первые 2-4 недели держать жёсткий мониторинг и ручное подтверждение переходов
- [ ] Включить полный live только после стабильного micro-live периода
- **Status:** pending

### Phase 16: Delivery
- [ ] План утверждён и понятен как рабочий roadmap
- [ ] Каждая следующая инженерная задача должна ссылаться на соответствующую фазу
- [ ] Реализацию начинать строго с Phase 3, не прыгая сразу к live или ML
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