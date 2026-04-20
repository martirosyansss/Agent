# Sentinel — техническое README

Это README — для разработчиков. Пользовательская документация —
в [`../docs/`](../docs/) и в [корневом `README.md`](../README.md).

## Стек

- Python 3.12+
- FastAPI (dashboard)
- SQLite (через `aiosqlite`)
- scikit-learn, lightgbm, xgboost (ML-ансамбль)
- pandas / numpy (индикаторы)
- python-binance, python-telegram-bot
- pytest (тесты)

## Структура

```
sentinel/
├── main.py                  # Точка входа, event loop, wiring
├── config.py                # Pydantic-схема для .env
├── analyzer/                # ML-слой (Clean Architecture)
│   ├── ml/                  # Layered ML: domain → features → models → training → prediction → persistence
│   ├── ml_predictor.py      # Публичный фасад поверх ml/
│   ├── ml_ensemble.py       # VotingEnsemble (RF + LGBM + XGB + ElasticNet)
│   ├── ml_walk_forward.py   # MLWalkForwardValidator
│   ├── ml_stacking.py       # StackingHead
│   ├── ml_regime_router.py  # RegimeRouter
│   └── ml_bootstrap.py      # Monte Carlo bootstrap CI
├── strategy/                # Торговые стратегии (одна стратегия = один файл)
├── features/                # Feature engineering, 32 канонических признака
├── risk/                    # Риск-гейты (daily loss, position count, correlation, etc.)
├── guards/                  # Circuit breakers, watchdog
├── execution/               # Постановка ордеров, trailing-stop, partial TP
├── position/                # Учёт позиций, PnL
├── collector/               # Подписка на Binance WebSocket, свечи
├── database/                # SQLite-слой, миграции, async queries
├── monitoring/              # Метрики, Telegram-алерты, events.jsonl
├── telegram_bot/            # Telegram-команды (/status, /pause, /close_all)
├── dashboard/               # FastAPI + статика (см. dashboard/README.md)
├── backtest/                # Walk-forward бэктестинг, Probabilistic Sharpe Ratio
├── core/                    # Общие абстракции (интерфейсы, DTO)
└── tests/                   # pytest suite — 700+ тестов
```

## Установка для разработки

```powershell
cd sentinel
python -m venv venv
.\venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, ruff, mypy (если есть файл)
```

## Запуск

```powershell
cp .env.example .env
# отредактировать .env
python main.py
```

Бот поднимет:
- торговый loop (1h-цикл по закрытию свечей),
- dashboard на `http://127.0.0.1:8080`,
- (опционально) Telegram-бот,
- watchdog-heartbeat,
- ML-retraining-task (каждые N дней).

## Тесты

```powershell
python -m pytest sentinel/tests/ -v
```

Тесты разделены:

- **`test_ml_*`** — ML-слой (walk-forward, stacking, bootstrap, ensemble, predictor).
- **`test_database.py`** — SQL-слой.
- **`test_feature_engine.py`** — feature engineering.
- **`test_dashboard_ml_endpoints.py`** — API dashboard (via `starlette.TestClient`).
- **`test_integration.py`** — end-to-end smoke-test: запуск, закрытие свечи, сигнал, сделка.
- **`test_ml_audit_*`** — регрессионные тесты на правки из audit-раундов.
- **`test_main_runtime.py`** — проверка wiring в `main.py`.

### Селективные прогоны

```powershell
python -m pytest sentinel/tests/test_ml_ensemble.py -v
python -m pytest sentinel/tests/ -k walk_forward -v
python -m pytest sentinel/tests/ -m "not slow"
```

### Coverage

```powershell
python -m pytest sentinel/tests/ --cov=sentinel --cov-report=html
# открыть htmlcov/index.html
```

## Архитектурные принципы

### Clean Architecture в ML-слое

[`analyzer/ml/`](analyzer/ml/) разделён на слои, зависимость идёт
строго **внутрь**:

```
persistence → training → prediction
              ↓           ↓
              models ← features ← domain
```

- **`domain/`** — pydantic/dataclass-объекты, без фреймворков.
- **`features/`** — `FeatureEngine`, адаптивный селектор, канонический
  порядок.
- **`models/`** — RF, LGBM, XGB, ElasticNet, калибровка, стекинг.
- **`training/`** — train-loop, walk-forward, metrics, threshold tuning.
- **`prediction/`** — `MLPredictor`-runtime, regime routing, shadow mode.
- **`persistence/`** — pickle save/load с `_RestrictedUnpickler`.

`ml_predictor.py` — **фасад** над этой структурой, сохраняющий
обратную совместимость старого API.

### Pickle backwards compatibility

Старые модели писались под `analyzer.ml_predictor.*`, новые — под
`analyzer.ml.*`. `_RestrictedUnpickler` поддерживает **оба префикса**
через dual-whitelist, чтобы после рефакторинга старые `.pkl`-файлы
продолжали грузиться.

### Feature flags

Все новые ML-фичи включаются через конфиг, default **OFF**:

- `use_stacking`, `use_regime_routing`, `use_elastic_net`,
  `walk_forward_enabled`.

Выключенный путь должен **полностью совпадать** с поведением до
добавления фичи. Это тестируется.

### Observability-политика

Каждая ошибка в ML-слое:
1. Логируется структурированно (`events.jsonl`).
2. Не роняет процесс — гасится на уровне публичного фасада.
3. Явно отмечается как `fail-open` (разрешаем сделку) или
   `fail-closed` (блокируем). Default — **fail-closed**.

## Линтинг и типы

```powershell
ruff check sentinel/
mypy sentinel/analyzer/  # если настроен
```

## База данных

SQLite в `sentinel/data/sentinel.db`. Схема описана в
`database/schema.sql`. Миграции автоматические при старте — если
таблицы нет, создаётся.

Ключевые таблицы:
- `trades` — все сделки (paper + live) с полным контекстом.
- `signals` — все сгенерированные сигналы, включая отклонённые.
- `ml_predictions` — предсказания модели, с features-snapshot.
- `events` — лог системных событий.
- `positions` — текущие открытые позиции.

Для аудита — SQL напрямую удобнее, чем dashboard:

```powershell
sqlite3 sentinel/data/sentinel.db
> SELECT symbol, pnl FROM trades WHERE created_at > date('now','-7 days');
```

## Полезные скрипты

В [`scripts/`](scripts/) — служебные утилиты (бэкап базы, миграции,
экспорт сделок в CSV). Каждый скрипт — документирован в docstring,
запускать через `python -m sentinel.scripts.<name>`.

## Коммиты и PR

- Commits в формате `<scope>: <imperative summary>` (например,
  `ml: fix threshold drift in walk-forward`).
- Breaking changes помечаются `BREAKING:` в начале.
- PR не мержим без зелёного pytest.
- Крупные рефакторы — отдельными коммитами с описанием **почему**,
  не только что.

## Куда смотреть

- [`../docs/`](../docs/) — пользовательская документация.
- [`dashboard/README.md`](dashboard/README.md) — внутрянка веб-интерфейса.
- `analyzer/ml/README.md` (если появится) — детали ML-архитектуры.
- Inline docstring-и на публичных функциях — первоисточник правды.
