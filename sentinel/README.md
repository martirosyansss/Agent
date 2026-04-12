# SENTINEL V1.5 — Self-Learning Trading System

Автоматическая торговая система для Binance Spot (BTC/USDT, ETH/USDT).

## Быстрый старт

```powershell
# 1. Создать виртуальное окружение
python -m venv venv
.\venv\Scripts\activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Настроить конфигурацию
copy .env.example .env
# → Заполнить .env реальными API-ключами

# 4. Запустить систему
python main.py

# 5. В отдельном терминале — Watchdog (обязательно)
python watchdog.py
```

## Режимы работы

| Режим | Описание |
|-------|----------|
| `paper` | Виртуальная торговля на реальных данных (по умолчанию) |
| `live` | Реальная торговля (требует 2+ недели paper trading) |

## Безопасность

- API ключи хранятся **только** в `.env` (никогда в git)
- Абсолютные лимиты зашиты в код (`core/absolute_limits.py`)
- Withdrawal через API **запрещён**
- 8 автоматических Circuit Breakers
- Независимый Watchdog-процесс
- Pre-flight checklist при каждом запуске

## Структура проекта

```
sentinel/
├── main.py              # Точка входа
├── watchdog.py          # Сторожевой процесс
├── config.py            # Конфигурация из .env
├── core/                # Модели, события, константы, лимиты
├── collector/           # WebSocket → Binance
├── database/            # SQLite (WAL mode)
├── features/            # Технические индикаторы
├── strategy/            # 6 торговых стратегий
├── risk/                # Risk Sentinel + Circuit Breakers
├── execution/           # Paper / Live исполнение ордеров
├── position/            # Трекинг позиций
├── guards/              # Data Integrity, Anti-Corruption, Safe Math
├── analyzer/            # Trade Analyzer (L1/L2/L3)
├── backtest/            # Бэктестирование
├── telegram_bot/        # Telegram интерфейс
├── dashboard/           # Web Dashboard (localhost:8080)
└── tests/               # Тесты
```
