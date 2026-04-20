# Dashboard — разработка

Пользовательское описание вкладок — в
[`../../docs/06-dashboard.md`](../../docs/06-dashboard.md). Этот
файл — для разработчиков.

## Стек

- **Backend**: FastAPI в [`app.py`](app.py), монтируется при старте
  `main.py`.
- **Frontend**: статические HTML + vanilla JS, без сборки (без
  webpack / vite / npm). Это сознательно — dashboard должен
  работать в браузере «из коробки», без build-step.
- **Auth**: сессия через cookie, пароль из `DASHBOARD_PASSWORD`
  (bcrypt hash в памяти).
- **Realtime**: WebSocket для живых обновлений (heartbeat,
  последние сделки, события).
- **CSRF**: токен в cookie + заголовке, защищает state-changing
  эндпоинты.

## Структура

```
dashboard/
├── app.py                # FastAPI app, routing, auth, state provider
├── __init__.py
├── static/               # HTML-страницы + assets, отдаются как есть
│   ├── index.html        # Overview
│   ├── trades.html
│   ├── analytics.html    # Metrics, performance
│   ├── observability.html
│   ├── logs.html
│   ├── news.html
│   ├── settings.html     # Read-only view of .env
│   ├── ml-robustness.html # ML metrics: walk-forward, bootstrap CI, regime matrix
│   ├── login.html
│   ├── css/
│   ├── js/
│   └── favicon.ico
└── frontend/             # (legacy) старая версия, не используется
```

## Ключевые endpoints

| Путь | Метод | Что |
|---|---|---|
| `/` | GET | redirect → `/index.html` |
| `/login` | POST | аутентификация, выставляет cookie |
| `/api/state` | GET | overview: balance, positions, PnL |
| `/api/trades` | GET | пагинированная история сделок |
| `/api/signals` | GET | сигналы, включая отклонённые |
| `/api/ml/status` | GET | состояние ML: модель, retrain info |
| `/api/ml/walk-forward` | GET | `WFReport.fold_results` |
| `/api/ml/regime-performance` | GET | per-regime metrics |
| `/api/ml/bootstrap-ci` | GET | `BootstrapCI` на метрики |
| `/api/risk` | GET | circuit breakers, лимиты |
| `/api/events` | GET | фильтрация по уровню и категории |
| `/api/position/{id}/close` | POST | ручное закрытие (нужен PIN) |
| `/ws/events` | WS | live-поток событий |

## State provider pattern

`app.py` **не держит** торгового состояния сам. При старте `main.py`
регистрирует `StateProvider` — объект с методами вроде
`get_balance()`, `get_open_positions()`, `get_recent_trades()`. Все
endpoint'ы в dashboard дёргают этот провайдер, а он уже читает
свежие данные (БД или in-memory state).

Это изолирует dashboard от торговой логики: тесты dashboard
используют `FakeStateProvider`, не трогая Binance и SQLite.

## Тесты

Тесты — в [`../tests/test_dashboard_ml_endpoints.py`](../tests/test_dashboard_ml_endpoints.py).
Используют `starlette.TestClient`:

```python
from starlette.testclient import TestClient

def test_ml_walk_forward_endpoint(client: TestClient, fake_state):
    fake_state.set_wf_report(sample_report)
    response = client.get("/api/ml/walk-forward")
    assert response.status_code == 200
    assert response.json()["mean_auc"] == pytest.approx(0.62)
```

## Frontend conventions

- **Vanilla JS**: ничего не компилируем. Модули через ES6 `import`,
  загружаемые браузером напрямую.
- **Chart.js**: для графиков (через CDN, без npm).
- **Fetch + CSRF**: все state-changing запросы идут через
  `_csrfFetch()` helper в `static/js/auth.js` — он подставляет
  токен из cookie в заголовок `X-CSRF-Token`.
- **WebSocket**: реконнект при разрыве в `static/js/ws.js`,
  экспоненциальный backoff.
- **Без JS-фреймворков**: React / Vue / Svelte — **нет**. Hand-written
  DOM manipulation. Это сознательный выбор — dashboard простой,
  overhead от фреймворка не оправдан.

## Добавление новой страницы

1. Создать `static/new-page.html` по шаблону `analytics.html`.
2. Добавить sidebar-ссылку в `static/index.html` и других страницах
   (или унести sidebar в shared `_sidebar.html`, если ещё не
   унесён).
3. Добавить API-endpoint в `app.py`, если нужны данные с backend.
4. Написать тест в `tests/test_dashboard_*.py`.

## Безопасность

- **CSRF-токен обязателен** на все state-changing запросы.
- **Pin protection** на `close_position`, `pause_bot`, `resume_bot`
  — пин сверяется с `TELEGRAM_PIN` из конфига.
- **API-ключи никогда не отдаются** во `/api/settings` — маскируются
  до `BINANCE_API_KEY=****...last4` в ответе.
- **HTTPS**: `app.py` не занимается TLS сам. В продакшне —
  reverse proxy (nginx / caddy) перед ним.

## Что **не** делаем в dashboard

- **Не меняем `.env`** через UI. Только просмотр. Любые изменения —
  через файл и перезапуск бота.
- **Не изменяем ML-модель** из UI. Переобучение — scheduled job в
  `main.py`.
- **Не выставляем ручные лимит-ордера**. Бот торгует по своей
  логике; если хочется ручной контроль — Binance UI.

## Производительность

- Endpoints отдают JSON размером < 100 KB в 99% случаев.
- Пагинация на `/api/trades` и `/api/signals` — обязательна, default
  limit = 100.
- Heavy-запросы (ML metrics, bootstrap CI) **кэшируются** на стороне
  provider'а — recompute раз в 5 минут, не каждый GET.

## Куда смотреть

- [`../../docs/06-dashboard.md`](../../docs/06-dashboard.md) —
  описание вкладок для пользователей.
- [`app.py`](app.py) — все routing и auth.
- [`static/`](static/) — HTML/JS/CSS, редактируемые напрямую.
