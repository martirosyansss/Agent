# SENTINEL Dashboard

FastAPI + vanilla-JS web console for the SENTINEL trading bot.
Runs at `http://127.0.0.1:8888/` by default.

## Layout

```
sentinel/dashboard/
├── app.py                  # FastAPI app, middlewares, REST + WebSocket endpoints
├── static/
│   ├── index.html          # Main dashboard page
│   ├── login.html, trades.html, analytics.html, news.html, logs.html, settings.html
│   ├── css/
│   │   ├── index.css       # Main dashboard styles (+ page-specific sections)
│   │   └── login.css       # Login form styles
│   └── js/
│       ├── index.js        # Shared dashboard bundle (loaded by all HTML pages)
│       └── lib/            # Pure, unit-tested helpers — UMD wrappers
│           ├── format.js        # formatPnl / formatUsd / formatVol / pnlClass
│           ├── risk.js          # riskBarColor / riskTextClass / clampPct
│           ├── time.js          # parseOpenedAt / formatDuration / timeAgoSeconds
│           └── market-score.js  # scoreMarketDirection
└── frontend/
    ├── package.json        # Vitest + coverage dev-deps
    ├── vitest.config.js
    └── tests/              # Unit tests for lib/*.js (Vitest)
```

## Running the dashboard

The backend starts together with the bot (`python -m sentinel.main`) — the
dashboard is mounted on the FastAPI app in `app.py`. No separate frontend
build step exists; static files are served as-is from `static/`.

## Security model

- **Password gate**: `dashboard_password` in settings. When set, unauthenticated
  HTML GETs are redirected to `/login`. WebSocket upgrade checks the
  `sentinel_auth` HttpOnly cookie (falls back to `?token=` query-param for
  legacy clients).
- **CSRF double-submit cookie**: `sentinel_csrf` (non-HttpOnly so JS can echo
  it into `X-CSRF-Token` on mutating requests). Rotated every 60 minutes.
- **Rate limit**: 10 mutating actions per minute per client IP on `/api/control/*`
  and `/api/positions/*/close` — honours `X-Forwarded-For` / `X-Real-IP` for
  reverse-proxy setups.
- **Content Security Policy**: `default-src 'self'`, inline scripts/styles
  currently allowed (`'unsafe-inline'`) for Chart.js tooltip styles and
  inline onclick on dynamic pipeline sections. Work-in-progress to remove.
- **Security headers**: `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: same-origin`, `Permissions-Policy: geolocation=() …`.

## Observability

Front-end runtime errors (including `window.onerror` and unhandled promise
rejections) are forwarded to `POST /api/client-error`. The server logs them
at `logging.WARNING` so they surface in normal dashboard logs without
contaminating ERROR channels used for backend faults. The client throttles
duplicates in a 5-second window.

## Frontend development

### Running unit tests

```bash
cd sentinel/dashboard/frontend
npm install        # one-time
npm test           # runs Vitest once
npm run test:watch # re-runs on file change
npm run coverage   # HTML + text coverage report
```

### Adding a new pure helper

1. Add a UMD-wrapped function to `static/js/lib/<name>.js`.
2. Register a `<script src="/static/js/lib/<name>.js">` line in every HTML
   page that uses it (before `index.js`).
3. Bind it in `index.js` via `var foo = SENTINEL.<name>.foo;`.
4. Write a `frontend/tests/<name>.test.js` covering the interesting shapes
   (happy path + edge cases + invalid input).
5. `npm test` must stay green.

**Rule**: anything non-trivial with branches or number-crunching lives in
`lib/` and has tests. DOM code stays in `index.js`.

### CI

`.github/workflows/dashboard-ci.yml` runs on any change under
`sentinel/dashboard/`:

1. `frontend` job — `npm ci` + Vitest (must pass 100%).
2. `backend-syntax` — `python -c "ast.parse(app.py)"`.
3. `js-syntax` — `new Function(...)` parse check on `index.js` and each
   `lib/*.js`.

## Endpoints (cheat sheet)

| Method | Path                                 | Purpose                              |
|--------|--------------------------------------|--------------------------------------|
| GET    | `/`                                  | Dashboard page (auth required if password set) |
| GET    | `/login`                             | Login form                           |
| POST   | `/api/login`                         | Issue `sentinel_auth` cookie         |
| POST   | `/api/logout`                        | Clear auth cookie                    |
| GET    | `/api/status`                        | Main state snapshot                  |
| GET    | `/api/positions`                     | Open positions list                  |
| GET    | `/api/trades`                        | Recent trades (24h)                  |
| GET    | `/api/pnl-history`                   | Equity curve points                  |
| GET    | `/api/market-chart?interval=1m`      | OHLC candles for the chart           |
| POST   | `/api/control/{stop,resume,kill}`    | Lifecycle control (rate-limited)     |
| POST   | `/api/positions/{symbol}/close`      | Manual close (rate-limited)          |
| POST   | `/api/client-error`                  | Client-side error beacon             |
| WS     | `/ws`                                | Real-time `state_update` stream      |

## Troubleshooting

**Login page loops back to `/`** — you already have a valid `sentinel_auth`
cookie. Clear it via the header logout button or DevTools.

**`Сессия истекла — перезагрузите страницу (F5)`** — CSRF token expired
(24h max). Refresh the page; a new cookie will be issued.

**Blank chart after reconnect** — check browser console for `[SENTINEL]`
entries. Persistent issues are also visible in the backend logs (as
`client-error` WARNINGs) thanks to the beacon endpoint.
