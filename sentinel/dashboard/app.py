"""
Web Dashboard SENTINEL — FastAPI + HTML/JS.

Предоставляет:
- REST API для данных (статус, PnL, позиции, сделки, бэктест)
- WebSocket для real-time обновлений
- HTML-страницу с Chart.js (профессиональный UI/UX)
- Управляющие эндпоинты (start/stop/emergency)

Адрес: http://localhost:{dashboard_port}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

from fastapi import Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from config import get_editable_settings_payload, save_settings_updates
from core.constants import VERSION
from core.events import EventBus

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger(__name__)

_START_TIME = time.time()


class _RateLimiter:
    """Minimal in-memory sliding-window limiter keyed by (client_ip, action).

    Backed by an OrderedDict so LRU eviction keeps memory bounded even when
    many distinct clients hit the endpoint (otherwise a bot scanning IPs
    could grow the dict unbounded).

    Cleanup strategy:
      1. On every `allow()`, entries with all timestamps outside the window
         are dropped from the bucket.
      2. Every `_CLEANUP_EVERY` calls, a full sweep removes empty/stale
         buckets.
      3. If total keys exceed `_MAX_KEYS`, oldest-touched keys are evicted.
    """

    _MAX_KEYS = 4096
    _CLEANUP_EVERY = 256

    def __init__(self, max_calls: int, window_sec: float) -> None:
        from collections import OrderedDict
        self._max = int(max_calls)
        self._window = float(window_sec)
        self._hits: "OrderedDict[tuple[str, str], list[float]]" = OrderedDict()
        self._calls_since_sweep = 0

    def _sweep(self, now: float) -> None:
        cutoff = now - self._window
        stale = [k for k, ts in self._hits.items() if not ts or ts[-1] < cutoff]
        for k in stale:
            self._hits.pop(k, None)

    def allow(self, key: tuple[str, str]) -> tuple[bool, float]:
        now = time.monotonic()
        self._calls_since_sweep += 1
        if self._calls_since_sweep >= self._CLEANUP_EVERY:
            self._sweep(now)
            self._calls_since_sweep = 0

        bucket = self._hits.get(key)
        cutoff = now - self._window
        if bucket is None:
            self._hits[key] = [now]
        else:
            bucket = [t for t in bucket if t >= cutoff]
            if len(bucket) >= self._max:
                self._hits[key] = bucket
                self._hits.move_to_end(key)
                return False, max(0.0, bucket[0] + self._window - now)
            bucket.append(now)
            self._hits[key] = bucket
        self._hits.move_to_end(key)

        # LRU eviction — drop oldest-touched keys if we blew past the cap
        while len(self._hits) > self._MAX_KEYS:
            self._hits.popitem(last=False)
        return True, 0.0


# Mutating endpoints get their own limiter so a burst of /api/control/*
# can't wedge the engine. Tuned for humans — 10 actions per minute is
# plenty for manual clicks but blocks scripted abuse.
_CONTROL_LIMITER = _RateLimiter(max_calls=10, window_sec=60.0)


def _format_uptime() -> str:
    """Форматировать uptime системы."""
    elapsed = int(time.time() - _START_TIME)
    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


class Dashboard:
    """Web-дашборд SENTINEL на FastAPI."""

    def __init__(
        self,
        settings: Settings,
        event_bus: EventBus,
        state_provider: Optional[Callable] = None,
    ) -> None:
        self._settings = settings
        self._port = settings.dashboard_port
        self._password = settings.dashboard_password
        self._event_bus = event_bus
        self._state_provider = state_provider
        self._app = None
        self._server = None
        self._ws_clients: list[Any] = []

        # Callbacks для управления (устанавливаются из main.py)
        self.on_stop: Optional[Callable[[], Coroutine]] = None
        self.on_resume: Optional[Callable[[], Coroutine]] = None
        self.on_kill: Optional[Callable[[], Coroutine]] = None
        self.on_manual_close: Optional[Callable[[str], Coroutine]] = None
        self.on_settings_update: Optional[Callable[[Any], None]] = None
        self.market_chart_provider: Optional[Callable[[str], dict]] = None
        self.trade_history_provider: Optional[Callable[[], list]] = None
        self.news_collector = None  # устанавливается из main.py

    def _build_config_payload(self) -> dict[str, Any]:
        settings = self._settings
        strategies = [
            {
                "name": "Core Swing",
                "enabled": True,
                "summary": f"{settings.signal_timeframe} -> {settings.trend_timeframe} confirmation",
                "details": [
                    f"Min confidence {settings.min_confidence:.2f}",
                    f"SL {settings.stop_loss_pct:.1f}%",
                    f"TP {settings.take_profit_pct:.1f}%",
                    f"Max {settings.max_trades_per_day} trades/day",
                ],
            },
            {
                "name": "Grid Trading",
                "enabled": settings.grid_enabled,
                "summary": f"{settings.grid_num_levels} levels, capital {settings.grid_capital_pct:.0f}%",
                "details": [
                    f"Auto range {'on' if settings.grid_auto_range else 'off'}",
                    f"Min profit {settings.grid_min_profit_pct:.2f}%",
                    f"Max loss {settings.grid_max_loss_pct:.1f}%",
                ],
            },
            {
                "name": "Mean Reversion",
                "enabled": settings.meanrev_enabled,
                "summary": f"RSI {settings.meanrev_rsi_oversold:.0f}/{settings.meanrev_rsi_overbought:.0f}",
                "details": [
                    f"Capital {settings.meanrev_capital_pct:.0f}%",
                    f"SL {settings.meanrev_stop_loss_pct:.1f}%",
                    f"TP {settings.meanrev_take_profit_pct:.1f}%",
                ],
            },
            {
                "name": "Bollinger Breakout",
                "enabled": settings.bb_breakout_enabled,
                "summary": f"BB({settings.bb_period}, {settings.bb_std_dev:.1f}) with squeeze filter",
                "details": [
                    f"Volume x{settings.bb_volume_confirm_mult:.1f}",
                    f"Trail {settings.bb_trailing_stop_pct:.1f}%",
                    f"TP {settings.bb_take_profit_pct:.1f}%",
                ],
            },
            {
                "name": "DCA Bot",
                "enabled": settings.dca_enabled,
                "summary": f"${settings.dca_base_amount_usd:.2f} every {settings.dca_interval_hours}h",
                "details": [
                    f"Max buys/day {settings.dca_max_daily_buys}",
                    f"Invested {settings.dca_max_invested_pct:.0f}%",
                    f"TP {settings.dca_take_profit_pct:.1f}%",
                ],
            },
            {
                "name": "MACD Divergence",
                "enabled": settings.macd_div_enabled,
                "summary": f"MACD {settings.macd_fast}/{settings.macd_slow}/{settings.macd_signal_period}",
                "details": [
                    f"Lookback {settings.macd_lookback_candles} candles",
                    f"RSI confirm {'on' if settings.macd_require_rsi_confirm else 'off'}",
                    f"Volume confirm {'on' if settings.macd_require_vol_confirm else 'off'}",
                ],
            },
        ]

        return {
            "control_center": {
                "mode": settings.trading_mode,
                "symbols": settings.trading_symbols,
                "symbols_display": ", ".join(settings.trading_symbols),
                "signal_timeframe": settings.signal_timeframe,
                "trend_timeframe": settings.trend_timeframe,
                "min_confidence": settings.min_confidence,
                "auto_strategy_selection": settings.auto_strategy_selection,
                "enabled_strategies": sum(1 for strategy in strategies if strategy["enabled"]),
            },
            "risk_limits": [
                {"label": "Max daily loss", "value": f"${settings.max_daily_loss_usd:.2f}", "tone": "negative"},
                {"label": "Daily loss cap", "value": f"{settings.max_daily_loss_pct:.1f}%", "tone": "negative"},
                {"label": "Max position size", "value": f"{settings.max_position_pct:.1f}%", "tone": "warning"},
                {"label": "Total exposure", "value": f"{settings.max_total_exposure_pct:.1f}%", "tone": "warning"},
                {"label": "Max open positions", "value": str(settings.max_open_positions), "tone": "neutral"},
                {"label": "Max order size", "value": f"${settings.max_order_usd:.2f}", "tone": "neutral"},
                {"label": "Trades per hour", "value": str(settings.max_trades_per_hour), "tone": "neutral"},
                {"label": "Trades per day", "value": str(settings.max_trades_per_day), "tone": "neutral"},
                {"label": "Resume cooldown", "value": f"{settings.resume_cooldown_min} min", "tone": "neutral"},
            ],
            "execution_profile": [
                {"label": "Execution mode", "value": settings.trading_mode.upper(), "tone": "positive" if settings.trading_mode == "live" else "neutral"},
                {"label": "Paper balance", "value": f"${settings.paper_initial_balance:.2f}", "tone": "neutral"},
                {"label": "Commission", "value": f"{settings.paper_commission_pct:.3f}%", "tone": "neutral"},
                {"label": "Slippage", "value": f"{settings.paper_slippage_pct:.3f}%", "tone": "neutral"},
                {"label": "Dashboard port", "value": str(settings.dashboard_port), "tone": "neutral"},
                {"label": "Dashboard password", "value": "Configured" if bool(self._password) else "Not configured", "tone": "warning" if bool(self._password) else "neutral"},
            ],
            "system_profile": [
                {"label": "Data max age", "value": f"{settings.max_data_age_sec}s", "tone": "neutral"},
                {"label": "Cross-check interval", "value": f"{settings.price_cross_validation_interval}s", "tone": "neutral"},
                {"label": "Watchdog heartbeat", "value": f"{settings.watchdog_heartbeat_interval}s", "tone": "neutral"},
                {"label": "Watchdog timeout", "value": f"{settings.watchdog_timeout}s", "tone": "warning"},
                {"label": "DB backup interval", "value": f"{settings.db_backup_interval_hours}h", "tone": "neutral"},
                {"label": "RAM ceiling", "value": f"{settings.max_ram_mb} MB", "tone": "neutral"},
                {"label": "Analyzer stats", "value": "Enabled" if settings.analyzer_stats_enabled else "Disabled", "tone": "neutral"},
                {"label": "ML shadow mode", "value": "Enabled" if settings.analyzer_ml_shadow_mode else "Disabled", "tone": "neutral"},
            ],
            "strategies": strategies,
        }

    def _create_app(self):
        """Создать и настроить FastAPI-приложение."""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from starlette.middleware.base import BaseHTTPMiddleware
        import pathlib

        app = FastAPI(title="SENTINEL Dashboard", version=VERSION)

        # ── CORS — restrict to localhost ──────────
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                f"http://localhost:{self._port}",
                f"http://127.0.0.1:{self._port}",
            ],
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            allow_credentials=True,
        )

        # ── Auth middleware ───────────────────────
        dashboard_password = self._password

        class AuthMiddleware(BaseHTTPMiddleware):
            """Token auth for mutating endpoints when password is configured.

            Flow when password IS configured:
              • /login, /api/login, /api/logout, /api/health, /ws → always public
              • Other HTML pages → redirect to /login when no/invalid cookie
              • Other /api/* → 401 JSON
              • Read-only GETs in _READ_ONLY still work without auth for embed use
            """
            _ALWAYS_PUBLIC = {"/api/health", "/api/login", "/api/logout", "/login", "/ws"}
            _HTML_PAGES = {"/", "/settings", "/trades", "/analytics", "/news", "/logs", "/ml-robustness", "/observability"}
            _READ_ONLY = {"/api/status", "/api/positions", "/api/trades",
                          "/api/pnl-history", "/api/market-chart",
                          "/api/backtest-results", "/api/config",
                          "/api/settings/editable", "/api/strategy-performance",
                          "/api/trades/export", "/api/trades/history", "/api/news",
                          "/api/logs", "/api/logs/list",
                          "/api/ml/status", "/api/ml/walk-forward",
                          "/api/ml/regime-performance", "/api/ml/bootstrap-ci",
                          "/api/ml/member-correlation",
                          "/api/ml/training-progress"}

            async def dispatch(self, request: Request, call_next):
                from starlette.responses import RedirectResponse
                path = request.url.path
                # Static files and always-public endpoints — no auth
                if path.startswith("/static") or path in self._ALWAYS_PUBLIC:
                    return await call_next(request)
                # If no password configured, allow everything
                if not dashboard_password:
                    return await call_next(request)
                # Check token. Header or cookie only — NEVER query-params,
                # which leak into browser history, server access logs, and
                # Referer headers sent to third parties (CDNs, font providers).
                token = (request.headers.get("X-Auth-Token")
                         or request.cookies.get("sentinel_auth")
                         or "")
                if not secrets.compare_digest(token, dashboard_password):
                    # HTML pages → send user to /login
                    if request.method == "GET" and path in self._HTML_PAGES:
                        return RedirectResponse(url="/login", status_code=303)
                    # Allow read-only GET without auth for dashboard panels / embeds
                    if request.method == "GET" and path in self._READ_ONLY:
                        return await call_next(request)
                    return JSONResponse(
                        content={"error": "Unauthorized"},
                        status_code=401,
                    )
                return await call_next(request)

        app.add_middleware(AuthMiddleware)

        # ── CSRF double-submit cookie ─────────────
        # Sets a non-HttpOnly `sentinel_csrf` cookie on HTML page loads.
        # For any mutating request (POST/PUT/PATCH/DELETE) the client must
        # echo the cookie value back in `X-CSRF-Token`. Blocks cross-origin
        # POST attacks that can't read same-origin cookies.
        class CsrfMiddleware(BaseHTTPMiddleware):
            _HTML_PATHS = {"/", "/settings", "/trades", "/analytics", "/news", "/logs", "/ml-robustness", "/observability"}
            _MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
            _COOKIE = "sentinel_csrf"
            _HEADER = "X-CSRF-Token"
            _ISSUED = "sentinel_csrf_iat"        # companion cookie holding issue-time (unix seconds)
            _ROTATE_AFTER_SEC = 60 * 60          # rotate once per hour to limit token lifetime
            _MAX_AGE_SEC = 60 * 60 * 24

            async def dispatch(self, request: Request, call_next):
                path = request.url.path
                method = request.method.upper()

                # WebSocket upgrades, static assets, and login (pre-session) bypass CSRF
                if (path.startswith("/static")
                        or path == "/ws"
                        or path == "/api/login"):
                    return await call_next(request)

                # Enforce on mutating requests
                if method in self._MUTATING:
                    cookie_tok = request.cookies.get(self._COOKIE, "")
                    header_tok = request.headers.get(self._HEADER, "")
                    if (not cookie_tok or not header_tok
                            or not secrets.compare_digest(cookie_tok, header_tok)):
                        return JSONResponse(
                            content={"error": "CSRF token missing or invalid"},
                            status_code=403,
                        )

                response = await call_next(request)

                # Issue/refresh CSRF cookie on HTML page loads.
                # Rotates the token value once per hour — shortens the window in
                # which a leaked cookie remains valid, while still re-using the
                # same token across rapid navigations so open tabs don't break.
                if method == "GET" and path in self._HTML_PATHS:
                    existing = request.cookies.get(self._COOKIE)
                    issued_at_raw = request.cookies.get(self._ISSUED, "")
                    try:
                        issued_at = int(issued_at_raw) if issued_at_raw else 0
                    except ValueError:
                        issued_at = 0
                    now_s = int(time.time())
                    should_rotate = (
                        not existing
                        or issued_at <= 0
                        or (now_s - issued_at) >= self._ROTATE_AFTER_SEC
                    )
                    token = secrets.token_urlsafe(32) if should_rotate else existing
                    response.set_cookie(
                        key=self._COOKIE,
                        value=token,
                        secure=False,   # local dashboard over http
                        httponly=False, # JS must read it for header echo
                        samesite="strict",
                        max_age=self._MAX_AGE_SEC,
                        path="/",
                    )
                    # Separate companion cookie tracks issue-time; we never
                    # compare it in a digest check so timing-safety is moot here.
                    response.set_cookie(
                        key=self._ISSUED,
                        value=str(now_s if should_rotate else issued_at),
                        secure=False,
                        httponly=False,
                        samesite="strict",
                        max_age=self._MAX_AGE_SEC,
                        path="/",
                    )
                return response

        app.add_middleware(CsrfMiddleware)

        # Cache-Control for static assets — HTML must not be cached aggressively
        # because we hot-swap it during releases (content hash would be cleaner
        # but that's a bigger refactor). CSS/JS get short cache to balance
        # freshness against repeat-visit speed.
        class CacheHeadersMiddleware:
            def __init__(self, app): self.app = app

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    return await self.app(scope, receive, send)
                path = scope.get("path", "")
                is_html = path.endswith(".html") or path == "/" or path in ("/settings", "/trades", "/analytics", "/news", "/logs")
                is_static_asset = path.startswith("/static/css/") or path.startswith("/static/js/")
                is_static_other = path.startswith("/static/")

                async def send_wrapper(message):
                    if message.get("type") == "http.response.start":
                        headers = list(message.get("headers", []))
                        if is_html:
                            headers.append((b"cache-control", b"no-cache, max-age=0"))
                        elif is_static_asset:
                            headers.append((b"cache-control", b"public, max-age=60"))
                        elif is_static_other:
                            headers.append((b"cache-control", b"public, max-age=3600"))
                        message["headers"] = headers
                    await send(message)

                await self.app(scope, receive, send_wrapper)

        app.add_middleware(CacheHeadersMiddleware)

        # ── Security headers (CSP, frame, referrer) ────────────────
        # Permissive CSP — keeps 'unsafe-inline' because the current HTML
        # uses inline styles and Chart.js tooltip plugins inject styles.
        # Tightening to nonces is a follow-up refactor.
        class SecurityHeadersMiddleware:
            def __init__(self, app): self.app = app

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    return await self.app(scope, receive, send)

                async def send_wrapper(message):
                    if message.get("type") == "http.response.start":
                        headers = list(message.get("headers", []))
                        csp = (
                            "default-src 'self'; "
                            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                            "font-src 'self' https://fonts.gstatic.com data:; "
                            "img-src 'self' data: blob:; "
                            "connect-src 'self' ws: wss: https://cdn.jsdelivr.net; "
                            "frame-ancestors 'none'; "
                            "base-uri 'self'; "
                            "form-action 'self'"
                        )
                        headers.append((b"content-security-policy", csp.encode("ascii")))
                        headers.append((b"x-frame-options", b"DENY"))
                        headers.append((b"x-content-type-options", b"nosniff"))
                        headers.append((b"referrer-policy", b"same-origin"))
                        headers.append((b"permissions-policy", b"geolocation=(), microphone=(), camera=()"))
                        message["headers"] = headers
                    await send(message)

                await self.app(scope, receive, send_wrapper)

        app.add_middleware(SecurityHeadersMiddleware)

        static_dir = pathlib.Path(__file__).parent / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        # ── Health & API ──────────────────────────

        @app.get("/api/health")
        async def health():
            return {
                "status": "ok",
                "version": VERSION,
                "uptime": _format_uptime(),
                "timestamp": int(time.time() * 1000),
            }

        # ── Auth: login / logout ─────────────────
        # When dashboard_password is configured, the UI's AuthMiddleware
        # requires the `sentinel_auth` cookie. These two endpoints set and
        # clear that cookie; when no password is configured they succeed
        # trivially so the flow works identically in paper/dev.
        @app.post("/api/login")
        async def login(request: Request):
            try:
                body = await request.json()
            except Exception:
                body = {}
            submitted = str(body.get("password", "") or "")
            if not dashboard_password:
                # No password configured — auth is effectively disabled
                return JSONResponse(content={"ok": True, "auth_required": False})
            if not secrets.compare_digest(submitted, dashboard_password):
                return JSONResponse(content={"ok": False, "error": "Invalid password"}, status_code=401)
            response = JSONResponse(content={"ok": True, "auth_required": True})
            # HttpOnly + SameSite=Strict: browser auto-sends to same origin
            # (including WebSocket upgrade), but JS cannot read it → safer vs XSS.
            response.set_cookie(
                key="sentinel_auth",
                value=dashboard_password,
                httponly=True,
                samesite="strict",
                secure=False,
                max_age=60 * 60 * 12,
                path="/",
            )
            return response

        @app.post("/api/logout")
        async def logout():
            response = JSONResponse(content={"ok": True})
            response.delete_cookie(key="sentinel_auth", path="/")
            return response

        @app.post("/api/client-error")
        async def client_error(request: Request):
            """Collect frontend errors for server-side diagnostics.

            Each payload is one error record from _logError() in index.js.
            We log at WARNING so they appear in normal dashboard logs without
            flooding ERROR channels used for backend faults.
            """
            try:
                body = await request.json()
            except Exception:
                body = {}
            scope = str(body.get("scope", "unknown"))[:64]
            msg = str(body.get("msg", ""))[:400]
            extra = body.get("extra")
            logger.warning(
                "client-error scope=%s msg=%s extra=%s",
                scope, msg, extra,
            )
            return JSONResponse(content={"ok": True})

        @app.get("/api/status")
        async def status():
            state = self._get_state()
            return JSONResponse(content={
                "mode": state.get("mode", "paper"),
                "risk_state": state.get("risk_state", "NORMAL"),
                "trading_paused": state.get("trading_paused", False),
                "uptime": state.get("uptime", _format_uptime()),
                "pnl_today": state.get("pnl_today", 0.0),
                "pnl_total": state.get("pnl_total", 0.0),
                "open_positions": state.get("open_positions", 0),
                "trades_today": state.get("trades_today", 0),
                "balance": state.get("balance", 0.0),
                "win_rate": state.get("win_rate", 0.0),
                "profit_factor": state.get("profit_factor", 0.0),
                "avg_rr_ratio": state.get("avg_rr_ratio", 0.0),
                "max_drawdown_pct": state.get("max_drawdown_pct", 0.0),
                "current_drawdown_pct": state.get("current_drawdown_pct", 0.0),
                "peak_balance": state.get("peak_balance", 0.0),
                "total_wins": state.get("total_wins", 0),
                "total_losses": state.get("total_losses", 0),
                "risk_details": state.get("risk_details", {}),
                "activity": state.get("activity", {}),
                "indicators": state.get("indicators", {}),
                "indicators_per_symbol": state.get("indicators_per_symbol", {}),
                "trading_symbols": state.get("trading_symbols", []),
                "win_rate_per_symbol": state.get("win_rate_per_symbol", {}),
                "readiness": state.get("readiness", {}),
                "strategy_log": state.get("strategy_log", []),
                "ml_status": state.get("ml_status", {}),
                "version": VERSION,
            })

        @app.get("/api/positions")
        async def positions():
            state = self._get_state()
            pos_list = state.get("positions", [])
            result = []
            for p in pos_list:
                if hasattr(p, "symbol"):
                    entry = p.entry_price or 0.0
                    current = p.current_price or 0.0
                    qty = p.quantity or 0.0
                    pnl = p.unrealized_pnl or 0.0
                    notional = abs(entry * qty) if entry and qty else 0.0
                    pnl_pct = (pnl / notional * 100) if notional > 0 else 0.0
                    sl = getattr(p, "stop_loss_price", 0.0) or 0.0
                    tp = getattr(p, "take_profit_price", 0.0) or 0.0
                    # Risk:Reward ratio
                    side = p.side
                    if side == "BUY" or side == "LONG":
                        risk = abs(entry - sl) if sl > 0 else 0.0
                        reward = abs(tp - entry) if tp > 0 else 0.0
                    else:
                        risk = abs(sl - entry) if sl > 0 else 0.0
                        reward = abs(entry - tp) if tp > 0 else 0.0
                    rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0
                    # SL/TP progress: how far price moved toward TP (0-100) or SL (negative)
                    if side in ("BUY", "LONG") and tp > 0 and sl > 0 and tp != entry and tp != sl:
                        sl_tp_progress = round((current - entry) / (tp - entry) * 100, 1)
                    elif side in ("SELL", "SHORT") and tp > 0 and sl > 0 and entry != tp and tp != sl:
                        sl_tp_progress = round((entry - current) / (entry - tp) * 100, 1)
                    else:
                        sl_tp_progress = 0.0
                    result.append({
                        "symbol": p.symbol,
                        "side": side,
                        "strategy_name": getattr(p, "strategy_name", ""),
                        "entry_price": entry,
                        "current_price": current,
                        "quantity": qty,
                        "stop_loss_price": sl,
                        "take_profit_price": tp,
                        "unrealized_pnl": pnl,
                        "pnl_pct": round(pnl_pct, 2),
                        "notional": round(notional, 2),
                        "rr_ratio": rr_ratio,
                        "sl_tp_progress": sl_tp_progress,
                        "opened_at": getattr(p, "opened_at", ""),
                        "signal_reason": getattr(p, "signal_reason", ""),
                        "position_id": getattr(p, "position_id", ""),
                        "is_paper": getattr(p, "is_paper", True),
                    })
                elif isinstance(p, dict):
                    result.append(p)
            return JSONResponse(content=result)

        @app.get("/api/trades")
        async def trades():
            state = self._get_state()
            return JSONResponse(content=state.get("recent_trades", []))

        @app.get("/api/pnl-history")
        async def pnl_history():
            state = self._get_state()
            return JSONResponse(content=state.get("pnl_history", []))

        @app.get("/api/market-chart")
        async def market_chart(interval: str = "1m", symbol: str = "", end: int = 0):
            """Return chart candles. `end` (ms, optional) shifts the window back
            in time for historical browsing; 0 or missing = live tail."""
            if self.market_chart_provider:
                try:
                    return JSONResponse(content=self.market_chart_provider(interval, symbol, end))
                except TypeError:
                    # Legacy provider without end_ts arg — fall back silently
                    return JSONResponse(content=self.market_chart_provider(interval, symbol))
            state = self._get_state()
            return JSONResponse(content=state.get("market_chart", {"candles": []}))

        @app.get("/api/backtest-results")
        async def backtest_results():
            state = self._get_state()
            return JSONResponse(content=state.get("backtest_results", {}))

        @app.get("/api/strategy-performance")
        async def strategy_performance():
            state = self._get_state()
            return JSONResponse(content=state.get("strategy_performance", []))

        @app.get("/api/trades/export")
        async def trades_export():
            """CSV export of all strategy trades."""
            import csv
            import io
            from fastapi.responses import StreamingResponse

            state = self._get_state()
            rows = state.get("trades_export", [])

            output = io.StringIO()
            if rows:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            else:
                output.write("No trades to export\n")

            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=sentinel_trades.csv"},
            )

        @app.get("/api/trades/history")
        async def trades_history(strategy: str = "", symbol: str = "", limit: int = 500):
            """Full strategy trade history with all details."""
            state = self._get_state()
            all_trades = state.get("trades_export_full", [])
            if strategy:
                all_trades = [t for t in all_trades if t.get("strategy_name") == strategy]
            if symbol:
                all_trades = [t for t in all_trades if t.get("symbol") == symbol]
            return JSONResponse(content=all_trades[:limit])

        @app.get("/api/news")
        async def news_feed():
            """Крипто-новости с анализом влияния на курс."""
            if not self.news_collector:
                return JSONResponse(content={
                    "news": [],
                    "sentiment": {"fear_greed_index": 50, "fear_greed_label": "N/A", "overall_score": 0},
                    "impact": {"status": "disabled", "message": "News collector not initialized"},
                })
            return JSONResponse(content={
                "news": self.news_collector.get_news(limit=200),
                "sentiment": self.news_collector.get_sentiment(),
                "impact": self.news_collector.get_impact_summary(),
                "signal": self.news_collector.get_news_signal(),
            })

        @app.get("/api/logs/list")
        async def logs_list():
            """Список доступных лог-файлов с метаданными."""
            logs_dir = pathlib.Path(__file__).parent.parent / "logs"
            files = []
            if logs_dir.exists():
                for f in sorted(logs_dir.glob("*.log")) + sorted(logs_dir.glob("*.jsonl")):
                    try:
                        stat = f.stat()
                        files.append({
                            "name": f.name,
                            "size": stat.st_size,
                            "modified": int(stat.st_mtime * 1000),
                        })
                    except OSError:
                        continue
            return JSONResponse(content={"files": files})

        @app.get("/api/logs")
        async def logs_read(name: str = "sentinel.log", tail: int = 500, level: str = ""):
            """Прочитать хвост лог-файла. tail — число последних строк (макс 5000)."""
            logs_dir = (pathlib.Path(__file__).parent.parent / "logs").resolve()
            # Защита от path traversal: разрешаем только имя файла без разделителей
            if "/" in name or "\\" in name or ".." in name or not name:
                return JSONResponse(content={"error": "invalid log name"}, status_code=400)
            log_path = (logs_dir / name).resolve()
            try:
                log_path.relative_to(logs_dir)
            except ValueError:
                return JSONResponse(content={"error": "path outside logs dir"}, status_code=400)
            if not log_path.exists() or not log_path.is_file():
                return JSONResponse(content={"error": "log not found", "name": name}, status_code=404)

            tail = max(1, min(int(tail or 500), 5000))

            ansi_re = re.compile(r"\x1b\[[0-9;]*m")

            # Читаем последние N строк эффективно — с конца файла блоками
            def read_tail_lines(path: pathlib.Path, n: int) -> list[str]:
                try:
                    size = path.stat().st_size
                    if size == 0:
                        return []
                    with path.open("rb") as fh:
                        block = 64 * 1024
                        data = b""
                        pos = size
                        while pos > 0 and data.count(b"\n") <= n:
                            read_size = min(block, pos)
                            pos -= read_size
                            fh.seek(pos)
                            data = fh.read(read_size) + data
                    text = data.decode("utf-8", errors="replace")
                    lines = [ansi_re.sub("", ln) for ln in text.splitlines()]
                    return lines[-n:]
                except OSError as exc:
                    logger.warning("read_tail_lines failed for %s: %s", path, exc)
                    return []

            lines = read_tail_lines(log_path, tail)

            if level:
                lvl = level.upper()
                keep = {"DEBUG":   {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
                        "INFO":    {"INFO", "WARNING", "ERROR", "CRITICAL"},
                        "WARNING": {"WARNING", "ERROR", "CRITICAL"},
                        "ERROR":   {"ERROR", "CRITICAL"}}.get(lvl)
                if keep:
                    level_re = re.compile(r"\|\s*(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\s*\|")
                    filtered: list[str] = []
                    keep_cont = False
                    for ln in lines:
                        m = level_re.search(ln)
                        if m:
                            lv = m.group(1).upper()
                            if lv == "WARN":
                                lv = "WARNING"
                            elif lv == "FATAL":
                                lv = "CRITICAL"
                            keep_cont = lv in keep
                            if keep_cont:
                                filtered.append(ln)
                        elif keep_cont:
                            filtered.append(ln)
                    lines = filtered

            try:
                stat = log_path.stat()
                size = stat.st_size
                mtime = int(stat.st_mtime * 1000)
            except OSError:
                size = 0
                mtime = 0

            return JSONResponse(content={
                "name": name,
                "size": size,
                "modified": mtime,
                "lines": lines,
                "truncated": len(lines) >= tail,
            })

        @app.delete("/api/logs")
        async def logs_clear(name: str = ""):
            """Очистить (truncate) лог-файл. Для events.jsonl и прочих .log/.jsonl."""
            logs_dir = (pathlib.Path(__file__).parent.parent / "logs").resolve()
            if "/" in name or "\\" in name or ".." in name or not name:
                return JSONResponse(content={"error": "invalid log name"}, status_code=400)
            log_path = (logs_dir / name).resolve()
            try:
                log_path.relative_to(logs_dir)
            except ValueError:
                return JSONResponse(content={"error": "path outside logs dir"}, status_code=400)
            if not log_path.exists() or not log_path.is_file():
                return JSONResponse(content={"error": "log not found", "name": name}, status_code=404)
            if log_path.suffix.lower() not in {".log", ".jsonl"}:
                return JSONResponse(content={"error": "only .log/.jsonl allowed"}, status_code=400)

            try:
                # Truncate in-place (loguru append handle продолжит писать с позиции 0+).
                with log_path.open("r+b") as fh:
                    fh.truncate(0)
                logger.warning("Log file cleared via dashboard: %s", name)
                return JSONResponse(content={"status": "ok", "name": name, "size": 0})
            except PermissionError as exc:
                return JSONResponse(
                    content={"error": f"file busy / locked: {exc}"},
                    status_code=409,
                )
            except OSError as exc:
                return JSONResponse(
                    content={"error": f"truncate failed: {exc}"},
                    status_code=500,
                )

        @app.get("/api/ml/status")
        async def ml_status():
            """ML model status, metrics, and mode."""
            ml = self._state_provider() if self._state_provider else {}
            predictor = ml.get("ml_predictor") if ml else None
            if predictor is None:
                return JSONResponse(content={"enabled": False, "reason": "not initialized"})
            m = predictor.metrics
            # M-5: read through public properties instead of private attrs
            # so the dashboard stays decoupled from MLPredictor internals.
            last_train_ms = getattr(predictor, "_last_train_ts", 0) or 0
            # PSR / DSR fields were added in Phase-1 of the ML overhaul but
            # the status endpoint predated them. Exposing via getattr keeps
            # older pickled MLMetrics objects (no psr/dsr fields) loadable
            # without breaking the payload shape.
            _psr = getattr(m, "psr", None) if m else None
            _dsr = getattr(m, "dsr", None) if m else None
            _psr_gate = bool(getattr(m, "psr_gate_passed", False)) if m else False
            _psr_n_trials = int(getattr(m, "psr_n_trials", 1)) if m else 1
            _fi_stab = getattr(m, "feature_importance_stability", None) if m else None
            return JSONResponse(content={
                "enabled": True,
                "ready": predictor.is_ready,
                "mode": predictor.rollout_mode,
                "version": predictor.model_version or "none",
                "needs_retrain": predictor.needs_retrain(),
                "last_trained_at": (last_train_ms / 1000.0) if last_train_ms else None,
                "metrics": {
                    "precision": round(m.precision, 4) if m else None,
                    "recall": round(m.recall, 4) if m else None,
                    "roc_auc": round(m.roc_auc, 4) if m else None,
                    "skill_score": round(m.skill_score, 4) if m else None,
                    "train_samples": m.train_samples if m else 0,
                    "test_samples": m.test_samples if m else 0,
                    # López de Prado Phase-1 fields — PSR/DSR tell whether
                    # the observed Sharpe is statistically distinguishable
                    # from zero (PSR) + from multi-testing inflation (DSR).
                    "psr": round(_psr, 4) if _psr is not None else None,
                    "dsr": round(_dsr, 4) if _dsr is not None else None,
                    "psr_gate_passed": _psr_gate,
                    "psr_n_trials": _psr_n_trials,
                    # Phase-4 feature-importance stability across ensemble
                    # members. Empty dict when < 2 members contributed.
                    "feature_importance_stability": _fi_stab or {},
                } if m else None,
                "threshold": predictor.block_threshold,
                "block_threshold": predictor.block_threshold,
                "reduce_threshold": predictor.reduce_threshold,
            })

        @app.post("/api/ml/retrain")
        async def ml_retrain():
            """Trigger manual ML retraining (runs in background)."""
            ml = self._state_provider() if self._state_provider else {}
            retrain_fn = ml.get("ml_retrain_fn") if ml else None
            if retrain_fn is None:
                return JSONResponse(content={"status": "error", "message": "retrain function not wired"}, status_code=503)
            progress_fn = ml.get("ml_training_progress_fn") if ml else None
            progress_set_fn = ml.get("ml_training_progress_set_fn") if ml else None
            if progress_fn is not None and progress_fn().get("active"):
                return JSONResponse(
                    content={"status": "busy", "message": "Обучение уже выполняется"},
                    status_code=409,
                )
            # Pre-arm progress state synchronously so the very first poll from the
            # dashboard sees an active job even if the background coroutine hasn't
            # scheduled its first await yet.
            if progress_set_fn is not None:
                progress_set_fn(
                    active=True,
                    phase="queued",
                    message="Задача поставлена в очередь…",
                    symbols_total=0,
                    symbols_done=0,
                    current_symbol=None,
                    percent=0,
                    started_at=time.time(),
                    finished_at=None,
                    ok=None,
                    metrics=None,
                )
            asyncio.create_task(retrain_fn())
            return JSONResponse(content={"status": "started", "message": "ML retraining triggered in background"})

        @app.get("/api/ml/training-progress")
        async def ml_training_progress():
            """Live snapshot of the manual retrain job."""
            ml = self._state_provider() if self._state_provider else {}
            progress_fn = ml.get("ml_training_progress_fn") if ml else None
            if progress_fn is None:
                return JSONResponse(content={
                    "active": False,
                    "phase": "unavailable",
                    "message": "progress tracker not wired",
                })
            return JSONResponse(content=progress_fn())

        # ──────────────────────────────────────────────────────
        # Phase-7: ML Robustness — WF / regime / bootstrap / correlations
        # Each route mirrors the /api/ml/status pattern above (lookup through
        # state_provider, tolerate missing predictor, serialize metrics).
        # Routes are intentionally read-only so the unauthenticated embed
        # case works — see AuthMiddleware._READ_ONLY for the mechanism.
        # ──────────────────────────────────────────────────────

        @app.get("/api/ml/walk-forward")
        async def ml_walk_forward():
            """Walk-forward stability report (per-fold AUC/precision + summary)."""
            ml = self._state_provider() if self._state_provider else {}
            predictor = ml.get("ml_predictor") if ml else None
            if predictor is None:
                return JSONResponse(content={"enabled": False, "reason": "not initialized"})
            report = getattr(predictor, "walk_forward_report", None)
            if report is None:
                return JSONResponse(content={"enabled": True, "available": False,
                                             "reason": "walk-forward not yet run"})
            folds = [{
                "fold_idx": r.fold_idx,
                "train_range": [r.train_start, r.train_end],
                "test_range": [r.test_start, r.test_end],
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
                "roc_auc": round(r.roc_auc, 4),
                "skill_score": round(r.skill_score, 4),
                "train_precision": round(r.train_precision, 4),
                "n_train": r.n_train,
                "n_test": r.n_test,
                "calibrated_threshold": round(r.calibrated_threshold, 4),
            } for r in report.fold_results]
            return JSONResponse(content={
                "enabled": True,
                "available": True,
                "summary": report.summary(),
                "folds": folds,
            })

        @app.get("/api/ml/regime-performance")
        async def ml_regime_performance():
            """Per-regime specialist statistics from RegimeRouter."""
            ml = self._state_provider() if self._state_provider else {}
            predictor = ml.get("ml_predictor") if ml else None
            if predictor is None:
                return JSONResponse(content={"enabled": False, "reason": "not initialized"})
            router = getattr(predictor, "regime_router", None)
            if router is None or not router.is_ready:
                return JSONResponse(content={"enabled": True, "available": False,
                                             "reason": "regime routing not active"})
            return JSONResponse(content={
                "enabled": True,
                "available": True,
                "regimes": router.get_regime_stats(),
                "trained_regimes": router.trained_regimes,
            })

        @app.get("/api/ml/bootstrap-ci")
        async def ml_bootstrap_ci():
            """Bootstrap confidence intervals for precision/recall/AUC."""
            ml = self._state_provider() if self._state_provider else {}
            predictor = ml.get("ml_predictor") if ml else None
            if predictor is None:
                return JSONResponse(content={"enabled": False, "reason": "not initialized"})
            ci = getattr(predictor, "bootstrap_ci", {}) or {}
            if not ci:
                return JSONResponse(content={"enabled": True, "available": False,
                                             "reason": "bootstrap_ci flag off or model not trained"})
            return JSONResponse(content={
                "enabled": True,
                "available": True,
                "intervals": ci,
            })

        @app.get("/api/ml/member-correlation")
        async def ml_member_correlation():
            """Pairwise error correlation between ensemble members."""
            ml = self._state_provider() if self._state_provider else {}
            predictor = ml.get("ml_predictor") if ml else None
            if predictor is None:
                return JSONResponse(content={"enabled": False, "reason": "not initialized"})
            corr = getattr(predictor, "member_error_correlation", {}) or {}
            if not corr:
                return JSONResponse(content={"enabled": True, "available": False,
                                             "reason": "model not trained or single member"})
            return JSONResponse(content={
                "enabled": True,
                "available": True,
                "correlations": {k: round(v, 4) for k, v in corr.items()},
                "warn_threshold": 0.85,
            })

        # 10-second TTL cache shared across requests. The endpoint is auto-
        # refreshed every 30s by each open dashboard tab — without a cache,
        # N tabs × 30s × file-IO would tail-read events.jsonl repeatedly.
        # Keyed by (events_path, window_hours, max_events) so different
        # paths (e.g. test tmpfiles) and different views don't collide.
        _OBS_CACHE: dict[tuple[str, float, int], tuple[float, dict]] = {}
        _OBS_CACHE_TTL_SEC = 10.0

        @app.get("/api/observability/summary")
        async def observability_summary(window_hours: float = 24.0, max_events: int = 500):
            """Aggregate recent observability events for the dashboard.

            Reads ``logs/events.jsonl`` (tail-only — bounded by ``max_events``
            so a multi-GB archive never blocks the request) and groups by:

            - ``signal_rejected.gate`` — which risk-gate blocks most signals
            - ``component_error.component`` — which subsystem fails most
            - ``guard_tripped.guard`` — recent guard activations
            - ``signal_approved`` vs ``signal_rejected`` — overall + BUY/SELL split

            ``window_hours`` filters by event timestamp (default 24h). When
            the window exceeds 24h, rotated backups (``events.jsonl.1``,
            ``.2``, …) are also scanned in chronological order so multi-day
            views aren't silently truncated at the rotation boundary.

            Path resolution honours ``SENTINEL_EVENTS_LOG_PATH`` so tests
            can isolate from the real production file via ``monkeypatch.setenv``.

            Results are cached in-process for ~10s — multiple open dashboard
            tabs auto-refreshing every 30s would otherwise tail-read the file
            repeatedly. Keyed by ``(window_hours, max_events)``.
            """
            window_hours = max(0.5, min(float(window_hours), 168.0))  # 30min..7d
            max_events = max(50, min(int(max_events), 5000))

            now_wall = time.time()

            # Path resolution: env override wins so test fixtures can point
            # at a tmp file without clobbering production logs.
            import os as _os
            env_path = _os.environ.get("SENTINEL_EVENTS_LOG_PATH", "").strip()
            if env_path:
                events_path = pathlib.Path(env_path).resolve()
            else:
                events_path = (
                    pathlib.Path(__file__).parent.parent / "logs" / "events.jsonl"
                ).resolve()

            # Cache key includes the resolved path so test fixtures with
            # distinct ``SENTINEL_EVENTS_LOG_PATH`` values get distinct
            # cache entries (otherwise consecutive tests would see stale
            # data from the prior test's cache fill).
            cache_key = (str(events_path), window_hours, max_events)
            cached = _OBS_CACHE.get(cache_key)
            if cached is not None and (now_wall - cached[0]) < _OBS_CACHE_TTL_SEC:
                return JSONResponse(content=cached[1])

            cutoff_ms = int(now_wall * 1000) - int(window_hours * 3600 * 1000)

            def _read_tail(path: pathlib.Path, n_lines: int) -> list[dict]:
                """Tail-read up to n_lines from a JSONL file. Each line one JSON object.
                Skips malformed lines silently (a corrupt last write must not break the dashboard)."""
                if not path.exists() or not path.is_file():
                    return []
                try:
                    size = path.stat().st_size
                    if size == 0:
                        return []
                    block = 64 * 1024
                    data = b""
                    pos = size
                    with path.open("rb") as fh:
                        while pos > 0 and data.count(b"\n") <= n_lines:
                            read_size = min(block, pos)
                            pos -= read_size
                            fh.seek(pos)
                            data = fh.read(read_size) + data
                    text = data.decode("utf-8", errors="replace")
                    out: list[dict] = []
                    for ln in text.splitlines()[-n_lines:]:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            out.append(json.loads(ln))
                        except (ValueError, TypeError):
                            continue
                    return out
                except OSError as exc:
                    logger.warning("observability read_tail failed: %s", exc)
                    return []

            # Backup-file inclusion: when the requested window exceeds 24h,
            # rotated backups (.1, .2, ...) may contain in-window events.
            # Read newest backup first (just-rotated has the most relevant
            # data) up to the configured count, then merge with active file.
            files_to_read: list[pathlib.Path] = [events_path]
            if window_hours > 24.0:
                # Iterate possible backup suffixes — EventLog rotates up to
                # ``backup_count`` (default 5). Stop on first missing file
                # so we don't waste stat() calls.
                for i in range(1, 11):
                    backup = events_path.with_suffix(events_path.suffix + f".{i}")
                    if not backup.exists():
                        break
                    files_to_read.append(backup)

            # Read newest file first (events_path), oldest last. Combined
            # tail is capped at max_events so a 7-day window with 6 huge
            # backups still bounds memory.
            collected: list[dict] = []
            remaining = max_events
            for fp in files_to_read:
                if remaining <= 0:
                    break
                chunk = _read_tail(fp, remaining)
                collected = chunk + collected  # older file → prepended
                remaining = max_events - len(collected)
            events = [e for e in collected if int(e.get("ts", 0)) >= cutoff_ms]

            # Aggregations
            from collections import Counter
            rejected_by_gate: Counter = Counter()
            errors_by_component: Counter = Counter()
            errors_by_severity: Counter = Counter()
            guards_tripped: Counter = Counter()
            # Per-direction signal counters: SELL is never gated (close-out
            # always allowed) so it dominates the overall approval rate
            # and would otherwise mask BUY-gating health. Splitting BUY vs
            # SELL gives the dashboard an honest BUY-rate signal.
            approved = {"BUY": 0, "SELL": 0, "OTHER": 0}
            rejected = {"BUY": 0, "SELL": 0, "OTHER": 0}
            recent_errors: list[dict] = []
            recent_trips: list[dict] = []

            def _bucket(direction: Any) -> str:
                d = str(direction or "").upper()
                return d if d in ("BUY", "SELL") else "OTHER"

            for ev in events:
                t = ev.get("type")
                if t == "signal_rejected":
                    rejected[_bucket(ev.get("direction"))] += 1
                    rejected_by_gate[str(ev.get("gate") or "unknown")] += 1
                elif t == "signal_approved":
                    approved[_bucket(ev.get("direction"))] += 1
                elif t == "component_error":
                    component = str(ev.get("component") or "unknown")
                    severity = str(ev.get("severity") or "error")
                    errors_by_component[component] += 1
                    errors_by_severity[severity] += 1
                    recent_errors.append({
                        "ts": ev.get("ts"),
                        "component": component,
                        "severity": severity,
                        "exc_type": ev.get("exc_type", ""),
                        "reason": ev.get("reason", ""),
                        "suppressed_count": ev.get("suppressed_count", 0),
                    })
                elif t == "guard_tripped":
                    guard = str(ev.get("guard") or "unknown")
                    guards_tripped[guard] += 1
                    recent_trips.append({
                        "ts": ev.get("ts"),
                        "guard": guard,
                        "name": ev.get("name", ""),
                        "reason": ev.get("reason", ""),
                    })

            def _rate(a: int, r: int) -> Optional[float]:
                tot = a + r
                return round(a / tot, 4) if tot else None

            total_approved = sum(approved.values())
            total_rejected = sum(rejected.values())
            payload = {
                "window_hours": window_hours,
                "events_scanned": len(events),
                "events_path_exists": events_path.exists(),
                "files_read": len(files_to_read),
                "signals": {
                    "approved": total_approved,
                    "rejected": total_rejected,
                    # Headline overall rate kept for back-compat; meaningful
                    # only when SELL volume is small.
                    "approval_rate": _rate(total_approved, total_rejected),
                    # BUY-only rate is the honest gating-health metric
                    # (SELLs bypass entry gates and always approve, so an
                    # overall rate near 100% can hide BUY rejection storms).
                    "buy": {
                        "approved": approved["BUY"],
                        "rejected": rejected["BUY"],
                        "approval_rate": _rate(approved["BUY"], rejected["BUY"]),
                    },
                    "sell": {
                        "approved": approved["SELL"],
                        "rejected": rejected["SELL"],
                        "approval_rate": _rate(approved["SELL"], rejected["SELL"]),
                    },
                },
                "top_blocking_gates": [
                    {"gate": g, "count": c, "pct_of_rejections": round(c / total_rejected * 100, 1) if total_rejected else 0.0}
                    for g, c in rejected_by_gate.most_common(10)
                ],
                "errors_by_component": [
                    {"component": k, "count": v} for k, v in errors_by_component.most_common(10)
                ],
                "errors_by_severity": dict(errors_by_severity),
                "recent_component_errors": recent_errors[-25:][::-1],
                "guards_tripped": [
                    {"guard": g, "count": c} for g, c in guards_tripped.most_common(10)
                ],
                "recent_guards_tripped": recent_trips[-15:][::-1],
            }
            # Bound the cache size: a misbehaving client could call with
            # arbitrary (window_hours, max_events) tuples and balloon it.
            if len(_OBS_CACHE) > 64:
                # Drop the oldest half — cheap, no need for true LRU here.
                for k in sorted(_OBS_CACHE, key=lambda k: _OBS_CACHE[k][0])[:32]:
                    _OBS_CACHE.pop(k, None)
            _OBS_CACHE[cache_key] = (now_wall, payload)
            return JSONResponse(content=payload)

        @app.get("/api/observability/slo")
        async def observability_slo(window_hours: float = 24.0):
            """SLO / error-budget computation against documented targets.

            Targets are simple operator commitments — adjust by env vars
            so on-call can re-tune without code changes:

            - ``SENTINEL_SLO_BUY_APPROVAL_RATE`` (default 0.40) — fraction
              of BUY signals that must pass the risk pipeline. Below this,
              gating is too aggressive *or* upstream signals are bad.
            - ``SENTINEL_SLO_MAX_ERRORS_PER_HR`` (default 5.0) — error-rate
              ceiling for ``component_error`` events.
            - ``SENTINEL_SLO_MAX_CRITICAL`` (default 0) — critical events
              are always a budget breach.

            The endpoint reuses the same data the summary endpoint reads
            (events.jsonl tail) and computes burn-rate as
            ``actual / budget`` per dimension. burn>=1.0 means the budget
            is exhausted for the window.
            """
            import os as _os
            target_buy_rate = float(_os.environ.get("SENTINEL_SLO_BUY_APPROVAL_RATE", "0.40"))
            target_errors_per_hr = float(_os.environ.get("SENTINEL_SLO_MAX_ERRORS_PER_HR", "5.0"))
            target_max_critical = int(_os.environ.get("SENTINEL_SLO_MAX_CRITICAL", "0"))

            # Reuse the summary endpoint's machinery via direct call. This
            # benefits from the same cache (within TTL) so SLO refreshes
            # are cheap when the summary panel just refreshed.
            summary_resp = await observability_summary(window_hours=window_hours, max_events=2000)
            import json as _json
            summary = _json.loads(summary_resp.body.decode("utf-8"))

            buy = summary["signals"]["buy"]
            buy_rate = buy.get("approval_rate")
            buy_burn = None
            if buy_rate is not None and 0.0 < target_buy_rate < 1.0:
                # SRE-style burn rate: error_budget = 1 - target.
                # actual_bad = 1 - actual. burn = actual_bad / error_budget.
                # burn=1.0 means budget exactly consumed; >1.0 means breach.
                error_budget = 1.0 - target_buy_rate
                actual_bad = max(0.0, 1.0 - buy_rate)
                buy_burn = round(actual_bad / error_budget, 4) if error_budget > 0 else None

            errors_total = sum(summary["errors_by_severity"].get(k, 0) for k in ("warning", "error", "critical"))
            errors_per_hr = errors_total / max(window_hours, 0.001)
            error_burn = round(errors_per_hr / target_errors_per_hr, 4) if target_errors_per_hr > 0 else None

            critical = summary["errors_by_severity"].get("critical", 0)
            critical_burn = round(critical / max(target_max_critical, 1), 4) if critical > target_max_critical else 0.0

            slos = [
                {"name": "buy_approval_rate", "target": target_buy_rate,
                 "actual": buy_rate, "burn": buy_burn,
                 "status": "ok" if (buy_burn is None or buy_burn < 0.5) else ("warn" if buy_burn < 1.0 else "breach")},
                {"name": "errors_per_hour", "target": target_errors_per_hr,
                 "actual": round(errors_per_hr, 3), "burn": error_burn,
                 "status": "ok" if (error_burn is None or error_burn < 0.5) else ("warn" if error_burn < 1.0 else "breach")},
                {"name": "critical_events", "target": target_max_critical,
                 "actual": critical, "burn": critical_burn,
                 "status": "ok" if critical <= target_max_critical else "breach"},
            ]
            return JSONResponse(content={
                "window_hours": window_hours,
                "slos": slos,
                "any_breach": any(s["status"] == "breach" for s in slos),
            })

        @app.get("/api/observability/metrics")
        async def observability_metrics():
            """Prometheus text-format metrics for scraping.

            Exposes counters derived from a 15-minute window over
            ``events.jsonl`` so a Prometheus instance can scrape this
            endpoint at its usual 15-30s cadence and build dashboards /
            alerts in Grafana.

            We deliberately keep the metric set small — gate counts,
            error counts by component, severity histograms — because each
            metric line is a label cardinality dimension that pricing-
            sensitive Prometheus tiers will charge for.
            """
            summary_resp = await observability_summary(window_hours=0.25, max_events=5000)
            import json as _json
            summary = _json.loads(summary_resp.body.decode("utf-8"))

            lines: list[str] = []
            # HELP/TYPE headers per metric family; values follow.
            lines.append("# HELP sentinel_signals_total Risk pipeline decisions in the last 15 min.")
            lines.append("# TYPE sentinel_signals_total counter")
            for direction in ("BUY", "SELL"):
                key = direction.lower()
                lines.append(f'sentinel_signals_total{{direction="{direction}",outcome="approved"}} {summary["signals"][key]["approved"]}')
                lines.append(f'sentinel_signals_total{{direction="{direction}",outcome="rejected"}} {summary["signals"][key]["rejected"]}')

            lines.append("# HELP sentinel_gate_rejections_total Rejections by risk-gate.")
            lines.append("# TYPE sentinel_gate_rejections_total counter")
            for g in summary["top_blocking_gates"]:
                lines.append(f'sentinel_gate_rejections_total{{gate="{g["gate"]}"}} {g["count"]}')

            lines.append("# HELP sentinel_component_errors_total Component errors by name.")
            lines.append("# TYPE sentinel_component_errors_total counter")
            for c in summary["errors_by_component"]:
                lines.append(f'sentinel_component_errors_total{{component="{c["component"]}"}} {c["count"]}')

            lines.append("# HELP sentinel_component_errors_by_severity Error counts by severity.")
            lines.append("# TYPE sentinel_component_errors_by_severity counter")
            for sev, n in summary["errors_by_severity"].items():
                lines.append(f'sentinel_component_errors_by_severity{{severity="{sev}"}} {n}')

            lines.append("# HELP sentinel_guard_trips_total Guard activations by guard.")
            lines.append("# TYPE sentinel_guard_trips_total counter")
            for g in summary["guards_tripped"]:
                lines.append(f'sentinel_guard_trips_total{{guard="{g["guard"]}"}} {g["count"]}')

            from fastapi.responses import PlainTextResponse
            return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

        @app.get("/api/config")
        async def config_snapshot():
            return JSONResponse(content=self._build_config_payload())

        @app.get("/api/settings/editable")
        async def editable_settings_snapshot():
            return JSONResponse(content={
                "values": get_editable_settings_payload(self._settings),
                "restart_required": True,
            })

        @app.post("/api/settings/update")
        async def update_settings(request: Request):
            if not hasattr(self._settings, "model_dump"):
                return JSONResponse(
                    content={"error": "settings backend is not writable in this runtime"},
                    status_code=503,
                )

            try:
                payload = await request.json()
            except json.JSONDecodeError:
                return JSONResponse(content={"error": "invalid JSON payload"}, status_code=400)

            if not isinstance(payload, dict):
                return JSONResponse(content={"error": "payload must be an object"}, status_code=400)

            try:
                updated_settings = save_settings_updates(self._settings, payload)
            except Exception as exc:
                return JSONResponse(content={"error": str(exc)}, status_code=400)

            self._settings = updated_settings
            self._password = updated_settings.dashboard_password

            # Propagate runtime-safe settings (risk limits) without restart
            if self.on_settings_update:
                try:
                    self.on_settings_update(updated_settings)
                except Exception as cb_err:
                    logger.warning("on_settings_update callback failed: %s", cb_err)

            return JSONResponse(content={
                "result": "saved",
                "restart_required": False,
                "message": "Settings saved and applied.",
                "values": get_editable_settings_payload(self._settings),
            })

        # ── Control ──────────────────────────────

        def _client_ip(request: Request) -> str:
            """Extract the real client IP, honoring reverse-proxy headers
            so a shared nginx/cloudflare setup doesn't bucket all users
            together. Takes the leftmost entry of X-Forwarded-For since
            that's the originator (downstream proxies append to the right)."""
            xff = request.headers.get("x-forwarded-for", "")
            if xff:
                return xff.split(",")[0].strip()
            real = request.headers.get("x-real-ip", "")
            if real:
                return real.strip()
            return request.client.host if request.client else "unknown"

        def _check_rate_limit(request: Request, action: str):
            """Return a 429 JSONResponse when the caller exceeds the window,
            otherwise None. Keyed by client IP so shared-office setups still
            protect each other rather than globally locking on one user."""
            ok, retry_after = _CONTROL_LIMITER.allow((_client_ip(request), action))
            if ok:
                return None
            return JSONResponse(
                content={
                    "error": "Too many requests",
                    "retry_after_sec": round(retry_after, 1),
                },
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        @app.post("/api/control/stop")
        async def control_stop(request: Request):
            limited = _check_rate_limit(request, "stop")
            if limited is not None:
                return limited
            if self.on_stop:
                await self.on_stop()
                return {"result": "stopped"}
            return JSONResponse(content={"error": "stop handler not set"}, status_code=503)

        @app.post("/api/control/resume")
        async def control_resume(request: Request):
            limited = _check_rate_limit(request, "resume")
            if limited is not None:
                return limited
            if self.on_resume:
                await self.on_resume()
                return {"result": "resumed"}
            return JSONResponse(content={"error": "resume handler not set"}, status_code=503)

        @app.post("/api/control/kill")
        async def control_kill(request: Request):
            limited = _check_rate_limit(request, "kill")
            if limited is not None:
                return limited
            if self.on_kill:
                await self.on_kill()
                return {"result": "killed"}
            return JSONResponse(content={"error": "kill handler not set"}, status_code=503)

        @app.post("/api/positions/{symbol}/close")
        async def control_close_position(symbol: str, request: Request):
            limited = _check_rate_limit(request, "close")
            if limited is not None:
                return limited
            if not self.on_manual_close:
                return JSONResponse(
                    content={"error": "manual close handler not set"},
                    status_code=503,
                )
            symbol = (symbol or "").upper().strip()
            if not symbol:
                return JSONResponse(content={"error": "symbol required"}, status_code=400)
            try:
                result = await self.on_manual_close(symbol)
            except Exception as exc:
                logger.error("manual close %s failed: %s", symbol, exc)
                return JSONResponse(content={"error": str(exc)}, status_code=500)
            if isinstance(result, dict) and not result.get("ok", False):
                return JSONResponse(content=result, status_code=404)
            return JSONResponse(content=result if isinstance(result, dict) else {"ok": True})

        # ── WebSocket ────────────────────────────

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            # Auth check: prefer cookie (auto-sent on WS upgrade, doesn't leak to logs),
            # fall back to query param for legacy clients.
            if self._password:
                token = (websocket.cookies.get("sentinel_auth")
                         or websocket.query_params.get("token", "")
                         or "")
                if not secrets.compare_digest(token, self._password):
                    await websocket.close(code=4001, reason="Unauthorized")
                    return
            await websocket.accept()
            self._ws_clients.append(websocket)
            try:
                while True:
                    state = self._get_state()
                    await websocket.send_json({
                        "type": "state_update",
                        "data": {
                            "mode": state.get("mode", "paper"),
                            "risk_state": state.get("risk_state", "NORMAL"),
                            "trading_paused": state.get("trading_paused", False),
                            "uptime": state.get("uptime", _format_uptime()),
                            "pnl_today": state.get("pnl_today", 0.0),
                            "pnl_total": state.get("pnl_total", 0.0),
                            "open_positions": state.get("open_positions", 0),
                            "trades_today": state.get("trades_today", 0),
                            "balance": state.get("balance", 0.0),
                            "win_rate": state.get("win_rate", 0.0),
                            "profit_factor": state.get("profit_factor", 0.0),
                            "avg_rr_ratio": state.get("avg_rr_ratio", 0.0),
                            "max_drawdown_pct": state.get("max_drawdown_pct", 0.0),
                            "current_drawdown_pct": state.get("current_drawdown_pct", 0.0),
                            "peak_balance": state.get("peak_balance", 0.0),
                            "total_wins": state.get("total_wins", 0),
                            "total_losses": state.get("total_losses", 0),
                            "risk_details": state.get("risk_details", {}),
                            "activity": state.get("activity", {}),
                            "indicators": state.get("indicators", {}),
                            "indicators_per_symbol": state.get("indicators_per_symbol", {}),
                            "trading_symbols": state.get("trading_symbols", []),
                            "win_rate_per_symbol": state.get("win_rate_per_symbol", {}),
                            "readiness": state.get("readiness", {}),
                            "strategy_log": state.get("strategy_log", []),
                            "ml_status": state.get("ml_status", {}),
                            "standing_ml_per_symbol": state.get("standing_ml_per_symbol", {}),
                            "last_cycle_ts_per_symbol": state.get("last_cycle_ts_per_symbol", {}),
                        },
                    })
                    await asyncio.sleep(2)
            except WebSocketDisconnect:
                logger.debug("Dashboard websocket client disconnected")
            finally:
                if websocket in self._ws_clients:
                    self._ws_clients.remove(websocket)

        # ── HTML Dashboard ───────────────────────

        @app.get("/login")
        async def login_page(request: Request):
            from starlette.responses import RedirectResponse
            # Already authenticated? Skip the form and go straight to the dashboard.
            if dashboard_password:
                tok = request.cookies.get("sentinel_auth", "")
                if tok and secrets.compare_digest(tok, dashboard_password):
                    return RedirectResponse(url="/", status_code=303)
            return HTMLResponse(_LOGIN_HTML)

        @app.get("/", response_class=HTMLResponse)
        async def dashboard_page():
            return _DASHBOARD_HTML

        @app.get("/settings", response_class=HTMLResponse)
        async def settings_page():
            return _SETTINGS_HTML

        @app.get("/trades", response_class=HTMLResponse)
        async def trades_page():
            return _TRADES_HTML

        @app.get("/analytics", response_class=HTMLResponse)
        async def analytics_page():
            return _ANALYTICS_HTML

        @app.get("/news", response_class=HTMLResponse)
        async def news_page():
            return _NEWS_HTML

        @app.get("/logs", response_class=HTMLResponse)
        async def logs_page():
            return _LOGS_HTML

        @app.get("/ml-robustness", response_class=HTMLResponse)
        async def ml_robustness_page():
            return _ML_ROBUSTNESS_HTML

        @app.get("/observability", response_class=HTMLResponse)
        async def observability_page():
            return _OBSERVABILITY_HTML

        self._app = app
        return app

    # ──────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────

    async def start(self) -> None:
        """Запуск dashboard."""
        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn not installed. Run: pip install uvicorn")
            return

        app = self._create_app()
        bind_ip = getattr(self._settings, "dashboard_bind_ip", "127.0.0.1")
        if bind_ip == "0.0.0.0" and not self._password:
            logger.warning(
                "Dashboard bound to 0.0.0.0 WITHOUT password — anyone on the "
                "network can control the bot. Set DASHBOARD_PASSWORD or "
                "change DASHBOARD_BIND_IP to 127.0.0.1."
            )
        config = uvicorn.Config(
            app,
            host=bind_ip,
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        logger.info("Dashboard starting on http://localhost:%d", self._port)
        asyncio.create_task(self._server.serve())

    async def stop(self) -> None:
        """Остановка dashboard."""
        if self._server:
            self._server.should_exit = True
            logger.info("Dashboard stopped")

    # ──────────────────────────────────────────────
    # State
    # ──────────────────────────────────────────────

    def _get_state(self) -> dict:
        if self._state_provider:
            try:
                result = self._state_provider()
                return result if isinstance(result, dict) else {}
            except Exception as exc:
                logger.error("State provider error: %s", exc)
        return {}

    # ──────────────────────────────────────────────
    # Broadcast to WS clients
    # ──────────────────────────────────────────────

    async def broadcast(self, event_type: str, data: dict) -> None:
        """Отправить событие всем WS-клиентам."""
        dead = []
        for ws in self._ws_clients:
            try:
                await ws.send_json({"type": event_type, "data": data})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.remove(ws)



# Load HTML from static/index.html
import pathlib as _pathlib
_STATIC_DIR = _pathlib.Path(__file__).parent / "static"
_DASHBOARD_HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8") if (_STATIC_DIR / "index.html").exists() else "<h1>Dashboard HTML not found</h1>"
_SETTINGS_HTML = (_STATIC_DIR / "settings.html").read_text(encoding="utf-8") if (_STATIC_DIR / "settings.html").exists() else "<h1>Settings HTML not found</h1>"
_TRADES_HTML = (_STATIC_DIR / "trades.html").read_text(encoding="utf-8") if (_STATIC_DIR / "trades.html").exists() else "<h1>Trades HTML not found</h1>"
_ANALYTICS_HTML = (_STATIC_DIR / "analytics.html").read_text(encoding="utf-8") if (_STATIC_DIR / "analytics.html").exists() else "<h1>Analytics HTML not found</h1>"
_NEWS_HTML = (_STATIC_DIR / "news.html").read_text(encoding="utf-8") if (_STATIC_DIR / "news.html").exists() else "<h1>News HTML not found</h1>"
_LOGS_HTML = (_STATIC_DIR / "logs.html").read_text(encoding="utf-8") if (_STATIC_DIR / "logs.html").exists() else "<h1>Logs HTML not found</h1>"
_LOGIN_HTML = (_STATIC_DIR / "login.html").read_text(encoding="utf-8") if (_STATIC_DIR / "login.html").exists() else "<h1>Login HTML not found</h1>"
_ML_ROBUSTNESS_HTML = (_STATIC_DIR / "ml-robustness.html").read_text(encoding="utf-8") if (_STATIC_DIR / "ml-robustness.html").exists() else "<h1>ML Robustness HTML not found</h1>"
_OBSERVABILITY_HTML = (_STATIC_DIR / "observability.html").read_text(encoding="utf-8") if (_STATIC_DIR / "observability.html").exists() else "<h1>Observability HTML not found</h1>"
