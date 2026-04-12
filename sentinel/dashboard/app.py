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
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

from core.constants import VERSION
from core.events import EventBus

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger(__name__)

_START_TIME = time.time()


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

    def _create_app(self):
        """Создать и настроить FastAPI-приложение."""
        from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        import pathlib

        app = FastAPI(title="SENTINEL Dashboard", version=VERSION)

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

        @app.get("/api/status")
        async def status():
            state = self._get_state()
            return JSONResponse(content={
                "mode": state.get("mode", "paper"),
                "risk_state": state.get("risk_state", "NORMAL"),
                "uptime": state.get("uptime", _format_uptime()),
                "pnl_today": state.get("pnl_today", 0.0),
                "pnl_total": state.get("pnl_total", 0.0),
                "open_positions": state.get("open_positions", 0),
                "trades_today": state.get("trades_today", 0),
                "balance": state.get("balance", 0.0),
                "win_rate": state.get("win_rate", 0.0),
                "version": VERSION,
            })

        @app.get("/api/positions")
        async def positions():
            state = self._get_state()
            pos_list = state.get("positions", [])
            result = []
            for p in pos_list:
                if hasattr(p, "symbol"):
                    result.append({
                        "symbol": p.symbol,
                        "side": p.side,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "quantity": p.quantity,
                        "unrealized_pnl": p.unrealized_pnl,
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

        @app.get("/api/backtest-results")
        async def backtest_results():
            state = self._get_state()
            return JSONResponse(content=state.get("backtest_results", {}))

        # ── Control ──────────────────────────────

        @app.post("/api/control/stop")
        async def control_stop():
            if self.on_stop:
                await self.on_stop()
                return {"result": "stopped"}
            return JSONResponse(content={"error": "stop handler not set"}, status_code=503)

        @app.post("/api/control/resume")
        async def control_resume():
            if self.on_resume:
                await self.on_resume()
                return {"result": "resumed"}
            return JSONResponse(content={"error": "resume handler not set"}, status_code=503)

        @app.post("/api/control/kill")
        async def control_kill():
            if self.on_kill:
                await self.on_kill()
                return {"result": "killed"}
            return JSONResponse(content={"error": "kill handler not set"}, status_code=503)

        # ── WebSocket ────────────────────────────

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
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
                            "uptime": state.get("uptime", _format_uptime()),
                            "pnl_today": state.get("pnl_today", 0.0),
                            "pnl_total": state.get("pnl_total", 0.0),
                            "open_positions": state.get("open_positions", 0),
                            "trades_today": state.get("trades_today", 0),
                            "balance": state.get("balance", 0.0),
                            "win_rate": state.get("win_rate", 0.0),
                        },
                    })
                    await asyncio.sleep(5)
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self._ws_clients:
                    self._ws_clients.remove(websocket)

        # ── HTML Dashboard ───────────────────────

        @app.get("/", response_class=HTMLResponse)
        async def dashboard_page():
            return _DASHBOARD_HTML

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
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
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


# ══════════════════════════════════════════════════
# HTML Template — Professional Fintech Dashboard
# UI/UX: Inter font, SVG icons, accessibility,
# prefers-reduced-motion, 4.5:1 contrast, 44px
# touch targets, skeleton loading, responsive grid
# ══════════════════════════════════════════════════

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENTINEL Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        /* ── Reset & Base ──────────────────────── */
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
        html { font-size: 16px; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: #0b0e11;
            color: #e8eaed;
            min-height: 100vh;
            line-height: 1.6;
        }
        .mono { font-family: 'JetBrains Mono', 'Cascadia Code', monospace; }

        /* ── Z-index scale ─────────────────────── */
        :root {
            --z-base: 0;
            --z-dropdown: 10;
            --z-sticky: 20;
            --z-overlay: 30;
            --z-modal: 50;
            /* Colors */
            --bg-primary: #0b0e11;
            --bg-card: #141821;
            --bg-elevated: #1c2333;
            --bg-hover: #222b3a;
            --border: #2a3040;
            --border-light: #343e52;
            --text-primary: #e8eaed;
            --text-secondary: #8b95a5;
            --text-muted: #5a6577;
            --accent: #3b82f6;
            --accent-dim: rgba(59, 130, 246, 0.15);
            --green: #10b981;
            --green-dim: rgba(16, 185, 129, 0.12);
            --red: #ef4444;
            --red-dim: rgba(239, 68, 68, 0.12);
            --amber: #f59e0b;
            --amber-dim: rgba(245, 158, 11, 0.12);
            --cyan: #06b6d4;
        }

        /* ── Animations ────────────────────────── */
        @media (prefers-reduced-motion: no-preference) {
            .animate-fade { animation: fadeIn 0.3s ease-out; }
            .animate-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }

        /* ── Skeleton loader ───────────────────── */
        .skeleton {
            background: linear-gradient(90deg, var(--bg-elevated) 25%, var(--bg-hover) 50%, var(--bg-elevated) 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s ease-in-out infinite;
            border-radius: 6px;
        }
        .skeleton-text { height: 14px; width: 60%; margin-bottom: 8px; }
        .skeleton-value { height: 32px; width: 40%; }

        /* ── Header ────────────────────────────── */
        .header {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 0 24px;
            height: 60px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: var(--z-sticky);
        }
        .header-left { display: flex; align-items: center; gap: 12px; }
        .header-logo {
            display: flex; align-items: center; gap: 8px;
            font-size: 18px; font-weight: 700; color: var(--accent);
            letter-spacing: -0.02em;
        }
        .header-logo svg { flex-shrink: 0; }
        .header-version {
            font-size: 11px; font-weight: 500; color: var(--text-muted);
            padding: 2px 8px; border: 1px solid var(--border);
            border-radius: 4px; letter-spacing: 0.02em;
        }
        .header-right { display: flex; align-items: center; gap: 10px; }
        .header-uptime {
            display: flex; align-items: center; gap: 6px;
            font-size: 13px; color: var(--text-secondary);
        }
        .header-uptime svg { color: var(--text-muted); }

        /* ── Badges ─────────────────────────────── */
        .badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 12px; border-radius: 6px;
            font-size: 12px; font-weight: 600;
            letter-spacing: 0.03em; text-transform: uppercase;
        }
        .badge-dot {
            width: 6px; height: 6px; border-radius: 50%;
            display: inline-block; flex-shrink: 0;
        }
        .badge-paper { background: var(--accent-dim); color: var(--accent); }
        .badge-paper .badge-dot { background: var(--accent); }
        .badge-live { background: var(--green-dim); color: var(--green); }
        .badge-live .badge-dot { background: var(--green); }
        .badge-normal { background: var(--green-dim); color: var(--green); }
        .badge-normal .badge-dot { background: var(--green); }
        .badge-reduced { background: var(--amber-dim); color: var(--amber); }
        .badge-reduced .badge-dot { background: var(--amber); }
        .badge-safe { background: rgba(249,115,22,0.12); color: #f97316; }
        .badge-safe .badge-dot { background: #f97316; }
        .badge-stop { background: var(--red-dim); color: var(--red); }
        .badge-stop .badge-dot { background: var(--red); }

        /* ── Container & Layout ────────────────── */
        .container { max-width: 1280px; margin: 0 auto; padding: 20px 24px; }

        .grid-6 {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 14px;
            margin-bottom: 20px;
        }

        /* ── Cards ─────────────────────────────── */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .card:hover { border-color: var(--border-light); }
        .card-header {
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 10px;
        }
        .card-icon {
            width: 36px; height: 36px;
            display: flex; align-items: center; justify-content: center;
            border-radius: 8px; flex-shrink: 0;
        }
        .card-icon svg { width: 18px; height: 18px; }
        .card-icon-blue { background: var(--accent-dim); color: var(--accent); }
        .card-icon-green { background: var(--green-dim); color: var(--green); }
        .card-icon-red { background: var(--red-dim); color: var(--red); }
        .card-icon-amber { background: var(--amber-dim); color: var(--amber); }
        .card-icon-cyan { background: rgba(6,182,212,0.12); color: var(--cyan); }
        .card-icon-purple { background: rgba(139,92,246,0.12); color: #8b5cf6; }
        .card-label {
            font-size: 12px; font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .card-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 26px; font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        .card-sub {
            font-size: 12px; color: var(--text-muted);
            margin-top: 4px;
        }
        .positive { color: var(--green); }
        .negative { color: var(--red); }
        .neutral { color: var(--text-primary); }

        /* ── Chart section ─────────────────────── */
        .chart-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .section-title {
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 16px;
        }
        .section-title h3 {
            font-size: 14px; font-weight: 600; color: var(--text-primary);
            display: flex; align-items: center; gap: 8px;
        }
        .section-title h3 svg { color: var(--text-muted); width: 16px; height: 16px; }
        .chart-wrap { height: 260px; position: relative; }

        /* ── Risk Panel ────────────────────────── */
        .risk-panel {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .risk-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }
        .risk-item {
            background: var(--bg-elevated);
            border-radius: 8px;
            padding: 14px;
            text-align: center;
        }
        .risk-item-label {
            font-size: 11px; font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .risk-item-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px; font-weight: 600;
        }

        /* ── Tables ────────────────────────────── */
        .table-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        .table-section .section-title { padding: 18px 20px 0; }
        table { width: 100%; border-collapse: collapse; }
        thead th {
            padding: 12px 16px;
            text-align: left;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border-bottom: 1px solid var(--border);
            background: var(--bg-elevated);
        }
        tbody td {
            padding: 12px 16px;
            font-size: 13px;
            color: var(--text-secondary);
            border-bottom: 1px solid rgba(42, 48, 64, 0.5);
        }
        tbody tr { transition: background-color 0.15s ease; }
        tbody tr:hover { background: var(--bg-hover); }
        tbody tr:last-child td { border-bottom: none; }
        .td-mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
        .empty-state {
            padding: 40px 16px;
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
        }
        .empty-state svg { margin-bottom: 8px; opacity: 0.4; }

        /* ── Controls ──────────────────────────── */
        .controls-section {
            display: flex; align-items: center; gap: 12px;
            flex-wrap: wrap;
        }
        .btn {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 0 20px; height: 44px;
            border: none; border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 13px; font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.1s ease, opacity 0.2s ease;
            white-space: nowrap;
            letter-spacing: 0.01em;
        }
        .btn:focus-visible {
            outline: 2px solid var(--accent);
            outline-offset: 2px;
        }
        .btn:active { transform: scale(0.97); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; pointer-events: none; }
        .btn svg { width: 16px; height: 16px; flex-shrink: 0; }
        .btn-start { background: var(--green); color: #fff; }
        .btn-start:hover { background: #059669; }
        .btn-stop { background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border); }
        .btn-stop:hover { background: var(--bg-hover); border-color: var(--border-light); }
        .btn-kill { background: var(--red-dim); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }
        .btn-kill:hover { background: rgba(239,68,68,0.2); }
        .controls-spacer { flex: 1; }

        /* ── Connection indicator ───────────────── */
        .conn-indicator {
            display: flex; align-items: center; gap: 6px;
            font-size: 12px; color: var(--text-muted);
        }
        .conn-dot {
            width: 8px; height: 8px; border-radius: 50%;
            transition: background-color 0.3s ease;
        }
        .conn-dot.connected { background: var(--green); box-shadow: 0 0 6px rgba(16,185,129,0.4); }
        .conn-dot.disconnected { background: var(--red); }

        /* ── Toast ─────────────────────────────── */
        .toast {
            position: fixed; bottom: 24px; right: 24px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px 20px;
            display: flex; align-items: center; gap: 10px;
            font-size: 13px; color: var(--text-primary);
            z-index: var(--z-modal);
            transform: translateY(100px); opacity: 0;
            transition: transform 0.3s ease, opacity 0.3s ease;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        .toast.show { transform: translateY(0); opacity: 1; }
        .toast-success { border-left: 3px solid var(--green); }
        .toast-error { border-left: 3px solid var(--red); }
        .toast-warning { border-left: 3px solid var(--amber); }

        /* ── Responsive ────────────────────────── */
        @media (max-width: 1024px) {
            .grid-6 { grid-template-columns: repeat(3, 1fr); }
            .risk-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 768px) {
            .header { padding: 0 16px; }
            .container { padding: 16px; }
            .grid-6 { grid-template-columns: repeat(2, 1fr); gap: 10px; }
            .risk-grid { grid-template-columns: repeat(2, 1fr); }
            .card { padding: 14px; }
            .card-value { font-size: 22px; }
            .controls-section { justify-content: stretch; }
            .controls-section .btn { flex: 1; justify-content: center; }
        }
        @media (max-width: 480px) {
            .grid-6 { grid-template-columns: 1fr; }
            .risk-grid { grid-template-columns: 1fr; }
            .header-version { display: none; }
        }
    </style>
</head>
<body>
    <!-- ── Header ──────────────────────────────── -->
    <header class="header" role="banner">
        <div class="header-left">
            <div class="header-logo">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                SENTINEL
            </div>
            <span class="header-version" id="version-label">v1.5.0</span>
        </div>
        <div class="header-right">
            <div class="header-uptime" aria-label="Uptime">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                </svg>
                <span id="uptime-label">0m 0s</span>
            </div>
            <span id="mode-badge" class="badge badge-paper"><span class="badge-dot"></span>PAPER</span>
            <span id="state-badge" class="badge badge-normal"><span class="badge-dot"></span>NORMAL</span>
            <div class="conn-indicator" aria-label="WebSocket connection">
                <span class="conn-dot disconnected" id="conn-dot"></span>
                <span id="conn-label">Offline</span>
            </div>
        </div>
    </header>

    <!-- ── Main Content ────────────────────────── -->
    <main class="container" role="main">

        <!-- Metric Cards -->
        <div class="grid-6 animate-fade">
            <!-- PnL Today -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon card-icon-green">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
                    </div>
                    <span class="card-label">PnL Сегодня</span>
                </div>
                <div class="card-value neutral" id="pnl-today">$0.00</div>
            </div>

            <!-- PnL Total -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon card-icon-blue">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                    </div>
                    <span class="card-label">PnL Всего</span>
                </div>
                <div class="card-value neutral" id="pnl-total">$0.00</div>
            </div>

            <!-- Balance -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon card-icon-cyan">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/></svg>
                    </div>
                    <span class="card-label">Баланс</span>
                </div>
                <div class="card-value neutral" id="balance">$0.00</div>
            </div>

            <!-- Open Positions -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon card-icon-amber">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                    </div>
                    <span class="card-label">Позиции</span>
                </div>
                <div class="card-value neutral" id="open-positions">0</div>
            </div>

            <!-- Trades Today -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon card-icon-purple">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                    </div>
                    <span class="card-label">Сделок сегодня</span>
                </div>
                <div class="card-value neutral" id="trades-today">0</div>
            </div>

            <!-- Win Rate -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon card-icon-green">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                    </div>
                    <span class="card-label">Win Rate</span>
                </div>
                <div class="card-value neutral" id="win-rate">0%</div>
            </div>
        </div>

        <!-- PnL Chart -->
        <div class="chart-section animate-fade">
            <div class="section-title">
                <h3>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                    PnL History
                </h3>
            </div>
            <div class="chart-wrap">
                <canvas id="pnlChart" aria-label="PnL history chart" role="img"></canvas>
            </div>
        </div>

        <!-- Risk State Panel -->
        <div class="risk-panel animate-fade">
            <div class="section-title">
                <h3>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                    Risk Overview
                </h3>
                <span id="risk-state-big" class="badge badge-normal"><span class="badge-dot"></span>NORMAL</span>
            </div>
            <div class="risk-grid" id="risk-grid">
                <div class="risk-item">
                    <div class="risk-item-label">Daily Loss</div>
                    <div class="risk-item-value neutral" id="risk-daily-loss">$0.00</div>
                </div>
                <div class="risk-item">
                    <div class="risk-item-label">Max Drawdown</div>
                    <div class="risk-item-value neutral" id="risk-max-dd">0.0%</div>
                </div>
                <div class="risk-item">
                    <div class="risk-item-label">Exposure</div>
                    <div class="risk-item-value neutral" id="risk-exposure">0.0%</div>
                </div>
                <div class="risk-item">
                    <div class="risk-item-label">Trade Freq</div>
                    <div class="risk-item-value neutral" id="risk-freq">0/h</div>
                </div>
            </div>
        </div>

        <!-- Positions Table -->
        <div class="table-section animate-fade">
            <div class="section-title">
                <h3>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                    Open Positions
                </h3>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>Qty</th>
                        <th>PnL</th>
                    </tr>
                </thead>
                <tbody id="positions-table">
                    <tr><td colspan="6" class="empty-state">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/></svg>
                        <div>No open positions</div>
                    </td></tr>
                </tbody>
            </table>
        </div>

        <!-- Trades Table -->
        <div class="table-section animate-fade">
            <div class="section-title">
                <h3>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                    Recent Trades
                </h3>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Price</th>
                        <th>PnL</th>
                    </tr>
                </thead>
                <tbody id="trades-table">
                    <tr><td colspan="5" class="empty-state">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/></svg>
                        <div>No trades yet</div>
                    </td></tr>
                </tbody>
            </table>
        </div>

        <!-- Controls -->
        <div class="controls-section animate-fade">
            <button class="btn btn-start" id="btn-start" onclick="controlAction('resume')" aria-label="Start trading">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                Start
            </button>
            <button class="btn btn-stop" id="btn-stop" onclick="controlAction('stop')" aria-label="Stop trading">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                Stop
            </button>
            <div class="controls-spacer"></div>
            <button class="btn btn-kill" id="btn-kill" onclick="confirmKill()" aria-label="Emergency stop">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                Emergency Stop
            </button>
        </div>
    </main>

    <!-- Toast container -->
    <div class="toast" id="toast" role="alert" aria-live="assertive"></div>

    <script>
    'use strict';

    /* ── Helpers ──────────────────────────────── */
    function formatPnl(val) {
        if (val >= 0) return '+$' + val.toFixed(2);
        return '-$' + Math.abs(val).toFixed(2);
    }
    function formatUsd(val) { return '$' + Number(val).toFixed(2); }
    function pnlClass(val) { return val > 0 ? 'positive' : val < 0 ? 'negative' : 'neutral'; }
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    /* ── Toast notifications ─────────────────── */
    let toastTimer = null;
    function showToast(message, type) {
        type = type || 'success';
        const t = document.getElementById('toast');
        t.textContent = message;
        t.className = 'toast toast-' + type + ' show';
        clearTimeout(toastTimer);
        toastTimer = setTimeout(function() { t.className = 'toast'; }, 3500);
    }

    /* ── Chart setup ─────────────────────────── */
    const ctx = document.getElementById('pnlChart').getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 260);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.25)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

    const pnlChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'PnL ($)',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: gradient,
                fill: true,
                tension: 0.35,
                borderWidth: 2,
                pointRadius: 0,
                pointHitRadius: 10,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: '#3b82f6',
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1c2333',
                    titleColor: '#8b95a5',
                    bodyColor: '#e8eaed',
                    borderColor: '#2a3040',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false,
                    titleFont: { family: 'Inter', size: 11 },
                    bodyFont: { family: 'JetBrains Mono', size: 13, weight: '600' },
                    callbacks: {
                        label: function(ctx) { return formatPnl(ctx.parsed.y); }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(42,48,64,0.5)', drawBorder: false },
                    ticks: { color: '#5a6577', font: { family: 'Inter', size: 11 }, maxRotation: 0 },
                    border: { display: false }
                },
                y: {
                    grid: { color: 'rgba(42,48,64,0.5)', drawBorder: false },
                    ticks: {
                        color: '#5a6577',
                        font: { family: 'JetBrains Mono', size: 11 },
                        callback: function(v) { return '$' + v; }
                    },
                    border: { display: false }
                }
            }
        }
    });

    /* ── State update ────────────────────────── */
    function updateUI(data) {
        /* PnL Today */
        const pt = document.getElementById('pnl-today');
        pt.textContent = formatPnl(data.pnl_today || 0);
        pt.className = 'card-value ' + pnlClass(data.pnl_today || 0);

        /* PnL Total */
        const pp = document.getElementById('pnl-total');
        pp.textContent = formatPnl(data.pnl_total || 0);
        pp.className = 'card-value ' + pnlClass(data.pnl_total || 0);

        /* Balance */
        document.getElementById('balance').textContent = formatUsd(data.balance || 0);

        /* Counters */
        document.getElementById('open-positions').textContent = data.open_positions || 0;
        document.getElementById('trades-today').textContent = data.trades_today || 0;

        /* Win Rate */
        const wr = data.win_rate || 0;
        const wrEl = document.getElementById('win-rate');
        wrEl.textContent = wr.toFixed(1) + '%';
        wrEl.className = 'card-value ' + (wr >= 50 ? 'positive' : wr > 0 ? 'negative' : 'neutral');

        /* Uptime */
        if (data.uptime) document.getElementById('uptime-label').textContent = data.uptime;

        /* Mode badge */
        const mb = document.getElementById('mode-badge');
        const mode = (data.mode || 'paper').toLowerCase();
        mb.innerHTML = '<span class="badge-dot"></span>' + escapeHtml(mode.toUpperCase());
        mb.className = 'badge badge-' + mode;

        /* Risk state badge (header) */
        const sb = document.getElementById('state-badge');
        const rs = (data.risk_state || 'NORMAL').toLowerCase();
        sb.innerHTML = '<span class="badge-dot"></span>' + escapeHtml((data.risk_state || 'NORMAL').toUpperCase());
        sb.className = 'badge badge-' + rs;

        /* Risk state badge (panel) */
        const rb = document.getElementById('risk-state-big');
        rb.innerHTML = '<span class="badge-dot"></span>' + escapeHtml((data.risk_state || 'NORMAL').toUpperCase());
        rb.className = 'badge badge-' + rs;

        /* Risk details */
        if (data.risk_details) {
            const rd = data.risk_details;
            const dlEl = document.getElementById('risk-daily-loss');
            dlEl.textContent = formatPnl(rd.daily_loss || 0);
            dlEl.className = 'risk-item-value ' + pnlClass(-(Math.abs(rd.daily_loss || 0)));

            document.getElementById('risk-max-dd').textContent = ((rd.max_drawdown || 0) * 100).toFixed(1) + '%';
            document.getElementById('risk-exposure').textContent = ((rd.exposure || 0) * 100).toFixed(1) + '%';
            document.getElementById('risk-freq').textContent = (rd.trade_freq || 0) + '/h';
        }
    }

    /* ── WebSocket ────────────────────────────── */
    let ws = null;
    let wsRetry = 0;
    const MAX_RETRY_DELAY = 30000;

    function connectWS() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(proto + '//' + location.host + '/ws');

        ws.onopen = function() {
            wsRetry = 0;
            document.getElementById('conn-dot').className = 'conn-dot connected';
            document.getElementById('conn-label').textContent = 'Live';
        };

        ws.onmessage = function(e) {
            try {
                const msg = JSON.parse(e.data);
                if (msg.type === 'state_update') updateUI(msg.data);
            } catch(err) {}
        };

        ws.onclose = function() {
            document.getElementById('conn-dot').className = 'conn-dot disconnected';
            document.getElementById('conn-label').textContent = 'Offline';
            var delay = Math.min(1000 * Math.pow(2, wsRetry), MAX_RETRY_DELAY);
            wsRetry++;
            setTimeout(connectWS, delay);
        };

        ws.onerror = function() { ws.close(); };
    }
    connectWS();

    /* ── REST Polling (fallback) ─────────────── */
    async function fetchStatus() {
        try {
            const r = await fetch('/api/status');
            const data = await r.json();
            updateUI(data);
        } catch(e) {}
    }

    async function fetchPositions() {
        try {
            const r = await fetch('/api/positions');
            const data = await r.json();
            const tbody = document.getElementById('positions-table');
            if (!data.length) {
                tbody.innerHTML = '<tr><td colspan="6" class="empty-state"><div>No open positions</div></td></tr>';
                return;
            }
            tbody.innerHTML = data.map(function(p) {
                const pnl = p.unrealized_pnl || 0;
                return '<tr>' +
                    '<td class="td-mono" style="color:var(--text-primary);font-weight:500;">' + escapeHtml(p.symbol) + '</td>' +
                    '<td><span class="badge ' + (p.side === 'BUY' ? 'badge-live' : 'badge-stop') + '" style="font-size:11px;padding:2px 8px;">' + escapeHtml(p.side) + '</span></td>' +
                    '<td class="td-mono">' + formatUsd(p.entry_price || 0) + '</td>' +
                    '<td class="td-mono">' + formatUsd(p.current_price || 0) + '</td>' +
                    '<td class="td-mono">' + Number(p.quantity || 0).toFixed(6) + '</td>' +
                    '<td class="td-mono ' + pnlClass(pnl) + '">' + formatPnl(pnl) + '</td></tr>';
            }).join('');
        } catch(e) {}
    }

    async function fetchTrades() {
        try {
            const r = await fetch('/api/trades');
            const data = await r.json();
            const tbody = document.getElementById('trades-table');
            if (!data.length) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state"><div>No trades yet</div></td></tr>';
                return;
            }
            tbody.innerHTML = data.slice(0, 15).map(function(t) {
                const pnl = t.pnl || 0;
                return '<tr>' +
                    '<td class="td-mono">' + escapeHtml(t.time || '-') + '</td>' +
                    '<td class="td-mono" style="color:var(--text-primary);">' + escapeHtml(t.symbol || '-') + '</td>' +
                    '<td><span class="badge ' + (t.side === 'BUY' ? 'badge-live' : 'badge-stop') + '" style="font-size:11px;padding:2px 8px;">' + escapeHtml(t.side || '-') + '</span></td>' +
                    '<td class="td-mono">' + formatUsd(t.price || 0) + '</td>' +
                    '<td class="td-mono ' + pnlClass(pnl) + '">' + (t.pnl != null ? formatPnl(pnl) : '-') + '</td></tr>';
            }).join('');
        } catch(e) {}
    }

    async function fetchPnlHistory() {
        try {
            const r = await fetch('/api/pnl-history');
            const data = await r.json();
            if (!data.length) return;
            pnlChart.data.labels = data.map(function(d) { return d.date || d.label || ''; });
            pnlChart.data.datasets[0].data = data.map(function(d) { return d.pnl || d.value || 0; });
            pnlChart.update('none');
        } catch(e) {}
    }

    /* ── Control actions ─────────────────────── */
    async function controlAction(action) {
        const btn = document.getElementById('btn-' + (action === 'resume' ? 'start' : action));
        if (btn) btn.disabled = true;
        try {
            const r = await fetch('/api/control/' + action, { method: 'POST' });
            const data = await r.json();
            if (r.ok) {
                showToast(action === 'resume' ? 'Trading started' : action === 'stop' ? 'Trading stopped' : 'Emergency stop executed',
                    action === 'kill' ? 'warning' : 'success');
            } else {
                showToast(data.error || 'Action failed', 'error');
            }
            fetchStatus();
        } catch(e) {
            showToast('Connection error: ' + e.message, 'error');
        } finally {
            if (btn) setTimeout(function() { btn.disabled = false; }, 2000);
        }
    }

    function confirmKill() {
        if (confirm('EMERGENCY STOP\n\nThis will immediately cancel all orders and close all positions.\n\nAre you sure?')) {
            controlAction('kill');
        }
    }

    /* ── Init & intervals ────────────────────── */
    fetchStatus();
    fetchPositions();
    fetchTrades();
    fetchPnlHistory();
    setInterval(fetchStatus, 5000);
    setInterval(fetchPositions, 8000);
    setInterval(fetchTrades, 8000);
    setInterval(fetchPnlHistory, 30000);
    </script>
</body>
</html>
"""
