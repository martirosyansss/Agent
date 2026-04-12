"""
Web Dashboard SENTINEL — FastAPI + HTML/JS.

Предоставляет:
- REST API для данных (статус, PnL, позиции, сделки)
- WebSocket для real-time обновлений
- HTML-страницу с Chart.js
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
            return {"status": "ok", "version": VERSION, "timestamp": int(time.time() * 1000)}

        @app.get("/api/status")
        async def status():
            state = self._get_state()
            return JSONResponse(content={
                "mode": state.get("mode", "paper"),
                "risk_state": state.get("risk_state", "NORMAL"),
                "uptime": state.get("uptime", "N/A"),
                "pnl_today": state.get("pnl_today", 0.0),
                "pnl_total": state.get("pnl_total", 0.0),
                "open_positions": state.get("open_positions", 0),
                "trades_today": state.get("trades_today", 0),
                "balance": state.get("balance", 0.0),
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
                    # Периодически отправляем состояние
                    state = self._get_state()
                    await websocket.send_json({
                        "type": "state_update",
                        "data": {
                            "mode": state.get("mode", "paper"),
                            "risk_state": state.get("risk_state", "NORMAL"),
                            "pnl_today": state.get("pnl_today", 0.0),
                            "pnl_total": state.get("pnl_total", 0.0),
                            "open_positions": state.get("open_positions", 0),
                            "trades_today": state.get("trades_today", 0),
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
        # Запуск в фоне — не блокирует event loop
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


# ──────────────────────────────────────────────
# HTML Template (встроенный, с Chart.js)
# ──────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENTINEL Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0f1117;
            color: #e1e4e8;
            min-height: 100vh;
        }
        .header {
            background: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 20px; color: #58a6ff; }
        .header .status-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
        }
        .status-normal { background: #238636; color: #fff; }
        .status-reduced { background: #d29922; color: #000; }
        .status-safe { background: #da6d28; color: #fff; }
        .status-stop { background: #da3633; color: #fff; }

        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }

        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }

        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
        }
        .card-label { font-size: 12px; color: #8b949e; text-transform: uppercase; margin-bottom: 4px; }
        .card-value { font-size: 24px; font-weight: 700; }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }

        .chart-container { height: 280px; margin-bottom: 20px; }

        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #21262d;
        }
        th { color: #8b949e; font-size: 12px; text-transform: uppercase; }

        .controls {
            display: flex;
            gap: 12px;
            margin-top: 20px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
        }
        .btn-start { background: #238636; color: #fff; }
        .btn-stop { background: #da3633; color: #fff; }
        .btn-kill { background: #6e1010; color: #fff; border: 1px solid #da3633; }
        .btn:hover { opacity: 0.85; }

        @media (max-width: 768px) {
            .grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ SENTINEL</h1>
        <div>
            <span id="mode-badge" class="status-badge status-normal">PAPER</span>
            <span id="state-badge" class="status-badge status-normal">NORMAL</span>
        </div>
    </div>

    <div class="container">
        <div class="grid">
            <div class="card">
                <div class="card-label">PnL Сегодня</div>
                <div class="card-value" id="pnl-today">$0.00</div>
            </div>
            <div class="card">
                <div class="card-label">PnL Всего</div>
                <div class="card-value" id="pnl-total">$0.00</div>
            </div>
            <div class="card">
                <div class="card-label">Открытых позиций</div>
                <div class="card-value" id="open-positions">0</div>
            </div>
            <div class="card">
                <div class="card-label">Сделок сегодня</div>
                <div class="card-value" id="trades-today">0</div>
            </div>
        </div>

        <div class="card chart-container">
            <canvas id="pnlChart"></canvas>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <h3 style="margin-bottom: 12px; color: #8b949e; font-size: 14px;">ОТКРЫТЫЕ ПОЗИЦИИ</h3>
            <table>
                <thead>
                    <tr>
                        <th>Символ</th>
                        <th>Сторона</th>
                        <th>Вход</th>
                        <th>Текущая</th>
                        <th>PnL</th>
                    </tr>
                </thead>
                <tbody id="positions-table"></tbody>
            </table>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <h3 style="margin-bottom: 12px; color: #8b949e; font-size: 14px;">ПОСЛЕДНИЕ СДЕЛКИ</h3>
            <table>
                <thead>
                    <tr>
                        <th>Время</th>
                        <th>Действие</th>
                        <th>Символ</th>
                        <th>Цена</th>
                        <th>PnL</th>
                    </tr>
                </thead>
                <tbody id="trades-table"></tbody>
            </table>
        </div>

        <div class="controls">
            <button class="btn btn-start" onclick="controlAction('resume')">🟢 START</button>
            <button class="btn btn-stop" onclick="controlAction('stop')">🔴 STOP</button>
            <button class="btn btn-kill" onclick="if(confirm('Аварийная остановка?')) controlAction('kill')">☠️ EMERGENCY</button>
        </div>
    </div>

    <script>
        // PnL Chart
        const ctx = document.getElementById('pnlChart').getContext('2d');
        const pnlChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'PnL ($)',
                    data: [],
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    fill: true,
                    tension: 0.3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' } },
                    y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' } }
                }
            }
        });

        function formatPnl(val) {
            const sign = val >= 0 ? '+' : '';
            return sign + '$' + val.toFixed(2);
        }

        function updateUI(data) {
            const pnlToday = document.getElementById('pnl-today');
            const pnlTotal = document.getElementById('pnl-total');
            pnlToday.textContent = formatPnl(data.pnl_today || 0);
            pnlToday.className = 'card-value ' + ((data.pnl_today || 0) >= 0 ? 'positive' : 'negative');
            pnlTotal.textContent = formatPnl(data.pnl_total || 0);
            pnlTotal.className = 'card-value ' + ((data.pnl_total || 0) >= 0 ? 'positive' : 'negative');
            document.getElementById('open-positions').textContent = data.open_positions || 0;
            document.getElementById('trades-today').textContent = data.trades_today || 0;

            const modeBadge = document.getElementById('mode-badge');
            modeBadge.textContent = (data.mode || 'paper').toUpperCase();

            const stateBadge = document.getElementById('state-badge');
            const rs = (data.risk_state || 'NORMAL').toLowerCase();
            stateBadge.textContent = data.risk_state || 'NORMAL';
            stateBadge.className = 'status-badge status-' + rs;
        }

        // WebSocket
        let ws;
        function connectWS() {
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = function(e) {
                const msg = JSON.parse(e.data);
                if (msg.type === 'state_update') updateUI(msg.data);
            };
            ws.onclose = function() { setTimeout(connectWS, 3000); };
            ws.onerror = function() { ws.close(); };
        }
        connectWS();

        // Polling fallback
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
                tbody.innerHTML = data.map(p =>
                    `<tr><td>${p.symbol}</td><td>${p.side}</td><td>$${(p.entry_price||0).toFixed(2)}</td>` +
                    `<td>$${(p.current_price||0).toFixed(2)}</td>` +
                    `<td class="${(p.unrealized_pnl||0)>=0?'positive':'negative'}">${formatPnl(p.unrealized_pnl||0)}</td></tr>`
                ).join('');
            } catch(e) {}
        }
        async function fetchTrades() {
            try {
                const r = await fetch('/api/trades');
                const data = await r.json();
                const tbody = document.getElementById('trades-table');
                tbody.innerHTML = data.slice(0,10).map(t =>
                    `<tr><td>${t.time||'-'}</td><td>${t.side||'-'}</td><td>${t.symbol||'-'}</td>` +
                    `<td>$${(t.price||0).toFixed(2)}</td>` +
                    `<td class="${(t.pnl||0)>=0?'positive':'negative'}">${t.pnl != null ? formatPnl(t.pnl) : '-'}</td></tr>`
                ).join('');
            } catch(e) {}
        }
        async function fetchPnlHistory() {
            try {
                const r = await fetch('/api/pnl-history');
                const data = await r.json();
                pnlChart.data.labels = data.map(d => d.date || d.label || '');
                pnlChart.data.datasets[0].data = data.map(d => d.pnl || d.value || 0);
                pnlChart.update();
            } catch(e) {}
        }

        // Initial fetch + interval
        fetchStatus(); fetchPositions(); fetchTrades(); fetchPnlHistory();
        setInterval(fetchStatus, 5000);
        setInterval(fetchPositions, 10000);
        setInterval(fetchTrades, 10000);
        setInterval(fetchPnlHistory, 60000);

        async function controlAction(action) {
            try {
                await fetch('/api/control/' + action, { method: 'POST' });
                fetchStatus();
            } catch(e) { alert('Ошибка: ' + e.message); }
        }
    </script>
</body>
</html>
"""
