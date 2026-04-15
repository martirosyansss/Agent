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
        self.market_chart_provider: Optional[Callable[[str], dict]] = None
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
            """Token auth for mutating endpoints when password is configured."""
            _PUBLIC = {"/api/health", "/", "/settings", "/ws"}
            _READ_ONLY = {"/api/status", "/api/positions", "/api/trades",
                          "/api/pnl-history", "/api/market-chart",
                          "/api/backtest-results", "/api/config",
                          "/api/settings/editable", "/api/strategy-performance",
                          "/api/trades/export", "/api/news"}

            async def dispatch(self, request: Request, call_next):
                path = request.url.path
                # Static files and public endpoints — no auth
                if path.startswith("/static") or path in self._PUBLIC:
                    return await call_next(request)
                # If no password configured, allow everything
                if not dashboard_password:
                    return await call_next(request)
                # Check token
                token = (request.headers.get("X-Auth-Token")
                         or request.query_params.get("token")
                         or "")
                if not secrets.compare_digest(token, dashboard_password):
                    # Allow read-only GET without auth for dashboard panels
                    if request.method == "GET" and path in self._READ_ONLY:
                        return await call_next(request)
                    return JSONResponse(
                        content={"error": "Unauthorized"},
                        status_code=401,
                    )
                return await call_next(request)

        app.add_middleware(AuthMiddleware)

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
                "trading_paused": state.get("trading_paused", False),
                "uptime": state.get("uptime", _format_uptime()),
                "pnl_today": state.get("pnl_today", 0.0),
                "pnl_total": state.get("pnl_total", 0.0),
                "open_positions": state.get("open_positions", 0),
                "trades_today": state.get("trades_today", 0),
                "balance": state.get("balance", 0.0),
                "win_rate": state.get("win_rate", 0.0),
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
        async def market_chart(interval: str = "1m", symbol: str = ""):
            if self.market_chart_provider:
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

        @app.get("/api/ml/status")
        async def ml_status():
            """ML model status, metrics, and mode."""
            ml = self._state_provider() if self._state_provider else {}
            predictor = ml.get("ml_predictor") if ml else None
            if predictor is None:
                return JSONResponse(content={"enabled": False, "reason": "not initialized"})
            m = predictor.metrics
            return JSONResponse(content={
                "enabled": True,
                "ready": predictor.is_ready,
                "mode": predictor.rollout_mode,
                "version": predictor._model_version or "none",
                "needs_retrain": predictor.needs_retrain(),
                "metrics": {
                    "precision": round(m.precision, 4) if m else None,
                    "recall": round(m.recall, 4) if m else None,
                    "roc_auc": round(m.roc_auc, 4) if m else None,
                    "skill_score": round(m.skill_score, 4) if m else None,
                    "train_samples": m.train_samples if m else 0,
                    "test_samples": m.test_samples if m else 0,
                } if m else None,
                "threshold": predictor._cfg.block_threshold,
                "block_threshold": predictor._cfg.block_threshold,
                "reduce_threshold": getattr(predictor._cfg, "reduce_threshold", 0.65),
            })

        @app.post("/api/ml/retrain")
        async def ml_retrain():
            """Trigger manual ML retraining (runs in background)."""
            ml = self._state_provider() if self._state_provider else {}
            retrain_fn = ml.get("ml_retrain_fn") if ml else None
            if retrain_fn is None:
                return JSONResponse(content={"status": "error", "message": "retrain function not wired"}, status_code=503)
            asyncio.create_task(retrain_fn())
            return JSONResponse(content={"status": "started", "message": "ML retraining triggered in background"})

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

            return JSONResponse(content={
                "result": "saved",
                "restart_required": True,
                "message": "Settings saved to .env. Restart the bot to apply engine-level changes.",
                "values": get_editable_settings_payload(self._settings),
            })

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
            # Auth check for WebSocket (token via query param)
            if self._password:
                token = websocket.query_params.get("token", "")
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
                            "risk_details": state.get("risk_details", {}),
                            "activity": state.get("activity", {}),
                            "indicators": state.get("indicators", {}),
                            "indicators_per_symbol": state.get("indicators_per_symbol", {}),
                            "trading_symbols": state.get("trading_symbols", []),
                            "win_rate_per_symbol": state.get("win_rate_per_symbol", {}),
                            "readiness": state.get("readiness", {}),
                            "strategy_log": state.get("strategy_log", []),
                            "ml_status": state.get("ml_status", {}),
                        },
                    })
                    await asyncio.sleep(2)
            except WebSocketDisconnect:
                logger.debug("Dashboard websocket client disconnected")
            finally:
                if websocket in self._ws_clients:
                    self._ws_clients.remove(websocket)

        # ── HTML Dashboard ───────────────────────

        @app.get("/", response_class=HTMLResponse)
        async def dashboard_page():
            return _DASHBOARD_HTML

        @app.get("/settings", response_class=HTMLResponse)
        async def settings_page():
            return _SETTINGS_HTML

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



# Load HTML from static/index.html
import pathlib as _pathlib
_STATIC_DIR = _pathlib.Path(__file__).parent / "static"
_DASHBOARD_HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8") if (_STATIC_DIR / "index.html").exists() else "<h1>Dashboard HTML not found</h1>"
_SETTINGS_HTML = (_STATIC_DIR / "settings.html").read_text(encoding="utf-8") if (_STATIC_DIR / "settings.html").exists() else "<h1>Settings HTML not found</h1>"
