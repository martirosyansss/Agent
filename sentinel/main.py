"""
SENTINEL V1.5 — Точка входа.

Последовательность запуска (21 шаг) описана в ТЗ §26.2.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

from loguru import logger

# Корень проекта — директория, где лежит main.py
BASE_DIR = Path(__file__).resolve().parent

# Добавляем корень проекта в sys.path, чтобы импорты работали
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import load_settings, Settings  # noqa: E402
from core.absolute_limits import FORBIDDEN_API_PERMISSIONS  # noqa: E402
from core.constants import (  # noqa: E402
    APP_NAME,
    EVENT_NEW_CANDLE,
    EVENT_NEW_SIGNAL,
    EVENT_NEW_TRADE,
    EVENT_ORDER_FILLED,
    VERSION,
    LOG_ROTATION_SIZE,
    LOG_ROTATION_COUNT,
    PID_FILE,
    STATE_FILE,
    HEARTBEAT_FILE,
)
from core.events import EventBus  # noqa: E402
from core.models import Direction, RiskState  # noqa: E402


BINANCE_REST_BASE = "https://api.binance.com"


# ──────────────────────────────────────────────
# Logging (loguru)
# ──────────────────────────────────────────────

def setup_logging(settings: Settings) -> None:
    """Настройка loguru: файлы + консоль."""
    logger.remove()  # убираем default stderr handler

    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "{extra[module]:<12} | {message}"
    )

    # Основной лог
    logger.add(
        log_dir / "sentinel.log",
        format=fmt,
        level=settings.log_level,
        rotation=LOG_ROTATION_SIZE,
        retention=LOG_ROTATION_COUNT,
        encoding="utf-8",
    )
    # Только ошибки
    logger.add(
        log_dir / "errors.log",
        format=fmt,
        level="ERROR",
        rotation=LOG_ROTATION_SIZE,
        retention=LOG_ROTATION_COUNT,
        encoding="utf-8",
    )
    # Сделки
    logger.add(
        log_dir / "trades.log",
        format=fmt,
        level="INFO",
        rotation=LOG_ROTATION_SIZE,
        retention=LOG_ROTATION_COUNT,
        filter=lambda r: r["extra"].get("module") in ("executor", "position"),
        encoding="utf-8",
    )
    # Risk
    logger.add(
        log_dir / "risk.log",
        format=fmt,
        level="INFO",
        rotation=LOG_ROTATION_SIZE,
        retention=LOG_ROTATION_COUNT,
        filter=lambda r: r["extra"].get("module") in ("risk", "circuit_breaker"),
        encoding="utf-8",
    )
    # Консоль (кратко)
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level=settings.log_level,
        colorize=True,
    )

    logger.configure(extra={"module": "main"})


# ──────────────────────────────────────────────
# PID Lock
# ──────────────────────────────────────────────

def acquire_pid_lock() -> bool:
    """Проверяет, не запущен ли уже инстанс. Создаёт PID-файл."""
    pid_path = BASE_DIR / PID_FILE
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
            # Проверяем жив ли процесс (Windows-совместимо)
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x100000, False, old_pid)  # SYNCHRONIZE
            if handle:
                kernel32.CloseHandle(handle)
                logger.error("Другой инстанс уже запущен (PID {})", old_pid)
                return False
        except (ValueError, OSError, AttributeError) as exc:
            logger.warning("PID файл будет перезаписан: {}", exc)

    pid_path.write_text(str(os.getpid()))
    return True


def release_pid_lock() -> None:
    pid_path = BASE_DIR / PID_FILE
    pid_path.unlink(missing_ok=True)


# ──────────────────────────────────────────────
# State persistence
# ──────────────────────────────────────────────

def save_state(state: dict) -> None:
    """Сохранить текущее состояние системы в JSON."""
    state_path = BASE_DIR / STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state() -> dict:
    """Загрузить последнее состояние системы."""
    state_path = BASE_DIR / STATE_FILE
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Не удалось загрузить state.json — стартуем с чистого состояния")
    return {}


# ──────────────────────────────────────────────
# Heartbeat
# ──────────────────────────────────────────────

async def heartbeat_writer(settings: Settings) -> None:
    """Пишет timestamp в heartbeat-файл каждые N секунд для Watchdog."""
    hb_path = BASE_DIR / HEARTBEAT_FILE
    hb_path.parent.mkdir(parents=True, exist_ok=True)
    interval = settings.watchdog_heartbeat_interval
    while True:
        # Atomic write: write to temp file, then rename
        tmp_path = hb_path.with_suffix(".tmp")
        tmp_path.write_text(str(int(time.time())), encoding="utf-8")
        tmp_path.replace(hb_path)
        await asyncio.sleep(interval)


def _build_signed_binance_params(
    settings: Settings,
    extra_params: dict[str, str | int] | None = None,
) -> list[tuple[str, str | int]]:
    """Собирает signed query params для Binance REST API."""
    params: list[tuple[str, str | int]] = list((extra_params or {}).items())
    params.extend([
        ("timestamp", int(time.time() * 1000)),
        ("recvWindow", 5000),
    ])
    query = urlencode(params)
    signature = hmac.new(
        settings.binance_api_secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    params.append(("signature", signature))
    return params


async def _binance_get_json(
    client,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    params: list[tuple[str, str | int]] | None = None,
) -> dict:
    response = await client.get(path, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def _evaluate_binance_permissions(
    account_data: dict | None,
    restrictions_data: dict | None,
) -> tuple[bool, str]:
    """Проверяет, что API-ключ ограничен только разрешёнными правами."""
    if account_data and account_data.get("canTrade") is False:
        return False, "spot trading disabled for API key"

    if restrictions_data:
        forbidden_enabled: list[str] = []
        if restrictions_data.get("enableWithdrawals"):
            forbidden_enabled.append("withdraw")
        if restrictions_data.get("enableFutures"):
            forbidden_enabled.append("futures")
        if restrictions_data.get("enableMargin"):
            forbidden_enabled.append("margin")
        if forbidden_enabled:
            return False, f"forbidden API rights enabled: {', '.join(forbidden_enabled)}"
        return True, "api restrictions verified"

    permissions = {
        str(permission).lower()
        for permission in (account_data or {}).get("permissions", [])
    }
    forbidden = sorted(permissions.intersection(FORBIDDEN_API_PERMISSIONS))
    if forbidden:
        return False, f"forbidden permissions present: {', '.join(forbidden)}"

    if account_data and account_data.get("canWithdraw") is True:
        return False, "withdrawal access is enabled"

    return True, "account permissions verified"


async def _run_binance_preflight(settings: Settings) -> dict[str, int | bool | str]:
    """Выполняет сетевые pre-flight проверки Binance REST API."""
    import httpx

    result: dict[str, int | bool | str] = {
        "ping_ok": False,
        "auth_ok": False,
        "permissions_ok": False,
        "time_sync_ok": False,
        "time_drift_ms": -1,
        "permissions_reason": "not checked",
    }

    headers = {"X-MBX-APIKEY": settings.binance_api_key}
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(base_url=BINANCE_REST_BASE, timeout=timeout) as client:
        await _binance_get_json(client, "/api/v3/ping")
        result["ping_ok"] = True

        time_payload = await _binance_get_json(client, "/api/v3/time")
        server_time = int(time_payload.get("serverTime", 0))
        drift_ms = abs(server_time - int(time.time() * 1000)) if server_time else -1
        result["time_drift_ms"] = drift_ms
        result["time_sync_ok"] = 0 <= drift_ms <= 5_000

        account_payload = await _binance_get_json(
            client,
            "/api/v3/account",
            headers=headers,
            params=_build_signed_binance_params(settings),
        )
        result["auth_ok"] = True

        restrictions_payload: dict | None = None
        try:
            restrictions_payload = await _binance_get_json(
                client,
                "/sapi/v1/account/apiRestrictions",
                headers=headers,
                params=_build_signed_binance_params(settings),
            )
        except Exception as _restr_err:
            log.debug("API restrictions query failed (non-critical): {}", _restr_err)
            restrictions_payload = None

        permissions_ok, permissions_reason = _evaluate_binance_permissions(
            account_payload,
            restrictions_payload,
        )
        result["permissions_ok"] = permissions_ok
        result["permissions_reason"] = permissions_reason

    return result


def _check_sqlite_integrity(settings: Settings) -> bool:
    """Запускает реальную integrity-check для SQLite."""
    from database.db import Database

    db = Database(BASE_DIR / settings.db_path)
    try:
        db.connect()
        return db.integrity_check()
    finally:
        db.close()


# ──────────────────────────────────────────────
# Pre-flight checklist
# ──────────────────────────────────────────────

async def preflight_check(settings: Settings) -> bool:
    """
    Pre-flight checklist (ТЗ §26.3): 10 проверок.
    Возвращает True если все критичные прошли.
    """
    log = logger.bind(module="preflight")
    ok = True

    # [1/10] .env существует
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        log.info("[1/10] .env файл — ✅")
    else:
        log.error("[1/10] .env файл — ❌ (не найден)")
        ok = False

    # [2/10] API ключи формат
    if len(settings.binance_api_key) >= 10 and len(settings.binance_api_secret) >= 10:
        log.info("[2/10] API ключи формат — ✅")
    else:
        log.error("[2/10] API ключи формат — ❌ (слишком короткие)")
        ok = False

    try:
        binance_result = await _run_binance_preflight(settings)
    except Exception as exc:
        binance_result = {
            "ping_ok": False,
            "auth_ok": False,
            "permissions_ok": False,
            "time_sync_ok": False,
            "time_drift_ms": -1,
            "permissions_reason": str(exc),
        }

    if binance_result["ping_ok"]:
        log.info("[3/10] Binance API ping — ✅")
    else:
        log.error("[3/10] Binance API ping — ❌")
        ok = False

    if binance_result["auth_ok"]:
        log.info("[4/10] API auth test — ✅")
    else:
        log.error("[4/10] API auth test — ❌")
        ok = False

    if binance_result["permissions_ok"]:
        log.info("[5/10] API permissions — ✅ ({})", binance_result["permissions_reason"])
    else:
        log.error("[5/10] API permissions — ❌ ({})", binance_result["permissions_reason"])
        ok = False

    if binance_result["time_sync_ok"]:
        log.info("[6/10] Time sync — ✅ (drift={} ms)", binance_result["time_drift_ms"])
    else:
        log.error("[6/10] Time sync — ❌ (drift={} ms)", binance_result["time_drift_ms"])
        ok = False

    try:
        if _check_sqlite_integrity(settings):
            log.info("[7/10] SQLite integrity — ✅")
        else:
            log.error("[7/10] SQLite integrity — ❌")
            ok = False
    except Exception as exc:
        log.error("[7/10] SQLite integrity — ❌ ({})", exc)
        ok = False

    # [8/10] Свободное место на диске
    import shutil
    usage = shutil.disk_usage(BASE_DIR)
    free_gb = usage.free / (1024**3)
    if free_gb > 1.0:
        log.info("[8/10] Диск свободно {:.1f} GB — ✅", free_gb)
    else:
        log.error("[8/10] Диск свободно {:.1f} GB — ❌ (нужно > 1 GB)", free_gb)
        ok = False

    # [9/10] Telegram — не блокирует
    if settings.telegram_bot_token:
        log.info("[9/10] Telegram token задан — ✅")
    else:
        log.warning("[9/10] Telegram token не задан — ⚠️")

    # [10/10] PID lock
    if acquire_pid_lock():
        log.info("[10/10] PID lock — ✅")
    else:
        log.error("[10/10] PID lock — ❌ (другой инстанс?)")
        ok = False

    return ok


# ──────────────────────────────────────────────
# Graceful Shutdown
# ──────────────────────────────────────────────

class GracefulShutdown:
    """Обработчик Ctrl+C / SIGTERM."""

    def __init__(self) -> None:
        self._shutdown_event = asyncio.Event()

    @property
    def event(self) -> asyncio.Event:
        return self._shutdown_event

    def trigger(self) -> None:
        self._shutdown_event.set()

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.trigger)
            except NotImplementedError:
                # Windows не поддерживает add_signal_handler
                signal.signal(sig, lambda s, f: self.trigger())


# ──────────────────────────────────────────────
# Uptime
# ──────────────────────────────────────────────

_BOOT_TIME = time.time()


def _format_uptime() -> str:
    elapsed = int(time.time() - _BOOT_TIME)
    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def run() -> None:
    """Основной цикл SENTINEL."""
    settings = load_settings()
    setup_logging(settings)

    log = logger.bind(module="main")
    log.info("=== {} v{} ===", APP_NAME, VERSION)
    log.info("Режим: {}", settings.trading_mode.upper())

    # Pre-flight
    if not await preflight_check(settings):
        log.critical("Pre-flight check FAILED — система НЕ запускается")
        return

    # Загрузка состояния
    state = load_state()
    if state:
        log.info("Загружено предыдущее состояние: {} ключей", len(state))

    # EventBus
    bus = EventBus()

    # Graceful shutdown
    shutdown = GracefulShutdown()
    loop = asyncio.get_running_loop()
    shutdown.install_signal_handlers(loop)

    # ------------------------------------------------------------------
    # Инициализация модулей
    # ------------------------------------------------------------------

    # 6. SQLite (WAL mode)
    from database.db import Database
    from database.repository import Repository
    db = Database(BASE_DIR / settings.db_path)
    db.connect()
    repo = Repository(db)
    log.info("[Module] Database initialized")

    # 9. Data Collector (WebSocket)
    collector = None
    collector_task: asyncio.Task | None = None
    try:
        from collector.binance_ws import BinanceWebSocketCollector
        collector = BinanceWebSocketCollector(
            symbols=settings.trading_symbols,
            repo=repo,
            bus=bus,
        )
        collector_task = asyncio.create_task(collector.start())
        log.info("[Module] Collector started for {}", settings.trading_symbols)
    except Exception as e:
        log.warning("[Module] Collector failed to start: {}", e)

    # 15. Execution Engine (paper mode)
    executor = None
    try:
        from execution.paper_executor import PaperExecutor
        executor = PaperExecutor(
            event_bus=bus,
            commission_pct=settings.paper_commission_pct,
        )
        log.info("[Module] Paper Executor initialized")
    except Exception as e:
        log.warning("[Module] Paper Executor failed: {}", e)

    # 16. Position / Risk runtime
    position_manager = None
    risk_state_machine = None
    risk_sentinel = None
    try:
        from position.manager import PositionManager
        from risk.sentinel import RiskLimits, RiskSentinel
        from risk.state_machine import RiskStateMachine

        position_manager = PositionManager(
            event_bus=bus,
            initial_balance=settings.paper_initial_balance,
            max_open_positions=settings.max_open_positions,
        )
        risk_state_machine = RiskStateMachine(
            event_bus=bus,
            max_daily_loss=settings.max_daily_loss_usd,
        )
        risk_sentinel = RiskSentinel(
            limits=RiskLimits(
                max_daily_loss_usd=settings.max_daily_loss_usd,
                max_daily_loss_pct=settings.max_daily_loss_pct,
                max_daily_trades=settings.max_trades_per_day,
                max_position_pct=settings.max_position_pct,
                max_total_exposure_pct=settings.max_total_exposure_pct,
                max_open_positions=settings.max_open_positions,
                max_trades_per_hour=settings.max_trades_per_hour,
                min_trade_interval_sec=settings.resume_cooldown_min * 60,
                max_order_usd=settings.max_order_usd,
                max_loss_per_trade_pct=settings.stop_loss_pct,
                max_daily_commission_pct=settings.cb_commission_alert_pct,
            ),
            state_machine=risk_state_machine,
        )

        # Stores {symbol: ml_probability} for open positions so we can feed the
        # outcome back to LivePerformanceTracker when the trade closes.
        _ml_prob_at_entry: dict[str, float] = {}

        async def _on_order_filled(order):
            if not position_manager:
                return

            if order.side == Direction.BUY:
                opened = await position_manager.open_position(order)
                if opened and risk_sentinel:
                    risk_sentinel.record_trade(order.commission, increment_trade=True)
            elif order.side == Direction.SELL:
                closed = await position_manager.close_position(order)
                if closed and risk_sentinel:
                    risk_sentinel.record_trade(order.commission, increment_trade=False)
                # Feed realized outcome back to ML tracker for concept drift detection
                if closed and _ml_predictor and order.symbol in _ml_prob_at_entry:
                    _entry_prob, _entry_px = _ml_prob_at_entry.pop(order.symbol, (0.5, 0.0))
                    _actual_win = (order.fill_price > _entry_px) if _entry_px > 0 else False
                    _ml_predictor.record_outcome(_entry_prob, _actual_win)

            if risk_state_machine:
                await risk_state_machine.update(position_manager.total_realized_pnl)

        async def _on_market_trade(trade):
            if not position_manager:
                return
            position_manager.update_price(trade.symbol, trade.price)
            if risk_state_machine:
                await risk_state_machine.update(position_manager.total_realized_pnl)

        bus.subscribe(EVENT_ORDER_FILLED, _on_order_filled)
        bus.subscribe(EVENT_NEW_TRADE, _on_market_trade)
        log.info("[Module] Position & Risk runtime initialized")
    except Exception as e:
        log.warning("[Module] Position/Risk runtime failed: {}", e)

    # ------------------------------------------------------------------
    # 17. Trading Loop — стратегии, фичи, сигналы, исполнение
    # ------------------------------------------------------------------
    trading_paused = False  # управляется через Start/Stop в Dashboard

    # Инициализация стратегий
    strategies = {}
    feature_builder = None
    try:
        from features.feature_builder import FeatureBuilder
        from strategy.ema_crossover_rsi import EMACrossoverRSI, EMAConfig
        from strategy.bollinger_breakout import BollingerBreakout
        from strategy.mean_reversion import MeanReversion
        from strategy.dca_bot import DCABot
        from strategy.macd_divergence import MACDDivergence
        from strategy.grid_trading import GridTrading
        from strategy.market_regime import detect_regime
        from strategy.strategy_selector import get_active_strategies, get_strategy_budget_pct, AdaptiveAllocator

        feature_builder = FeatureBuilder()

        strategies["ema_crossover_rsi"] = EMACrossoverRSI(EMAConfig(
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
            min_confidence=settings.min_confidence,
        ))
        strategies["bollinger_breakout"] = BollingerBreakout()
        strategies["mean_reversion"] = MeanReversion()
        strategies["dca_bot"] = DCABot()
        strategies["macd_divergence"] = MACDDivergence()
        strategies["grid_trading"] = GridTrading()

        log.info("[Module] Strategies initialized: {}", list(strategies.keys()))
    except Exception as e:
        log.warning("[Module] Strategies failed to initialize: {}", e)

    _current_regime = None
    _last_features = None  # последний FeatureVector для дашборда
    _adaptive_allocator = None
    try:
        _adaptive_allocator = AdaptiveAllocator(lookback_trades=50)
        log.info("[Module] AdaptiveAllocator initialized")
    except Exception as e:
        log.warning("[Module] AdaptiveAllocator failed: {}", e)

    # Position sizer & dynamic SL/TP & alerts
    _position_sizer = None
    _alert_monitor = None
    try:
        from risk.position_sizer import calculate_position_size, SizingInput
        from risk.dynamic_sltp import calculate_dynamic_sltp
        from monitoring.alerts import AlertMonitor
        _position_sizer = True  # flag — functions imported
        _alert_monitor = AlertMonitor()
        log.info("[Module] Position sizer + Dynamic SL/TP + Alert monitor initialized")
    except Exception as e:
        log.warning("[Module] Advanced risk modules failed: {}", e)

    # ML Predictor — фильтрация сигналов
    _ml_predictor = None
    try:
        from analyzer.ml_predictor import MLPredictor, MLConfig
        _ml_predictor = MLPredictor(MLConfig(
            block_threshold=settings.analyzer_ml_block_threshold,
            min_precision=settings.analyzer_ml_min_precision,
            min_recall=settings.analyzer_ml_min_recall,
            min_roc_auc=settings.analyzer_ml_min_roc_auc,
            min_skill_score=settings.analyzer_ml_min_skill_score,
            min_trades=settings.analyzer_min_trades_ml,
            retrain_days=settings.analyzer_ml_retrain_days,
            test_window_days=settings.analyzer_ml_test_window_days,
        ))
        if settings.analyzer_ml_shadow_mode:
            _ml_predictor.rollout_mode = "shadow"
        elif settings.analyzer_ml_enabled:
            _ml_predictor.rollout_mode = "block"
        # Load previously trained model from disk
        _ml_model_path = Path(__file__).parent / "data" / "ml_models" / "ml_predictor.pkl"
        if _ml_model_path.exists():
            loaded = _ml_predictor.load_from_file(_ml_model_path)
            if loaded:
                log.info("[Module] ML model loaded from disk (version={})", _ml_predictor._model_version)
            else:
                log.warning("[Module] ML model load failed — running without model until retrain")
        else:
            log.warning("[Module] No saved ML model found at {} — will train on startup", _ml_model_path)
        log.info("[Module] ML Predictor initialized (mode={})", _ml_predictor.rollout_mode)
    except Exception as e:
        log.warning("[Module] ML Predictor failed: {}", e)

    # ── ML retrain function (defined early so get_system_state can reference it) ──
    _ml_model_path_early = Path(__file__).parent / "data" / "ml_models" / "ml_predictor.pkl"
    _ml_retrain_lock_early = asyncio.Lock()

    async def _run_ml_training() -> bool:
        """Load trades from DB, train ML model, save to disk. Returns True on success."""
        if not _ml_predictor or not repo:
            return False
        try:
            import sys as _sys
            _scripts_dir = str(Path(__file__).parent / "scripts")
            if _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)
            from scripts.train_ml import build_all_trades
            log.info("ML auto-retrain: collecting training data...")
            all_trades = await asyncio.to_thread(build_all_trades, repo, settings)
            if not all_trades:
                log.warning("ML auto-retrain: no trades available for training")
                return False
            log.info("ML auto-retrain: training on {} trades...", len(all_trades))
            metrics = await asyncio.to_thread(_ml_predictor.train, all_trades)
            if metrics is None or not _ml_predictor.is_ready:
                log.warning("ML auto-retrain: training failed or metrics below threshold")
                return False
            saved = await asyncio.to_thread(_ml_predictor.save_to_file, _ml_model_path_early)
            if saved:
                log.info(
                    "ML auto-retrain: ✅ saved (skill={:.3f} prec={:.3f} rec={:.3f} auc={:.3f})",
                    metrics.skill_score, metrics.precision, metrics.recall, metrics.roc_auc,
                )
                return True
            log.warning("ML auto-retrain: model below quality gate — not saved")
            return False
        except Exception as exc:
            log.error("ML auto-retrain error: {}", exc)
            return False

    # ── Strategy Decision Log (ring buffer) ──
    from collections import deque
    _strategy_log: deque[dict] = deque(maxlen=50)

    # news_collector будет инициализирован позже (вместе с Dashboard)
    news_collector = None

    async def _on_new_candle(candle):
        """Главный торговый цикл — вызывается при закрытии каждой свечи."""
        nonlocal _current_regime, trading_paused, _last_features

        if trading_paused:
            return

        if not feature_builder or not executor or not position_manager or not risk_sentinel:
            return

        # Только 1h свечи запускают стратегии (signal_timeframe)
        if candle.interval != settings.signal_timeframe:
            return

        symbol = candle.symbol
        ts_now = int(time.time() * 1000)
        log.info("Trading loop triggered: {} {} candle closed @ {:.2f}", symbol, candle.interval, candle.close)

        # 1. Собрать свечи из БД
        try:
            candles_1h_raw = await asyncio.to_thread(
                repo.get_candles, symbol, "1h", limit=60
            )
            candles_4h_raw = await asyncio.to_thread(
                repo.get_candles, symbol, "4h", limit=60
            )
            candles_1d_raw = await asyncio.to_thread(
                repo.get_candles, symbol, "1d", limit=60
            )
        except Exception as e:
            log.warning("Failed to fetch candles for {}: {}", symbol, e)
            _strategy_log.append({"ts": ts_now, "symbol": symbol, "event": "error", "msg": f"Failed to fetch candles: {e}"})
            return

        if not candles_1h_raw or not candles_4h_raw:
            log.info("Not enough candles for {} (1h={}, 4h={})", symbol, len(candles_1h_raw or []), len(candles_4h_raw or []))
            _strategy_log.append({"ts": ts_now, "symbol": symbol, "event": "skip", "msg": f"Not enough candles (1h={len(candles_1h_raw or [])}, 4h={len(candles_4h_raw or [])})"})
            return

        # Конвертировать dict → Candle
        from core.models import Candle as CandleModel
        candles_1h = [
            CandleModel(
                timestamp=c["timestamp"], symbol=c["symbol"], interval=c["interval"],
                open=float(c["open"]), high=float(c["high"]), low=float(c["low"]),
                close=float(c["close"]), volume=float(c.get("volume", 0)),
                trades_count=int(c.get("trades_count", 0)),
            ) for c in candles_1h_raw
        ]
        candles_4h = [
            CandleModel(
                timestamp=c["timestamp"], symbol=c["symbol"], interval=c["interval"],
                open=float(c["open"]), high=float(c["high"]), low=float(c["low"]),
                close=float(c["close"]), volume=float(c.get("volume", 0)),
                trades_count=int(c.get("trades_count", 0)),
            ) for c in candles_4h_raw
        ]

        # Convert daily candles (may be empty, feature_builder handles None)
        candles_1d = [
            CandleModel(
                timestamp=c["timestamp"], symbol=c["symbol"], interval=c["interval"],
                open=float(c["open"]), high=float(c["high"]), low=float(c["low"]),
                close=float(c["close"]), volume=float(c.get("volume", 0)),
                trades_count=int(c.get("trades_count", 0)),
            ) for c in (candles_1d_raw or [])
        ] or None

        # 2. FeatureBuilder (with daily candles for multi-TF)
        features = feature_builder.build(symbol, candles_1h, candles_4h, candles_1d)
        if features is None:
            log.info("FeatureBuilder returned None for {} (1h={}, 4h={} candles)", symbol, len(candles_1h), len(candles_4h))
            _strategy_log.append({"ts": ts_now, "symbol": symbol, "event": "skip", "msg": "FeatureBuilder: not enough data to compute indicators"})
            return

        _last_features = features

        # 2.5. Обогащение FeatureVector данными из NewsCollector
        if news_collector:
            try:
                sentiment_data = news_collector.get_sentiment()
                impact_data = news_collector.get_impact_summary()
                news_signal = news_collector.get_news_signal()
                features.news_sentiment = sentiment_data.get("overall_score", 0.0)
                features.fear_greed_index = sentiment_data.get("fear_greed_index", 50)
                features.news_impact_pct = impact_data.get("avg_impact_pct", 0.0)
                features.high_impact_news = impact_data.get("high_impact_count", 0)
                # Pro fields
                features.news_composite_score = news_signal.get("composite_score", 0.0)
                features.news_signal_strength = news_signal.get("signal_strength", 0.0)
                features.news_critical_alert = news_signal.get("critical_alert", False)
                features.news_actionable = news_signal.get("actionable", False)
                features.news_dominant_category = news_signal.get("dominant_category", "")
            except Exception as e:
                log.debug("News sentiment enrichment failed: {}", e)

        # 3. Market Regime Detection
        try:
            _current_regime = detect_regime(features)
            log.debug("Regime for {}: {}", symbol, _current_regime.regime.value)
        except Exception as e:
            log.warning("Regime detection failed: {}", e)

        regime_name = _current_regime.regime.value if _current_regime else "unknown"

        # 4. Определить активные стратегии
        if settings.auto_strategy_selection and _current_regime:
            if _adaptive_allocator and _adaptive_allocator._skill_scores:
                allocs = _adaptive_allocator.get_adaptive_allocations(_current_regime)
                active_names = [a.strategy_name for a in allocs if a.is_active]
            else:
                active_names = get_active_strategies(_current_regime)
        else:
            # Запускаем все стратегии, у которых есть инициализированный объект
            # (стратегии фильтруются через .env флаги при инициализации)
            active_names = list(strategies.keys())

        # 5. Проверить позицию
        has_position = position_manager.has_position(symbol)
        pos = position_manager.get_position(symbol)
        entry_price = pos.entry_price if pos else None

        # 6. Генерировать сигналы от каждой активной стратегии
        signal_found = False
        strat_results = []
        for strat_name in active_names:
            strat = strategies.get(strat_name)
            if not strat:
                continue

            try:
                signal = strat.generate_signal(
                    features,
                    has_open_position=has_position,
                    entry_price=entry_price,
                )
            except Exception as e:
                log.warning("Strategy {} error: {}", strat_name, e)
                strat_results.append({"strategy": strat_name, "result": "error", "detail": str(e)})
                continue

            if signal is None:
                strat_results.append({"strategy": strat_name, "result": "no_signal"})
                continue

            # Compute position size with real Kelly params from recent trades
            if signal.suggested_quantity <= 0 and signal.direction == Direction.BUY:
                balance = position_manager.balance
                if _position_sizer and features.atr > 0:
                    _kelly_win_rate = 0.45
                    _kelly_avg_win = 2.5
                    _kelly_avg_loss = 2.5
                    try:
                        _recent = await asyncio.to_thread(
                            repo.get_strategy_trades, strat_name, limit=50
                        ) if repo else []
                        if len(_recent) >= 30:
                            _wins = [t for t in _recent if t.get("pnl_pct", 0) > 0]
                            _losses = [t for t in _recent if t.get("pnl_pct", 0) <= 0]
                            _kelly_win_rate = len(_wins) / len(_recent) if _recent else 0.5
                            _kelly_avg_win = sum(t.get("pnl_pct", 0) for t in _wins) / len(_wins) if _wins else 3.0
                            _kelly_avg_loss = abs(sum(t.get("pnl_pct", 0) for t in _losses) / len(_losses)) if _losses else 2.0
                    except Exception as _kelly_err:
                        log.debug("Kelly stats fetch failed: {}", _kelly_err)

                    sizing = calculate_position_size(SizingInput(
                        balance=balance,
                        price=features.close,
                        atr=features.atr,
                        win_rate=_kelly_win_rate,
                        avg_win_pct=_kelly_avg_win,
                        avg_loss_pct=max(_kelly_avg_loss, 0.1),
                        regime_adx=features.adx,
                        max_position_pct=settings.max_position_pct,
                        max_order_usd=settings.max_order_usd,
                    ))
                    signal.suggested_quantity = sizing.quantity
                    log.debug("Position sizer: {} budget=${:.2f} ({:.1f}%) kelly={:.3f}",
                              sizing.method, sizing.budget_usd, sizing.budget_pct, sizing.kelly_fraction)
                else:
                    if _current_regime and settings.auto_strategy_selection:
                        budget_pct = get_strategy_budget_pct(_current_regime, strat_name)
                    else:
                        budget_pct = settings.max_position_pct
                    budget_usd = balance * budget_pct / 100
                    budget_usd = min(budget_usd, settings.max_order_usd)
                    if features.close > 0:
                        signal.suggested_quantity = budget_usd / features.close

            # Dynamic SL/TP
            if _position_sizer and signal.direction == Direction.BUY:
                sltp = calculate_dynamic_sltp(
                    entry_price=features.close,
                    atr=features.atr,
                    strategy_name=strat_name,
                    fallback_sl_pct=settings.stop_loss_pct,
                    fallback_tp_pct=settings.take_profit_pct,
                )
                signal.stop_loss_price = sltp.stop_loss_price
                signal.take_profit_price = sltp.take_profit_price
                log.debug("Dynamic SL/TP [{}]: SL={:.2f} ({:.1f}%) TP={:.2f} ({:.1f}%)",
                          sltp.method, sltp.stop_loss_price, sltp.stop_loss_pct,
                          sltp.take_profit_price, sltp.take_profit_pct)

            # 6.5 ML Predictor filter
            if _ml_predictor and settings.analyzer_ml_enabled and _ml_predictor.rollout_mode != "off":
                try:
                    from core.models import StrategyTrade as _ST
                    _ml_trade = _ST(
                        trade_id="pending",
                        symbol=symbol,
                        strategy_name=strat_name,
                        market_regime=regime_name,
                        entry_price=features.close,
                        confidence=signal.confidence,
                        hour_of_day=int(time.strftime("%H")),
                        day_of_week=int(time.strftime("%w")),
                        rsi_at_entry=features.rsi_14,
                        adx_at_entry=features.adx,
                        volume_ratio_at_entry=features.volume_ratio,
                        news_sentiment=features.news_sentiment,
                        fear_greed_index=features.fear_greed_index,
                    )
                    _prev_trades_for_ml = []
                    if repo:
                        try:
                            _raw = await asyncio.to_thread(repo.get_strategy_trades, strat_name, limit=20)
                            _prev_trades_for_ml = [_ST(**t) if isinstance(t, dict) else t for t in _raw]
                        except Exception as _ml_hist_err:
                            log.debug("ML history fetch failed: {}", _ml_hist_err)
                    _ml_features = _ml_predictor.extract_features(_ml_trade, _prev_trades_for_ml)
                    _ml_pred = _ml_predictor.predict(_ml_features)
                    log.info("ML prediction: {} prob={:.2f} decision={} mode={}",
                             strat_name, _ml_pred.probability, _ml_pred.decision, _ml_pred.rollout_mode)
                    # Track ML probability at entry so we can record outcome on close.
                    # Store (prob, entry_price) tuple — used in _on_order_filled SELL path.
                    if signal.direction.value == "BUY":
                        _ml_prob_at_entry[symbol] = (_ml_pred.probability, features.close)
                    if _ml_pred.decision == "block":
                        log.info("ML BLOCKED signal: {} {} {} prob={:.2f}",
                                 strat_name, signal.direction.value, symbol, _ml_pred.probability)
                        strat_results.append({"strategy": strat_name, "result": "ml_blocked",
                                              "detail": f"ML blocked: prob={_ml_pred.probability:.2f}"})
                        if repo:
                            try:
                                repo.insert_signal_execution(
                                    timestamp=ts_now, symbol=symbol, strategy_name=strat_name,
                                    direction=signal.direction.value, confidence=signal.confidence,
                                    outcome="ml_blocked", reason=f"ML prob={_ml_pred.probability:.2f}",
                                )
                            except Exception as _ml_audit_err:
                                log.debug("ML block audit write failed: {}", _ml_audit_err)
                        continue
                except Exception as e:
                    log.debug("ML prediction failed: {}", e)

            # 7. Risk check
            daily_pnl = position_manager.total_realized_pnl
            check = risk_sentinel.check_signal(
                signal=signal,
                daily_pnl=daily_pnl,
                open_positions_count=position_manager.open_positions_count,
                total_exposure_pct=(position_manager.total_exposure_usd / position_manager.balance * 100)
                    if position_manager.balance > 0 else 0.0,
                balance=position_manager.balance,
                current_market_price=features.close,
            )

            if not check.approved:
                log.info("Signal REJECTED by Risk: {} {} {} — {}",
                         strat_name, signal.direction.value, symbol, check.reason)
                strat_results.append({"strategy": strat_name, "result": "rejected", "detail": f"{signal.direction.value} rejected: {check.reason}"})
                # Audit trail
                if repo:
                    try:
                        repo.insert_signal_execution(
                            timestamp=ts_now, symbol=symbol, strategy_name=strat_name,
                            direction=signal.direction.value, confidence=signal.confidence,
                            outcome="rejected", reason=check.reason,
                        )
                    except Exception as _rej_audit_err:
                        log.debug("Rejection audit write failed: {}", _rej_audit_err)
                if _alert_monitor:
                    _alert_monitor.record_signal_rejection(check.reason)
                continue

            # 8. Emit signal event
            await bus.emit(EVENT_NEW_SIGNAL, signal)
            if _alert_monitor:
                _alert_monitor.record_signal_accepted()

            # 9. Execute order
            _exec_start = time.time()
            log.info("Signal APPROVED: {} {} {} conf={:.2f} reason={}",
                     strat_name, signal.direction.value, symbol, signal.confidence, signal.reason)
            order_msg = f"{signal.direction.value} conf={signal.confidence:.2f}"
            try:
                order = await executor.execute_order(
                    signal=signal,
                    quantity=signal.suggested_quantity,
                    current_price=features.close,
                )
                if order:
                    _exec_ms = int((time.time() - _exec_start) * 1000)
                    log.info("Order FILLED: {} {} qty={:.6f} @ {:.2f}",
                             order.side.value, order.symbol, order.fill_quantity, order.fill_price)
                    order_msg += f" → FILLED @ ${order.fill_price:.2f}"
                    # Audit trail — filled
                    if repo:
                        try:
                            repo.insert_signal_execution(
                                timestamp=ts_now, symbol=symbol, strategy_name=strat_name,
                                direction=signal.direction.value, confidence=signal.confidence,
                                outcome="filled", reason=signal.reason, latency_ms=_exec_ms,
                            )
                        except Exception as _fill_audit_err:
                            log.debug("Fill audit write failed: {}", _fill_audit_err)
                    if _alert_monitor:
                        _alert_monitor.check_execution_latency(ts_now, ts_now + _exec_ms)
            except Exception as e:
                log.error("Execution error: {}", e)
                order_msg += f" → exec error: {e}"
                # Audit trail — error
                if repo:
                    try:
                        repo.insert_signal_execution(
                            timestamp=ts_now, symbol=symbol, strategy_name=strat_name,
                            direction=signal.direction.value, confidence=signal.confidence,
                            outcome="error", reason=str(e),
                        )
                    except Exception as _err_audit_err:
                        log.debug("Error audit write failed: {}", _err_audit_err)

            strat_results.append({"strategy": strat_name, "result": "signal", "detail": order_msg})
            signal_found = True
            # Только один сигнал на символ за свечу
            break

        # ── Запись в strategy log ──
        _strategy_log.append({
            "ts": ts_now,
            "symbol": symbol,
            "event": "signal" if signal_found else "scan",
            "regime": regime_name,
            "price": round(features.close, 2),
            "active_strategies": active_names,
            "strategies": strat_results,
            "msg": f"Signal found via {strat_results[-1]['strategy']}" if signal_found else f"Scanned {len(active_names)} strategies — no trade",
        })

    # SL/TP проверка на каждый тик
    async def _check_sl_tp(trade):
        """Проверить stop-loss / take-profit на каждый маркет-трейд."""
        if trading_paused or not position_manager or not executor:
            return

        symbol = trade.symbol
        if not position_manager.has_position(symbol):
            return

        trigger = position_manager.check_stop_loss_take_profit(symbol)
        if trigger is None:
            return

        pos = position_manager.get_position(symbol)
        if not pos:
            return

        direction = Direction.SELL
        reason = f"SL/TP triggered: {trigger}"
        signal = Signal(
            timestamp=int(time.time() * 1000),
            symbol=symbol,
            direction=direction,
            confidence=1.0,
            strategy_name=pos.strategy_name or "sl_tp",
            reason=reason,
            suggested_quantity=pos.quantity,
            stop_loss_price=pos.stop_loss_price,
            take_profit_price=pos.take_profit_price,
        )

        try:
            order = await executor.execute_order(
                signal=signal,
                quantity=pos.quantity,
                current_price=trade.price,
            )
            if order:
                log.info("{} {} @ {:.2f} — {}", trigger, symbol, trade.price, reason)
        except Exception as e:
            log.error("SL/TP execution error: {}", e)

    # Импорт Signal для SL/TP
    from core.models import Signal

    # Подписываемся на события
    if feature_builder and strategies:
        bus.subscribe(EVENT_NEW_CANDLE, _on_new_candle)
        bus.subscribe(EVENT_NEW_TRADE, _check_sl_tp)
        log.info("[Module] Trading loop active — listening for {} candles", settings.signal_timeframe)
    else:
        log.warning("[Module] Trading loop DISABLED — strategies not initialized")

    # 18. Web Dashboard
    dashboard = None

    _ALLOWED_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}
    _INTERVAL_LIMITS = {"1m": 120, "5m": 120, "15m": 96, "1h": 96, "4h": 120, "1d": 90}

    def build_market_chart(interval: str = "1m") -> dict:
        primary_symbol = settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"
        if interval not in _ALLOWED_INTERVALS:
            interval = "1m"
        limit = _INTERVAL_LIMITS.get(interval, 120)

        try:
            candles = repo.get_candles(primary_symbol, interval, limit=limit)
        except Exception as _candle_err:
            log.debug("Chart candle fetch failed: {}", _candle_err)
            candles = []

        time_fmt = "%H:%M" if interval in ("1m", "5m", "15m") else (
            "%d %b %H:%M" if interval in ("1h", "4h") else "%d %b"
        )

        # Fallback: aggregate 1m candles into 5m/15m when native candles not yet available
        if len(candles) < 2 and interval in ("5m", "15m"):
            bucket_mins = 5 if interval == "5m" else 15
            raw_limit = bucket_mins * limit
            try:
                raw = repo.get_candles(primary_symbol, "1m", limit=raw_limit)
            except Exception as _agg_err:
                log.debug("Aggregation candle fetch failed: {}", _agg_err)
                raw = []
            if len(raw) >= bucket_mins:
                aggregated = []
                bucket_ms = bucket_mins * 60_000
                for i in range(0, len(raw) - bucket_mins + 1, bucket_mins):
                    chunk = raw[i:i + bucket_mins]
                    t0 = chunk[0]["timestamp"]
                    # align to bucket boundary
                    aligned = (t0 // bucket_ms) * bucket_ms
                    aggregated.append({
                        "timestamp": aligned,
                        "open": chunk[0]["open"],
                        "high": max(float(c["high"]) for c in chunk),
                        "low": min(float(c["low"]) for c in chunk),
                        "close": chunk[-1]["close"],
                        "volume": sum(float(c.get("volume", 0)) for c in chunk),
                    })
                candles = aggregated

        if len(candles) >= 2:
            return {
                "symbol": primary_symbol,
                "interval": interval,
                "source": f"candles_{interval}",
                "candles": [
                    {
                        "t": c["timestamp"],
                        "label": time.strftime(time_fmt, time.localtime(c["timestamp"] / 1000)),
                        "o": float(c["open"]),
                        "h": float(c["high"]),
                        "l": float(c["low"]),
                        "c": float(c["close"]),
                        "v": float(c.get("volume", 0)),
                    }
                    for c in candles
                ],
            }

        # Fallback to trades for 1m only
        if interval == "1m":
            try:
                trades = repo.get_recent_trades(primary_symbol, limit=120)
            except Exception as _trade_err:
                log.debug("Chart trades fetch failed: {}", _trade_err)
                trades = []
            trades = list(reversed(trades[-60:])) if trades else []
            if len(trades) >= 2:
                return {
                    "symbol": primary_symbol,
                    "interval": interval,
                    "source": "trades",
                    "candles": [
                        {
                            "t": t["timestamp"],
                            "label": time.strftime("%H:%M:%S", time.localtime(t["timestamp"] / 1000)),
                            "o": float(t["price"]),
                            "h": float(t["price"]),
                            "l": float(t["price"]),
                            "c": float(t["price"]),
                            "v": 0,
                        }
                        for t in trades
                    ],
                }

        return {
            "symbol": primary_symbol,
            "interval": interval,
            "source": "none",
            "candles": [],
        }

    def _build_indicators_snapshot() -> dict:
        """Собирает текущие значения индикаторов для дашборда."""
        f = _last_features
        # Если ещё нет features от торгового цикла — вычислить напрямую из 1h свечей
        if f is None and repo:
            try:
                from features import indicators as ind
                primary_symbol = settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"
                c1r = repo.get_candles(primary_symbol, "1h", limit=60)
                if not c1r or len(c1r) < 30:
                    return {}
                closes = [float(c["close"]) for c in c1r]
                highs = [float(c["high"]) for c in c1r]
                lows = [float(c["low"]) for c in c1r]
                volumes = [float(c.get("volume", 0)) for c in c1r]

                ema_9 = ind.ema(closes, 9)
                ema_21 = ind.ema(closes, 21)
                ema_50 = ind.ema(closes, 50)
                rsi_val = ind.rsi(closes, 14)
                macd_r = ind.macd(closes, 12, 26, 9)
                adx_val = ind.adx(highs, lows, closes, 14)
                bb = ind.bollinger_bands(closes, 20, 2.0)
                atr_val = ind.atr(highs, lows, closes, 14)
                stoch_rsi = ind.stochastic_rsi(closes, 14, 14)
                vol_ratio = ind.volume_ratio(volumes, 20)
                mom = ind.momentum(closes, 10)

                if ema_9 is None or ema_21 is None or rsi_val is None:
                    return {}

                trend = "neutral"
                if ema_9 > ema_21:
                    trend = "bullish"
                elif ema_9 < ema_21:
                    trend = "bearish"
                adx_v = adx_val or 0.0
                trend_strength = "strong" if adx_v >= 40 else "moderate" if adx_v >= 25 else "weak"

                return {
                    "symbol": primary_symbol,
                    "close": round(closes[-1], 2),
                    "trend": trend,
                    "trend_strength": trend_strength,
                    "ema_9": round(ema_9, 2),
                    "ema_21": round(ema_21, 2),
                    "ema_50": round(ema_50, 2) if ema_50 else 0.0,
                    "rsi_14": round(rsi_val, 2),
                    "macd": round(macd_r[0], 4) if macd_r else 0.0,
                    "macd_signal": round(macd_r[1], 4) if macd_r else 0.0,
                    "macd_histogram": round(macd_r[2], 4) if macd_r else 0.0,
                    "adx": round(adx_v, 2),
                    "bb_upper": round(bb[0], 2) if bb else 0.0,
                    "bb_middle": round(bb[1], 2) if bb else 0.0,
                    "bb_lower": round(bb[2], 2) if bb else 0.0,
                    "bb_bandwidth": round(bb[3], 4) if bb else 0.0,
                    "atr": round(atr_val, 2) if atr_val else 0.0,
                    "volume_ratio": round(vol_ratio, 2) if vol_ratio else 0.0,
                    "stoch_rsi": round(stoch_rsi, 2) if stoch_rsi else 0.0,
                    "momentum": round(mom, 4) if mom else 0.0,
                }
            except Exception as e:
                log.debug("Indicators on-demand calc failed: {}", e)
                return {}
        if f is None:
            return {}
        # Определяем тренд по EMA
        trend = "neutral"
        if f.ema_9 > 0 and f.ema_21 > 0:
            if f.ema_9 > f.ema_21:
                trend = "bullish"
            elif f.ema_9 < f.ema_21:
                trend = "bearish"
        # Сила тренда по ADX
        trend_strength = "weak"
        if f.adx >= 40:
            trend_strength = "strong"
        elif f.adx >= 25:
            trend_strength = "moderate"
        return {
            "symbol": f.symbol,
            "close": round(f.close, 2),
            "trend": trend,
            "trend_strength": trend_strength,
            "ema_9": round(f.ema_9, 2),
            "ema_21": round(f.ema_21, 2),
            "ema_50": round(f.ema_50, 2),
            "rsi_14": round(f.rsi_14, 2),
            "macd": round(f.macd, 4),
            "macd_signal": round(f.macd_signal, 4),
            "macd_histogram": round(f.macd_histogram, 4),
            "adx": round(f.adx, 2),
            "bb_upper": round(f.bb_upper, 2),
            "bb_middle": round(f.bb_middle, 2),
            "bb_lower": round(f.bb_lower, 2),
            "bb_bandwidth": round(f.bb_bandwidth, 4),
            "atr": round(f.atr, 2),
            "volume_ratio": round(f.volume_ratio, 2),
            "stoch_rsi": round(f.stoch_rsi, 2),
            "momentum": round(f.momentum, 4),
        }

    def get_system_state() -> dict:
        position_state = position_manager.get_state() if position_manager else {}
        balance = float(position_state.get("balance", settings.paper_initial_balance))
        pnl_today = float(position_state.get("pnl_today", 0.0))
        risk_metrics = (
            risk_sentinel.get_runtime_metrics(balance=balance)
            if risk_sentinel else {}
        )

        market_data_age_sec = -1.0
        if collector:
            age = collector.last_data_age_sec
            if age != float("inf"):
                market_data_age_sec = round(age, 1)

        # Readiness — прогресс сбора данных для торговли
        readiness = {"ready": False, "pct": 0, "steps": []}
        try:
            from features.feature_builder import MIN_CANDLES_1H, MIN_CANDLES_4H
            primary = settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"
            n_1h = len(repo.get_candles(primary, "1h", limit=MIN_CANDLES_1H)) if repo else 0
            n_4h = len(repo.get_candles(primary, "4h", limit=MIN_CANDLES_4H)) if repo else 0
            pct_1h = min(n_1h / MIN_CANDLES_1H * 100, 100)
            pct_4h = min(n_4h / MIN_CANDLES_4H * 100, 100)
            ws_ok = collector._ws is not None and collector._running if collector else False
            strats_ok = len(strategies) > 0
            overall = (pct_1h * 0.4 + pct_4h * 0.4 + (100 if ws_ok else 0) * 0.1 + (100 if strats_ok else 0) * 0.1)
            readiness = {
                "ready": pct_1h >= 100 and pct_4h >= 100 and ws_ok and strats_ok,
                "pct": round(overall, 1),
                "symbol": primary,
                "timeframe": settings.signal_timeframe,
                "steps": [
                    {"name": "Binance WS", "done": ws_ok, "detail": "connected" if ws_ok else "disconnected"},
                    {"name": "Strategies", "done": strats_ok, "detail": f"{len(strategies)} loaded"},
                    {"name": f"1h candles", "done": pct_1h >= 100, "detail": f"{n_1h}/{MIN_CANDLES_1H}", "pct": round(pct_1h, 1)},
                    {"name": f"4h candles", "done": pct_4h >= 100, "detail": f"{n_4h}/{MIN_CANDLES_4H}", "pct": round(pct_4h, 1)},
                ],
            }
        except Exception as _ready_err:
            log.debug("Readiness check failed: {}", _ready_err)

        return {
            "mode": settings.trading_mode,
            "risk_state": risk_state_machine.state.value if risk_state_machine else "NORMAL",
            "trading_paused": trading_paused,
            "uptime": _format_uptime(),
            "pnl_today": pnl_today,
            "pnl_total": float(position_state.get("pnl_total", 0.0)),
            "open_positions": int(position_state.get("open_positions", 0)),
            "trades_today": int(position_state.get("trades_today", 0)),
            "balance": balance,
            "win_rate": float(position_state.get("win_rate", 0.0)),
            "positions": position_state.get("positions", []),
            "recent_trades": position_state.get("recent_trades", []),
            "pnl_history": position_state.get("pnl_history", []),
            "market_chart": build_market_chart("1m"),
            "activity": {
                "collector": collector.stats if collector else {},
                "events": bus.get_event_stats(),
                "strategies_loaded": list(strategies.keys()) if strategies else [],
                "current_regime": _current_regime.regime.value if _current_regime else None,
            },
            "readiness": readiness,
            "strategy_log": list(_strategy_log),
            "risk_details": {
                "daily_loss": min(pnl_today, 0.0),
                "max_drawdown": float(position_state.get("max_drawdown_pct", 0.0)) / 100,
                "exposure": float(position_state.get("exposure_pct", 0.0)) / 100,
                "trade_freq": int(risk_metrics.get("trades_last_hour", 0)),
                "daily_commission": float(risk_metrics.get("daily_commission", 0.0)),
                "market_data_age_sec": market_data_age_sec,
                "cooldown_remaining_sec": int(risk_metrics.get("cooldown_remaining_sec", 0)),
            },
            "indicators": _build_indicators_snapshot(),
            "strategy_performance": repo.get_strategy_performance() if repo else [],
            "trades_export": repo.get_all_trades_for_export() if repo else [],
            "alerts": _alert_monitor.get_recent_alerts() if _alert_monitor else [],
            "signal_exec_stats": repo.get_signal_execution_stats() if repo else {},
            "ml_predictor": _ml_predictor,
            "ml_retrain_fn": _run_ml_training,
        }

    try:
        from dashboard.app import Dashboard
        dashboard = Dashboard(settings, bus, state_provider=get_system_state)

        async def _handle_stop():
            nonlocal trading_paused
            log.warning("STOP requested from dashboard — trading paused")
            trading_paused = True

        async def _handle_resume():
            nonlocal trading_paused
            log.info("RESUME requested from dashboard — trading resumed")
            trading_paused = False
            if risk_state_machine:
                risk_state_machine.reset()

        async def _handle_kill():
            log.warning("KILL requested from dashboard — shutting down")
            shutdown.trigger()

        dashboard.on_stop = _handle_stop
        dashboard.on_resume = _handle_resume
        dashboard.on_kill = _handle_kill
        dashboard.market_chart_provider = build_market_chart

        # News collector
        from collector.news_collector import NewsCollector
        news_collector = NewsCollector(
            update_interval=300,
            groq_api_key=settings.groq_api_key,
            openrouter_api_key=settings.openrouter_api_key,
            db=db,
        )
        dashboard.news_collector = news_collector
        await news_collector.start()
        log.info("[Module] NewsCollector started (5min interval)")

        await dashboard.start()
        log.info("[Module] Dashboard started on http://localhost:{}", settings.dashboard_port)
    except Exception as e:
        log.warning("[Module] Dashboard failed: {}", e)

    # ── ML Auto-Retrain loop ─────────────────────────────────────────────────
    async def _ml_retrain_loop():
        """Background task: retrain ML model every ANALYZER_ML_RETRAIN_DAYS days."""
        # Small delay so system fully starts before first retrain check
        await asyncio.sleep(60)
        while True:
            try:
                async with _ml_retrain_lock_early:
                    if _ml_predictor:
                        if not _ml_predictor.is_ready:
                            log.info("ML auto-retrain: no model loaded — triggering initial training")
                            await _run_ml_training()
                        elif _ml_predictor.needs_retrain():
                            log.info("ML auto-retrain: model is stale — retraining")
                            await _run_ml_training()
            except Exception as exc:
                log.error("ML retrain loop error: {}", exc)
            # Check every 6 hours
            await asyncio.sleep(6 * 3600)

    ml_retrain_task = asyncio.create_task(_ml_retrain_loop())

    # Periodic equity snapshot (для графика PnL)
    async def _equity_snapshot_loop():
        while True:
            await asyncio.sleep(30)
            if position_manager:
                position_manager._record_equity_snapshot(force=True)

    equity_task = asyncio.create_task(_equity_snapshot_loop())

    # Heartbeat writer (шаг 19)
    heartbeat_task = asyncio.create_task(heartbeat_writer(settings))

    # Monitor background tasks for unexpected failures
    _background_tasks: list[tuple[str, asyncio.Task]] = [
        ("equity_snapshot", equity_task),
        ("heartbeat", heartbeat_task),
        ("ml_retrain", ml_retrain_task),
    ]
    if collector_task:
        _background_tasks.append(("collector", collector_task))

    async def _task_watchdog():
        """Monitor background tasks and log if any fail unexpectedly."""
        while True:
            await asyncio.sleep(10)
            for name, task in _background_tasks:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        log.error("Background task '{}' crashed: {} — system may be inconsistent!", name, exc)
                        if _alert_monitor:
                            _alert_monitor.send_critical_alert(f"Task '{name}' crashed: {exc}")

    watchdog_task = asyncio.create_task(_task_watchdog())

    log.info("🟢 Система запущена. Режим: {} TRADING", settings.trading_mode.upper())

    # Ожидание сигнала остановки
    await shutdown.event.wait()

    # Graceful shutdown sequence
    log.info("Остановка системы...")
    watchdog_task.cancel()
    heartbeat_task.cancel()
    equity_task.cancel()
    ml_retrain_task.cancel()
    await asyncio.gather(watchdog_task, heartbeat_task, equity_task, ml_retrain_task, return_exceptions=True)
    if collector:
        await collector.stop()
    if collector_task:
        collector_task.cancel()
        await asyncio.gather(collector_task, return_exceptions=True)
    if dashboard:
        await dashboard.stop()
    db.close()
    save_state({"stopped_at": int(time.time()), "mode": settings.trading_mode})
    release_pid_lock()
    log.info("🔴 Система остановлена. Состояние сохранено.")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
    finally:
        release_pid_lock()


if __name__ == "__main__":
    main()
