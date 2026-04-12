"""
Константы системы SENTINEL.
"""

# === Версия ===
VERSION = "1.5.0"
APP_NAME = "SENTINEL"

# === Имена событий (EventBus) ===
EVENT_NEW_TRADE = "new_trade"
EVENT_NEW_CANDLE = "new_candle"
EVENT_NEW_SIGNAL = "new_signal"
EVENT_ORDER_FILLED = "order_filled"
EVENT_POSITION_OPENED = "position_opened"
EVENT_POSITION_CLOSED = "position_closed"
EVENT_RISK_STATE_CHANGED = "risk_state_changed"
EVENT_CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
EVENT_HEARTBEAT = "heartbeat"
EVENT_EMERGENCY_STOP = "emergency_stop"

# === Таймфреймы ===
TIMEFRAME_1M = "1m"
TIMEFRAME_1H = "1h"
TIMEFRAME_4H = "4h"
TIMEFRAME_1D = "1d"

# === WebSocket ===
WS_PING_INTERVAL_SEC = 30
WS_STALE_DATA_TIMEOUT_SEC = 60
WS_RECONNECT_DELAYS = [1, 3, 10, 30, 60]

# === Healthcheck ===
HEALTHCHECK_INTERVAL_SEC = 60
MIN_FREE_DISK_GB = 1.0

# === Logging ===
LOG_ROTATION_SIZE = "10 MB"
LOG_ROTATION_COUNT = 5

# === Данные ===
TRADES_RETENTION_DAYS = 7
CANDLES_RETENTION_DAYS = 90
FIFO_BUFFER_MAX_TRADES = 10_000

# === PID ===
PID_FILE = "sentinel.pid"
STATE_FILE = "data/state.json"
HEARTBEAT_FILE = "data/heartbeat"
