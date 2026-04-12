"""
АБСОЛЮТНЫЕ ЛИМИТЫ — ЗАШИТЫ В КОД, НЕ МЕНЯЮТСЯ ЧЕРЕЗ ENV

Даже если пользователь поставит max_daily_loss=999999 в .env,
система всё равно ограничит до ABSOLUTE_MAX_DAILY_LOSS_USD.
"""

# === ФИНАНСОВЫЕ АБСОЛЮТЫ ===
ABSOLUTE_MAX_DAILY_LOSS_USD: float = 100.0
ABSOLUTE_MAX_ORDER_USD: float = 200.0
ABSOLUTE_MAX_POSITION_PCT: float = 50.0
ABSOLUTE_MAX_EXPOSURE_PCT: float = 80.0
ABSOLUTE_MAX_LEVERAGE: int = 1  # ТОЛЬКО спот, x1

# === ЧАСТОТНЫЕ АБСОЛЮТЫ (V1.2: снижены) ===
ABSOLUTE_MAX_TRADES_PER_HOUR: int = 10
ABSOLUTE_MAX_TRADES_PER_DAY: int = 20

# === РАЗРЕШЁННЫЕ СИМВОЛЫ ===
ALLOWED_SYMBOLS: list[str] = ["BTCUSDT", "ETHUSDT"]

# === ЗАПРЕЩЁННЫЕ API ПРАВА ===
FORBIDDEN_API_PERMISSIONS: list[str] = ["withdraw", "futures", "margin"]

# === ЦЕНА: АДЕКВАТНЫЕ ДИАПАЗОНЫ ===
PRICE_RANGES: dict[str, tuple[float, float]] = {
    "BTCUSDT": (1_000.0, 1_000_000.0),
    "ETHUSDT": (50.0, 100_000.0),
}
