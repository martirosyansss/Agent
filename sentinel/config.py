"""
Загрузка настроек из .env с валидацией через pydantic-settings.

Все настраиваемые лимиты проверяются на соответствие absolute_limits
при инициализации — пользователь НЕ может обойти захардкоженные пределы.
"""

from __future__ import annotations

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

from core.absolute_limits import (
    ABSOLUTE_MAX_DAILY_LOSS_USD,
    ABSOLUTE_MAX_EXPOSURE_PCT,
    ABSOLUTE_MAX_ORDER_USD,
    ABSOLUTE_MAX_POSITION_PCT,
    ABSOLUTE_MAX_TRADES_PER_DAY,
    ABSOLUTE_MAX_TRADES_PER_HOUR,
)


class Settings(BaseSettings):
    """Центральная конфигурация SENTINEL — читает .env при старте."""

    # === Binance ===
    binance_api_key: str = ""
    binance_api_secret: str = ""

    # === Telegram ===
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_pin: str = ""

    # === Trading ===
    trading_mode: str = "paper"
    trading_symbols: list[str] = ["BTCUSDT", "ETHUSDT"]

    # === Risk Limits (настраиваемые, но ≤ absolute limits) ===
    max_daily_loss_usd: float = 50.0
    max_daily_loss_pct: float = 10.0
    max_position_pct: float = 20.0
    max_total_exposure_pct: float = 60.0
    max_open_positions: int = 2
    max_order_usd: float = 100.0
    max_trades_per_hour: int = 2

    # === Circuit Breakers ===
    cb_price_anomaly_pct: float = 5.0
    cb_consecutive_losses: int = 3
    cb_spread_anomaly_pct: float = 0.5
    cb_volume_anomaly_mult: float = 10.0
    cb_api_error_count: int = 5
    cb_latency_threshold_sec: float = 5.0
    cb_balance_mismatch_pct: float = 1.0
    cb_commission_alert_pct: float = 1.0

    # === Data Integrity ===
    max_data_age_sec: int = 30
    price_cross_validation_interval: int = 300

    # === Watchdog ===
    watchdog_heartbeat_interval: int = 10
    watchdog_timeout: int = 120

    # === Strategy (V1.2: swing trading) ===
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 5.0
    min_confidence: float = 0.75
    signal_timeframe: str = "1h"
    trend_timeframe: str = "4h"
    max_trades_per_day: int = 6

    # === Grid Trading (V1.3) ===
    grid_enabled: bool = False
    grid_num_levels: int = 8
    grid_capital_pct: float = 30.0
    grid_auto_range: bool = True
    grid_min_profit_pct: float = 0.3
    grid_max_loss_pct: float = 5.0

    # === Mean Reversion (V1.3) ===
    meanrev_enabled: bool = False
    meanrev_rsi_oversold: float = 25.0
    meanrev_rsi_overbought: float = 75.0
    meanrev_stop_loss_pct: float = 4.0
    meanrev_take_profit_pct: float = 6.0
    meanrev_capital_pct: float = 15.0

    # === Strategy Selector (V1.3) ===
    auto_strategy_selection: bool = False
    regime_check_interval_hours: int = 4
    adx_trending_threshold: float = 25.0
    adx_sideways_threshold: float = 20.0

    # === Bollinger Breakout (V1.4) ===
    bb_breakout_enabled: bool = False
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_volume_confirm_mult: float = 1.5
    bb_squeeze_threshold: float = 0.05
    bb_stop_loss_pct: float = 3.0
    bb_take_profit_pct: float = 6.0
    bb_trailing_stop_pct: float = 2.0

    # === DCA Bot (V1.4) ===
    dca_enabled: bool = False
    dca_base_amount_usd: float = 10.0
    dca_interval_hours: int = 24
    dca_max_daily_buys: int = 3
    dca_max_invested_pct: float = 40.0
    dca_stop_drawdown_pct: float = 15.0
    dca_take_profit_pct: float = 8.0
    dca_partial_tp_pct: float = 5.0

    # === MACD Divergence (V1.4) ===
    macd_div_enabled: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal_period: int = 9
    macd_lookback_candles: int = 30
    macd_div_stop_loss_pct: float = 3.5
    macd_div_take_profit_pct: float = 7.0
    macd_require_rsi_confirm: bool = True
    macd_require_vol_confirm: bool = True

    # === Trade Analyzer (V1.5) ===
    analyzer_stats_enabled: bool = True
    analyzer_weekly_report: bool = True
    analyzer_monthly_report: bool = True
    analyzer_optimizer_enabled: bool = False
    analyzer_min_trades: int = 100
    analyzer_max_changes_week: int = 1
    analyzer_paper_test_days: int = 14
    analyzer_ml_enabled: bool = False
    analyzer_ml_shadow_mode: bool = True
    analyzer_min_trades_ml: int = 500
    analyzer_ml_retrain_days: int = 30
    analyzer_ml_block_threshold: float = 0.40
    analyzer_ml_history_days: int = 180
    analyzer_ml_test_window_days: int = 60
    analyzer_ml_min_skill_score: float = 0.55
    analyzer_ml_min_precision: float = 0.55
    analyzer_ml_min_recall: float = 0.50
    analyzer_ml_min_roc_auc: float = 0.58

    # === Paper Trading ===
    paper_initial_balance: float = 500.0
    paper_commission_pct: float = 0.1
    paper_slippage_pct: float = 0.05

    # === System ===
    log_level: str = "INFO"
    dashboard_port: int = 8080
    dashboard_password: str = ""
    db_path: str = "data/sentinel.db"
    db_backup_interval_hours: int = 6
    max_ram_mb: int = 2048

    # === Cooling Period ===
    resume_cooldown_min: int = 30
    live_first_day_max_order: float = 20.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ------------------------------------------------------------------
    # Валидация: настраиваемые лимиты ≤ absolute limits
    # ------------------------------------------------------------------

    @field_validator("trading_mode")
    @classmethod
    def validate_trading_mode(cls, v: str) -> str:
        if v not in ("paper", "live"):
            raise ValueError("trading_mode must be 'paper' or 'live'")
        return v

    @model_validator(mode="after")
    def clamp_to_absolute_limits(self) -> "Settings":
        """Зажимаем пользовательские лимиты до абсолютных максимумов."""
        self.max_daily_loss_usd = min(self.max_daily_loss_usd, ABSOLUTE_MAX_DAILY_LOSS_USD)
        self.max_order_usd = min(self.max_order_usd, ABSOLUTE_MAX_ORDER_USD)
        self.max_position_pct = min(self.max_position_pct, ABSOLUTE_MAX_POSITION_PCT)
        self.max_total_exposure_pct = min(self.max_total_exposure_pct, ABSOLUTE_MAX_EXPOSURE_PCT)
        self.max_trades_per_hour = min(self.max_trades_per_hour, ABSOLUTE_MAX_TRADES_PER_HOUR)
        self.max_trades_per_day = min(self.max_trades_per_day, ABSOLUTE_MAX_TRADES_PER_DAY)
        return self


def load_settings() -> Settings:
    """Создаёт и возвращает провалидированные настройки."""
    return Settings()
