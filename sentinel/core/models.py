"""
Базовые модели данных SENTINEL — dataclasses для всей системы.

Все модули оперируют этими типами; они не содержат бизнес-логики.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    REDUCED = "REDUCED"
    SAFE = "SAFE"
    STOP = "STOP"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class MarketRegimeType(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────
# Market data
# ──────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MarketTrade:
    """Сырая сделка с биржи."""
    timestamp: int        # Unix ms
    symbol: str           # "BTCUSDT"
    price: float
    quantity: float
    is_buyer_maker: bool  # True = продажа, False = покупка


@dataclass(frozen=True, slots=True)
class Candle:
    """OHLCV свеча."""
    timestamp: int    # Unix ms начала свечи
    symbol: str
    interval: str     # "1m", "1h", "4h", "1d"
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades_count: int = 0


# ──────────────────────────────────────────────
# Feature Engine output
# ──────────────────────────────────────────────

@dataclass(slots=True)
class FeatureVector:
    """Результат Feature Engine — снимок индикаторов."""
    timestamp: int
    symbol: str

    # Трендовые
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    adx: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # Осцилляторы
    rsi_14: float = 0.0
    stoch_rsi: float = 0.0

    # Волатильность
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_bandwidth: float = 0.0
    atr: float = 0.0

    # Объём
    volume: float = 0.0
    volume_sma_20: float = 0.0
    volume_ratio: float = 0.0
    obv: float = 0.0

    # Производные
    price_change_1m: float = 0.0
    price_change_5m: float = 0.0
    price_change_15m: float = 0.0
    momentum: float = 0.0
    spread: float = 0.0

    # Текущая цена (удобный доступ)
    close: float = 0.0

    # News sentiment (от NewsCollector)
    news_sentiment: float = 0.0       # -1.0 .. +1.0 (overall weighted score)
    fear_greed_index: int = 50         # 0-100 (Fear & Greed Index)
    news_impact_pct: float = 0.0      # средний impact_pct новостей
    high_impact_news: int = 0          # количество новостей с |impact| >= 1.5%
    # Pro news fields
    news_composite_score: float = 0.0  # -1.0..+1.0 weighted effective_impact
    news_signal_strength: float = 0.0  # 0.0..1.0 agreement + data depth
    news_critical_alert: bool = False  # есть ли critical urgency
    news_actionable: bool = False      # достаточно ли strong для торговли
    news_dominant_category: str = ""   # macro/regulatory/adoption/technical/etc

    # Phase 1: Enhanced indicators
    cci: float = 0.0                   # Commodity Channel Index
    roc: float = 0.0                   # Rate of Change 12-period (%)
    vroc: float = 0.0                  # Volume Rate of Change (%)
    cmf: float = 0.0                   # Chaikin Money Flow (-1..+1)
    bb_pct_b: float = 0.5             # Bollinger %B (0..1)
    vwap: float = 0.0                  # Volume Weighted Average Price
    hist_volatility: float = 0.0       # Historical Volatility (std of returns)
    dmi_spread: float = 0.0            # +DI minus -DI
    trend_alignment: float = 0.5       # Multi-TF trend alignment (0..1)

    # Daily timeframe
    ema_50_daily: float = 0.0          # EMA 50 на дневном TF
    rsi_14_daily: float = 0.0          # RSI 14 на дневном TF


# ──────────────────────────────────────────────
# Strategy / Signals
# ──────────────────────────────────────────────

@dataclass(slots=True)
class Signal:
    """Торговый сигнал, генерируемый стратегией."""
    timestamp: int
    symbol: str
    direction: Direction
    confidence: float
    strategy_name: str
    reason: str
    suggested_quantity: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    features: Optional[FeatureVector] = None
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# ──────────────────────────────────────────────
# Orders
# ──────────────────────────────────────────────

@dataclass(slots=True)
class Order:
    """Ордер (paper или live)."""
    timestamp: int
    symbol: str
    side: Direction
    order_type: OrderType
    quantity: float
    price: Optional[float] = None        # None для MARKET
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    commission: float = 0.0
    is_paper: bool = True
    signal_id: str = ""
    strategy_name: str = ""
    signal_reason: str = ""
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# ──────────────────────────────────────────────
# Positions
# ──────────────────────────────────────────────

@dataclass(slots=True)
class Position:
    """Открытая или закрытая позиция."""
    symbol: str
    side: str = "LONG"
    entry_price: float = 0.0
    quantity: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    strategy_name: str = ""
    signal_id: str = ""
    signal_reason: str = ""
    status: PositionStatus = PositionStatus.OPEN
    opened_at: str = ""
    closed_at: Optional[str] = None
    is_paper: bool = True
    position_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# ──────────────────────────────────────────────
# Strategy trade (завершённая сделка для Analyzer)
# ──────────────────────────────────────────────

@dataclass(slots=True)
class StrategyTrade:
    """Завершённая сделка стратегии — основа для Trade Analyzer."""
    trade_id: str
    signal_id: Optional[str] = None
    symbol: str = ""
    strategy_name: str = ""
    market_regime: str = ""
    timestamp_open: str = ""
    timestamp_close: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    is_win: bool = False
    confidence: float = 0.0
    hour_of_day: int = 0
    day_of_week: int = 0
    rsi_at_entry: float = 0.0
    adx_at_entry: float = 0.0
    volume_ratio_at_entry: float = 0.0
    exit_reason: str = ""
    hold_duration_hours: float = 0.0
    max_drawdown_during_trade: float = 0.0
    max_profit_during_trade: float = 0.0
    commission_usd: float = 0.0
    news_sentiment: float = 0.0       # -1.0 .. +1.0 at entry
    fear_greed_index: int = 50         # 0-100 at entry
    ema_9_at_entry: float = 0.0
    ema_21_at_entry: float = 0.0
    bb_bandwidth_at_entry: float = 0.0
    macd_histogram_at_entry: float = 0.0
    atr_at_entry: float = 0.0


# ──────────────────────────────────────────────
# Risk
# ──────────────────────────────────────────────

@dataclass(slots=True)
class RiskCheckResult:
    """Результат проверки Risk Sentinel."""
    approved: bool
    reason: str = ""
    state: RiskState = RiskState.NORMAL
    daily_pnl: float = 0.0


# ──────────────────────────────────────────────
# Market Regime
# ──────────────────────────────────────────────

@dataclass(slots=True)
class MarketRegime:
    """Текущий режим рынка."""
    regime: MarketRegimeType = MarketRegimeType.UNKNOWN
    adx: float = 0.0
    atr_ratio: float = 0.0
    determined_at: int = 0  # Unix ms
