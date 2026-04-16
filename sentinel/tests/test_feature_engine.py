"""Тесты Feature Engine и стратегии EMA Crossover RSI — Phase 5."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Candle, Direction, FeatureVector
from features import indicators as ind
from features.feature_builder import FeatureBuilder
from strategy.ema_crossover_rsi import EMACrossoverRSI, EMAConfig


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_closes(base: float, changes: list[float]) -> list[float]:
    """Генерирует массив цен из базовой и % изменений."""
    prices = [base]
    for pct in changes:
        prices.append(prices[-1] * (1 + pct / 100))
    return prices


def make_candles(
    symbol: str,
    interval: str,
    prices: list[float],
    base_ts: int = 1700000000000,
    interval_ms: int = 3_600_000,
) -> list[Candle]:
    """Генерирует список свечей из цен закрытия."""
    candles = []
    for i, close in enumerate(prices):
        ts = base_ts + i * interval_ms
        high = close * 1.005
        low = close * 0.995
        candles.append(Candle(
            timestamp=ts,
            symbol=symbol,
            interval=interval,
            open=close * 0.999,
            high=high,
            low=low,
            close=close,
            volume=100.0 + i * 0.5,
            trades_count=50,
        ))
    return candles


# ──────────────────────────────────────────────
# Indicator tests
# ──────────────────────────────────────────────

class TestEMA:
    def test_basic(self):
        closes = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08]
        result = ind.ema(closes, 5)
        assert result is not None
        assert 45 < result < 47

    def test_not_enough_data(self):
        assert ind.ema([1, 2, 3], 5) is None

    def test_ema_series_length(self):
        closes = list(range(1, 21))
        series = ind.ema_series(closes, 5)
        # Длина: len(closes) - period + 1
        assert len(series) == 16


class TestRSI:
    def test_rising_prices(self):
        closes = [i * 1.0 for i in range(1, 20)]
        val = ind.rsi(closes, 14)
        assert val is not None
        assert val > 90  # Постоянный рост → RSI → 100

    def test_falling_prices(self):
        closes = [100.0 - i for i in range(20)]
        val = ind.rsi(closes, 14)
        assert val is not None
        assert val < 10

    def test_not_enough_data(self):
        assert ind.rsi([1, 2, 3], 14) is None


class TestMACD:
    def test_basic(self):
        # 50 точек восходящего тренда
        closes = [100 + i * 0.5 for i in range(50)]
        result = ind.macd(closes, 12, 26, 9)
        assert result is not None
        macd_line, signal_line, histogram = result
        assert macd_line > 0  # Восходящий тренд → MACD положительный

    def test_not_enough_data(self):
        assert ind.macd([1, 2, 3], 12, 26, 9) is None


class TestBollingerBands:
    def test_basic(self):
        closes = [100.0 + (i % 5) * 0.2 for i in range(25)]
        result = ind.bollinger_bands(closes, 20, 2.0)
        assert result is not None
        upper, middle, lower, bandwidth = result
        assert upper > middle > lower
        assert bandwidth > 0

    def test_not_enough(self):
        assert ind.bollinger_bands([1, 2], 20) is None


class TestATR:
    def test_basic(self):
        n = 30
        highs = [100 + i + 1 for i in range(n)]
        lows = [100 + i - 1 for i in range(n)]
        closes = [100 + i for i in range(n)]
        val = ind.atr(highs, lows, closes, 14)
        assert val is not None
        assert val > 0

    def test_not_enough(self):
        assert ind.atr([1], [1], [1], 14) is None


class TestADX:
    def test_trending(self):
        # Сильный восходящий тренд
        n = 50
        highs = [100 + i * 2 + 1 for i in range(n)]
        lows = [100 + i * 2 - 1 for i in range(n)]
        closes = [100 + i * 2 for i in range(n)]
        val = ind.adx(highs, lows, closes, 14)
        assert val is not None
        assert val > 20  # Должен указывать на тренд

    def test_not_enough(self):
        assert ind.adx([1, 2], [1, 2], [1, 2], 14) is None


class TestVolume:
    def test_volume_ratio(self):
        vols = [100.0] * 20 + [200.0]
        ratio = ind.volume_ratio(vols, 20)
        assert ratio is not None
        assert 1.9 < ratio < 2.1

    def test_obv_rising(self):
        closes = [100, 101, 102, 103, 104]
        volumes = [10, 10, 10, 10, 10]
        val = ind.obv(closes, volumes)
        assert val is not None
        assert val == 40  # Все вверх → 4 × 10

    def test_stochastic_rsi(self):
        closes = [50 + i * 0.5 for i in range(50)]
        val = ind.stochastic_rsi(closes, 14, 14)
        assert val is not None
        assert 0 <= val <= 100


class TestPriceChange:
    def test_basic(self):
        closes = [100, 105]
        pct = ind.price_change_pct(closes, 1)
        assert pct is not None
        assert abs(pct - 5.0) < 0.01

    def test_not_enough(self):
        assert ind.price_change_pct([100], 1) is None


# ──────────────────────────────────────────────
# FeatureBuilder tests
# ──────────────────────────────────────────────

class TestFeatureBuilder:
    def test_returns_none_on_insufficient_data(self):
        fb = FeatureBuilder()
        candles_1h = make_candles("BTCUSDT", "1h", [67000 + i * 10 for i in range(10)])
        candles_4h = make_candles("BTCUSDT", "4h", [67000 + i * 10 for i in range(10)])
        result = fb.build("BTCUSDT", candles_1h, candles_4h)
        assert result is None

    def test_returns_feature_vector(self):
        fb = FeatureBuilder()
        # Достаточно данных (60 свечей)
        prices = [67000 + i * 20 + (i % 7) * 10 for i in range(60)]
        candles_1h = make_candles("BTCUSDT", "1h", prices)
        candles_4h = make_candles("BTCUSDT", "4h", prices, interval_ms=14_400_000)
        result = fb.build("BTCUSDT", candles_1h, candles_4h)
        assert result is not None
        assert isinstance(result, FeatureVector)
        assert result.symbol == "BTCUSDT"
        assert result.ema_9 > 0
        assert result.ema_21 > 0
        assert 0 <= result.rsi_14 <= 100
        assert result.close > 0


# ──────────────────────────────────────────────
# Strategy EMA Crossover RSI tests
# ──────────────────────────────────────────────

class TestEMACrossoverRSI:
    def _make_features(self, **overrides) -> FeatureVector:
        """Создать FeatureVector с дефолтными значениями."""
        defaults = dict(
            timestamp=1700000000000,
            symbol="BTCUSDT",
            ema_9=67100.0,
            ema_21=67000.0,
            ema_50=66500.0,
            adx=30.0,
            macd=5.0,
            macd_signal=3.0,
            macd_histogram=2.0,
            rsi_14=55.0,
            stoch_rsi=50.0,
            bb_upper=68000.0,
            bb_middle=67000.0,
            bb_lower=66000.0,
            bb_bandwidth=0.03,
            atr=500.0,
            volume=150.0,
            volume_sma_20=100.0,
            volume_ratio=1.5,
            obv=10000.0,
            price_change_1m=0.1,
            price_change_5m=0.5,
            price_change_15m=1.0,
            momentum=2.0,
            spread=0.01,
            close=67100.0,
            market_regime="trending_up",
        )
        defaults.update(overrides)
        return FeatureVector(**defaults)

    def test_buy_signal_on_crossover(self):
        strat = EMACrossoverRSI()
        sym = "BTCUSDT"

        # Первый тик: EMA9 < EMA21 (до кроссовера)
        f1 = self._make_features(ema_9=66900.0, ema_21=67000.0, close=66900.0)
        result1 = strat.generate_signal(f1, has_open_position=False)
        assert result1 is None  # Первый тик — нет данных для crossover

        # Второй тик: EMA9 > EMA21 (кроссовер!) — разница должна быть > ATR*0.3
        f2 = self._make_features(ema_9=67600.0, ema_21=67000.0, close=67100.0, atr=500.0)
        result2 = strat.generate_signal(f2, has_open_position=False)
        assert result2 is not None
        assert result2.direction == Direction.BUY
        # Confidence threshold tuned for grouped_confidence with
        # correlation_penalty=0.12 — weaker evidence groups now contribute
        # with geometric attenuation (no longer full sum), so the bar is
        # lower but the score is a more honest estimate of independent info.
        assert result2.confidence >= 0.72
        assert result2.strategy_name == "ema_crossover_rsi"

    def test_no_buy_if_rsi_overbought(self):
        strat = EMACrossoverRSI()

        f1 = self._make_features(ema_9=66900.0, ema_21=67000.0)
        strat.generate_signal(f1)

        f2 = self._make_features(ema_9=67100.0, ema_21=67000.0, rsi_14=75.0)
        result = strat.generate_signal(f2)
        assert result is None

    def test_no_buy_if_below_ema50(self):
        strat = EMACrossoverRSI()

        f1 = self._make_features(ema_9=66900.0, ema_21=67000.0)
        strat.generate_signal(f1)

        f2 = self._make_features(ema_9=67100.0, ema_21=67000.0, ema_50=68000.0, close=67100.0)
        result = strat.generate_signal(f2)
        assert result is None

    def test_sell_stop_loss(self):
        strat = EMACrossoverRSI()

        # Инициализируем prev_diff
        f1 = self._make_features(ema_9=67100.0, ema_21=67000.0)
        strat.generate_signal(f1, has_open_position=True, entry_price=67000.0)

        # Цена упала -4% → stop-loss
        drop_price = 67000.0 * 0.96
        f2 = self._make_features(close=drop_price)
        result = strat.generate_signal(f2, has_open_position=True, entry_price=67000.0)
        assert result is not None
        assert result.direction == Direction.SELL
        assert "Stop-loss" in result.reason

    def test_sell_take_profit(self):
        strat = EMACrossoverRSI()

        f1 = self._make_features()
        strat.generate_signal(f1, has_open_position=True, entry_price=64000.0)

        profit_price = 64000.0 * 1.07  # +7% exceeds TP 6.25% (R:R 2.5)
        f2 = self._make_features(close=profit_price)
        result = strat.generate_signal(f2, has_open_position=True, entry_price=64000.0)
        assert result is not None
        assert result.direction == Direction.SELL
        assert "Take-profit" in result.reason

    def test_no_buy_if_low_volume(self):
        strat = EMACrossoverRSI()

        f1 = self._make_features(ema_9=66900.0, ema_21=67000.0)
        strat.generate_signal(f1)

        f2 = self._make_features(ema_9=67100.0, ema_21=67000.0, volume_ratio=0.5)
        result = strat.generate_signal(f2)
        assert result is None

    def test_hold_when_no_crossover(self):
        strat = EMACrossoverRSI()

        # EMA9 уже выше EMA21, нет кроссовера
        f1 = self._make_features(ema_9=67100.0, ema_21=67000.0)
        strat.generate_signal(f1)

        f2 = self._make_features(ema_9=67200.0, ema_21=67000.0)  # Всё ещё выше
        result = strat.generate_signal(f2)
        assert result is None  # HOLD
