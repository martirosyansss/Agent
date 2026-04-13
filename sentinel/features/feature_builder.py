"""
Feature Builder — собирает FeatureVector из свечей по таймфреймам.

Один вызов build() возвращает готовый FeatureVector или None
(если недостаточно данных для вычисления базовых индикаторов).
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from core.models import Candle, FeatureVector
from features import indicators as ind
from guards.safe_math import safe_value

import math

log = logger.bind(module="features")

# Минимальное количество свечей для стабильного расчёта
MIN_CANDLES_1H = 55   # EMA 50 + запас
MIN_CANDLES_4H = 55
MIN_CANDLES_1D = 55   # для daily EMA 50


class FeatureBuilder:
    """Строит FeatureVector из свечей."""

    def build(
        self,
        symbol: str,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        candles_1d: list[Candle] | None = None,
    ) -> Optional[FeatureVector]:
        """
        Собрать FeatureVector.

        Возвращает None, если недостаточно данных.
        """
        if len(candles_1h) < MIN_CANDLES_1H:
            log.debug("Мало 1h свечей ({}/{})", len(candles_1h), MIN_CANDLES_1H)
            return None
        if len(candles_4h) < MIN_CANDLES_4H:
            log.debug("Мало 4h свечей ({}/{})", len(candles_4h), MIN_CANDLES_4H)
            return None

        # ── Извлекаем массивы ──
        c1h = _extract(candles_1h)
        c4h = _extract(candles_4h)

        closes_1h = c1h["close"]
        highs_1h = c1h["high"]
        lows_1h = c1h["low"]
        volumes_1h = c1h["volume"]

        closes_4h = c4h["close"]
        highs_4h = c4h["high"]
        lows_4h = c4h["low"]
        volumes_4h = c4h["volume"]

        # ── Трендовые (1h) ──
        ema_9 = ind.ema(closes_1h, 9)
        ema_21 = ind.ema(closes_1h, 21)
        ema_50 = ind.ema(closes_4h, 50)  # 4h trend filter

        # ADX (4h — для Strategy Selector)
        adx_val = ind.adx(highs_4h, lows_4h, closes_4h, 14)

        # MACD (1h)
        macd_result = ind.macd(closes_1h, 12, 26, 9)
        macd_val = macd_result[0] if macd_result else 0.0
        macd_sig = macd_result[1] if macd_result else 0.0
        macd_hist = macd_result[2] if macd_result else 0.0

        # ── Осцилляторы (1h) ──
        rsi_val = ind.rsi(closes_1h, 14)
        stoch_rsi_val = ind.stochastic_rsi(closes_1h, 14, 14)

        # ── Волатильность (4h) ──
        bb = ind.bollinger_bands(closes_4h, 20, 2.0)
        bb_upper = bb[0] if bb else 0.0
        bb_middle = bb[1] if bb else 0.0
        bb_lower = bb[2] if bb else 0.0
        bb_bw = bb[3] if bb else 0.0

        atr_val = ind.atr(highs_4h, lows_4h, closes_4h, 14)

        # ── Объём (1h) ──
        vol_sma = ind.volume_sma(volumes_1h, 20)
        vol_ratio = ind.volume_ratio(volumes_1h, 20)
        obv_val = ind.obv(closes_1h, volumes_1h)

        # ── Производные (1h) ──
        pc_1 = ind.price_change_pct(closes_1h, 1)
        pc_5 = ind.price_change_pct(closes_1h, 5)
        pc_15 = ind.price_change_pct(closes_1h, 15)
        mom = ind.momentum(closes_1h, 10)

        # ── Новые индикаторы (Phase 1) ──
        cci_val = ind.cci(highs_1h, lows_1h, closes_1h, 20)
        roc_val = ind.roc(closes_1h, 12)
        vroc_val = ind.vroc(volumes_1h, 12)
        cmf_val = ind.cmf(highs_1h, lows_1h, closes_1h, volumes_1h, 20)
        bb_pct_b_val = ind.bollinger_pct_b(closes_4h, 20, 2.0)
        vwap_val = ind.vwap(closes_1h, volumes_1h, 20)
        hvol_val = ind.historical_volatility(closes_1h, 20)
        dmi_val = ind.dmi_spread(highs_4h, lows_4h, closes_4h, 14)

        # ── Daily timeframe (если есть) ──
        ema_50_daily = None
        rsi_14_daily = None
        if candles_1d and len(candles_1d) >= MIN_CANDLES_1D:
            c1d = _extract(candles_1d)
            ema_50_daily = ind.ema(c1d["close"], 50)
            rsi_14_daily = ind.rsi(c1d["close"], 14)

        # ── Trend alignment (multi-TF) ──
        ta_val = ind.trend_alignment(ema_9, ema_21, closes_1h[-1], ema_50_daily)

        # Проверка: базовые индикаторы должны быть не None
        if any(v is None for v in (ema_9, ema_21, rsi_val)):
            log.debug("Базовые индикаторы не готовы для {}", symbol)
            return None

        fv = FeatureVector(
            timestamp=candles_1h[-1].timestamp,
            symbol=symbol,
            # Трендовые
            ema_9=ema_9,
            ema_21=ema_21,
            ema_50=safe_value(ema_50),
            adx=safe_value(adx_val),
            macd=macd_val,
            macd_signal=macd_sig,
            macd_histogram=macd_hist,
            # Осцилляторы
            rsi_14=rsi_val,
            stoch_rsi=safe_value(stoch_rsi_val),
            # Волатильность
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_bandwidth=bb_bw,
            atr=safe_value(atr_val),
            # Объём
            volume=volumes_1h[-1] if volumes_1h else 0.0,
            volume_sma_20=safe_value(vol_sma),
            volume_ratio=safe_value(vol_ratio),
            obv=safe_value(obv_val),
            # Производные
            price_change_1m=safe_value(pc_1),
            price_change_5m=safe_value(pc_5),
            price_change_15m=safe_value(pc_15),
            momentum=safe_value(mom),
            spread=0.0,  # Обновляется отдельно через order book
            # Текущая цена
            close=closes_1h[-1],
            # Phase 1: Enhanced indicators
            cci=safe_value(cci_val),
            roc=safe_value(roc_val),
            vroc=safe_value(vroc_val),
            cmf=safe_value(cmf_val),
            bb_pct_b=safe_value(bb_pct_b_val) if bb_pct_b_val is not None else 0.5,
            vwap=safe_value(vwap_val),
            hist_volatility=safe_value(hvol_val),
            dmi_spread=safe_value(dmi_val),
            trend_alignment=ta_val,
            # Daily timeframe
            ema_50_daily=safe_value(ema_50_daily),
            rsi_14_daily=safe_value(rsi_14_daily),
        )
        return fv


def _extract(candles: list[Candle]) -> dict[str, list[float]]:
    """Извлечь массивы OHLCV из списка свечей (отсортированных по времени)."""
    return {
        "open": [0.0 if math.isnan(c.open) or math.isinf(c.open) else c.open for c in candles],
        "high": [0.0 if math.isnan(c.high) or math.isinf(c.high) else c.high for c in candles],
        "low": [0.0 if math.isnan(c.low) or math.isinf(c.low) else c.low for c in candles],
        "close": [0.0 if math.isnan(c.close) or math.isinf(c.close) else c.close for c in candles],
        "volume": [0.0 if math.isnan(c.volume) or math.isinf(c.volume) else c.volume for c in candles],
    }
