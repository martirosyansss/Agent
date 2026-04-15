"""
Стратегия V1: EMA Crossover + RSI Filter (Swing Trading, 1h/4h).

Логика:
  BUY:  EMA9 пересекает EMA21 снизу вверх (1h) + RSI < 70 + Volume > avg + цена > EMA50 (4h)
  SELL: EMA9 пересекает EMA21 сверху вниз (1h) + RSI > 30 + Volume > 0.8× avg
        ИЛИ stop-loss / take-profit / trailing stop / time exit

Confidence = нормализованная сумма факторов (0.0 – 1.0).
Сигналы с confidence < min_confidence (0.75) игнорируются.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from core.models import Direction, FeatureVector, Signal
from strategy.base_strategy import (
    BaseStrategy,
    news_confidence_adjustment,
    news_should_accelerate_exit,
    news_should_block_entry,
    news_adjust_sl_tp,
)

log = logger.bind(module="strategy")


@dataclass
class EMAConfig:
    """Параметры EMA Crossover RSI стратегии."""
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    min_volume_ratio: float = 1.0
    stop_loss_pct: float = 2.5
    take_profit_pct: float = 5.0
    trailing_stop_pct: float = 1.5
    trailing_activate_pct: float = 2.5
    max_hold_hours: int = 72
    min_confidence: float = 0.60
    max_position_pct: float = 20.0

    def __post_init__(self):
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 50:
            raise ValueError(f"stop_loss_pct must be (0, 50], got {self.stop_loss_pct}")
        if self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be > 0, got {self.take_profit_pct}")


class EMACrossoverRSI(BaseStrategy):
    """Стратегия V1: EMA Crossover + RSI Filter."""

    NAME = "ema_crossover_rsi"

    def __init__(self, config: EMAConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or EMAConfig()
        # Для детекции crossover храним предыдущее значение EMA-разницы
        self._prev_ema_diff: dict[str, float | None] = {}
        # Trailing stop: track max price per symbol
        self._max_price: dict[str, float] = {}
        # Time exit: track entry timestamp per symbol
        self._entry_ts: dict[str, int] = {}

    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        cfg = self._cfg
        sym = features.symbol
        now_ms = int(time.time() * 1000)

        # ── SELL логика (если есть позиция) ──
        if has_open_position and entry_price is not None:
            sell_signal = self._check_sell(features, entry_price, now_ms)
            if sell_signal:
                # Clean up tracking state
                self._max_price.pop(sym, None)
                self._entry_ts.pop(sym, None)
                return sell_signal

        # ── BUY логика (если нет позиции) ──
        if not has_open_position:
            buy_signal = self._check_buy(features, now_ms)
            if buy_signal:
                # Record entry for trailing stop & time exit
                self._max_price[sym] = features.close
                self._entry_ts[sym] = now_ms
                return buy_signal

        # Обновить prev diff для crossover detection
        self._prev_ema_diff[sym] = features.ema_9 - features.ema_21

        return None

    # ------------------------------------------------------------------
    # BUY
    # ------------------------------------------------------------------

    def _check_buy(self, f: FeatureVector, now_ms: int) -> Optional[Signal]:
        cfg = self._cfg

        # Crossover: EMA9 пересекает EMA21 снизу вверх
        current_diff = f.ema_9 - f.ema_21
        prev_diff = self._prev_ema_diff.get(f.symbol)

        if prev_diff is None:
            self._prev_ema_diff[f.symbol] = current_diff
            return None  # Первый тик — нет данных для crossover

        self._prev_ema_diff[f.symbol] = current_diff

        is_crossover = prev_diff <= 0 and current_diff > 0

        if not is_crossover:
            return None

        # ATR-based crossover threshold — prevent whipsaw in sideways (softened)
        min_cross_threshold = f.atr * 0.1 if f.atr > 0 else f.close * 0.0003
        if current_diff < min_cross_threshold:
            log.debug("{} BUY skip: crossover too weak {:.4f} < {:.4f}", f.symbol, current_diff, min_cross_threshold)
            return None

        # RSI filter
        if f.rsi_14 >= cfg.rsi_overbought:
            log.debug("{} BUY skip: RSI {:.1f} >= {}", f.symbol, f.rsi_14, cfg.rsi_overbought)
            return None

        # Volume filter
        if f.volume_ratio < cfg.min_volume_ratio:
            log.debug("{} BUY skip: vol_ratio {:.2f} < {}", f.symbol, f.volume_ratio, cfg.min_volume_ratio)
            return None

        # Trend filter: цена > EMA50 на 4h
        if f.ema_50 > 0 and f.close < f.ema_50:
            log.debug("{} BUY skip: close {:.2f} < EMA50 {:.2f}", f.symbol, f.close, f.ema_50)
            return None

        # ── Расчёт confidence ──
        confidence = 0.50  # base

        # RSI далёк от overbought (+0.10)
        if f.rsi_14 < 50:
            confidence += 0.10
        elif f.rsi_14 < 60:
            confidence += 0.05

        # Сильный объём (+0.10)
        if f.volume_ratio > 2.0:
            confidence += 0.10
        elif f.volume_ratio > 1.5:
            confidence += 0.05

        # Тренд подтверждён EMA50 (+0.10)
        if f.ema_50 > 0 and f.close > f.ema_50:
            confidence += 0.10

        # MACD подтверждает (+0.10)
        if f.macd_histogram > 0:
            confidence += 0.10

        # ADX сильный тренд (+0.05)
        if f.adx > 25:
            confidence += 0.05

        # Trend alignment (multi-TF) boost (+0.10 / +0.05 / -0.10)
        if hasattr(f, 'trend_alignment'):
            if f.trend_alignment >= 0.8:
                confidence += 0.10
            elif f.trend_alignment >= 0.6:
                confidence += 0.05
            elif f.trend_alignment <= 0.2:
                confidence -= 0.10

        # News: block entry on critical events (black swan / security)
        blocked, block_reason = news_should_block_entry(f)
        if blocked:
            log.debug("{} BUY {}", f.symbol, block_reason)
            return None

        # News confidence adjustment (composite_score, strength, category, F&G)
        news_delta, news_reason = news_confidence_adjustment(f, "buy", "trend")
        confidence += news_delta

        confidence = min(confidence, 0.95)

        if confidence < cfg.min_confidence:
            log.debug("{} BUY skip: confidence {:.2f} < {}", f.symbol, confidence, cfg.min_confidence)
            return None

        # SL / TP (adjusted for news-driven volatility)
        sl_price, tp_price = news_adjust_sl_tp(f, f.close, cfg.stop_loss_pct, cfg.take_profit_pct)

        reasons = []
        reasons.append(f"EMA crossover: EMA9={f.ema_9:.2f} > EMA21={f.ema_21:.2f}")
        reasons.append(f"RSI={f.rsi_14:.1f}")
        reasons.append(f"vol_ratio={f.volume_ratio:.2f}x")
        if f.macd_histogram > 0:
            reasons.append("MACD+")
        if hasattr(f, 'trend_alignment') and f.trend_alignment >= 0.8:
            reasons.append(f"trend_align={f.trend_alignment:.2f}")
        if news_delta != 0:
            reasons.append(news_reason)

        return Signal(
            timestamp=now_ms,
            symbol=f.symbol,
            direction=Direction.BUY,
            confidence=confidence,
            strategy_name=self.NAME,
            reason="; ".join(reasons),
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            features=f,
        )

    # ------------------------------------------------------------------
    # SELL
    # ------------------------------------------------------------------

    def _check_sell(self, f: FeatureVector, entry_price: float, now_ms: int) -> Optional[Signal]:
        cfg = self._cfg
        reasons: list[str] = []

        # Guard: invalid entry price → force exit
        if entry_price <= 0:
            return self._make_sell_signal(f, now_ms, 0.99, ["SAFETY: invalid entry_price"])

        pnl_pct = (f.close - entry_price) / entry_price * 100

        # News-driven emergency exit (critical bearish / security event / profit lock)
        exit_now, exit_conf, exit_reason = news_should_accelerate_exit(f, pnl_pct)
        if exit_now:
            self._max_price.pop(f.symbol, None)
            self._entry_ts.pop(f.symbol, None)
            return self._make_sell_signal(f, now_ms, exit_conf, [exit_reason])

        # Stop-loss
        if pnl_pct <= -cfg.stop_loss_pct:
            reasons.append(f"Stop-loss: {pnl_pct:.2f}% <= -{cfg.stop_loss_pct}%")
            return self._make_sell_signal(f, now_ms, 0.90, reasons)

        # Take-profit
        if pnl_pct >= cfg.take_profit_pct:
            reasons.append(f"Take-profit: {pnl_pct:.2f}% >= +{cfg.take_profit_pct}%")
            return self._make_sell_signal(f, now_ms, 0.90, reasons)

        # Trailing stop: activate after +trailing_activate_pct%, trail at trailing_stop_pct%
        sym = f.symbol
        self._max_price[sym] = max(self._max_price.get(sym, entry_price), f.close)
        if pnl_pct >= cfg.trailing_activate_pct:
            max_p = self._max_price.get(sym, f.close)
            drawdown_from_max = (max_p - f.close) / max_p * 100 if max_p > 0 else 0
            if drawdown_from_max >= cfg.trailing_stop_pct:
                reasons.append(f"Trailing stop: {drawdown_from_max:.1f}% from max (pnl={pnl_pct:+.1f}%)")
                return self._make_sell_signal(f, now_ms, 0.85, reasons)

        # Time-based exit: close stale positions after max_hold_hours
        entry_ts = self._entry_ts.get(sym, 0)
        if entry_ts > 0 and cfg.max_hold_hours > 0:
            hold_ms = now_ms - entry_ts
            hold_hours = hold_ms / (3600 * 1000)
            if hold_hours >= cfg.max_hold_hours and abs(pnl_pct) < cfg.take_profit_pct * 0.5:
                reasons.append(f"Time exit: held {hold_hours:.0f}h > {cfg.max_hold_hours}h (pnl={pnl_pct:+.1f}%)")
                return self._make_sell_signal(f, now_ms, 0.70, reasons)

        # EMA death cross
        current_diff = f.ema_9 - f.ema_21
        prev_diff = self._prev_ema_diff.get(f.symbol)
        self._prev_ema_diff[f.symbol] = current_diff

        if prev_diff is not None and prev_diff >= 0 and current_diff < 0:
            # RSI не перепродан
            if f.rsi_14 > cfg.rsi_oversold:
                if f.volume_ratio > 0.8:
                    reasons.append(f"EMA death cross: EMA9={f.ema_9:.2f} < EMA21={f.ema_21:.2f}")
                    reasons.append(f"RSI={f.rsi_14:.1f}, vol={f.volume_ratio:.2f}x")
                    confidence = 0.75
                    if f.rsi_14 > 60:
                        confidence += 0.05
                    if f.volume_ratio > 1.5:
                        confidence += 0.05
                    return self._make_sell_signal(f, now_ms, confidence, reasons)

        return None

    def _make_sell_signal(
        self,
        f: FeatureVector,
        now_ms: int,
        confidence: float,
        reasons: list[str],
    ) -> Signal:
        return Signal(
            timestamp=now_ms,
            symbol=f.symbol,
            direction=Direction.SELL,
            confidence=min(confidence, 0.95),
            strategy_name=self.NAME,
            reason="; ".join(reasons),
            features=f,
        )
