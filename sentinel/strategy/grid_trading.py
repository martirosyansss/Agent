"""
Стратегия V2: Grid Trading (ATR-adaptive, BB-based range).

Логика:
  Авто-определение границ через Bollinger Bands (20, 2.0).
  N уровней от lower BB до upper BB, spacing адаптируется к ATR.
  BUY: цена ниже grid level + spread filter + regime check.
  SELL: ATR-based TP (min 0.5%) + trailing + SL + max hold.

Защита:
  - max 30% капитала
  - стоп при выходе цены за границы >2% (только для НОВЫХ входов;
    открытые позиции продолжают обслуживаться SL/trailing/time-exit)
  - spread filter (не входить при широком спреде)
  - rebuild при изменении волатильности >50%
  - commission-aware TP: cfg.min_profit_pct трактуется как net-after-commission
Режим рынка: sideways (основной)

Состояние стратегии (in-memory) не персистится между рестартами процесса.
После рестарта `_filled_buys` обнуляется → возможен повторный заход на
уровень в текущем 24h-цикле; `_entry_ts` тоже сбрасывается → таймер
max_hold_hours стартует заново для уже открытой позиции. Если это станет
проблемой — переехать на StrategyStateRepository по аналогии с DCA Bot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from core.models import Direction, FeatureVector, Signal
from monitoring.event_log import emit_rejection
from strategy.base_strategy import (
    BaseStrategy,
    news_confidence_adjustment,
    news_should_block_entry,
    news_should_accelerate_exit,
    news_adjust_sl_tp,
    adaptive_min_confidence,
)

COMMISSION_ROUND_TRIP_PCT = 0.20  # Binance spot: 0.1% maker + 0.1% taker


def _news_sl_tp_multipliers(f: FeatureVector) -> tuple[float, float]:
    """Зеркало логики мультипликаторов из ``news_adjust_sl_tp``.

    Нужно, чтобы внутренние пороги выхода (pnl_pct vs sl/tp_pct) совпадали
    с абсолютными SL/TP, которые мы кладём в Signal для executor — иначе
    источник истины двоится и срабатывания расходятся.
    """
    sl_mult = 1.0
    tp_mult = 1.0

    impact = abs(f.news_impact_pct)
    score = f.news_composite_score

    if impact > 2.0:
        sl_mult = 1.15
        tp_mult = 1.25
    elif impact > 1.0:
        sl_mult = 1.08
        tp_mult = 1.12

    if f.news_critical_alert and score < -0.2:
        sl_mult = 0.75

    if score > 0.3 and f.news_actionable:
        tp_mult = max(tp_mult, 1.15)

    return sl_mult, tp_mult


@dataclass
class GridConfig:
    num_grids: int = 8
    capital_pct: float = 30.0
    min_profit_pct: float = 1.2          # NET после 0.2% round-trip комиссии (R:R ≥ 1.4 vs SL=4%)
    max_loss_pct: float = 4.0            # tighter SL for grid (was 5%)
    trailing_activate_pct: float = 1.2   # activate trailing at +1.2%
    trailing_stop_pct: float = 0.4       # tight trail for grid scalping
    max_hold_hours: int = 48             # prevent stuck positions
    min_confidence: float = 0.80
    min_volume_ratio: float = 0.8
    max_spread_pct: float = 0.15         # skip entry when spread > 0.15%
    rebuild_vol_change_pct: float = 50.0 # rebuild grid when BB width changes >50%

    def __post_init__(self):
        if self.num_grids <= 0:
            raise ValueError(f"num_grids must be > 0, got {self.num_grids}")
        if self.max_loss_pct <= 0 or self.max_loss_pct > 50:
            raise ValueError(f"max_loss_pct must be (0, 50], got {self.max_loss_pct}")


class GridTrading(BaseStrategy):
    """Стратегия V2: Grid Trading — ATR-adaptive grid."""

    NAME = "grid_trading"

    def __init__(self, config: GridConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or GridConfig()
        self._grid_levels: dict[str, list[float]] = {}
        self._last_rebuild: dict[str, int] = {}
        # Уровни считаются заполненными до следующей перестройки сетки
        # (по таймеру 24h или vol-shift >50%) — это намеренное классическое
        # поведение grid-стратегии, а не утечка состояния.
        self._filled_buys: dict[str, set[int]] = {}
        self._bb_width_at_build: dict[str, float] = {}  # track BB width for rebuild trigger
        self._max_price: dict[str, float] = {}           # trailing stop
        self._entry_ts: dict[str, int] = {}               # hold time tracking
        # SL/TP-мультипликаторы, зафиксированные на момент входа: используем
        # одни и те же значения и для абсолютных цен в Signal, и для
        # внутренних %-проверок выхода — иначе executor и стратегия будут
        # принимать решения по разным порогам.
        self._news_sl_mult: dict[str, float] = {}
        self._news_tp_mult: dict[str, float] = {}

    def _build_grid(self, f: FeatureVector) -> list[float]:
        """Build grid levels from lower BB to upper BB."""
        low = f.bb_lower
        high = f.bb_upper
        if high <= low or low <= 0 or self._cfg.num_grids <= 0:
            return []
        step = (high - low) / self._cfg.num_grids
        return [low + step * i for i in range(self._cfg.num_grids + 1)]

    def _should_rebuild(self, sym: str, now_ms: int, features: FeatureVector) -> bool:
        """Rebuild grid on timer (24h) OR when volatility shifts >50%.

        Возвращает False, если для символа ещё не было первой сборки —
        вызывающий код самостоятельно строит сетку через ``sym not in
        self._grid_levels``. Это убирает зависимость от того, что
        ``last_rebuild=0`` всегда даёт truthy diff.
        """
        last = self._last_rebuild.get(sym)
        if last is None:
            return False
        if (now_ms - last) > 24 * 3600 * 1000:
            return True
        # Volatility-based rebuild
        old_width = self._bb_width_at_build.get(sym, 0)
        if old_width > 0 and features.bb_bandwidth > 0:
            change_pct = abs(features.bb_bandwidth - old_width) / old_width * 100
            if change_pct > self._cfg.rebuild_vol_change_pct:
                return True
        return False

    def _cleanup(self, sym: str) -> None:
        self._max_price.pop(sym, None)
        self._entry_ts.pop(sym, None)
        self._news_sl_mult.pop(sym, None)
        self._news_tp_mult.pop(sym, None)

    def _atr_based_tp(self, features: FeatureVector, tp_mult: float = 1.0) -> float:
        """ATR-based take-profit, NET после комиссий.

        ``cfg.min_profit_pct`` уже трактуется как net-target (1.2% при
        round-trip 0.2%). News-driven ``tp_mult`` ≥ 1.0 расширяет цель
        синхронно с тем, что мы передали в Signal.
        """
        if features.atr > 0 and features.close > 0:
            atr_pct = features.atr / features.close * 100
            # Grid TP = 35% of ATR (capture a fraction of expected range)
            raw_tp = atr_pct * 0.35
        else:
            raw_tp = self._cfg.min_profit_pct
        net_tp = max(raw_tp, self._cfg.min_profit_pct)
        return net_tp * tp_mult

    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        cfg = self._cfg
        sym = features.symbol
        now_ms = features.timestamp or int(time.time() * 1000)

        # Rebuild grid if needed (timer or volatility shift)
        if sym not in self._grid_levels or self._should_rebuild(sym, now_ms, features):
            levels = self._build_grid(features)
            if not levels:
                return None
            self._grid_levels[sym] = levels
            self._last_rebuild[sym] = now_ms
            self._filled_buys[sym] = set()
            self._bb_width_at_build[sym] = features.bb_bandwidth

        levels = self._grid_levels[sym]
        if not levels:
            return None

        price = features.close
        grid_low, grid_high = levels[0], levels[-1]

        # ── SELL ──
        # ВАЖНО: exit-логика должна работать даже если цена ушла за границы
        # сетки — иначе позиция «замораживается» и SL/trailing не сработают.
        if has_open_position and entry_price is not None:
            if entry_price <= 0:
                self._cleanup(sym)
                emit_rejection(
                    "grid_trading", "invalid entry_price",
                    symbol=sym, entry_price=entry_price,
                )
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.99, strategy_name=self.NAME,
                              reason=f"SAFETY: invalid entry_price={entry_price}",
                              features=features)
            pnl_pct = (price - entry_price) / entry_price * 100
            self._max_price[sym] = max(self._max_price.get(sym, entry_price), price)

            # News-driven emergency exit
            exit_now, exit_conf, exit_reason = news_should_accelerate_exit(features, pnl_pct)
            if exit_now:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=exit_conf, strategy_name=self.NAME,
                              reason=exit_reason, features=features)

            # Внутренние пороги выхода используют те же news-мультипликаторы,
            # что и абсолютные SL/TP в Signal — чтобы executor и strategy
            # принимали решения по одним и тем же значениям.
            sl_mult = self._news_sl_mult.get(sym, 1.0)
            tp_mult = self._news_tp_mult.get(sym, 1.0)
            tp_pct = self._atr_based_tp(features, tp_mult=tp_mult)
            sl_pct = cfg.max_loss_pct * sl_mult

            if pnl_pct >= tp_pct:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.75, strategy_name=self.NAME,
                              reason=f"Grid TP: PnL {pnl_pct:.2f}% >= {tp_pct:.2f}% (ATR-based)",
                              features=features)

            # Trailing stop
            max_p = self._max_price.get(sym, entry_price)
            max_gain = (max_p - entry_price) / entry_price * 100
            if max_gain >= cfg.trailing_activate_pct:
                drop = (max_p - price) / max_p * 100
                if drop >= cfg.trailing_stop_pct:
                    self._cleanup(sym)
                    return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                                  confidence=0.78, strategy_name=self.NAME,
                                  reason=f"Grid trailing: drop {drop:.1f}% from max",
                                  features=features)

            # Stop loss
            if pnl_pct <= -sl_pct:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.90, strategy_name=self.NAME,
                              reason=f"Grid SL: PnL {pnl_pct:.2f}% <= -{sl_pct:.2f}%",
                              features=features)

            # Max hold time
            entry_ts = self._entry_ts.get(sym, now_ms)
            hours_held = (now_ms - entry_ts) / 3_600_000
            if hours_held >= cfg.max_hold_hours and pnl_pct < tp_pct * 0.5:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.65, strategy_name=self.NAME,
                              reason=f"Grid time exit: held {hours_held:.0f}h, pnl={pnl_pct:+.1f}%",
                              features=features)

            return None

        # ── BUY ──
        if not has_open_position:
            # Safety: новые входы запрещены, если цена улетела за пределы сетки.
            # Открытые позиции эту проверку обходят (см. SELL-блок выше).
            if price < grid_low * 0.98 or price > grid_high * 1.02:
                emit_rejection(
                    "grid_trading", "price outside grid range",
                    symbol=sym, price=price,
                    grid_low=grid_low, grid_high=grid_high,
                )
                return None

            # Sideways invalidation: если цена ушла НИЖЕ нижней BB полосы —
            # боковик сломан, дальнейшее усреднение = ловля падающего ножа
            # (живой кейс ETH: 4 BUY подряд при rsi=42→22, последний вход
            # close=2263 < bb_lower=2266, все закрылись по −0.7%).
            if price < features.bb_lower:
                emit_rejection(
                    "grid_trading", "price below BB lower (sideways invalidated)",
                    symbol=sym, price=price, bb_lower=features.bb_lower,
                )
                return None

            # Bearish-trend stack: ema_9 < ema_21 < ema_50 при цене ниже ema_50 —
            # формальный регим может оставаться sideways по ADX, но скользящие
            # уже выстроились медвежьим веером. Mean-reversion в такой среде
            # стабильно сливает.
            if (features.ema_9 > 0 and features.ema_21 > 0 and features.ema_50 > 0
                    and features.ema_9 < features.ema_21 < features.ema_50
                    and price < features.ema_50):
                emit_rejection(
                    "grid_trading", "bearish ema stack",
                    symbol=sym, price=price,
                    ema_9=features.ema_9, ema_21=features.ema_21, ema_50=features.ema_50,
                )
                return None

            # Falling knife: oversold + сильное превосходство −DI. RSI<30 сам по
            # себе для grid — нормальная зона входа, но при dmi_spread < −5
            # это означает, что давление продавцов не остыло.
            if features.rsi_14 > 0 and features.rsi_14 < 30 and features.dmi_spread < -5:
                emit_rejection(
                    "grid_trading", "falling knife (oversold + strong bears)",
                    symbol=sym,
                    rsi=features.rsi_14, dmi_spread=features.dmi_spread,
                )
                return None

            # News block (стандартный — black swan, security)
            blocked, block_reason = news_should_block_entry(features)
            if blocked:
                emit_rejection(
                    "grid_trading", block_reason,
                    symbol=sym, direction="buy",
                )
                return None

            # Bullish critical news: для mean-reversion стратегии сильная
            # позитивная новость = цена побежит вверх, не вернётся к
            # нижним уровням сетки. Не входим — `news_should_block_entry`
            # ловит только медвежьи critical alerts.
            if (features.news_critical_alert
                    and features.news_composite_score > 0.3
                    and features.news_actionable):
                emit_rejection(
                    "grid_trading", "bullish critical news (no reversion)",
                    symbol=sym, direction="buy",
                    news_score=features.news_composite_score,
                )
                return None

            # Volume filter
            if features.volume_ratio < cfg.min_volume_ratio:
                # Ноль-вход без подтверждения объёма не считаем "missed
                # opportunity" — слишком шумно. Молчаливый skip.
                return None

            # Spread filter: don't enter when spread is too wide
            spread_pct = features.spread / price * 100 if price > 0 and features.spread > 0 else 0
            if spread_pct > cfg.max_spread_pct:
                emit_rejection(
                    "grid_trading", "spread too wide",
                    symbol=sym, direction="buy",
                    spread_pct=round(spread_pct, 4),
                    max_spread_pct=cfg.max_spread_pct,
                )
                return None

            filled = self._filled_buys.get(sym, set())
            for i, level in enumerate(levels):
                if i in filled:
                    continue
                if price <= level:
                    self._filled_buys.setdefault(sym, set()).add(i)

                    # News-adjusted SL/TP — те же мультипликаторы попадут
                    # и во внутренние exit-проверки (см. SELL-блок).
                    sl_mult, tp_mult = _news_sl_tp_multipliers(features)
                    sl, tp = news_adjust_sl_tp(features, price, cfg.max_loss_pct, cfg.min_profit_pct)

                    # Confidence: базовый 0.80 — сейчас совпадает с
                    # adaptive_min_confidence("grid", "sideways") = 0.80 −
                    # 0.05 = 0.75, без новостей фильтр пропускает. С 0.75
                    # (как было раньше) стратегия в своём целевом режиме не
                    # стреляла.
                    # ВАЖНО: grid — контртрендовая (mean-reversion) стратегия;
                    # дефолтный trend-вес инвертирует знак (бычьи новости
                    # повышали бы уверенность в покупке низа сетки, хотя
                    # цена при этом убегает вверх и не возвращается).
                    news_delta, news_reason = news_confidence_adjustment(
                        features, direction="buy", strategy_type="mean_reversion",
                    )
                    grid_conf = max(0.0, min(1.0, 0.80 + news_delta))

                    regime = getattr(features, 'market_regime', 'unknown')
                    eff_threshold = adaptive_min_confidence(cfg.min_confidence, regime, "grid")
                    if grid_conf < eff_threshold:
                        emit_rejection(
                            "grid_trading", "confidence below threshold",
                            symbol=sym, direction="buy",
                            confidence=round(grid_conf, 3),
                            threshold=round(eff_threshold, 3),
                            level_index=i, regime=regime,
                        )
                        continue

                    self._entry_ts[sym] = now_ms
                    self._max_price[sym] = price
                    self._news_sl_mult[sym] = sl_mult
                    self._news_tp_mult[sym] = tp_mult

                    reason = f"Grid BUY at level {i}/{len(levels)-1}, price={price:.2f}"
                    if news_delta != 0:
                        reason += f", {news_reason}"

                    return Signal(
                        timestamp=now_ms, symbol=sym, direction=Direction.BUY,
                        confidence=grid_conf, strategy_name=self.NAME,
                        reason=reason, stop_loss_price=sl, take_profit_price=tp,
                        features=features,
                    )

        return None
