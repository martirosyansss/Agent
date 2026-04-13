# SENTINEL V1.5 — ПОЛНЫЙ АУДИТ ПРОГНОЗИРОВАНИЯ V3

> **Дата**: 13 апреля 2026  
> **Формат**: Профессиональный трейдерский аудит  
> **Методология**: Построчный анализ каждого модуля, влияющего на Win Rate  
> **Цель**: Определить ВСЕ факторы, снижающие точность, и дать пошаговый план повышения Win Rate

---

## EXECUTIVE SUMMARY

### Текущее состояние системы

| Метрика | Статус | Комментарий |
|---------|--------|-------------|
| Архитектура | ✅ Хорошая | 6 стратегий, ML фильтр, 8 CB, news sentiment, 30+ features |
| Интеграция | ⚠️ Частичная | Ключевые модули подключены, но ML НЕ ВЫЗЫВАЕТСЯ НИГДЕ в main.py |
| Daily Candles | ✅ Исправлено | `build(symbol, candles_1h, candles_4h, candles_1d)` — передаются |
| Auto Strategy Selection | ✅ Исправлено | `auto_strategy_selection: bool = True` в config.py |
| Kelly Criterion | ✅ Частично исправлено | Реальная статистика подтягивается из БД (≥10 сделок) |
| ML Predictor | 🔴 НЕ ПОДКЛЮЧЕН | НИ ОДНОЙ ссылки на ML в main.py — полностью мёртвый код |
| AdaptiveAllocator | ⚠️ Создан, не вызывается | Инициализируется, но `.update_skills()` и `.get_adaptive_allocations()` не вызываются |
| Forward-Looking Bias | ✅ Исправлено | `extract_features()` использует только pre-trade данные |
| Бэктест | ⚠️ Без daily candles | `engine.run()` не принимает candles_1d — бэктест не использует дневной тренд |

### Оценочный Win Rate

| Сценарий | Win Rate | Profit Factor |
|----------|----------|---------------|
| Сейчас (только EMA + auto select) | ~52-55% | ~1.1-1.3 |
| + ML фильтрация (shadow→block) | ~60-65% | ~1.5-1.8 |
| + AdaptiveAllocator | ~62-67% | ~1.6-2.0 |
| + Все улучшения из этого аудита | ~65-72% | ~1.8-2.5 |

---

## РАЗДЕЛ 1: КРИТИЧЕСКИЕ ПРОБЛЕМЫ (БЛОКЕРЫ WIN RATE)

### 🔴 ПРОБЛЕМА #1: ML Predictor НЕ ПОДКЛЮЧЕН В MAIN.PY

**Уровень критичности**: МАКСИМАЛЬНЫЙ  
**Влияние на Win Rate**: +8-15% при правильном подключении

**Факт**: В `main.py` отсутствует ЛЮБОЕ упоминание ML. Строку `"ml"`, `"ML"`, `"predict"`, `"analyzer"`, `"shadow"` — ничего нет.

**Что написано в коде:**
- `config.py`: `analyzer_ml_enabled: bool = True` — настройка есть
- `config.py`: `analyzer_ml_shadow_mode: bool = True` — тоже есть
- `analyzer/ml_predictor.py`: Полный класс `MLPredictor` с 20 features, RandomForest, TimeSeriesCV
- `main.py`: **НОЛЬ** импортов, **НОЛЬ** вызовов

**Что должно быть:**
```python
# В main.py — инициализация
from analyzer.ml_predictor import MLPredictor, MLConfig
ml_predictor = MLPredictor(MLConfig(
    block_threshold=settings.analyzer_ml_block_threshold,
    min_precision=settings.analyzer_ml_min_precision,
    min_recall=settings.analyzer_ml_min_recall,
    min_roc_auc=settings.analyzer_ml_min_roc_auc,
    min_skill_score=settings.analyzer_ml_min_skill_score,
    min_trades=settings.analyzer_min_trades_ml,
))

# В торговом цикле — после генерации сигнала и перед Risk Check:
if settings.analyzer_ml_enabled and ml_predictor.is_ready:
    prediction = ml_predictor.predict(signal, features, previous_trades)
    if prediction.decision == "block":
        log.info("ML BLOCKED signal: {} prob={:.2f}", strat_name, prediction.probability)
        continue  # пропускаем этот сигнал

# Периодическое переобучение:
if settings.analyzer_ml_enabled and should_retrain:
    trades = repo.get_all_completed_trades(days=180)
    ml_predictor.train(trades)
```

**Почему это критично:**
ML фильтр предназначен для блокировки ПЛОХИХ сигналов. Без него система принимает КАЖДЫЙ сигнал, который прошёл базовые фильтры стратегии. ML мог бы отсекать ~20-30% ложных сигналов.

---

### 🔴 ПРОБЛЕМА #2: AdaptiveAllocator создан, но НЕ ВЫЗЫВАЕТСЯ

**Уровень критичности**: ВЫСОКИЙ  
**Влияние на Win Rate**: +3-5%

**Факт**: В `main.py` строка 637:
```python
_adaptive_allocator = AdaptiveAllocator(lookback_trades=50)
```

Но дальше — **НИ ОДНОГО** вызова:
- `.update_skills()` — не вызывается
- `.get_adaptive_allocations()` — не вызывается

Вместо этого используется **статическая** аллокация из `ALLOCATION_TABLE`.

**Что должно быть:**
```python
# Периодически (раз в час или при закрытии сделки):
if _adaptive_allocator and repo:
    recent_trades = repo.get_all_completed_trades(days=30)
    _adaptive_allocator.update_skills(recent_trades)

# В торговом цикле — вместо get_active_strategies():
if settings.auto_strategy_selection and _current_regime:
    if _adaptive_allocator:
        allocs = _adaptive_allocator.get_adaptive_allocations(_current_regime)
        active_names = [a.strategy_name for a in allocs if a.is_active]
    else:
        active_names = get_active_strategies(_current_regime)
```

---

### 🔴 ПРОБЛЕМА #3: Бэктест без Daily Candles

**Уровень критичности**: ВЫСОКИЙ  
**Влияние**: Невозможно корректно оценить стратегии

**Факт**: `BacktestEngine.run()` принимает только `candles_1h` и `candles_4h`:
```python
def run(self, strategy, candles_1h, candles_4h, symbol="BTCUSDT") -> BacktestResult:
```

Вызов `feature_builder.build()` внутри бэктеста:
```python
features = self._feature_builder.build(symbol, window_1h, window_4h)
```
— **без candles_1d**.

**Последствия:**
- `ema_50_daily = None` → `trend_alignment` = 0.5 (нейтральный)
- `rsi_14_daily = None` → дневной RSI не используется
- Бэктест НЕ ОТРАЖАЕТ реальные условия бота (бот использует daily candles)

---

## РАЗДЕЛ 2: ПРОБЛЕМЫ СТРАТЕГИЙ

### 2.1 EMA Crossover RSI — Главная стратегия

| Аспект | Текущее | Проблема | Решение |
|--------|---------|----------|---------|
| **Whipsaw filter** | `min_cross_threshold = close × 0.001` (0.1%) | Порог слишком мал для BTC. При BTC=$85000 это $85 — почти любой crossover проходит | Увеличить до 0.2-0.5% или сделать ATR-based: `min_cross = ATR × 0.3` |
| **RSI confirm** | `RSI < 70` для BUY | Слишком мягкий. RSI 65-70 = зона риска. Профессионалы делают покупку из зоны 30-60 | Добавить tiered confidence: RSI < 40 → +0.15, RSI 40-50 → +0.10, RSI 50-60 → +0.05 |
| **Volume confirm** | `volume_ratio ≥ 1.0` | 1.0 = средний объём. Это НЕ подтверждение. Breakout нужен при АНОМАЛЬНО высоком объёме | Увеличить до ≥1.3, а лучше ≥1.5 |
| **Trend confirm** | `close > EMA50 (4h)` | Хорошо, но нет проверки НАКЛОНА EMA50 | Добавить: `trend_rising = ema_50_current > ema_50_prev` |
| **MACD confirm** | `histogram > 0` | Даёт +0.10 conf. Но histogram может быть ничтожно мал | Заменить: `histogram > 0 and abs(histogram) > price × 0.0001` |
| **Sell: trailing stop** | НЕТ | Стратегия использует фикс SL/TP. Нет trailing stop | Добавить trailing stop при PnL > 3%: trail at 1.5× ATR |
| **Sell: time stop** | НЕТ | Позиция может висеть бесконечно | Закрывать при нулевом движении > 48h (dead trade) |
| **Multi-TF confirm** | Частичный | Использует 1h + 4h + daily EMA50, но НЕ проверяет согласованность | Добавить: `if trend_alignment < 0.6: confidence -= 0.10` |

### 2.2 Bollinger Breakout — Проблемы

| Аспект | Текущее | Проблема |
|--------|---------|----------|
| **False breakouts** | BUY при `close > upper_BB` | BB breakout имеет ВЫСОКИЙ false positive rate (~60%). Нужен фильтр: подождать 1-2 свечи подтверждения |
| **Squeeze detection** | `bb_bandwidth < hist_volatility × 0.5` | Разумная адаптивная логика, но проверяется ТОЛЬКО в момент breakout, а не "было ли squeeze ДО breakout" |
| **RSI cap** | `RSI < 80` | Для breakout stratgegy 80 — слишком высоко. Overbought breakout = ловушка. RSI < 70 лучше |
| **Volume confirm** | `volume_ratio ≥ 1.5` | Хорошо для BB, но нужен убывающий volume_ratio через 2-3 свечи (volume exhaustion check) |

### 2.3 Mean Reversion — Проблемы

| Аспект | Текущее | Проблема |
|--------|---------|----------|
| **RSI threshold** | `RSI_oversold = 25` | СЛИШКОМ строго. RSI < 25 бывает редко для BTC. Стратегия почти никогда не торгует. Увеличить до 30 |
| **Falling knife** | Есть фильтр: `ema9 < ema21 < ema50 + adx > 25` | ✅ Хорошо, но можно добавить: `price_change_15h < -10%` → опасно |
| **Exit: revert to EMA21** | Продажа при `close >= ema21` | Слишком ранний выход. Mean reversion к СРЕДНЕЙ (EMA21) = только пол пути. Лучше exit при EMA21 + RSI > 50 |
| **Min confidence** | `0.80` | Слишком высокий порог. С RSI < 25 + lower BB + vol confirm редко набирается 0.80. Снизить до 0.72 |

### 2.4 MACD Divergence — Проблемы

| Аспект | Текущее | Проблема |
|--------|---------|----------|
| **History buffer** | In-memory `_price_history`, `_macd_history` | При перезапуске бота — буфер ПУСТ. Стратегия не работает ~50 свечей (50 часов) |
| **Divergence detection** | `price_new_low = ph[-1] < min(ph[-recent:-1])` | Проверяет ТОЛЬКО последнюю свечу vs. предыдущие. Настоящая дивергенция — это ДВА минимума на разном расстоянии |
| **Zero-cross bonus** | +0.05 conf при MACD crossing zero | Хорошо, но должен быть PRIMARY trigger, а не бонус |
| **Min confidence** | `0.72` | С base=0.55 + RSI(+0.10) + vol(+0.08) = 0.73 max. Порог срабатывает ТОЛЬКО при идеальных условиях |

### 2.5 Grid Trading — Проблемы

| Аспект | Текущее | Проблема |
|--------|---------|----------|
| **Grid rebuild** | Каждые 24h | В volatile рынке BB меняются за часы. 24h = устаревшая сетка |
| **Profit target** | `min_profit_pct = 0.3%` | После комиссии (0.1% × 2 = 0.2%) остаётся 0.1% прибыли. Это МЕНЬШЕ slippage |
| **Position management** | Одна позиция на символ | Grid должен поддерживать НЕСКОЛЬКО позиций одновременно |

### 2.6 DCA Bot — Проблемы

| Аспект | Текущее | Проблема |
|--------|---------|----------|
| **Dip multiplier** | Использует `price_change_15m` (15 × 1h candles) | Переменная `price_change_15m` — это на самом деле % за последние 15 свечей (15 часов, не минут). Для DCA нужен lookback 24-72h |
| **Interval** | 24h фикс | Нет адаптации к волатильности. В crash — нужно чаще, в rally — реже |
| **Drawdown stop** | `15%` | Для DCA стратегии 15% = нормально на крипторынке. Слишком рано закрывать |

---

## РАЗДЕЛ 3: ПРОБЛЕМЫ ML ПРЕДИКТОРА (когда подключим)

### 3.1 Архитектура ML

| Компонент | Статус | Оценка |
|-----------|--------|--------|
| Алгоритм | RandomForest (200 trees, depth=8) | ✅ Хороший выбор для табличных данных |
| Scaling | StandardScaler | ✅ Корректно |
| CV | TimeSeriesSplit (5 folds) | ✅ Правильная валидация для временных рядов |
| Overfitting check | train_prec - test_prec > 5% → reject | ✅ Хорошая защита |
| Feature extraction | 20 features, все pre-trade | ✅ ИСПРАВЛЕНО (нет forward-looking) |
| Min trades | 500 | ⚠️ Может быть слишком много для начала. Снизить до 200 |

### 3.2 Проблемы с Features

| # | Feature | Проблема |
|---|---------|----------|
| 1 | `ema_trend_signal` = `regime_bias × (confidence / 5.0)` | **СИНТЕТИЧЕСКИЙ FEATURE** из других features. RandomForest может делать это сам. Лучше использовать RAW `ema_9 - ema_21` |
| 2 | `volatility_proxy` = `volume_ratio × adx_normalized` | Опять синтетический. Модель лучше обучится на отдельных `volume_ratio` и `adx` |
| 3 | `macd_trend_proxy` = `regime_bias × confidence` | Бесполезный feature — перемножение двух уже имеющихся features |
| 4 | `trend_strength` = `adx / 50` | Дубликат `adx` (feature #1) — просто другой scale. Бесполезен |
| 5 | `time_since_last_norm` = `hours_since_last / 24` | Дубликат `hours_since_last_trade` (feature #12). Двойной счёт |
| 6 | `loss_streak_norm` = `consecutive_losses / 5` | Дубликат `consecutive_losses` (feature #14). Двойной счёт |
| 7 | `confidence_at_entry` | Это OUTPUT стратегии, зависящий от ВСЕХ остальных features. Создаёт colinearity / information leakage |

**Рекомендация**: Заменить синтетические features на RAW значения:
```python
BETTER_FEATURES = [
    trade.rsi_at_entry,                          # RSI
    trade.adx_at_entry,                          # ADX (тренд сила)
    (trade.ema_9 - trade.ema_21) / trade.close,  # EMA diff normalized  
    trade.bb_bandwidth,                           # Volatility (real)
    trade.volume_ratio_at_entry,                  # Volume confirmation
    trade.macd_histogram,                         # MACD (real)
    trade.atr / trade.close,                      # ATR% (normalized volatility)
    float(trade.hour_of_day),                     # Time
    float(trade.day_of_week),                     # Day
    regime_encoded,                               # Regime
    strategy_encoded,                             # Strategy
    recent_win_rate,                              # Performance history
    hours_since_last_trade,                       # Trade spacing
    daily_pnl_so_far,                             # Daily context
    consecutive_losses,                           # Streak
    trade.news_sentiment,                         # Sentiment
    trade.fear_greed_index / 100.0,               # Market mood
    trade.trend_alignment,                        # Multi-TF alignment
    trade.cmf,                                    # Money flow
    trade.stoch_rsi / 100.0,                      # Stoch RSI
]
```

### 3.3 Метрики качества

| Порог | Текущее | Рекомендация |
|-------|---------|--------------|
| `min_precision` | 0.65 | ✅ Правильно для фильтра (precision important) |
| `min_recall` | 0.58 | ⚠️ Слишком низко. recall 0.58 = пропускает 42% хороших сделок |
| `min_roc_auc` | 0.65 | ✅ Ок |
| `min_skill_score` | 0.72 | ⚠️ Сложно достичь с recall 0.58. Формула: 0.4×prec + 0.25×rec + 0.25×auc + 0.1×acc |
| `block_threshold` | 0.60 | ✅ Ок — блокирует только низко-вероятные сигналы |
| `min_trades` | 500 | ⚠️ Слишком много. На paper trading с 6 сделок/день = 83 дня ожидания. Снизить до 200 |

---

## РАЗДЕЛ 4: ПРОБЛЕМЫ RISK MANAGEMENT

### 4.1 Kelly Criterion — Текущее состояние

**Хорошо**: Код в `main.py` теперь подтягивает реальные данные:
```python
_recent = repo.get_strategy_trades(strat_name, limit=50)
if len(_recent) >= 10:
    _kelly_win_rate = len(_wins) / len(_recent)
    _kelly_avg_win = sum(pnl_pct for wins) / len(wins)
    _kelly_avg_loss = abs(sum(pnl_pct for losses) / len(losses))
```

**Проблема #1**: Fallback на 10 сделок слишком мал. Kelly с 10 сделками = шум. Увеличить до 30.

**Проблема #2**: Kelly рассчитывается для каждой стратегии ОТДЕЛЬНО, но при `len(_recent) < 10` fallback = `win_rate=0.5, avg_win=3%, avg_loss=2%`. Это даёт Kelly fraction:
```
f = (0.5 × 1.5 - 0.5) / 1.5 = (0.75 - 0.5) / 1.5 = 0.167
Half-Kelly = 0.083 → 8.3% of balance
```
Это слишком агрессивно для стратегии с неизвестной статистикой. Fallback должен быть `win_rate=0.45` (пессимистичный).

### 4.2 Dynamic SL/TP

**Статус**: ✅ Подключён и работает
**Проблема**: Нет TRAILING STOP в core стратегиях (кроме Bollinger Breakout).

| Стратегия | SL | TP | Trailing? |
|-----------|----|----|-----------|
| EMA Crossover | 3% фикс → ATR dynamic | 5% фикс → ATR dynamic | ❌ НЕТ |
| Bollinger | 3% → ATR dynamic | 6% → ATR dynamic | ✅ (2% от max) |
| Mean Reversion | 4% → ATR dynamic | 6% → ATR dynamic | ❌ НЕТ |
| MACD Div | 3.5% → ATR dynamic | 7% → ATR dynamic | ❌ НЕТ |
| Grid | 5% фикс | 0.3% фикс | ❌ НЕТ |
| DCA | 15% drawdown | 8% profit | ❌ НЕТ |

**Решение**: Внедрить trailing stop для EMA Crossover и MACD Divergence (трендовые стратегии):
- Активация: при PnL > 2× ATR%
- Trail distance: 1.5× ATR
- Это позволяет "пускать прибыль расти" вместо фикс TP

### 4.3 Circuit Breakers

| CB | Триггер | Оценка |
|----|---------|--------|
| CB-1: Price Anomaly | >5% за 1 мин | ✅ Разумно |
| CB-2: Consecutive Loss | 3 подряд | ⚠️ Для 6 стратегий — 3 loosing trades happens regularly. Увеличить до 4-5 или per-strategy |
| CB-3: Spread | >0.5% | ✅ Ок для BTC |
| CB-4: Volume | >10x или <0.1x | ✅ Разумно |
| CB-5: API Errors | >5 за 5 мин | ✅ Ок |
| CB-6: Latency | >5s × 3 | ✅ Ок |
| CB-7: Balance Mismatch | >1% | ✅ Ок |
| CB-8: Commission Spike | >1% balance/day | ✅ Ок |

**Главная проблема CB-2**: cooldown = 30 минут после 3 подряд убытков. Но при 6 стратегиях, убытки РАЗНЫХ стратегий суммируются в один счётчик. Grid может проиграть 2 раза, потом Mean Rev проигрывает 1 раз = CB-2 срабатывает, блокируя ВСЕ стратегии.

**Решение**: Сделать CB-2 PER-STRATEGY или увеличить порог до 5.

---

## РАЗДЕЛ 5: ПРОБЛЕМЫ MARKET REGIME

### 5.1 Текущая логика

```
if atr_pct > 0.04:       → VOLATILE (приоритет)
elif EMA9 > EMA21 > EMA50 AND ADX > 25:  → TRENDING_UP
elif EMA9 < EMA21 < EMA50 AND ADX > 25:  → TRENDING_DOWN
elif ADX < 20 AND price inside BB:        → SIDEWAYS
else:                                     → UNKNOWN
```

### 5.2 Проблемы

| # | Проблема | Влияние |
|---|----------|---------|
| 1 | **VOLATILE имеет абсолютный приоритет**. Если ATR/price > 4%, всегда VOLATILE, даже если есть чёткий тренд | Во время сильных трендовых движений (которые ВСЕГДА волатильны) система переключается на VOLATILE и УБИРАЕТ EMA Crossover (5%) | 
| 2 | **UNKNOWN catchall слишком широкий**. ADX 20-25 + EMA не выстроены = UNKNOWN | В UNKNOWN аллокация: EMA=5%, DCA=5% = 10% total. Большая часть времени рынок = UNKNOWN |
| 3 | **Нет ACCUMULATION/DISTRIBUTION фазы** | Перед breakout рынок переходит в фазу накопления (low vol + narrow BB). Это ЛУЧШЕЕ время для подготовки |
| 4 | **Гистерезис 3 подтверждения** | 3 × 4h = 12 часов delay для смены режима. Тренд может начаться и закончиться за 12 часов |

**Решения:**

1. **VOLATILE должен проверяться ПОСЛЕ тренда**:
```python
if ema9 > ema21 > ema50 > 0 and adx > 25:
    regime = TRENDING_UP
elif ema9 < ema21 < ema50 and adx > 25:
    regime = TRENDING_DOWN
elif atr_pct > 0.04:  # volatile ПОСЛЕ тренда
    regime = VOLATILE
elif adx < 20:
    regime = SIDEWAYS
else:
    regime = UNKNOWN
```

2. **Снизить гистерезис до 2** (8 часов вместо 12)

3. **Добавить TRANSITIONAL фазу**: ADX 20-25 = переход, аллокация 50/50 между текущим и ожидаемым

---

## РАЗДЕЛ 6: ПРОБЛЕМЫ FEATURE ENGINEERING

### 6.1 Индикаторы — что работает

| Индикатор | Реализация | Оценка |
|-----------|------------|--------|
| EMA (9/21/50) | ✅ Корректная формула | Ок |
| RSI (Wilder smoothing) | ✅ Правильная формула | Ок |
| MACD | ✅ Правильно | Ок |
| ADX | ✅ Wilder smoothing | Ок |
| Bollinger Bands | ✅ 20-period, 2σ | Ок |
| ATR | ✅ Wilder smoothing | Ок |
| Stochastic RSI | ✅ Корректно | Ок |
| CCI | ✅ Правильная формула | Ок |
| CMF | ✅ Chaikin Money Flow | Ок |
| VWAP | ⚠️ Simplified (20-period) | Настоящий VWAP = от открытия дня |
| OBV | ✅ Кумулятивный | Ок |

### 6.2 Что ОТСУТСТВУЕТ (профессиональные трейдеры используют)

| Индикатор | Зачем | Сложность добавления |
|-----------|-------|---------------------|
| **Ichimoku Cloud** | Определяет тренд + support/resistance. Tenkan/Kijun cross = сильный сигнал | Средняя |
| **Pivots (S/R levels)** | Ключевые уровни, от которых цена отбивается. Повышает SL/TP точность | Низкая |
| **Order Book Imbalance** | Соотношение bid/ask. Предсказывает краткосрочное направление | Высокая (нужен WS book) |
| **Funding Rate** (futures) | Показывает перекос рыночных настроений | Низкая (API Binance) |
| **Open Interest** (futures) | Рост OI + рост цены = НАСТОЯЩИЙ тренд | Низкая (API Binance) |
| **Volume Profile (VPVR)** | Определяет зоны высокой ликвидности | Средняя |
| **Heikin-Ashi candles** | Сглаженные свечи для определения тренда | Низкая |
| **ATR% percentile** | ATR в контексте исторической волатильности (ATR сейчас vs ATR за 90 дней) | Низкая |

### 6.3 Trend Alignment Score

**Текущая логика** (indicators.py):
```python
def trend_alignment(ema_fast, ema_slow, price, ema_daily):
    score = 0.5
    if price > ema_fast > ema_slow:
        score += 0.25
    if ema_daily and price > ema_daily:
        score += 0.25
    return score  # 0.5 - 1.0
```

**Проблемы**:
- Score только 3 уровня: 0.5, 0.75, 1.0. Слишком грубая градация
- Не учитывает РАССТОЯНИЕ от EMA (сила тренда)
- Не учитывает направление движения EMA (наклон)

**Улучшенная версия:**
```python
def trend_alignment(ema_fast, ema_slow, price, ema_daily, prev_ema_daily=None):
    score = 0.0
    # Multi-TF alignment (0.0 to 1.0)
    if price > ema_fast: score += 0.15
    if ema_fast > ema_slow: score += 0.15
    if ema_daily and price > ema_daily: score += 0.15
    # EMA slope (momentum)
    if prev_ema_daily and ema_daily > prev_ema_daily: score += 0.15
    # Distance from EMA (trend strength)
    if ema_slow > 0:
        dist = (price - ema_slow) / ema_slow
        score += min(abs(dist) * 5, 0.20)  # max +0.20 for 4% distance
    # Alignment coherence
    all_bullish = price > ema_fast > ema_slow
    if ema_daily: all_bullish = all_bullish and price > ema_daily
    if all_bullish: score += 0.20
    return min(score, 1.0)
```

---

## РАЗДЕЛ 7: ПРОБЛЕМЫ БЭКТЕСТИНГА

### 7.1 Текущий бэктест

| Аспект | Статус | Проблема |
|--------|--------|----------|
| Комиссии | 0.1% | ✅ Ок (Binance VIP0 = 0.1%) |
| Slippage | 0.05% | ✅ Разумно для BTC |
| Safety discount | 0.7 | ✅ Консервативно |
| Daily candles | ❌ Не используются | Feature вектор неполный |
| Market impact | ❌ | Большие ордера двигают цену. На $100 ордерах — несущественно |
| Funding/fees | ❌ | Для spot — не нужно |

### 7.2 Скрытый баг в бэктесте

В `BacktestEngine.run()`, при SL/TP check:
```python
if stop_loss > 0 and price <= stop_loss:
    exit_price = stop_loss * (1 - cfg.slippage_pct / 100)
```

**Проблема**: Используется `candle.close`, но цена во время свечи могла коснуться SL раньше. Нужно проверять `candle.low <= stop_loss`, а exit_price = stop_loss.

Аналогично для TP: нужно проверять `candle.high >= take_profit`.

Текущая логика может ПРОПУСКАТЬ срабатывания SL/TP, если свеча закрылась выше SL но была ниже интрадей.

---

## РАЗДЕЛ 8: NEWS SENTIMENT

### 8.1 Текущее состояние

| Аспект | Статус |
|--------|--------|
| RSS источники | 10 крипто-изданий |
| LLM анализ | Groq (primary) + OpenRouter (fallback) |
| Keyword fallback | Отключён (только LLM) |
| Fear & Greed Index | Подключён |
| Дедупликация | n-gram fuzzy matching |
| Влияние на стратегии | ±0.05-0.10 confidence |

### 8.2 Проблемы

| # | Проблема | Влияние |
|---|----------|---------|
| 1 | **News latency**: RSS обновляется каждые 300s (5 мин) | К моменту получения новости рынок уже отреагировал |
| 2 | **LLM rate limits**: Groq free tier = ограничен | При исчерпании лимита → OpenRouter → при исчерпании → новости без анализа |
| 3 | **Sentiment weight слишком мал**: max ±0.10 confidence | На практике одна МОЩНАЯ новость (ETF approval, SEC lawsuit) двигает рынок на 5-15%. +0.10 confidence = капля в море |
| 4 | **Нет "kill switch" от news** | Экстренная новость (биржа взломана, крупный бан) должна НЕМЕДЛЕННО закрывать все позиции |
| 5 | **Нет кэширования LLM ответов** | Одна и та же новость может анализироваться повторно при перезапуске |

---

## РАЗДЕЛ 9: КОНКРЕТНЫЙ ПЛАН ПОВЫШЕНИЯ WIN RATE

### Фаза 1: Критические исправления (ожидаемый эффект: +8-12% Win Rate)

| # | Задача | Файл | Сложность | Влияние |
|---|--------|------|-----------|---------|
| 1.1 | **Подключить ML Predictor в main.py** — import, init, predict, retrain | main.py | Средняя | +8-10% |
| 1.2 | **Вызывать AdaptiveAllocator.update_skills()** при закрытии сделки |  main.py | Низкая | +2-3% |
| 1.3 | **Использовать adaptive allocations** вместо статических | main.py | Низкая | +1-2% |

### Фаза 2: Улучшение стратегий (ожидаемый эффект: +5-8%)

| # | Задача | Файл | Сложность | Влияние |
|---|--------|------|-----------|---------|
| 2.1 | **ATR-based whipsaw filter** для EMA Crossover | ema_crossover_rsi.py | Низкая | +2-3% |
| 2.2 | **Trailing stop** для EMA Crossover + MACD Divergence | ema_crossover_rsi.py, macd_divergence.py | Средняя | +3-5% |
| 2.3 | **Candle confirmation** для Bollinger Breakout (ждать 1 свечу) | bollinger_breakout.py | Низкая | +1-2% |
| 2.4 | **Снизить RSI oversold** Mean Reversion с 25 до 30 | mean_reversion.py | Тривиально | +1% |
| 2.5 | **Увеличить min_volume_ratio** EMA Crossover с 1.0 до 1.3 | ema_crossover_rsi.py | Тривиально | +1-2% |
| 2.6 | **Time stop**: закрытие позиций без движения > 48h | Все стратегии | Средняя | +1% |

### Фаза 3: ML улучшения (ожидаемый эффект: +3-5%)

| # | Задача | Файл | Сложность | Влияние |
|---|--------|------|-----------|---------|
| 3.1 | **Заменить синтетические features** на RAW индикаторы | ml_predictor.py | Средняя | +2-3% |
| 3.2 | **Снизить min_trades** с 500 до 200 | config.py | Тривиально | Ускорение активации ML |
| 3.3 | **Добавить trend_alignment, cmf, stoch_rsi** в features | ml_predictor.py | Низкая | +1-2% |
| 3.4 | **Убрать дубликат features** (norm versions) | ml_predictor.py | Тривиально | Чистота модели |

### Фаза 4: Market Regime (ожидаемый эффект: +2-4%)

| # | Задача | Файл | Сложность | Влияние |
|---|--------|------|-----------|---------|
| 4.1 | **Volatile после тренда** (поменять приоритет) | market_regime.py | Тривиально | +1-2% |
| 4.2 | **Снизить гистерезис** с 3 до 2 | market_regime.py | Тривиально | +0.5-1% |
| 4.3 | **Сделать CB-2 per-strategy** | circuit_breakers.py | Средняя | +1-2% |

### Фаза 5: Бэктест и валидация

| # | Задача | Файл | Сложность |
|---|--------|------|-----------|
| 5.1 | **Добавить candles_1d в бэктест** | backtest/engine.py | Низкая |
| 5.2 | **Исправить SL/TP check** — использовать high/low свечи | backtest/engine.py | Низкая |
| 5.3 | **Добавить Walk-Forward Analysis** | backtest/ | Высокая |

---

## РАЗДЕЛ 10: ПРАВИЛА ПРОФЕССИОНАЛЬНОГО ТРЕЙДЕРА

### 10 правил для повышения Win Rate

1. **НЕ ТОРГУЙ ПРОТИВ ТРЕНДА СТАРШЕГО ТАЙМФРЕЙМА**
   - Если Daily EMA50 падает → НЕ покупай, даже если 1h показывает crossover
   - Текущий `trend_alignment` даёт 0.5 если нет daily данных — это НЕЙТРАЛЬНОЕ решение при НЕИЗВЕСТНОМ тренде. Должно быть -0.10 confidence penalty

2. **VOLUME ПОДТВЕРЖДАЕТ, НЕ ИНИЦИИРУЕТ**
   - Breakout без объёма = ложный breakout. Volume ratio < 1.3 для EMA crossover = шум

3. **TRAILING STOP > FIXED TP**
   - Fixed TP в 5% может закрыть сделку, которая может пойти на +20%
   - Trailing stop позволяет "пускать прибыль расти"

4. **ОТБРАСЫВАЙ ПЕРВЫЙ СИГНАЛ**
   - Для BB Breakout: первый пробой верхнего BB = часто ложный. Жди retrace и подтверждение

5. **ИСПОЛЬЗУЙ ВРЕМЯ**
   - Crypto более волатилен в 13:00-17:00 UTC (US open). Лучшие тренды начинаются тут
   - Ночью (00:00-07:00 UTC) — low volume = ложные сигналы

6. **LIMIT CONCURRENT STRATEGIES**
   - Не запускай 6 стратегий одновременно. Макс 2-3 стратегии, лучшие для текущего режима

7. **ML ФИЛЬТР = ОБЯЗАТЕЛЕН**
   - Даже простая логистическая регрессия, фильтрующая 20% худших сигналов, повышает Win Rate на 5-10%

8. **АДАПТИРУЙСЯ К ВОЛАТИЛЬНОСТИ**
   - В volatile рынке: шире стопы (2× ATR), меньше позиция, выше порог confidence
   - В спокойном рынке: уже стопы, больше позиция

9. **НЕ ТОРГУЙ В НЕОПРЕДЕЛЁННОСТИ**
   - Если Market Regime = UNKNOWN → не торгуй (или только DCA). Текущая аллокация для UNKNOWN = 10% — уже ок

10. **JOURNAL КАЖДУЮ СДЕЛКУ**
    - Текущий `_strategy_log` — хорошо, но нужно АНАЛИЗИРОВАТЬ: какие ПРИЧИНЫ rejection чаще всего → те стратегии нужно чинить первыми

---

## ИТОГО: ДОРОЖНАЯ КАРТА

```
Текущий Win Rate: ~52-55%
                         
Фаза 1 (+8-12%):  → 60-67%    [ML + AdaptiveAllocator]
Фаза 2 (+5-8%):   → 65-72%    [Улучшение стратегий]  
Фаза 3 (+3-5%):   → 68-75%    [ML features]
Фаза 4 (+2-4%):   → 70-77%    [Regime + CB]

ЦЕЛЕВОЙ Win Rate: 65-72% (реалистичный)
МАКСИМАЛЬНЫЙ Win Rate: ~75% (идеальный, при всех оптимизациях)
```

> **Примечание**: Win Rate > 75% в крипто на длинном горизонте практически невозможен. Профессиональные алго-фонды работают с 55-65% Win Rate, компенсируя Profit Factor > 2.0 (средняя прибыль × 2 > средний убыток).

**Ключевой KPI — НЕ Win Rate, а Profit Factor:**
- Win Rate 55% + PF 2.0 = отличная система
- Win Rate 70% + PF 1.1 = плохая система (много маленьких побед, одно большое поражение)

**Формула для focus:**
$$\text{Expectancy} = (\text{Win Rate} \times \text{Avg Win}) - ((1 - \text{Win Rate}) \times \text{Avg Loss})$$

Цель: Expectancy > 1.5% per trade.
