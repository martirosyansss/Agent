"""
Backtest Engine — тестирование стратегий на исторических данных.

Прогоняет стратегию по историческим свечам, симулирует исполнение,
считает PnL и метрики. Применяет Safety Discount 0.7.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from core.models import Candle, Direction, FeatureVector, Signal
from features.feature_builder import FeatureBuilder
from strategy.base_strategy import BaseStrategy

from .execution_model import ExecutionConfig, FillReason, RealisticExecutionModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Конфигурация бэктеста."""
    initial_balance: float = 500.0
    commission_pct: float = 0.1
    slippage_pct: float = 0.05  # legacy fallback when realistic_execution=False
    safety_discount: float = 0.7
    position_size_pct: float = 20.0  # % от баланса на сделку
    # Realistic execution: spread + market impact + gap-through-stop modelling.
    # When True, slippage_pct is ignored in favour of ``execution_config``.
    realistic_execution: bool = True
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    # Apply production entry guards (multi-TF, regime, drawdown) in the
    # backtest too. Off by default to preserve existing regression-test
    # baselines; enable to close the backtest-vs-live optimism gap.
    apply_risk_guards: bool = False
    # DD thresholds mirror production defaults when apply_risk_guards=True.
    dd_daily_pct: float = 0.05
    dd_weekly_pct: float = 0.10
    dd_monthly_pct: float = 0.15


@dataclass
class BacktestTrade:
    """Одна сделка в бэктесте."""
    symbol: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    reason: str


@dataclass
class BacktestResult:
    """Результат бэктеста."""
    strategy_name: str
    symbol: str
    period_start: int
    period_end: int
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_pnl_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    safety_discount: float
    expected_real_pnl: float
    trades: list[BacktestTrade] = field(default_factory=list)
    guards_rejected: int = 0  # number of BUY signals blocked by pro guards
    # Probabilistic Sharpe Ratio (López de Prado, 2012). Reports the
    # probability that the strategy's *true* Sharpe exceeds 0 (or any
    # benchmark), correcting the point-estimate Sharpe for skewness and
    # excess kurtosis of the daily-returns sample. PSR ≥ 0.95 means the
    # observed Sharpe is significantly better than the benchmark; values
    # near 0.5 are inconclusive. Defaults to 0.0 when too few daily
    # returns exist for a stable estimate.
    psr: float = 0.0
    # Returns-distribution diagnostics: Shapiro-Wilk p-value flags whether the
    # daily-returns sample is normal enough to trust the naïve annualised
    # Sharpe (which assumes Gaussian). p < 0.05 → returns are non-normal,
    # Sharpe is biased — prefer PSR. ``returns_normal`` is the binary
    # decision; ``returns_shapiro_p`` is the raw p-value for transparency.
    # Both default to ``None`` when the sample is too small (n < 8) for the
    # test to be meaningful.
    returns_shapiro_p: Optional[float] = None
    returns_normal: Optional[bool] = None


class BacktestEngine:
    """Движок бэктестинга SENTINEL."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self._config = config or BacktestConfig()
        self._feature_builder = FeatureBuilder()
        self._exec_model = (
            RealisticExecutionModel(self._config.execution_config)
            if self._config.realistic_execution
            else None
        )

    def run(
        self,
        strategy: BaseStrategy,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        symbol: str = "BTCUSDT",
        candles_1d: list[Candle] | None = None,
    ) -> BacktestResult:
        """Прогнать стратегию по историческим свечам.

        Args:
            strategy: Экземпляр стратегии.
            candles_1h: 1h свечи (отсортированы по времени).
            candles_4h: 4h свечи.
            symbol: Торговый символ.
            candles_1d: 1d свечи (опционально, для multi-TF).

        Returns:
            BacktestResult с полными метриками.
        """
        cfg = self._config
        balance = cfg.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        trades: list[BacktestTrade] = []
        daily_returns: list[float] = []
        guards_rejected = 0

        # Optional production-style guards (disabled by default for
        # back-compat). Three guards are meaningful in a single-symbol
        # backtest: multi-TF trend confluence, regime, and drawdown.
        # News/liquidity/correlation are either unavailable or N/A here.
        _mtf = _regime = _dd = None
        _classify = None
        if cfg.apply_risk_guards:
            from strategy.multi_tf_gate import MultiTFGate, MultiTFGateConfig, classify_strategy
            from risk.regime_gate import RegimeGate
            from risk.drawdown_breaker import DrawdownBreaker, DrawdownThresholds
            _mtf = MultiTFGate(MultiTFGateConfig(fail_closed_on_missing_data=False))
            _regime = RegimeGate()
            _dd = DrawdownBreaker(DrawdownThresholds(
                daily_pct=cfg.dd_daily_pct,
                weekly_pct=cfg.dd_weekly_pct,
                monthly_pct=cfg.dd_monthly_pct,
            ))
            _classify = classify_strategy

        # Состояние позиции
        in_position = False
        entry_price = 0.0
        entry_time = 0
        quantity = 0.0
        stop_loss = 0.0
        take_profit = 0.0

        # Скользящее окно: нужно минимум 55 свечей для features
        min_history = 55

        for i in range(min_history, len(candles_1h)):
            candle = candles_1h[i]
            price = candle.close

            # Собрать features из окна
            window_1h = candles_1h[max(0, i - min_history):i]

            # Найти 4h свечи до текущего времени
            window_4h = [c for c in candles_4h if c.timestamp <= candle.timestamp]
            if len(window_4h) > min_history:
                window_4h = window_4h[-min_history:]

            # 1d свечи (если есть)
            window_1d = None
            if candles_1d:
                window_1d = [c for c in candles_1d if c.timestamp <= candle.timestamp]
                if len(window_1d) > min_history:
                    window_1d = window_1d[-min_history:]

            features = self._feature_builder.build(symbol, window_1h, window_4h, window_1d)
            if features is None:
                continue

            # Генерация сигнала
            signal = strategy.generate_signal(
                features,
                has_open_position=in_position,
                entry_price=entry_price if in_position else None,
            )

            # Проверить SL/TP для открытой позиции
            if in_position:
                # Gap-aware exit: stop-loss takes priority. If both SL and TP
                # are touched in the same candle we conservatively assume the
                # adverse one (SL) hit first — the candle's path is unknown.
                notional = quantity * price
                sl_fill = None
                tp_fill = None
                if stop_loss > 0:
                    if self._exec_model is not None:
                        sl_fill = self._exec_model.fill_stop_loss(
                            stop_price=stop_loss,
                            candle_open=candle.open,
                            candle_low=candle.low,
                            notional_usd=notional,
                        )
                    elif candle.low <= stop_loss:
                        # Legacy flat-slippage path.
                        from .execution_model import FillResult
                        exit_p = stop_loss * (1 - cfg.slippage_pct / 100)
                        sl_fill = FillResult(
                            fill_price=exit_p,
                            slippage_bps=cfg.slippage_pct * 100,
                            reason=FillReason.STOP_LOSS,
                        )

                if sl_fill is not None:
                    exit_price = sl_fill.fill_price
                    comm = quantity * exit_price * cfg.commission_pct / 100
                    pnl = (exit_price - entry_price) * quantity - comm
                    balance += pnl
                    reason_label = "Stop-loss (gap)" if sl_fill.reason == FillReason.STOP_LOSS_GAP else "Stop-loss"
                    trades.append(BacktestTrade(
                        symbol=symbol, entry_time=entry_time, exit_time=candle.timestamp,
                        entry_price=entry_price, exit_price=exit_price,
                        quantity=quantity, pnl=pnl,
                        pnl_pct=(exit_price - entry_price) / entry_price * 100,
                        commission=comm, reason=reason_label,
                    ))
                    daily_returns.append(pnl / balance * 100 if balance != 0 else 0)
                    in_position = False
                    continue

                if take_profit > 0:
                    if self._exec_model is not None:
                        tp_fill = self._exec_model.fill_take_profit(
                            tp_price=take_profit,
                            candle_open=candle.open,
                            candle_high=candle.high,
                            notional_usd=notional,
                        )
                    elif candle.high >= take_profit:
                        from .execution_model import FillResult
                        exit_p = take_profit * (1 - cfg.slippage_pct / 100)
                        tp_fill = FillResult(
                            fill_price=exit_p,
                            slippage_bps=cfg.slippage_pct * 100,
                            reason=FillReason.TAKE_PROFIT,
                        )

                if tp_fill is not None:
                    exit_price = tp_fill.fill_price
                    comm = quantity * exit_price * cfg.commission_pct / 100
                    pnl = (exit_price - entry_price) * quantity - comm
                    balance += pnl
                    reason_label = "Take-profit (gap)" if tp_fill.reason == FillReason.TAKE_PROFIT_GAP else "Take-profit"
                    trades.append(BacktestTrade(
                        symbol=symbol, entry_time=entry_time, exit_time=candle.timestamp,
                        entry_price=entry_price, exit_price=exit_price,
                        quantity=quantity, pnl=pnl,
                        pnl_pct=(exit_price - entry_price) / entry_price * 100,
                        commission=comm, reason=reason_label,
                    ))
                    daily_returns.append(pnl / balance * 100 if balance != 0 else 0)
                    in_position = False
                    continue

            if signal is None:
                continue

            # Risk guards on BUY (opt-in via apply_risk_guards).
            # Drawdown sees equity ticks on every candle so its windows
            # roll over in backtest time, not wall-clock time.
            if cfg.apply_risk_guards and _dd is not None:
                _dd.update(balance)
            if (
                cfg.apply_risk_guards
                and signal.direction == Direction.BUY
                and not in_position
            ):
                strat_name = getattr(strategy, "NAME", strategy.__class__.__name__)
                if _regime is not None:
                    rg = _regime.check(strat_name, features)
                    if not rg.approved:
                        guards_rejected += 1
                        continue
                if _mtf is not None and _classify is not None:
                    mtf = _mtf.check(
                        direction=signal.direction,
                        features=features,
                        strategy_type=_classify(strat_name),
                    )
                    if not mtf.approved:
                        guards_rejected += 1
                        continue
                if _dd is not None and not _dd.allows_new_entry():
                    guards_rejected += 1
                    continue

            # BUY
            if signal.direction == Direction.BUY and not in_position:
                position_value = balance * cfg.position_size_pct / 100
                if self._exec_model is not None:
                    fill = self._exec_model.fill_market_buy(
                        reference_price=price,
                        notional_usd=position_value,
                    )
                    entry_price = fill.fill_price
                else:
                    entry_price = price * (1 + cfg.slippage_pct / 100)
                quantity = position_value / entry_price
                comm = quantity * entry_price * cfg.commission_pct / 100
                balance -= comm  # Комиссия при входе
                entry_time = candle.timestamp
                stop_loss = signal.stop_loss_price
                take_profit = signal.take_profit_price
                in_position = True

            # SELL
            elif signal.direction == Direction.SELL and in_position:
                if self._exec_model is not None:
                    fill = self._exec_model.fill_market_sell(
                        reference_price=price,
                        notional_usd=quantity * price,
                    )
                    exit_price = fill.fill_price
                else:
                    exit_price = price * (1 - cfg.slippage_pct / 100)
                comm = quantity * exit_price * cfg.commission_pct / 100
                pnl = (exit_price - entry_price) * quantity - comm
                balance += pnl
                trades.append(BacktestTrade(
                    symbol=symbol, entry_time=entry_time, exit_time=candle.timestamp,
                    entry_price=entry_price, exit_price=exit_price,
                    quantity=quantity, pnl=pnl,
                    pnl_pct=(exit_price - entry_price) / entry_price * 100,
                    commission=comm, reason=signal.reason,
                ))
                daily_returns.append(pnl / balance * 100 if balance != 0 else 0)
                in_position = False

            # Обновить drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        # Close any open position at end of data
        if in_position and candles_1h:
            last_price = candles_1h[-1].close
            if self._exec_model is not None:
                eod_fill = self._exec_model.fill_market_sell(
                    reference_price=last_price,
                    notional_usd=quantity * last_price,
                )
                exit_price = eod_fill.fill_price
            else:
                exit_price = last_price * (1 - cfg.slippage_pct / 100)
            comm = quantity * exit_price * cfg.commission_pct / 100
            pnl = (exit_price - entry_price) * quantity - comm
            balance += pnl
            trades.append(BacktestTrade(
                symbol=symbol, entry_time=entry_time, exit_time=candles_1h[-1].timestamp,
                entry_price=entry_price, exit_price=exit_price,
                quantity=quantity, pnl=pnl,
                pnl_pct=(exit_price - entry_price) / entry_price * 100,
                commission=comm, reason="End-of-data close",
            ))
            daily_returns.append(pnl / balance * 100 if balance != 0 else 0)
            in_position = False

        # Расчёт метрик
        wins = [t for t in trades if t.pnl > 0]
        losses_list = [t for t in trades if t.pnl <= 0]
        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses_list)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        total_pnl = balance - cfg.initial_balance
        total_pnl_pct = total_pnl / cfg.initial_balance * 100

        avg_win = sum(t.pnl for t in wins) / win_count if wins else 0
        avg_loss = sum(t.pnl for t in losses_list) / loss_count if losses_list else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses_list))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.99

        # N-2 fix: Sharpe Ratio on actual DAILY returns (not per-trade)
        # Aggregate PnL by calendar date, then compute daily return series
        from collections import defaultdict
        daily_pnl_by_date: dict[int, float] = defaultdict(float)
        for t in trades:
            # Use exit_time day as the trade's settlement date
            trade_day = t.exit_time // 86_400_000  # ms → day bucket
            daily_pnl_by_date[trade_day] += t.pnl
        daily_returns_agg = list(daily_pnl_by_date.values())

        if daily_returns_agg and len(daily_returns_agg) > 1:
            # Convert PnL to % returns relative to initial balance
            daily_ret_pct = [pnl / cfg.initial_balance * 100 for pnl in daily_returns_agg]
            mean_return = sum(daily_ret_pct) / len(daily_ret_pct)
            std_return = math.sqrt(sum((r - mean_return) ** 2 for r in daily_ret_pct) / (len(daily_ret_pct) - 1))
            sharpe = (mean_return / std_return * math.sqrt(252)) if std_return > 0 else 0.0
        else:
            sharpe = 0.0

        # PSR — corrects Sharpe for non-normal returns (crypto has fat tails).
        # Imported lazily so this module's import graph stays light when only
        # the engine façade is needed (e.g. CLI report formatters).
        try:
            from backtest.walk_forward import probabilistic_sharpe_ratio
            psr = probabilistic_sharpe_ratio(daily_ret_pct) if daily_returns_agg and len(daily_returns_agg) > 3 else 0.0
        except Exception as _psr_err:  # noqa: BLE001
            psr = 0.0

        # Shapiro-Wilk normality test on daily returns. Sharpe ratio assumes
        # Gaussian returns; for crypto strategies with fat tails this test
        # almost always rejects normality at p < 0.05, signalling the user
        # to trust PSR over Sharpe. n < 8 makes Shapiro-Wilk unstable; we
        # report None in that case rather than a misleading p-value.
        returns_shapiro_p: Optional[float] = None
        returns_normal: Optional[bool] = None
        if daily_returns_agg and len(daily_returns_agg) >= 8:
            try:
                from scipy.stats import shapiro
                # Shapiro-Wilk's published validity range tops out at n=5000;
                # subsample to that ceiling deterministically (every k-th
                # sample) so larger backtests don't crash or warn.
                rets_arr = daily_ret_pct
                if len(rets_arr) > 5000:
                    step = max(1, len(rets_arr) // 5000)
                    rets_arr = rets_arr[::step][:5000]
                _, p_val = shapiro(rets_arr)
                returns_shapiro_p = float(p_val)
                returns_normal = bool(p_val >= 0.05)
            except Exception as _sw_err:  # noqa: BLE001
                # Test failed (degenerate sample, scipy missing) — leave None.
                returns_shapiro_p = None
                returns_normal = None

        expected_real = total_pnl * cfg.safety_discount

        period_start = candles_1h[0].timestamp if candles_1h else 0
        period_end = candles_1h[-1].timestamp if candles_1h else 0

        return BacktestResult(
            strategy_name=strategy.__class__.__name__,
            symbol=symbol,
            period_start=period_start,
            period_end=period_end,
            initial_balance=cfg.initial_balance,
            final_balance=balance,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            total_trades=total_trades,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            safety_discount=cfg.safety_discount,
            expected_real_pnl=expected_real,
            trades=trades,
            guards_rejected=guards_rejected,
            psr=psr,
            returns_shapiro_p=returns_shapiro_p,
            returns_normal=returns_normal,
        )

    @staticmethod
    def _normality_tag(result: "BacktestResult") -> str:
        """Compact human-readable label for the Shapiro-Wilk normality
        diagnostic. Used in the text report so operators know whether to
        trust the printed Sharpe at face value, or rely on PSR instead.
        """
        if result.returns_normal is None:
            return "n/a (мало данных)"
        if result.returns_normal:
            return f"да (Shapiro p={result.returns_shapiro_p:.3f})"
        return f"НЕТ (Shapiro p={result.returns_shapiro_p:.3f}) — используйте PSR"

    def format_report(self, result: BacktestResult) -> str:
        """Форматировать текстовый отчёт."""
        pf = f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "∞"
        return (
            f"{'═' * 45}\n"
            f"       BACKTEST REPORT\n"
            f"{'═' * 45}\n"
            f" Стратегия:    {result.strategy_name}\n"
            f" Символ:       {result.symbol}\n"
            f"{'─' * 45}\n"
            f" Начальный баланс:    ${result.initial_balance:.2f}\n"
            f" Конечный баланс:     ${result.final_balance:.2f}\n"
            f" Общий PnL:           ${result.total_pnl:.2f} ({result.total_pnl_pct:+.1f}%)\n"
            f"{'─' * 45}\n"
            f" Всего сделок:        {result.total_trades}\n"
            f" Прибыльных:          {result.wins} ({result.win_rate:.1f}%)\n"
            f" Убыточных:           {result.losses}\n"
            f"{'─' * 45}\n"
            f" Макс просадка:       {result.max_drawdown_pct:.1f}%\n"
            f" Sharpe Ratio:        {result.sharpe_ratio:.2f}\n"
            f" Sharpe trustworthy?  {self._normality_tag(result)}\n"
            f" PSR (P[Sharpe>0]):   {result.psr:.2f}  ({'значимо' if result.psr >= 0.95 else 'неубедительно'})\n"
            f" Profit Factor:       {pf}\n"
            f" Средний выигрыш:     ${result.avg_win:.2f}\n"
            f" Средний проигрыш:    ${result.avg_loss:.2f}\n"
            f"{'─' * 45}\n"
            f" ⚠️  Коэф. безопасности: {result.safety_discount}\n"
            f" Ожидаемый реальный PnL: ~${result.expected_real_pnl:.2f}\n"
            f"{'═' * 45}"
        )

    def run_walk_forward(
        self,
        strategy: BaseStrategy,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        symbol: str = "BTCUSDT",
        n_splits: int = 5,
        train_ratio: float = 0.7,
        candles_1d: list[Candle] | None = None,
    ) -> dict:
        """Walk-forward validation: split data into N folds, train on first part, test on rest.

        Returns dict with per-fold results and aggregate stats.
        """
        total = len(candles_1h)
        if total < 100 or n_splits < 2:
            return {"folds": [], "error": "Not enough data or splits"}

        fold_size = total // n_splits
        folds = []

        for i in range(n_splits):
            fold_start = i * fold_size
            fold_end = min(fold_start + fold_size, total)
            if i == n_splits - 1:
                fold_end = total

            split_point = fold_start + int((fold_end - fold_start) * train_ratio)
            test_candles_1h = candles_1h[split_point:fold_end]

            # Find 4h candles matching the test window
            if test_candles_1h:
                t_start = test_candles_1h[0].timestamp
                t_end = test_candles_1h[-1].timestamp
                test_candles_4h = [c for c in candles_4h if t_start <= c.timestamp <= t_end]
                # Also include history for feature building
                history_4h = [c for c in candles_4h if c.timestamp <= t_end]
                if len(history_4h) > 60:
                    history_4h = history_4h[-60:]
                # 1d candles for the fold
                fold_1d = None
                if candles_1d:
                    fold_1d = [c for c in candles_1d if c.timestamp <= t_end]
                    if len(fold_1d) > 60:
                        fold_1d = fold_1d[-60:]
            else:
                continue

            if len(test_candles_1h) < 60:
                continue

            # Run backtest on the test fold (strategy doesn't retrain — walk-forward test)
            result = self.run(strategy, test_candles_1h, history_4h, symbol, candles_1d=fold_1d)
            folds.append({
                "fold": i + 1,
                "test_start": test_candles_1h[0].timestamp if test_candles_1h else 0,
                "test_end": test_candles_1h[-1].timestamp if test_candles_1h else 0,
                "test_candles": len(test_candles_1h),
                "total_pnl": round(result.total_pnl, 2),
                "total_pnl_pct": round(result.total_pnl_pct, 2),
                "total_trades": result.total_trades,
                "win_rate": round(result.win_rate, 1),
                "max_drawdown_pct": round(result.max_drawdown_pct, 1),
                "sharpe_ratio": round(result.sharpe_ratio, 2),
            })

        if not folds:
            return {"folds": [], "error": "No valid folds generated"}

        # Aggregate
        profitable_folds = sum(1 for f in folds if f["total_pnl"] > 0)
        avg_pnl = sum(f["total_pnl"] for f in folds) / len(folds)
        avg_wr = sum(f["win_rate"] for f in folds) / len(folds)
        worst_dd = max(f["max_drawdown_pct"] for f in folds)

        return {
            "folds": folds,
            "n_splits": n_splits,
            "profitable_folds": profitable_folds,
            "avg_pnl": round(avg_pnl, 2),
            "avg_win_rate": round(avg_wr, 1),
            "worst_drawdown": round(worst_dd, 1),
            "consistency_score": round(profitable_folds / len(folds) * 100, 1),
        }
