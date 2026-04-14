"""
Strategy Auto-Tuner — Optuna-based hyperparameter optimization.

For each strategy, searches the optimal parameter set that maximizes
a combined objective: profit_factor * sqrt(win_rate) * sqrt(n_trades).

Constraints:
- min 30 trades (statistical significance)
- max 30% drawdown (risk control)
- SL range [1%, 10%], TP range [2%, 15%]

Usage:
    python -m analyzer.strategy_tuner
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import optuna

from backtest.engine import BacktestConfig, BacktestEngine
from core.models import Candle

logger = logging.getLogger(__name__)

# Suppress Optuna internal logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TunerConfig:
    n_trials: int = 25            # Number of Optuna trials per strategy (each ~15s)
    min_trades: int = 20          # Minimum trades for valid result
    max_drawdown_pct: float = 30  # Max acceptable drawdown
    max_candles_tune: int = 15000 # Use only last N candles for tuning (speed)
    backtest_balance: float = 500.0
    backtest_commission: float = 0.1
    backtest_slippage: float = 0.05


@dataclass
class TuneResult:
    strategy_name: str
    best_params: dict[str, Any]
    best_score: float
    best_trades: int
    best_win_rate: float
    best_pnl: float
    best_profit_factor: float
    n_trials: int


class StrategyTuner:
    """Optuna-based strategy parameter optimizer."""

    def __init__(self, config: TunerConfig | None = None) -> None:
        self._cfg = config or TunerConfig()

    def tune_ema_crossover(
        self,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        candles_1d: list[Candle],
        symbol: str,
    ) -> TuneResult:
        """Tune EMA Crossover RSI strategy."""
        from strategy.ema_crossover_rsi import EMACrossoverRSI, EMAConfig

        def objective(trial: optuna.Trial) -> float:
            cfg = EMAConfig(
                ema_fast=trial.suggest_int("ema_fast", 5, 15),
                ema_slow=trial.suggest_int("ema_slow", 15, 30),
                ema_trend=trial.suggest_int("ema_trend", 30, 100),
                rsi_overbought=trial.suggest_float("rsi_overbought", 65, 80),
                rsi_oversold=trial.suggest_float("rsi_oversold", 20, 40),
                min_volume_ratio=trial.suggest_float("min_volume_ratio", 0.8, 2.0, step=0.1),
                stop_loss_pct=trial.suggest_float("stop_loss_pct", 1.5, 6.0, step=0.5),
                take_profit_pct=trial.suggest_float("take_profit_pct", 3.0, 12.0, step=0.5),
                trailing_stop_pct=trial.suggest_float("trailing_stop_pct", 1.0, 3.0, step=0.5),
                trailing_activate_pct=trial.suggest_float("trailing_activate_pct", 1.5, 5.0, step=0.5),
                max_hold_hours=trial.suggest_int("max_hold_hours", 24, 168),
                min_confidence=trial.suggest_float("min_confidence", 0.50, 0.80, step=0.05),
            )
            strategy = EMACrossoverRSI(config=cfg)
            return self._evaluate(strategy, candles_1h, candles_4h, candles_1d, symbol)

        return self._run_study("ema_crossover_rsi", objective)

    def tune_bollinger(
        self,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        candles_1d: list[Candle],
        symbol: str,
    ) -> TuneResult:
        """Tune Bollinger Breakout strategy."""
        from strategy.bollinger_breakout import BollingerBreakout, BBBreakoutConfig

        def objective(trial: optuna.Trial) -> float:
            cfg = BBBreakoutConfig(
                volume_confirm_mult=trial.suggest_float("volume_confirm_mult", 1.0, 2.5, step=0.1),
                squeeze_threshold=trial.suggest_float("squeeze_threshold", 0.02, 0.10, step=0.01),
                stop_loss_pct=trial.suggest_float("stop_loss_pct", 1.5, 6.0, step=0.5),
                take_profit_pct=trial.suggest_float("take_profit_pct", 3.0, 12.0, step=0.5),
                trailing_stop_pct=trial.suggest_float("trailing_stop_pct", 1.0, 4.0, step=0.5),
                trailing_activate_pct=trial.suggest_float("trailing_activate_pct", 1.5, 6.0, step=0.5),
                min_confidence=trial.suggest_float("min_confidence", 0.55, 0.80, step=0.05),
            )
            strategy = BollingerBreakout(config=cfg)
            return self._evaluate(strategy, candles_1h, candles_4h, candles_1d, symbol)

        return self._run_study("bollinger_breakout", objective)

    def tune_mean_reversion(
        self,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        candles_1d: list[Candle],
        symbol: str,
    ) -> TuneResult:
        """Tune Mean Reversion strategy."""
        from strategy.mean_reversion import MeanReversion, MeanRevConfig

        def objective(trial: optuna.Trial) -> float:
            cfg = MeanRevConfig(
                rsi_oversold=trial.suggest_float("rsi_oversold", 20, 40, step=1),
                rsi_overbought=trial.suggest_float("rsi_overbought", 60, 80, step=1),
                stop_loss_pct=trial.suggest_float("stop_loss_pct", 2.0, 6.0, step=0.5),
                take_profit_pct=trial.suggest_float("take_profit_pct", 4.0, 15.0, step=0.5),
                min_volume_ratio=trial.suggest_float("min_volume_ratio", 1.0, 2.5, step=0.1),
                min_confidence=trial.suggest_float("min_confidence", 0.55, 0.80, step=0.05),
            )
            strategy = MeanReversion(config=cfg)
            return self._evaluate(strategy, candles_1h, candles_4h, candles_1d, symbol)

        return self._run_study("mean_reversion", objective)

    def tune_macd_divergence(
        self,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        candles_1d: list[Candle],
        symbol: str,
    ) -> TuneResult:
        """Tune MACD Divergence strategy."""
        from strategy.macd_divergence import MACDDivergence, MACDDivConfig

        def objective(trial: optuna.Trial) -> float:
            cfg = MACDDivConfig(
                lookback_candles=trial.suggest_int("lookback_candles", 30, 80),
                min_divergence_bars=trial.suggest_int("min_divergence_bars", 5, 25),
                rsi_oversold=trial.suggest_float("rsi_oversold", 25, 45, step=1),
                rsi_overbought=trial.suggest_float("rsi_overbought", 55, 75, step=1),
                min_volume_ratio=trial.suggest_float("min_volume_ratio", 1.0, 2.0, step=0.1),
                stop_loss_pct=trial.suggest_float("stop_loss_pct", 2.0, 6.0, step=0.5),
                take_profit_pct=trial.suggest_float("take_profit_pct", 4.0, 12.0, step=0.5),
                min_confidence=trial.suggest_float("min_confidence", 0.55, 0.80, step=0.05),
            )
            strategy = MACDDivergence(config=cfg)
            return self._evaluate(strategy, candles_1h, candles_4h, candles_1d, symbol)

        return self._run_study("macd_divergence", objective)

    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────

    def _evaluate(
        self,
        strategy,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        candles_1d: list[Candle],
        symbol: str,
    ) -> float:
        """Run backtest and return objective score."""
        # Slice candles for speed — use only the last N
        max_c = self._cfg.max_candles_tune
        c1h = candles_1h[-max_c:] if len(candles_1h) > max_c else candles_1h
        c4h = candles_4h[-(max_c // 4):] if len(candles_4h) > max_c // 4 else candles_4h
        c1d = candles_1d[-(max_c // 24):] if candles_1d and len(candles_1d) > max_c // 24 else candles_1d

        bt_config = BacktestConfig(
            initial_balance=self._cfg.backtest_balance,
            commission_pct=self._cfg.backtest_commission,
            slippage_pct=self._cfg.backtest_slippage,
        )
        engine = BacktestEngine(config=bt_config)
        result = engine.run(strategy, c1h, c4h, symbol, c1d)

        # Constraint: minimum trades
        if result.total_trades < self._cfg.min_trades:
            return 0.0

        # Constraint: max drawdown
        if result.max_drawdown_pct > self._cfg.max_drawdown_pct:
            return 0.0

        # Objective: profit_factor * sqrt(win_rate/100) * log(n_trades)
        import math
        pf = max(result.profit_factor, 0.01)
        wr = max(result.win_rate, 1.0) / 100.0
        n = max(result.total_trades, 1)

        # Bonus for positive PnL
        pnl_bonus = 1.0 + (result.total_pnl_pct / 100.0) if result.total_pnl_pct > 0 else 0.5

        score = pf * math.sqrt(wr) * math.log(n + 1) * pnl_bonus
        return score

    def _run_study(self, name: str, objective) -> TuneResult:
        """Run Optuna study and return best result."""
        import sys

        study = optuna.create_study(
            direction="maximize",
            study_name=name,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        def _log_trial(study, trial):
            """Log each completed trial for progress visibility."""
            if trial.value and trial.value > 0:
                msg = f"  Trial {trial.number + 1}/{self._cfg.n_trials}: score={trial.value:.3f}"
            else:
                msg = f"  Trial {trial.number + 1}/{self._cfg.n_trials}: score=0 (constraint violated)"
            print(msg, flush=True)

        study.optimize(
            objective,
            n_trials=self._cfg.n_trials,
            show_progress_bar=False,
            callbacks=[_log_trial],
        )

        best = study.best_trial
        best_params = best.params

        logger.info("Tuner [%s]: best score=%.3f after %d trials", name, best.value, len(study.trials))
        for k, v in sorted(best_params.items()):
            logger.info("  %s = %s", k, v)
        sys.stdout.flush()

        return TuneResult(
            strategy_name=name,
            best_params=best_params,
            best_score=best.value,
            best_trades=0,
            best_win_rate=0.0,
            best_pnl=0.0,
            best_profit_factor=0.0,
            n_trials=len(study.trials),
        )

    @staticmethod
    def save_params(results: list[TuneResult], path: str | Path) -> None:
        """Save tuned parameters to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for r in results:
            data[r.strategy_name] = {
                "params": r.best_params,
                "score": r.best_score,
                "n_trials": r.n_trials,
                "tuned_at": int(time.time()),
            }
        with path.open("w") as f:
            json.dump(data, f, indent=2)
        logger.info("Tuned parameters saved to %s", path)

    @staticmethod
    def load_params(path: str | Path) -> dict[str, dict]:
        """Load tuned parameters from JSON."""
        path = Path(path)
        if not path.exists():
            return {}
        with path.open() as f:
            return json.load(f)
