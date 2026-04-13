"""
ML Model Training Script — генерирует сделки через бэктест и обучает ML-фильтр.

Использует все исторические данные из БД, прогоняет стратегии,
конвертирует BacktestTrade → StrategyTrade, обучает MLPredictor.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from loguru import logger

from config import load_settings
from core.models import Candle, StrategyTrade, MarketRegimeType
from database.db import Database
from database.repository import Repository
from backtest.engine import BacktestEngine, BacktestConfig, BacktestTrade
from features.feature_builder import FeatureBuilder
from strategy.ema_crossover_rsi import EMACrossoverRSI, EMAConfig
from strategy.bollinger_breakout import BollingerBreakout
from strategy.mean_reversion import MeanReversion
from strategy.macd_divergence import MACDDivergence
from strategy.market_regime import detect_regime, reset_hysteresis as reset_regime
from analyzer.ml_predictor import MLPredictor, MLConfig, REGIME_ENCODING, STRATEGY_ENCODING


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_candles(repo: Repository, symbol: str, interval: str) -> list[Candle]:
    """Загрузить все свечи из БД и конвертировать в Candle."""
    raw = repo.get_candles(symbol, interval, limit=200_000)
    candles = []
    for c in raw:
        candles.append(Candle(
            timestamp=c["timestamp"],
            symbol=c["symbol"],
            interval=c["interval"],
            open=float(c["open"]),
            high=float(c["high"]),
            low=float(c["low"]),
            close=float(c["close"]),
            volume=float(c.get("volume", 0)),
            trades_count=int(c.get("trades_count", 0)),
        ))
    # Sort by time
    candles.sort(key=lambda x: x.timestamp)
    return candles


def backtest_trade_to_strategy_trade(
    bt: BacktestTrade,
    strategy_name: str,
    features_at_entry: dict | None = None,
    regime: str = "unknown",
    confidence: float = 0.75,
) -> StrategyTrade:
    """Конвертировать BacktestTrade → StrategyTrade для ML."""
    entry_dt = datetime.fromtimestamp(bt.entry_time / 1000, tz=timezone.utc)
    exit_dt = datetime.fromtimestamp(bt.exit_time / 1000, tz=timezone.utc)
    hold_hours = (bt.exit_time - bt.entry_time) / (3600 * 1000)

    f = features_at_entry or {}

    return StrategyTrade(
        trade_id=f"bt_{bt.entry_time}_{strategy_name}",
        symbol=bt.symbol,
        strategy_name=strategy_name,
        market_regime=regime,
        timestamp_open=entry_dt.isoformat(),
        timestamp_close=exit_dt.isoformat(),
        entry_price=bt.entry_price,
        exit_price=bt.exit_price,
        quantity=bt.quantity,
        pnl_usd=bt.pnl,
        pnl_pct=bt.pnl_pct,
        is_win=bt.pnl > 0,
        confidence=confidence,
        hour_of_day=entry_dt.hour,
        day_of_week=entry_dt.weekday(),
        rsi_at_entry=f.get("rsi", 50.0),
        adx_at_entry=f.get("adx", 20.0),
        volume_ratio_at_entry=f.get("volume_ratio", 1.0),
        exit_reason=bt.reason,
        hold_duration_hours=hold_hours,
        commission_usd=bt.commission,
        ema_9_at_entry=f.get("ema_9", 0.0),
        ema_21_at_entry=f.get("ema_21", 0.0),
        bb_bandwidth_at_entry=f.get("bb_bandwidth", 0.0),
        macd_histogram_at_entry=f.get("macd_histogram", 0.0),
        atr_at_entry=f.get("atr", 0.0),
    )


def run_backtest_with_features(
    strategy,
    strategy_name: str,
    candles_1h: list[Candle],
    candles_4h: list[Candle],
    candles_1d: list[Candle] | None,
    symbol: str,
) -> list[StrategyTrade]:
    """Бэктест с сохранением features на момент каждого входа."""
    cfg = BacktestConfig(
        initial_balance=10_000.0,
        commission_pct=0.1,
        slippage_pct=0.05,
        position_size_pct=20.0,
    )
    fb = FeatureBuilder()
    min_history = 55

    balance = cfg.initial_balance
    in_position = False
    entry_price = 0.0
    entry_time = 0
    quantity = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_features: dict = {}
    entry_regime = "unknown"
    entry_confidence = 0.75

    strategy_trades: list[StrategyTrade] = []

    for i in range(min_history, len(candles_1h)):
        candle = candles_1h[i]
        price = candle.close

        window_1h = candles_1h[max(0, i - min_history):i]
        window_4h = [c for c in candles_4h if c.timestamp <= candle.timestamp]
        if len(window_4h) > min_history:
            window_4h = window_4h[-min_history:]

        window_1d = None
        if candles_1d:
            window_1d = [c for c in candles_1d if c.timestamp <= candle.timestamp]
            if len(window_1d) > min_history:
                window_1d = window_1d[-min_history:]

        features = fb.build(symbol, window_1h, window_4h, window_1d)
        if features is None:
            continue

        # Regime detection
        try:
            regime = detect_regime(features)
            regime_str = regime.regime.value
        except Exception:
            regime_str = "unknown"

        signal = strategy.generate_signal(
            features,
            has_open_position=in_position,
            entry_price=entry_price if in_position else None,
        )

        # SL/TP check
        if in_position:
            if stop_loss > 0 and candle.low <= stop_loss:
                exit_price = stop_loss * (1 - cfg.slippage_pct / 100)
                comm = quantity * exit_price * cfg.commission_pct / 100
                pnl = (exit_price - entry_price) * quantity - comm
                balance += pnl
                bt = BacktestTrade(
                    symbol=symbol, entry_time=entry_time, exit_time=candle.timestamp,
                    entry_price=entry_price, exit_price=exit_price,
                    quantity=quantity, pnl=pnl,
                    pnl_pct=(exit_price - entry_price) / entry_price * 100,
                    commission=comm, reason="Stop-loss",
                )
                strategy_trades.append(backtest_trade_to_strategy_trade(
                    bt, strategy_name, entry_features, entry_regime, entry_confidence
                ))
                in_position = False
                continue

            if take_profit > 0 and candle.high >= take_profit:
                exit_price = take_profit * (1 - cfg.slippage_pct / 100)
                comm = quantity * exit_price * cfg.commission_pct / 100
                pnl = (exit_price - entry_price) * quantity - comm
                balance += pnl
                bt = BacktestTrade(
                    symbol=symbol, entry_time=entry_time, exit_time=candle.timestamp,
                    entry_price=entry_price, exit_price=exit_price,
                    quantity=quantity, pnl=pnl,
                    pnl_pct=(exit_price - entry_price) / entry_price * 100,
                    commission=comm, reason="Take-profit",
                )
                strategy_trades.append(backtest_trade_to_strategy_trade(
                    bt, strategy_name, entry_features, entry_regime, entry_confidence
                ))
                in_position = False
                continue

        if signal is None:
            continue

        if signal.direction.value == "BUY" and not in_position:
            entry_price = price * (1 + cfg.slippage_pct / 100)
            position_value = balance * cfg.position_size_pct / 100
            quantity = position_value / entry_price
            comm = quantity * entry_price * cfg.commission_pct / 100
            balance -= comm
            entry_time = candle.timestamp
            stop_loss = signal.stop_loss_price
            take_profit = signal.take_profit_price
            in_position = True
            entry_confidence = signal.confidence
            entry_regime = regime_str
            entry_features = {
                "rsi": features.rsi_14,
                "adx": features.adx,
                "volume_ratio": features.volume_ratio,
                "ema_9": features.ema_9,
                "ema_21": features.ema_21,
                "bb_bandwidth": features.bb_bandwidth,
                "macd_histogram": features.macd_histogram,
                "atr": features.atr,
            }

        elif signal.direction.value == "SELL" and in_position:
            exit_price = price * (1 - cfg.slippage_pct / 100)
            comm = quantity * exit_price * cfg.commission_pct / 100
            pnl = (exit_price - entry_price) * quantity - comm
            balance += pnl
            bt = BacktestTrade(
                symbol=symbol, entry_time=entry_time, exit_time=candle.timestamp,
                entry_price=entry_price, exit_price=exit_price,
                quantity=quantity, pnl=pnl,
                pnl_pct=(exit_price - entry_price) / entry_price * 100,
                commission=comm, reason=signal.reason,
            )
            strategy_trades.append(backtest_trade_to_strategy_trade(
                bt, strategy_name, entry_features, entry_regime, entry_confidence
            ))
            in_position = False

    return strategy_trades


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

    settings = load_settings()
    db = Database(BASE_DIR / settings.db_path)
    db.connect()
    repo = Repository(db)

    symbols = settings.trading_symbols
    strategies = {
        "ema_crossover_rsi": EMACrossoverRSI(),
        "bollinger_breakout": BollingerBreakout(),
        "mean_reversion": MeanReversion(),
        "macd_divergence": MACDDivergence(),
    }

    all_trades: list[StrategyTrade] = []

    for symbol in symbols:
        logger.info("Loading candles for {}...", symbol)
        candles_1h = load_candles(repo, symbol, "1h")
        candles_4h = load_candles(repo, symbol, "4h")
        candles_1d = load_candles(repo, symbol, "1d")
        logger.info("  1h={}, 4h={}, 1d={}", len(candles_1h), len(candles_4h), len(candles_1d))

        for strat_name, strategy in strategies.items():
            logger.info("Running backtest: {} on {}...", strat_name, symbol)
            reset_regime()

            try:
                trades = run_backtest_with_features(
                    strategy, strat_name, candles_1h, candles_4h, candles_1d, symbol
                )
                wins = sum(1 for t in trades if t.is_win)
                losses = len(trades) - wins
                total_pnl = sum(t.pnl_usd for t in trades)
                logger.info("  {} trades (W:{} L:{}) PnL=${:.2f}", len(trades), wins, losses, total_pnl)
                all_trades.extend(trades)
            except Exception as e:
                logger.error("  Backtest failed: {}", e)

    logger.info("=" * 50)
    logger.info("Total trades from all backtests: {}", len(all_trades))
    wins = sum(1 for t in all_trades if t.is_win)
    logger.info("Win rate: {:.1f}% ({}/{})", wins / len(all_trades) * 100 if all_trades else 0, wins, len(all_trades))

    if len(all_trades) < 100:
        logger.error("Too few trades for ML training ({}). Need at least 100.", len(all_trades))
        db.close()
        return

    # Sort by trade open time
    all_trades.sort(key=lambda t: t.timestamp_open)

    # Train ML model with relaxed thresholds for initial training
    ml_config = MLConfig(
        n_estimators=300,
        max_depth=6,
        min_child_samples=15,
        min_samples_split=10,
        min_trades=100,
        block_threshold=0.40,
        min_precision=0.50,
        min_recall=0.45,
        min_roc_auc=0.52,
        min_skill_score=0.50,
        cv_splits=5,
    )
    predictor = MLPredictor(config=ml_config)

    logger.info("Training ML model on {} trades...", len(all_trades))
    metrics = predictor.train(all_trades)

    if metrics is None:
        logger.error("ML training returned None — insufficient data or sklearn missing")
        db.close()
        return

    logger.info("=" * 50)
    logger.info("ML Training Results:")
    logger.info("  Precision:   {:.3f}", metrics.precision)
    logger.info("  Recall:      {:.3f}", metrics.recall)
    logger.info("  ROC AUC:     {:.3f}", metrics.roc_auc)
    logger.info("  Accuracy:    {:.3f}", metrics.accuracy)
    logger.info("  Skill Score: {:.3f}", metrics.skill_score)
    logger.info("  Train/Test:  {}/{}", metrics.train_samples, metrics.test_samples)

    # Top features
    if metrics.feature_importances:
        sorted_imp = sorted(metrics.feature_importances.items(), key=lambda x: x[1], reverse=True)
        logger.info("  Top features:")
        for name, imp in sorted_imp[:10]:
            logger.info("    {:<25} {:.4f}", name, imp)

    # Save model
    model_dir = BASE_DIR / "data" / "ml_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "ml_predictor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": predictor._model,
            "scaler": predictor._scaler,
            "version": predictor._model_version,
            "metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "roc_auc": metrics.roc_auc,
                "accuracy": metrics.accuracy,
                "skill_score": metrics.skill_score,
                "train_samples": metrics.train_samples,
                "test_samples": metrics.test_samples,
                "feature_importances": metrics.feature_importances,
            },
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "total_trades": len(all_trades),
        }, f)

    logger.info("Model saved to {}", model_path)

    # Save training report
    report_path = model_dir / "training_report.json"
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "total_trades": len(all_trades),
        "symbols": symbols,
        "strategies": list(strategies.keys()),
        "metrics": {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "roc_auc": metrics.roc_auc,
            "accuracy": metrics.accuracy,
            "skill_score": metrics.skill_score,
        },
        "feature_importances": metrics.feature_importances,
        "model_ready": predictor.is_ready,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report saved to {}", report_path)

    if predictor.is_ready:
        logger.info("✅ ML model is READY — can be used in shadow mode")
    else:
        logger.warning("⚠️ ML model metrics below threshold — model trained but not activated")

    db.close()


if __name__ == "__main__":
    main()
