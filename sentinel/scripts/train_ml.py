"""
ML Model Training Script — генерирует сделки через бэктест и обучает ML-фильтр.

Использует все исторические данные из БД, прогоняет стратегии,
конвертирует BacktestTrade → StrategyTrade, обучает MLPredictor.
"""

from __future__ import annotations

import json
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
from strategy.bollinger_breakout import BollingerBreakout, BBBreakoutConfig
from strategy.mean_reversion import MeanReversion, MeanRevConfig
from strategy.macd_divergence import MACDDivergence, MACDDivConfig
from strategy.market_regime import detect_regime, reset_hysteresis as reset_regime
from analyzer.ml_predictor import MLPredictor, MLConfig, REGIME_ENCODING, STRATEGY_REGIME_FIT
from analyzer.strategy_tuner import StrategyTuner, TunerConfig
from risk.dynamic_sltp import calculate_dynamic_sltp


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
        # Phase 2: Enhanced ML features
        cci_at_entry=f.get("cci", 0.0),
        roc_at_entry=f.get("roc", 0.0),
        cmf_at_entry=f.get("cmf", 0.0),
        bb_pct_b_at_entry=f.get("bb_pct_b", 0.5),
        hist_volatility_at_entry=f.get("hist_volatility", 0.0),
        dmi_spread_at_entry=f.get("dmi_spread", 0.0),
        stoch_rsi_at_entry=f.get("stoch_rsi", 0.0),
        price_change_5h_at_entry=f.get("price_change_5h", 0.0),
        momentum_at_entry=f.get("momentum", 0.0),
        rsi_daily_at_entry=f.get("rsi_daily", 0.0),
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
                # Phase 2: Enhanced features
                "cci": features.cci,
                "roc": features.roc,
                "cmf": features.cmf,
                "bb_pct_b": features.bb_pct_b,
                "hist_volatility": features.hist_volatility,
                "dmi_spread": features.dmi_spread,
                "stoch_rsi": features.stoch_rsi,
                "price_change_5h": features.price_change_5m,
                "momentum": features.momentum,
                "rsi_daily": features.rsi_14_daily,
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

def build_all_trades(repo: Repository, settings) -> list:
    """Build StrategyTrade list from backtests — called by both main() and auto-retrain loop.

    Returns flat list of all trades (backward compat).
    Use build_trades_per_symbol() for per-coin separation.
    """
    result = build_trades_per_symbol(repo, settings)
    all_trades: list[StrategyTrade] = []
    for sym_trades in result.values():
        all_trades.extend(sym_trades)
    all_trades.sort(key=lambda t: t.timestamp_open)
    logger.info("build_all_trades: total {} trades", len(all_trades))
    return all_trades


def build_trades_per_symbol(repo: Repository, settings) -> dict:
    """Build StrategyTrade list per symbol from backtests."""
    symbols = settings.trading_symbols

    # Load tuned params if available
    tuned_params = {}
    params_path = BASE_DIR / "data" / "ml_models" / "tuned_params.json"
    if params_path.exists():
        try:
            import json
            with open(params_path) as f:
                saved = json.load(f)
            tuned_params = {item["strategy"]: item["best_params"] for item in saved if "strategy" in item and "best_params" in item}
        except Exception:
            pass

    strategies_map = {
        "ema_crossover_rsi_default": EMACrossoverRSI(),
        "bollinger_breakout_default": BollingerBreakout(),
        "mean_reversion_default": MeanReversion(),
        "macd_divergence_default": MACDDivergence()
    }

    if "ema_crossover_rsi" in tuned_params:
        p = tuned_params["ema_crossover_rsi"]
        ema_cfg = EMAConfig(**{k: v for k, v in p.items() if hasattr(EMAConfig(), k)})
        strategies_map["ema_crossover_rsi"] = EMACrossoverRSI(config=ema_cfg)
    else:
        strategies_map["ema_crossover_rsi"] = strategies_map["ema_crossover_rsi_default"]

    if "bollinger_breakout" in tuned_params:
        p = tuned_params["bollinger_breakout"]
        bb_cfg = BBBreakoutConfig(**{k: v for k, v in p.items() if hasattr(BBBreakoutConfig(), k)})
        strategies_map["bollinger_breakout"] = BollingerBreakout(config=bb_cfg)
    else:
        strategies_map["bollinger_breakout"] = strategies_map["bollinger_breakout_default"]

    if "mean_reversion" in tuned_params:
        p = tuned_params["mean_reversion"]
        mr_cfg = MeanRevConfig(**{k: v for k, v in p.items() if hasattr(MeanRevConfig(), k)})
        strategies_map["mean_reversion"] = MeanReversion(config=mr_cfg)
    else:
        strategies_map["mean_reversion"] = strategies_map["mean_reversion_default"]

    if "macd_divergence" in tuned_params:
        p = tuned_params["macd_divergence"]
        md_cfg = MACDDivConfig(**{k: v for k, v in p.items() if hasattr(MACDDivConfig(), k)})
        strategies_map["macd_divergence"] = MACDDivergence(config=md_cfg)
    else:
        strategies_map["macd_divergence"] = strategies_map["macd_divergence_default"]

    trades_per_symbol: dict[str, list[StrategyTrade]] = {sym: [] for sym in symbols}
    for symbol in symbols:
        logger.info("build_trades_per_symbol: loading candles for {}...", symbol)
        candles_1h = load_candles(repo, symbol, "1h")
        candles_4h = load_candles(repo, symbol, "4h")
        candles_1d = load_candles(repo, symbol, "1d")
        logger.info("  1h={} 4h={} 1d={}", len(candles_1h), len(candles_4h), len(candles_1d))
        for strat_name, strategy in strategies_map.items():
            reset_regime()
            try:
                trades = run_backtest_with_features(strategy, strat_name, candles_1h, candles_4h, candles_1d, symbol)
                trades_per_symbol[symbol].extend(trades)
                logger.info("  {} / {}: {} trades", symbol, strat_name, len(trades))
            except Exception as e:
                logger.error("  Backtest failed {} {}: {}", symbol, strat_name, e)

    for sym, sym_trades in trades_per_symbol.items():
        sym_trades.sort(key=lambda t: t.timestamp_open)
    return trades_per_symbol


def main():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

    settings = load_settings()
    db = Database(BASE_DIR / settings.db_path)
    db.connect()
    repo = Repository(db)

    symbols = settings.trading_symbols

    # ── Auto-Tune strategies if --tune flag is passed ──
    do_tune = "--tune" in sys.argv
    tuned_params = {}

    if do_tune:
        logger.info("🔧 AUTO-TUNE MODE: optimizing strategy parameters with Optuna...")
        tuner = StrategyTuner(TunerConfig(n_trials=25, min_trades=15))

        # Use first symbol for tuning
        tune_symbol = symbols[0]
        logger.info("Loading candles for tuning ({})...", tune_symbol)
        tc_1h = load_candles(repo, tune_symbol, "1h")
        tc_4h = load_candles(repo, tune_symbol, "4h")
        tc_1d = load_candles(repo, tune_symbol, "1d")

        tune_results = []
        try:
            logger.info("Tuning ema_crossover_rsi...")
            r = tuner.tune_ema_crossover(tc_1h, tc_4h, tc_1d, tune_symbol)
            tune_results.append(r)
            tuned_params["ema_crossover_rsi"] = r.best_params
        except Exception as e:
            logger.error("Tuning ema_crossover_rsi failed: {}", e)

        try:
            logger.info("Tuning bollinger_breakout...")
            r = tuner.tune_bollinger(tc_1h, tc_4h, tc_1d, tune_symbol)
            tune_results.append(r)
            tuned_params["bollinger_breakout"] = r.best_params
        except Exception as e:
            logger.error("Tuning bollinger_breakout failed: {}", e)

        try:
            logger.info("Tuning mean_reversion...")
            r = tuner.tune_mean_reversion(tc_1h, tc_4h, tc_1d, tune_symbol)
            tune_results.append(r)
            tuned_params["mean_reversion"] = r.best_params
        except Exception as e:
            logger.error("Tuning mean_reversion failed: {}", e)

        try:
            logger.info("Tuning macd_divergence...")
            r = tuner.tune_macd_divergence(tc_1h, tc_4h, tc_1d, tune_symbol)
            tune_results.append(r)
            tuned_params["macd_divergence"] = r.best_params
        except Exception as e:
            logger.error("Tuning macd_divergence failed: {}", e)

        # Save tuned params
        params_path = BASE_DIR / "data" / "ml_models" / "tuned_params.json"
        tuner.save_params(tune_results, params_path)
        logger.info("Auto-tune complete. {} strategies tuned.", len(tune_results))

    # ── Build strategies with tuned AND default params (doubles training data) ──
    strategies_map = {
        "ema_crossover_rsi_default": EMACrossoverRSI(),
        "bollinger_breakout_default": BollingerBreakout(),
        "mean_reversion_default": MeanReversion(),
        "macd_divergence_default": MACDDivergence()
    }

    if "ema_crossover_rsi" in tuned_params:
        p = tuned_params["ema_crossover_rsi"]
        ema_cfg = EMAConfig(**{k: v for k, v in p.items() if hasattr(EMAConfig, k)})
        strategies_map["ema_crossover_rsi"] = EMACrossoverRSI(config=ema_cfg)
    else:
        # Also map base name to default if not tuned (for predict flow)
        strategies_map["ema_crossover_rsi"] = strategies_map["ema_crossover_rsi_default"]

    if "bollinger_breakout" in tuned_params:
        p = tuned_params["bollinger_breakout"]
        bb_cfg = BBBreakoutConfig(**{k: v for k, v in p.items() if hasattr(BBBreakoutConfig, k)})
        strategies_map["bollinger_breakout"] = BollingerBreakout(config=bb_cfg)
    else:
        strategies_map["bollinger_breakout"] = strategies_map["bollinger_breakout_default"]

    if "mean_reversion" in tuned_params:
        p = tuned_params["mean_reversion"]
        mr_cfg = MeanRevConfig(**{k: v for k, v in p.items() if hasattr(MeanRevConfig, k)})
        strategies_map["mean_reversion"] = MeanReversion(config=mr_cfg)
    else:
        strategies_map["mean_reversion"] = strategies_map["mean_reversion_default"]

    if "macd_divergence" in tuned_params:
        p = tuned_params["macd_divergence"]
        md_cfg = MACDDivConfig(**{k: v for k, v in p.items() if hasattr(MACDDivConfig, k)})
        strategies_map["macd_divergence"] = MACDDivergence(config=md_cfg)
    else:
        strategies_map["macd_divergence"] = strategies_map["macd_divergence_default"]

    strategies = strategies_map

    # ── Collect trades per symbol ──
    trades_per_symbol: dict[str, list[StrategyTrade]] = {sym: [] for sym in symbols}
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
                trades_per_symbol[symbol].extend(trades)
                all_trades.extend(trades)
            except Exception as e:
                logger.error("  Backtest failed: {}", e)

    logger.info("=" * 50)
    logger.info("Total trades from all backtests: {}", len(all_trades))
    for sym, sym_trades in trades_per_symbol.items():
        w = sum(1 for t in sym_trades if t.is_win)
        logger.info("  {}: {} trades, WR {:.1f}%", sym, len(sym_trades), w / len(sym_trades) * 100 if sym_trades else 0)

    if len(all_trades) < 100:
        logger.error("Too few trades for ML training ({}). Need at least 100.", len(all_trades))
        db.close()
        return

    model_dir = BASE_DIR / "data" / "ml_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # min_trades=80 (lowered from 150): per-symbol training gets ~half the data
    # Use settings where available, fallback to proven defaults
    ml_config = MLConfig(
        n_estimators=250,
        max_depth=4,                # conservative for per-symbol (~1200 samples)
        learning_rate=0.01,         # low LR + many trees = stable convergence
        min_child_samples=40,
        min_samples_split=25,
        max_features="sqrt",
        subsample=0.7,
        colsample_bytree=0.7,
        min_trades=max(80, getattr(settings, 'analyzer_min_trades_ml', 80)),
        block_threshold=getattr(settings, 'analyzer_ml_block_threshold', 0.45),
        reduce_threshold=0.60,
        min_precision=getattr(settings, 'analyzer_ml_min_precision', 0.65),
        min_recall=getattr(settings, 'analyzer_ml_min_recall', 0.45),
        min_roc_auc=getattr(settings, 'analyzer_ml_min_roc_auc', 0.55),
        min_skill_score=getattr(settings, 'analyzer_ml_min_skill_score', 0.72),
        cv_splits=5,
        use_lightgbm=True,
        use_xgboost=True,
        max_overfit_gap=0.10,       # relaxed from 0.08: builders now have L1/L2 + AUC-based guard
    )

    # ── Train per-symbol models ──
    all_reports = []
    for symbol in symbols:
        sym_trades = trades_per_symbol[symbol]
        sym_trades.sort(key=lambda t: t.timestamp_open)

        logger.info("=" * 50)
        logger.info("Training ML model for {} ({} trades)...", symbol, len(sym_trades))

        if len(sym_trades) < 50:
            logger.warning("⚠️ {} — too few trades ({}), skipping per-symbol model", symbol, len(sym_trades))
            continue

        predictor = MLPredictor(config=ml_config)
        metrics = predictor.train(sym_trades)

        if metrics is None:
            logger.error("{}: ML training returned None", symbol)
            continue

        logger.info("{} Training Results:", symbol)
        logger.info("  Precision:   {:.3f}", metrics.precision)
        logger.info("  Recall:      {:.3f}", metrics.recall)
        logger.info("  ROC AUC:     {:.3f}", metrics.roc_auc)
        logger.info("  Accuracy:    {:.3f}", metrics.accuracy)
        logger.info("  Skill Score: {:.3f}", metrics.skill_score)
        logger.info("  Train/Test:  {}/{}", metrics.train_samples, metrics.test_samples)

        if predictor._ensemble is not None:
            members = predictor._ensemble.get_member_info()
            logger.info("  Ensemble members ({}):", len(members))
            for m in members:
                logger.info("    [{}] weight={:.3f}", m['tag'], m['weight'])

        if hasattr(predictor, '_feature_selector') and predictor._feature_selector and predictor._feature_selector.is_fitted:
            dropped = predictor._feature_selector.dropped_names
            kept = predictor._feature_selector.selected_names
            logger.info("  Features kept: {}/{}", len(kept), len(kept) + len(dropped))

        if metrics.feature_importances:
            sorted_imp = sorted(metrics.feature_importances.items(), key=lambda x: x[1], reverse=True)
            logger.info("  Top features:")
            for name, imp in sorted_imp[:10]:
                logger.info("    {:<25} {:.4f}", name, imp)

        # Save per-symbol model
        model_path = model_dir / f"ml_predictor_{symbol}.pkl"
        saved = predictor.save_to_file(model_path)
        if saved:
            n_members = predictor._ensemble.member_count() if predictor._ensemble else 0
            logger.info("✅ {} model saved to {} (ensemble members={})", symbol, model_path, n_members)
        else:
            logger.warning("⚠️ {} model save failed — may not have passed quality gate", symbol)

        ensemble_info = predictor._ensemble.get_member_info() if predictor._ensemble else []
        feature_selector_info = {}
        if hasattr(predictor, '_feature_selector') and predictor._feature_selector and predictor._feature_selector.is_fitted:
            feature_selector_info = {
                "kept": predictor._feature_selector.selected_names,
                "dropped": predictor._feature_selector.dropped_names,
            }

        all_reports.append({
            "symbol": symbol,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "total_trades": len(sym_trades),
            "strategies": list(strategies.keys()),
            "model_version": predictor._model_version,
            "ensemble": ensemble_info,
            "feature_selector": feature_selector_info,
            "metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "roc_auc": metrics.roc_auc,
                "accuracy": metrics.accuracy,
                "skill_score": metrics.skill_score,
            },
            "feature_importances": metrics.feature_importances,
            "model_ready": predictor.is_ready,
        })

        if predictor.is_ready:
            logger.info("✅ {} ML model is READY", symbol)
        else:
            logger.warning("⚠️ {} ML model metrics below threshold", symbol)

    # ── Also train unified fallback model (backward compat) ──
    logger.info("=" * 50)
    logger.info("Training unified fallback model on all {} trades...", len(all_trades))
    all_trades.sort(key=lambda t: t.timestamp_open)
    predictor = MLPredictor(config=ml_config)
    metrics = predictor.train(all_trades)

    if metrics is not None:
        model_path = model_dir / "ml_predictor.pkl"
        saved = predictor.save_to_file(model_path)
        if saved:
            logger.info("✅ Unified fallback model saved to {}", model_path)

        all_reports.append({
            "symbol": "UNIFIED",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "total_trades": len(all_trades),
            "symbols": symbols,
            "strategies": list(strategies.keys()),
            "model_version": predictor._model_version,
            "metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "roc_auc": metrics.roc_auc,
                "accuracy": metrics.accuracy,
                "skill_score": metrics.skill_score,
            },
            "model_ready": predictor.is_ready,
        })

    # Save combined training report
    report_path = model_dir / "training_report.json"
    report_path.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Combined report saved to {}", report_path)

    db.close()


if __name__ == "__main__":
    main()
