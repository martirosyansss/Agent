"""
Trade Analyzer Level 1 — Statistician.

Собирает статистику по закрытым сделкам (StrategyTrade), формирует отчёты.
Не меняет параметры автоматически — только рекомендации и метрики.

Отчёты: weekly (воскресенье) и monthly.
Фильтры: strategy, symbol, market_regime, hour_range, day_of_week.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.models import StrategyTrade


@dataclass
class TradeStats:
    """Агрегированная статистика по набору сделок."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    avg_hold_hours: float = 0.0
    total_commission: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    best_hours: list[int] = field(default_factory=list)
    best_days: list[int] = field(default_factory=list)


class Statistician:
    """Level 1 Trade Analyzer — статистик."""

    def compute_stats(
        self,
        trades: list[StrategyTrade],
        strategy: str | None = None,
        symbol: str | None = None,
        market_regime: str | None = None,
        hour_range: tuple[int, int] | None = None,
        day_of_week: int | None = None,
    ) -> TradeStats:
        """Вычислить статистику с фильтрами."""
        filtered = self._filter(trades, strategy, symbol, market_regime, hour_range, day_of_week)

        if not filtered:
            return TradeStats()

        wins = [t for t in filtered if t.is_win]
        losses = [t for t in filtered if not t.is_win]

        total_won = sum(t.pnl_usd for t in wins) if wins else 0.0
        total_lost = abs(sum(t.pnl_usd for t in losses)) if losses else 0.0

        # Max drawdown (peak-to-trough of cumulative PnL)
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in filtered:
            cum_pnl += t.pnl_usd
            peak = max(peak, cum_pnl)
            dd = peak - cum_pnl
            max_dd = max(max_dd, dd)

        # Best hours / days
        hour_pnl: dict[int, float] = {}
        day_pnl: dict[int, float] = {}
        for t in filtered:
            hour_pnl[t.hour_of_day] = hour_pnl.get(t.hour_of_day, 0) + t.pnl_usd
            day_pnl[t.day_of_week] = day_pnl.get(t.day_of_week, 0) + t.pnl_usd

        best_hours = sorted(hour_pnl, key=hour_pnl.get, reverse=True)[:3]
        best_days = sorted(day_pnl, key=day_pnl.get, reverse=True)[:3]

        pnl_values = [t.pnl_usd for t in filtered]

        return TradeStats(
            total_trades=len(filtered),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(filtered) * 100 if filtered else 0,
            total_pnl=sum(pnl_values),
            avg_pnl=sum(pnl_values) / len(filtered),
            avg_win=total_won / len(wins) if wins else 0,
            avg_loss=-total_lost / len(losses) if losses else 0,
            profit_factor=total_won / total_lost if total_lost > 0 else float("inf") if total_won > 0 else 0,
            max_drawdown=max_dd,
            avg_hold_hours=sum(t.hold_duration_hours for t in filtered) / len(filtered),
            total_commission=sum(t.commission_usd for t in filtered),
            best_trade_pnl=max(pnl_values),
            worst_trade_pnl=min(pnl_values),
            best_hours=best_hours,
            best_days=best_days,
        )

    def compute_by_strategy(self, trades: list[StrategyTrade]) -> dict[str, TradeStats]:
        """Статистика по каждой стратегии."""
        strategies = set(t.strategy_name for t in trades)
        return {s: self.compute_stats(trades, strategy=s) for s in strategies}

    def compute_by_regime(self, trades: list[StrategyTrade]) -> dict[str, TradeStats]:
        """Статистика по каждому режиму рынка."""
        regimes = set(t.market_regime for t in trades if t.market_regime)
        return {r: self.compute_stats(trades, market_regime=r) for r in regimes}

    def format_report(self, stats: TradeStats, title: str = "Trade Report") -> str:
        """Форматировать отчёт в текст."""
        lines = [
            f"═══ {title} ═══",
            f"Trades: {stats.total_trades} | Win: {stats.wins} | Loss: {stats.losses}",
            f"Win Rate: {stats.win_rate:.1f}%",
            f"Total PnL: ${stats.total_pnl:.2f}",
            f"Avg PnL: ${stats.avg_pnl:.2f} | Avg Win: ${stats.avg_win:.2f} | Avg Loss: ${stats.avg_loss:.2f}",
            f"Profit Factor: {stats.profit_factor:.2f}",
            f"Max Drawdown: ${stats.max_drawdown:.2f}",
            f"Avg Hold: {stats.avg_hold_hours:.1f}h",
            f"Commission: ${stats.total_commission:.2f}",
            f"Best Trade: ${stats.best_trade_pnl:.2f} | Worst: ${stats.worst_trade_pnl:.2f}",
        ]
        if stats.best_hours:
            lines.append(f"Best Hours: {stats.best_hours}")
        if stats.best_days:
            lines.append(f"Best Days: {stats.best_days}")
        return "\n".join(lines)

    @staticmethod
    def _filter(
        trades: list[StrategyTrade],
        strategy: str | None = None,
        symbol: str | None = None,
        market_regime: str | None = None,
        hour_range: tuple[int, int] | None = None,
        day_of_week: int | None = None,
    ) -> list[StrategyTrade]:
        result = trades
        if strategy:
            result = [t for t in result if t.strategy_name == strategy]
        if symbol:
            result = [t for t in result if t.symbol == symbol]
        if market_regime:
            result = [t for t in result if t.market_regime == market_regime]
        if hour_range:
            lo, hi = hour_range
            result = [t for t in result if lo <= t.hour_of_day <= hi]
        if day_of_week is not None:
            result = [t for t in result if t.day_of_week == day_of_week]
        return result
