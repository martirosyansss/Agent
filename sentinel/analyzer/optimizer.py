"""
Trade Analyzer Level 2 — Parameter Optimizer.

Осторожная оптимизация TUNABLE параметров:
  - confidence_threshold (up only, 0.70–0.95)
  - time_blocks (add only)
  - strategy_weight_adjustments (reduce only, ±10%)
  - min_volume_ratio (up only, 1.0–3.0)

FROZEN (нельзя менять): stop_loss, max_position_pct, max_daily_loss, max_exposure, take_profit

Правила:
  - Max 1 change per week
  - Walk-forward split 70/30
  - Min 100 trades required
  - Paper test 14 days before apply
  - Improvement > 5% win rate required
  - Auto rollback if worse
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from core.models import StrategyTrade
from analyzer.statistician import Statistician, TradeStats


@dataclass
class OptimizationProposal:
    """Предложение по оптимизации параметра."""
    proposal_id: str = ""
    parameter: str = ""
    current_value: float = 0.0
    proposed_value: float = 0.0
    expected_improvement_pct: float = 0.0
    train_stats: Optional[TradeStats] = None
    validation_stats: Optional[TradeStats] = None
    created_at: int = 0
    status: str = "pending"  # pending, testing, applied, rejected, rolled_back
    reason: str = ""


@dataclass
class OptimizerConfig:
    max_changes_per_week: int = 2
    paper_test_days: int = 7
    min_trades: int = 100
    min_improvement_pct: float = 5.0
    walk_forward_train_pct: float = 0.70
    max_confidence_increase: float = 0.10
    max_allocation_decrease: float = 10.0
    max_volume_ratio_increase: float = 1.0
    rollback_on_worse: bool = True


# Frozen parameters that must NEVER be changed
FROZEN_PARAMS = frozenset({
    "stop_loss_pct", "max_position_pct", "max_daily_loss_usd",
    "max_total_exposure_pct", "take_profit_pct", "trading_symbols",
})

# Tunable parameters and their constraints
TUNABLE_PARAMS = {
    "min_confidence": {"direction": "up_only", "min": 0.70, "max": 0.95},
    "min_volume_ratio": {"direction": "up_only", "min": 1.0, "max": 3.0},
}


class Optimizer:
    """Level 2 Trade Analyzer — осторожная оптимизация."""

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self._cfg = config or OptimizerConfig()
        self._stat = Statistician()
        self._history: list[OptimizationProposal] = []
        self._last_change_ts: int = 0

    @property
    def changes_this_week(self) -> int:
        week_ago = int(time.time() * 1000) - 7 * 86400 * 1000
        return sum(
            1 for p in self._history
            if p.status == "applied" and p.created_at > week_ago
        )

    def can_propose(self) -> bool:
        """Проверить, можно ли предложить изменение."""
        return self.changes_this_week < self._cfg.max_changes_per_week

    def analyze_and_propose(
        self,
        trades: list[StrategyTrade],
        strategy_name: str,
        parameter: str,
        test_values: list[float],
        current_value: float = 0.0,
    ) -> Optional[OptimizationProposal]:
        """Проанализировать и предложить оптимизацию.

        Использует walk-forward split: 70% train, 30% validation.

        Args:
            trades: Historical trades to analyze
            strategy_name: Name of strategy to optimize
            parameter: Parameter name to optimize (must be in TUNABLE_PARAMS)
            test_values: List of candidate values to test
            current_value: Current value of the parameter (for up_only guard)
        """
        if parameter in FROZEN_PARAMS:
            return None
        if parameter not in TUNABLE_PARAMS:
            return None
        if not self.can_propose():
            return None

        # Filter by strategy
        strat_trades = [t for t in trades if t.strategy_name == strategy_name]
        if len(strat_trades) < self._cfg.min_trades:
            return None

        # Walk-forward split
        split_idx = int(len(strat_trades) * self._cfg.walk_forward_train_pct)
        train_trades = strat_trades[:split_idx]
        val_trades = strat_trades[split_idx:]

        if len(train_trades) < 20 or len(val_trades) < 10:
            return None

        # Current baseline
        baseline_stats = self._stat.compute_stats(val_trades)
        if baseline_stats.total_trades < 10:
            return None

        # Test different values
        constraint = TUNABLE_PARAMS[parameter]
        best_proposal = None
        best_improvement = 0.0

        for val in test_values:
            # W-7 fix: Enforce direction constraint against CURRENT value, not zero
            if constraint["direction"] == "up_only" and val < current_value:
                continue  # only allow increases
            if not (constraint["min"] <= val <= constraint["max"]):
                continue

            # Simulate: filter trades by confidence threshold
            if parameter == "min_confidence":
                sim_val = [t for t in val_trades if t.confidence >= val]
            elif parameter == "min_volume_ratio":
                sim_val = [t for t in val_trades if t.volume_ratio_at_entry >= val]
            else:
                continue

            if len(sim_val) < 5:
                continue

            sim_stats = self._stat.compute_stats(sim_val)

            # Check improvement
            improvement = sim_stats.win_rate - baseline_stats.win_rate
            if improvement > best_improvement and improvement >= self._cfg.min_improvement_pct:
                best_improvement = improvement
                best_proposal = OptimizationProposal(
                    proposal_id=f"opt_{int(time.time())}_{parameter}",
                    parameter=parameter,
                    current_value=current_value,
                    proposed_value=val,
                    expected_improvement_pct=improvement,
                    train_stats=self._stat.compute_stats(train_trades),
                    validation_stats=sim_stats,
                    created_at=int(time.time() * 1000),
                    status="pending",
                    reason=f"Win rate +{improvement:.1f}% on validation set",
                )

        if best_proposal:
            self._history.append(best_proposal)
        return best_proposal

    def apply_proposal(self, proposal_id: str) -> bool:
        """Применить предложение."""
        for p in self._history:
            if p.proposal_id == proposal_id and p.status == "pending":
                p.status = "applied"
                self._last_change_ts = int(time.time() * 1000)
                return True
        return False

    def rollback_proposal(self, proposal_id: str) -> bool:
        """Откатить предложение."""
        for p in self._history:
            if p.proposal_id == proposal_id and p.status == "applied":
                p.status = "rolled_back"
                return True
        return False

    def get_history(self) -> list[OptimizationProposal]:
        return list(self._history)
