"""
Quality Gates — условия перехода Paper → Live.

Проверяет метрики за последние N дней:
- Win Rate > 50%
- PnL > 0
- Max Drawdown < 5%
- Min 50 completed trades
- No critical errors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Результат проверки одного gate."""
    name: str
    passed: bool
    actual: float
    required: str
    message: str


@dataclass
class QualityReport:
    """Полный отчёт quality gates."""
    gates: list[GateResult]
    all_passed: bool
    summary: str


class QualityGates:
    """Ворота качества для перехода Paper → Live."""

    MIN_WIN_RATE: float = 50.0        # %
    MIN_PNL: float = 0.0              # $
    MAX_DRAWDOWN: float = 5.0         # %
    MIN_TRADES: int = 50

    def check(
        self,
        win_rate: float,
        total_pnl: float,
        max_drawdown_pct: float,
        total_trades: int,
        has_critical_errors: bool = False,
    ) -> QualityReport:
        """Проверить все ворота качества.

        Args:
            win_rate: Win Rate в % (0-100).
            total_pnl: Общий PnL в $.
            max_drawdown_pct: Макс просадка в % (0-100).
            total_trades: Количество завершённых сделок.
            has_critical_errors: Были ли критические ошибки.

        Returns:
            QualityReport с результатами всех проверок.
        """
        gates: list[GateResult] = []

        gates.append(GateResult(
            name="Win Rate",
            passed=win_rate > self.MIN_WIN_RATE,
            actual=win_rate,
            required=f"> {self.MIN_WIN_RATE}%",
            message=f"Win Rate: {win_rate:.1f}% (нужно > {self.MIN_WIN_RATE}%)",
        ))

        gates.append(GateResult(
            name="Total PnL",
            passed=total_pnl > self.MIN_PNL,
            actual=total_pnl,
            required=f"> ${self.MIN_PNL:.2f}",
            message=f"PnL: ${total_pnl:.2f} (нужно > ${self.MIN_PNL:.2f})",
        ))

        gates.append(GateResult(
            name="Max Drawdown",
            passed=max_drawdown_pct < self.MAX_DRAWDOWN,
            actual=max_drawdown_pct,
            required=f"< {self.MAX_DRAWDOWN}%",
            message=f"Drawdown: {max_drawdown_pct:.1f}% (нужно < {self.MAX_DRAWDOWN}%)",
        ))

        gates.append(GateResult(
            name="Trade Count",
            passed=total_trades >= self.MIN_TRADES,
            actual=float(total_trades),
            required=f">= {self.MIN_TRADES}",
            message=f"Сделок: {total_trades} (нужно >= {self.MIN_TRADES})",
        ))

        gates.append(GateResult(
            name="Critical Errors",
            passed=not has_critical_errors,
            actual=1.0 if has_critical_errors else 0.0,
            required="0",
            message=f"Критические ошибки: {'ДА ⛔' if has_critical_errors else 'НЕТ ✅'}",
        ))

        all_passed = all(g.passed for g in gates)

        lines = ["📋 QUALITY GATES REPORT", "─" * 35]
        for g in gates:
            icon = "✅" if g.passed else "❌"
            lines.append(f" {icon} {g.message}")
        lines.append("─" * 35)

        if all_passed:
            lines.append("🟢 ВСЕ ВОРОТА ПРОЙДЕНЫ — готов к Live")
            lines.append("⚠️  Требуется ручное подтверждение пользователя!")
        else:
            failed = sum(1 for g in gates if not g.passed)
            lines.append(f"🔴 НЕ ПРОЙДЕНО: {failed} из {len(gates)} ворот")

        summary = "\n".join(lines)

        return QualityReport(
            gates=gates,
            all_passed=all_passed,
            summary=summary,
        )


def test_strategy_skill_on_history(
    trades: list[dict],
    lookback_days: int = 180,
    train_ratio: float = 0.7,
) -> dict:
    """Тест навыка стратегии на исторических сделках.

    Разделяет сделки по времени (70/30), считает метрики
    на тестовой части, которую модель "не видела".

    Args:
        trades: Список сделок [{'pnl': float, 'timestamp': int, ...}].
        lookback_days: Период в днях.
        train_ratio: Доля для обучения.

    Returns:
        dict с skill_score, precision, recall, expected_pnl, confidence.
    """
    if not trades:
        return {
            "skill_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "expected_pnl": 0.0,
            "confidence": "low",
            "message": "Недостаточно данных",
        }

    # Сортировка по времени
    sorted_trades = sorted(trades, key=lambda t: t.get("timestamp", 0))

    # Разделение по времени (без shuffle!)
    split_idx = int(len(sorted_trades) * train_ratio)
    train_set = sorted_trades[:split_idx]
    test_set = sorted_trades[split_idx:]

    if len(test_set) < 10:
        return {
            "skill_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "expected_pnl": 0.0,
            "confidence": "low",
            "message": f"Мало данных в тестовом наборе ({len(test_set)})",
        }

    # Метрики на тестовой части
    wins_test = sum(1 for t in test_set if t.get("pnl", 0) > 0)
    total_test = len(test_set)
    precision = wins_test / total_test if total_test > 0 else 0

    # Win Rate на обучающей части
    wins_train = sum(1 for t in train_set if t.get("pnl", 0) > 0)
    total_train = len(train_set)
    train_wr = wins_train / total_train if total_train > 0 else 0

    # Recall = сколько прибыльных в тесте vs всего прибыльных
    total_wins = wins_train + wins_test
    recall = wins_test / total_wins if total_wins > 0 else 0

    # Skill score: среднее precision и консистентности
    consistency = 1.0 - abs(precision - train_wr) * 2  # чем ближе к train WR — тем лучше
    consistency = max(0, min(1, consistency))
    skill_score = (precision + consistency) / 2

    expected_pnl = sum(t.get("pnl", 0) for t in test_set)

    confidence = "high" if total_test >= 30 else ("medium" if total_test >= 15 else "low")

    return {
        "skill_score": round(skill_score, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "expected_pnl": round(expected_pnl, 2),
        "confidence": confidence,
        "train_trades": total_train,
        "test_trades": total_test,
        "message": f"Skill={skill_score:.1%}, WR test={precision:.1%}, WR train={train_wr:.1%}",
    }
