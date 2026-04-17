"""
Risk Sentinel — абсолютный защитный слой.

Ни один ордер не исполняется без одобрения Risk Sentinel.
Pipeline: Signal → 7 проверок → APPROVED / REJECTED.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from core.models import Direction, RiskCheckResult, RiskState, Signal
from .decision_tracer import (
    DecisionTrace,
    GateOutcome,
    GateTimer,
    feature_snapshot_dict,
)
from .state_machine import RiskStateMachine

if TYPE_CHECKING:
    from .correlation_guard import CorrelationGuard
    from .drawdown_breaker import DrawdownBreaker
    from .exposure_caps import ExposureCapGuard, OpenPositionExposure
    from strategy.multi_tf_gate import MultiTFGate

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Настраиваемые лимиты (зажаты absolute_limits в config)."""
    max_daily_loss_usd: float = 25.0      # $25 = 5% of $500 (safer for paper testing)
    max_daily_loss_pct: float = 5.0       # professional standard: max 5% daily drawdown
    max_daily_trades: int = 6
    max_position_pct: float = 20.0
    max_total_exposure_pct: float = 60.0
    max_open_positions: int = 5
    max_trades_per_hour: int = 2
    min_trade_interval_sec: int = 1800
    min_order_usd: float = 10.0
    max_order_usd: float = 100.0
    max_loss_per_trade_pct: float = 3.0   # legacy: used as fallback SL% cap
    max_risk_per_trade_pct: float = 3.0   # NEW: max % of PORTFOLIO at risk per trade
    mandatory_stop_loss: bool = True
    min_rr_ratio: float = 1.5               # minimum risk:reward ratio for BUY
    max_daily_commission_pct: float = 1.0


class RiskSentinel:
    """Главный модуль проверки рисков.

    Каждый сигнал проходит 7 проверок:
    1. Daily Loss
    2. Position Limit
    3. Exposure
    4. Frequency
    5. Order Size
    6. Stop-Loss
    7. Sanity Check
    """

    def __init__(
        self,
        limits: RiskLimits,
        state_machine: RiskStateMachine,
        drawdown_breaker: Optional["DrawdownBreaker"] = None,
        correlation_guard: Optional["CorrelationGuard"] = None,
        exposure_cap_guard: Optional["ExposureCapGuard"] = None,
        multi_tf_gate: Optional["MultiTFGate"] = None,
        regime_gate: Optional["object"] = None,
        news_cooldown: Optional["object"] = None,
        liquidity_gate: Optional["object"] = None,
        stale_data_gate: Optional["object"] = None,
        circuit_breakers: Optional["object"] = None,
    ) -> None:
        self._limits = limits
        self._sm = state_machine
        self._trades_timestamps: list[float] = []  # timestamps of recent trades
        self._last_trade_ts: float = 0.0
        self._daily_trades: int = 0
        self._daily_commission: float = 0.0
        # Optional guards: injected by main.py wiring. When None, the
        # corresponding pre-trade check is skipped (back-compat with existing tests).
        self._drawdown_breaker = drawdown_breaker
        self._correlation_guard = correlation_guard
        self._exposure_cap_guard = exposure_cap_guard
        self._multi_tf_gate = multi_tf_gate
        self._regime_gate = regime_gate
        self._news_cooldown = news_cooldown
        self._liquidity_gate = liquidity_gate
        self._stale_data_gate = stale_data_gate
        self._circuit_breakers = circuit_breakers

    # ──────────────────────────────────────────────
    # Guard wiring (used by main.py at startup)
    # ──────────────────────────────────────────────

    def attach_drawdown_breaker(self, breaker: "DrawdownBreaker") -> None:
        self._drawdown_breaker = breaker

    def attach_correlation_guard(self, guard: "CorrelationGuard") -> None:
        self._correlation_guard = guard

    def attach_exposure_cap_guard(self, guard: "ExposureCapGuard") -> None:
        self._exposure_cap_guard = guard

    def attach_multi_tf_gate(self, gate: "MultiTFGate") -> None:
        self._multi_tf_gate = gate

    def attach_regime_gate(self, gate) -> None:
        self._regime_gate = gate

    def attach_news_cooldown(self, cooldown) -> None:
        self._news_cooldown = cooldown

    def attach_liquidity_gate(self, gate) -> None:
        self._liquidity_gate = gate

    def attach_stale_data_gate(self, gate) -> None:
        self._stale_data_gate = gate

    def attach_circuit_breakers(self, cbs) -> None:
        self._circuit_breakers = cbs

    # ──────────────────────────────────────────────
    # Traced evaluation (structured decision audit)
    # ──────────────────────────────────────────────

    def evaluate_with_trace(
        self,
        signal: Signal,
        daily_pnl: float,
        open_positions_count: int,
        total_exposure_pct: float,
        balance: float,
        current_market_price: float,
        open_symbols: Optional[set[str]] = None,
        price_history: Optional[dict[str, list[float]]] = None,
        open_positions_exposure: Optional[list["OpenPositionExposure"]] = None,
        shadow_mode: bool = False,
        market_data_age_sec: Optional[float] = None,
    ) -> tuple[RiskCheckResult, DecisionTrace]:
        """Run all gates, recording each verdict on a DecisionTrace.

        Args:
            shadow_mode: if True, evaluate EVERY gate even after a rejection
                — useful for analytics ("would the regime gate also have
                blocked this trade the multi-TF gate just rejected?").
                Production callers leave it False (short-circuit on first
                REJECTED) to avoid wasted work and to match the semantics
                of ``check_signal``.

        Returns:
            (RiskCheckResult, DecisionTrace) — the result mirrors
            ``check_signal`` for back-compat; the trace carries the full
            structured audit for the EventLog and the decision_audit table.
        """
        trace = DecisionTrace(
            signal_id=getattr(signal, "signal_id", "") or "",
            symbol=signal.symbol,
            strategy=signal.strategy_name,
            direction=signal.direction.value,
            confidence=signal.confidence,
            feature_snapshot=feature_snapshot_dict(signal.features),
            short_circuit=not shadow_mode,
        )

        is_sell = signal.direction == Direction.SELL
        rejected_first: Optional[RiskCheckResult] = None

        def _maybe_short_circuit(result: RiskCheckResult) -> bool:
            """Track first rejection; return True if pipeline should stop."""
            nonlocal rejected_first
            if not result.approved and rejected_first is None:
                rejected_first = result
            return rejected_first is not None and not shadow_mode

        # ── [-8] Circuit Breakers ──
        # Hard-block BUY when any of the 8 CBs is tripped. Independent of the
        # other guards: reacts to fast anomalies (price jumps, spread blowouts,
        # API error storms) detected by feeds from collector/executor.
        with GateTimer(trace, "circuit_breakers") as t:
            if is_sell:
                t.skipped("SELL bypasses entry gates")
            elif self._circuit_breakers is None:
                t.skipped("not configured")
            else:
                if self._circuit_breakers.is_trading_allowed():
                    t.record(True, "no CB active")
                else:
                    active = self._circuit_breakers.get_active_breakers()
                    reason = f"Circuit breaker(s) active: {','.join(active)}"
                    t.record(False, reason, payload={"active": active})
                    if _maybe_short_circuit(RiskCheckResult(False, reason)):
                        return self._finalise(trace, rejected_first)

        # ── [-7] Stale-data gate ──
        # Block BUY when WS market-data freshness is degraded. SELL always passes.
        with GateTimer(trace, "stale_data") as t:
            if is_sell:
                t.skipped("SELL bypasses entry gates")
            elif self._stale_data_gate is None:
                t.skipped("not configured")
            else:
                d = self._stale_data_gate.check(
                    direction=signal.direction,
                    data_age_sec=market_data_age_sec,
                )
                t.record(d.approved, d.reason,
                         payload={"data_age_sec": getattr(d, "data_age_sec", None)})
                if _maybe_short_circuit(RiskCheckResult(d.approved, d.reason)):
                    return self._finalise(trace, rejected_first)

        # ── [-6] Multi-TF gate ──
        with GateTimer(trace, "multi_tf") as t:
            if is_sell:
                t.skipped("SELL bypasses entry gates")
            elif self._multi_tf_gate is None or signal.features is None:
                t.skipped("not configured")
            else:
                from strategy.multi_tf_gate import classify_strategy
                strat_type = classify_strategy(signal.strategy_name)
                d = self._multi_tf_gate.check(
                    direction=signal.direction,
                    features=signal.features,
                    strategy_type=strat_type,
                )
                t.record(d.approved, d.reason, payload={"checks": getattr(d, "checks", {})})
                if _maybe_short_circuit(RiskCheckResult(d.approved, d.reason)):
                    return self._finalise(trace, rejected_first)

        # ── [-5] Regime gate ──
        with GateTimer(trace, "regime") as t:
            if is_sell:
                t.skipped("SELL bypasses entry gates")
            elif self._regime_gate is None or signal.features is None:
                t.skipped("not configured")
            else:
                d = self._regime_gate.check(signal.strategy_name, signal.features)
                t.record(d.approved, d.reason,
                         payload={"strategy_type": d.strategy_type, "regime": d.regime})
                if _maybe_short_circuit(RiskCheckResult(d.approved, d.reason)):
                    return self._finalise(trace, rejected_first)

        # ── [-4] News cooldown ──
        with GateTimer(trace, "news_cooldown") as t:
            if is_sell:
                t.skipped("SELL bypasses entry gates")
            elif self._news_cooldown is None or signal.features is None:
                t.skipped("not configured")
            else:
                d = self._news_cooldown.check(features=signal.features)
                t.record(d.approved, d.reason,
                         payload={"cooldown_remaining_sec": d.cooldown_remaining_sec})
                if _maybe_short_circuit(RiskCheckResult(d.approved, d.reason)):
                    return self._finalise(trace, rejected_first)

        # ── [-3.5] Liquidity gate ──
        with GateTimer(trace, "liquidity") as t:
            if is_sell:
                t.skipped("SELL bypasses entry gates")
            elif self._liquidity_gate is None or signal.features is None:
                t.skipped("not configured")
            else:
                d = self._liquidity_gate.check(
                    direction=signal.direction,
                    features=signal.features,
                    candidate_notional_usd=signal.suggested_quantity * current_market_price,
                )
                t.record(d.approved, d.reason,
                         payload={"volume_ratio": d.volume_ratio,
                                  "order_pct_of_volume": d.order_pct_of_volume})
                if _maybe_short_circuit(RiskCheckResult(d.approved, d.reason)):
                    return self._finalise(trace, rejected_first)

        # ── [-3] Drawdown breaker ──
        with GateTimer(trace, "drawdown") as t:
            if is_sell:
                t.skipped("SELL bypasses entry gates")
            elif self._drawdown_breaker is None:
                t.skipped("not configured")
            else:
                self._drawdown_breaker.update(balance)
                if self._drawdown_breaker.allows_new_entry():
                    t.record(True, "no DD breach")
                else:
                    trips = ",".join(self._drawdown_breaker.active_trips())
                    t.record(False, f"Drawdown breaker tripped: {trips}",
                             payload={"active_trips": self._drawdown_breaker.active_trips(),
                                      "balance": balance})
                    if _maybe_short_circuit(RiskCheckResult(False, f"Drawdown breaker tripped: {trips}")):
                        return self._finalise(trace, rejected_first)

        # ── [-2] Correlation guard ──
        with GateTimer(trace, "correlation") as t:
            if is_sell or self._correlation_guard is None or not open_symbols or price_history is None:
                t.skipped("not configured / no open positions / no price history")
            else:
                d = self._correlation_guard.check(
                    candidate_symbol=signal.symbol,
                    open_symbols=open_symbols,
                    price_history=price_history,
                )
                t.record(d.approved, d.reason,
                         payload={"cluster": list(d.cluster),
                                  "effective_positions": d.effective_positions,
                                  "pair_correlations": d.pair_correlations})
                if _maybe_short_circuit(RiskCheckResult(d.approved, d.reason)):
                    return self._finalise(trace, rejected_first)

        # ── [-1] Asset-class exposure cap ──
        with GateTimer(trace, "exposure_cap") as t:
            if is_sell or self._exposure_cap_guard is None or open_positions_exposure is None:
                t.skipped("not configured / SELL")
            else:
                d = self._exposure_cap_guard.check(
                    candidate_symbol=signal.symbol,
                    candidate_notional_usd=signal.suggested_quantity * current_market_price,
                    equity_usd=balance,
                    open_positions=open_positions_exposure,
                )
                t.record(d.approved, d.reason,
                         payload={"asset_class": d.asset_class,
                                  "class_exposure_pct_after": d.class_exposure_pct_after,
                                  "class_cap_pct": d.class_cap_pct})
                if _maybe_short_circuit(RiskCheckResult(d.approved, d.reason)):
                    return self._finalise(trace, rejected_first)

        # ── Legacy account-level gates [0..8] — delegate to the shared
        # private helper, avoiding the prior detach-and-call dance which
        # re-executed every pro-guard (with side effects on news cooldown).
        with GateTimer(trace, "legacy_checks") as t:
            legacy_result = self._run_legacy_post_checks(
                signal=signal,
                daily_pnl=daily_pnl,
                open_positions_count=open_positions_count,
                total_exposure_pct=total_exposure_pct,
                balance=balance,
                current_market_price=current_market_price,
                open_symbols=open_symbols,
            )
            t.record(legacy_result.approved, legacy_result.reason)
            if not legacy_result.approved and rejected_first is None:
                rejected_first = legacy_result

        return self._finalise(trace, rejected_first)

    def _finalise(
        self,
        trace: DecisionTrace,
        rejected: Optional[RiskCheckResult],
    ) -> tuple[RiskCheckResult, DecisionTrace]:
        if rejected is not None:
            trace.final_outcome = GateOutcome.REJECTED
            trace.final_reason = rejected.reason
            return rejected, trace
        trace.final_outcome = GateOutcome.APPROVED
        trace.final_reason = "All gates passed"
        return RiskCheckResult(approved=True, reason=trace.final_reason), trace

    @property
    def state(self) -> RiskState:
        return self._sm.state

    @property
    def daily_trades(self) -> int:
        return self._daily_trades

    @property
    def daily_commission(self) -> float:
        return self._daily_commission

    @property
    def trades_last_hour(self) -> int:
        hour_ago = time.time() - 3600
        return sum(1 for ts in self._trades_timestamps if ts > hour_ago)

    @property
    def cooldown_remaining_sec(self) -> int:
        if self._last_trade_ts <= 0:
            return 0
        elapsed = time.time() - self._last_trade_ts
        remaining = self._limits.min_trade_interval_sec - elapsed
        return max(0, int(remaining))

    def get_runtime_metrics(self, balance: float = 0.0) -> dict[str, float | int | str]:
        commission_pct = self._daily_commission / balance * 100 if balance > 0 else 0.0
        return {
            "state": self.state.value,
            "daily_trades": self._daily_trades,
            "trades_last_hour": self.trades_last_hour,
            "daily_commission": self._daily_commission,
            "commission_pct": commission_pct,
            "cooldown_remaining_sec": self.cooldown_remaining_sec,
        }

    # ──────────────────────────────────────────────
    # Main check
    # ──────────────────────────────────────────────

    def check_signal(
        self,
        signal: Signal,
        daily_pnl: float,
        open_positions_count: int,
        total_exposure_pct: float,
        balance: float,
        current_market_price: float,
        open_symbols: Optional[set[str]] = None,
        price_history: Optional[dict[str, list[float]]] = None,
        open_positions_exposure: Optional[list["OpenPositionExposure"]] = None,
        market_data_age_sec: Optional[float] = None,
    ) -> RiskCheckResult:
        """Проверить сигнал по 7 правилам.

        Returns:
            RiskCheckResult(approved=True/False, reason=...)
        """
        # SELL (closing positions) must NEVER be blocked — holding a losing
        # position is always riskier than closing it.
        is_sell = signal.direction == Direction.SELL

        # [-8] Circuit breakers: hard-block BUY when any CB tripped.
        if not is_sell and self._circuit_breakers is not None:
            if not self._circuit_breakers.is_trading_allowed():
                active = self._circuit_breakers.get_active_breakers()
                return RiskCheckResult(
                    approved=False,
                    reason=f"Circuit breaker(s) active: {','.join(active)}",
                )

        # [-7] Stale-data gate: block BUY when WS data is too old.
        if not is_sell and self._stale_data_gate is not None:
            sd_decision = self._stale_data_gate.check(
                direction=signal.direction,
                data_age_sec=market_data_age_sec,
            )
            if not sd_decision.approved:
                return RiskCheckResult(approved=False, reason=sd_decision.reason)

        # [-6] Multi-TF AND-gate: trend strategies must agree across 4h + 1d.
        # The single biggest cause of small-account blow-ups is taking
        # higher-TF-counter-trend trades that the 1h indicator says are valid.
        if not is_sell and self._multi_tf_gate is not None and signal.features is not None:
            from strategy.multi_tf_gate import classify_strategy
            strat_type = classify_strategy(signal.strategy_name)
            tf_decision = self._multi_tf_gate.check(
                direction=signal.direction,
                features=signal.features,
                strategy_type=strat_type,
            )
            if not tf_decision.approved:
                return RiskCheckResult(approved=False, reason=tf_decision.reason)

        # [-5] Regime gate: hard-block trend BUY in adverse regimes
        # (trending_down / volatile / transitioning). Adaptive thresholds
        # in adaptive_min_confidence are SOFT — a strategy with high
        # confidence can override them. This gate is HARD.
        if not is_sell and self._regime_gate is not None and signal.features is not None:
            rg_decision = self._regime_gate.check(
                strategy_name=signal.strategy_name,
                features=signal.features,
            )
            if not rg_decision.approved:
                return RiskCheckResult(approved=False, reason=rg_decision.reason)

        # [-4] News cooldown: lock out new entries for N hours after a critical
        # bearish event, even after the live alert flag drops. Markets re-price
        # on event memory, not on the headline timestamp.
        if not is_sell and self._news_cooldown is not None and signal.features is not None:
            nc_decision = self._news_cooldown.check(features=signal.features)
            if not nc_decision.approved:
                return RiskCheckResult(approved=False, reason=nc_decision.reason)

        # [-3.5] Liquidity gate: thin-market protection. Block BUY when current
        # volume is well below average or the order would dominate available
        # liquidity (large slippage risk).
        if not is_sell and self._liquidity_gate is not None and signal.features is not None:
            lq_decision = self._liquidity_gate.check(
                direction=signal.direction,
                features=signal.features,
                candidate_notional_usd=signal.suggested_quantity * current_market_price,
            )
            if not lq_decision.approved:
                return RiskCheckResult(approved=False, reason=lq_decision.reason)

        # [-3] Drawdown breaker: blocks BUY when daily/weekly/monthly DD breached.
        if not is_sell and self._drawdown_breaker is not None:
            self._drawdown_breaker.update(balance)
            if not self._drawdown_breaker.allows_new_entry():
                trips = ",".join(self._drawdown_breaker.active_trips())
                return RiskCheckResult(
                    approved=False,
                    reason=f"Drawdown breaker tripped: {trips}",
                )

        # [-2] Correlation guard: blocks BUY that joins an oversized cluster.
        if (
            not is_sell
            and self._correlation_guard is not None
            and price_history is not None
            and open_symbols
        ):
            corr_decision = self._correlation_guard.check(
                candidate_symbol=signal.symbol,
                open_symbols=open_symbols,
                price_history=price_history,
            )
            if not corr_decision.approved:
                return RiskCheckResult(approved=False, reason=corr_decision.reason)

        # [-1] Asset-class exposure cap.
        if (
            not is_sell
            and self._exposure_cap_guard is not None
            and open_positions_exposure is not None
            and current_market_price > 0
        ):
            cap_decision = self._exposure_cap_guard.check(
                candidate_symbol=signal.symbol,
                candidate_notional_usd=signal.suggested_quantity * current_market_price,
                equity_usd=balance,
                open_positions=open_positions_exposure,
            )
            if not cap_decision.approved:
                return RiskCheckResult(approved=False, reason=cap_decision.reason)

        # Legacy account-level checks [0..8] — extracted so ``evaluate_with_trace``
        # can invoke them directly without the detach-and-call dance (which
        # re-ran every pro-guard).
        return self._run_legacy_post_checks(
            signal=signal,
            daily_pnl=daily_pnl,
            open_positions_count=open_positions_count,
            total_exposure_pct=total_exposure_pct,
            balance=balance,
            current_market_price=current_market_price,
            open_symbols=open_symbols,
        )

    def _run_legacy_post_checks(
        self,
        signal: Signal,
        daily_pnl: float,
        open_positions_count: int,
        total_exposure_pct: float,
        balance: float,
        current_market_price: float,
        open_symbols: Optional[set[str]] = None,
    ) -> RiskCheckResult:
        """Account-level legacy checks [0..8] — state/daily-loss/caps/SL/R:R.

        Pure function of the signal + account state; no dependency on any
        pro-guard. Callable independently from both ``check_signal`` and
        ``evaluate_with_trace``.
        """
        is_sell = signal.direction == Direction.SELL

        # [0] State check
        if self._sm.state == RiskState.STOP and not is_sell:
            return RiskCheckResult(approved=False, reason="Trading stopped: STOP state")

        if self._sm.state == RiskState.SAFE and signal.direction == Direction.BUY:
            return RiskCheckResult(approved=False, reason="SAFE state: only SELL allowed")

        if self._sm.state == RiskState.REDUCED and signal.confidence < 0.8 and not is_sell:
            return RiskCheckResult(
                approved=False,
                reason=f"REDUCED state: requires confidence >= 0.8, got {signal.confidence:.2f}",
            )

        # [1] Daily Loss (never block SELL — must be able to close losing positions)
        if daily_pnl <= -self._limits.max_daily_loss_usd and not is_sell:
            return RiskCheckResult(
                approved=False,
                reason=f"Daily loss limit exhausted: ${daily_pnl:.2f}",
            )

        # [2] Position Limit (only for BUY)
        if signal.direction == Direction.BUY:
            if open_positions_count >= self._limits.max_open_positions:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Max open positions reached: {open_positions_count}",
                )

        # [3] Exposure (only for BUY)
        if signal.direction == Direction.BUY:
            if balance <= 0:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Invalid balance: ${balance:.2f}",
                )
            order_value = signal.suggested_quantity * current_market_price
            new_exposure_pct = total_exposure_pct + (order_value / balance * 100)
            # Correlation cluster cap: BTC+ETH are ~0.85 correlated — treating
            # them as independent diversifies the paper but not the risk. Block
            # the second entry into the same cluster entirely.
            correlated_symbols = {"BTCUSDT", "ETHUSDT"}
            if signal.symbol in correlated_symbols and open_symbols:
                cluster_overlap = [
                    s for s in open_symbols
                    if s in correlated_symbols and s != signal.symbol
                ]
                if cluster_overlap:
                    return RiskCheckResult(
                        approved=False,
                        reason=(
                            f"Correlated cluster exposure: already long "
                            f"{','.join(cluster_overlap)} (~0.85 corr with {signal.symbol})"
                        ),
                    )
            if new_exposure_pct > self._limits.max_total_exposure_pct:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Exposure limit exceeded: {new_exposure_pct:.1f}% > {self._limits.max_total_exposure_pct}%",
                )

        # [4] Frequency
        now = time.time()
        if signal.direction == Direction.BUY:
            if self._daily_trades >= self._limits.max_daily_trades:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Daily trade limit reached: {self._daily_trades}",
                )

            # Trades this hour
            hour_ago = now - 3600
            trades_this_hour = sum(1 for ts in self._trades_timestamps if ts > hour_ago)
            if trades_this_hour >= self._limits.max_trades_per_hour:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Hourly trade limit reached: {trades_this_hour}",
                )

            # Min interval
            if self._last_trade_ts > 0:
                elapsed = now - self._last_trade_ts
                if elapsed < self._limits.min_trade_interval_sec:
                    remaining = int(self._limits.min_trade_interval_sec - elapsed)
                    return RiskCheckResult(
                        approved=False,
                        reason=f"Min trade interval: wait {remaining}s more",
                    )

        # [5] Order Size
        if signal.direction == Direction.BUY:
            order_usd = signal.suggested_quantity * current_market_price
            if order_usd < self._limits.min_order_usd:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Order too small: ${order_usd:.2f} < ${self._limits.min_order_usd}",
                )
            # Max check — reduce instead of reject
            if order_usd > self._limits.max_order_usd:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Order too large: ${order_usd:.2f} > ${self._limits.max_order_usd}",
                )

        # [6] Stop-Loss — constant dollar risk model
        # Instead of capping SL% (which kills ATR-adaptive SL), cap the DOLLAR risk:
        # risk_usd = sl_pct% × order_value ≤ balance × max_risk_per_trade_pct%
        # Wider SL is fine if position size is proportionally smaller.
        if self._limits.mandatory_stop_loss and signal.direction == Direction.BUY:
            if signal.stop_loss_price <= 0:
                return RiskCheckResult(
                    approved=False,
                    reason="Stop-loss is mandatory but not set",
                )
            sl_pct = abs(current_market_price - signal.stop_loss_price) / current_market_price * 100
            # Hard ceiling: SL > 8% is never acceptable (likely a bug)
            if sl_pct > 8.0:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Stop-loss extreme: {sl_pct:.1f}% > 8.0% hard ceiling",
                )
            # Constant dollar risk: risk_usd must be ≤ max_risk_per_trade_pct% of portfolio
            # Include 0.1% exit commission in risk (Binance spot = 0.1% per trade)
            if balance > 0:
                order_value = signal.suggested_quantity * current_market_price
                risk_usd = order_value * (sl_pct + 0.10) / 100  # SL% + 0.1% exit commission
                max_risk_usd = balance * self._limits.max_risk_per_trade_pct / 100
                if risk_usd > max_risk_usd:
                    return RiskCheckResult(
                        approved=False,
                        reason=f"Dollar risk too high: ${risk_usd:.2f} > ${max_risk_usd:.2f} "
                               f"(SL={sl_pct:.1f}%, order=${order_value:.2f})",
                    )

        # [7] Minimum Risk:Reward ratio
        if signal.direction == Direction.BUY and current_market_price > 0:
            if signal.stop_loss_price > 0 and signal.take_profit_price > 0:
                risk = abs(current_market_price - signal.stop_loss_price)
                reward = abs(signal.take_profit_price - current_market_price)
                if risk > 0:
                    rr_ratio = reward / risk
                    if rr_ratio < self._limits.min_rr_ratio:
                        return RiskCheckResult(
                            approved=False,
                            reason=f"R:R too low: {rr_ratio:.2f} < {self._limits.min_rr_ratio}",
                        )

        # [8] Sanity Check
        if signal.direction == Direction.BUY and current_market_price > 0:
            if signal.suggested_quantity <= 0:
                return RiskCheckResult(
                    approved=False,
                    reason="Invalid order quantity <= 0",
                )

        # ✅ All checks passed
        logger.info(
            "Risk APPROVED: %s %s conf=%.2f",
            signal.direction.value, signal.symbol, signal.confidence,
        )
        return RiskCheckResult(approved=True, reason="All risk checks passed")

    # ──────────────────────────────────────────────
    # Trade recording
    # ──────────────────────────────────────────────

    def record_trade(self, commission: float = 0.0, *, increment_trade: bool = True) -> None:
        """Зарегистрировать совершённую сделку."""
        now = time.time()
        if increment_trade:
            self._trades_timestamps.append(now)
            self._last_trade_ts = now
            self._daily_trades += 1
        self._daily_commission += commission

        # Очистка старых timestamps (>24h)
        cutoff = now - 86400
        self._trades_timestamps = [ts for ts in self._trades_timestamps if ts > cutoff]

    def reset_daily(self) -> None:
        """Сброс дневных счётчиков."""
        self._daily_trades = 0
        self._daily_commission = 0.0
        self._sm.reset()
        logger.info("Risk Sentinel daily reset")
