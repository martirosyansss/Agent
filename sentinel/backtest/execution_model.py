"""
Realistic execution model for backtests — slippage, market impact, gap-aware fills.

The default ``BacktestEngine`` uses a flat ``slippage_pct`` (e.g. 0.05%) on
every fill and assumes stop-loss exits at the trigger price minus that flat
slip. That model produces optimistic equity curves: it underestimates
slippage on large orders and ignores overnight / weekend gaps that punch
straight through stops.

This module replaces the flat assumption with three components that
compose into a realistic fill price:

1. **Spread cost** — half the bid-ask spread in basis points. For BTC/ETH
   this is ~1-3 bps in normal regimes; widens to 10-30 bps during news.
2. **Market impact** — square-root model: impact_bps = k · sqrt(qty / adv)
   where ADV is the average daily volume. Calibrated so a $10k order in a
   $1M ADV pair pays ~10 bps; scales with sqrt of size.
3. **Gap handling** — when a candle's open is already past the stop-loss
   level (overnight / weekend gap), the realistic fill is the OPEN price,
   not the stop price. This matters: a 10% gap-down on FB-style earnings
   converts a 3% stop into a 10% loss in real life.

The model is deterministic (no random component) by default. A controlled
``volatility_jitter_bps`` can be injected for stress-testing — disabled by
default to keep backtest results reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FillReason(str, Enum):
    """Why the fill happened, used by the engine for trade tagging."""
    SIGNAL = "signal"           # normal entry/exit on a generated signal
    STOP_LOSS = "stop_loss"
    STOP_LOSS_GAP = "stop_loss_gap"   # gap-through stop — worse fill than SL price
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_GAP = "take_profit_gap"  # gap-through TP — better fill than TP price


@dataclass(frozen=True)
class ExecutionConfig:
    """Tunables for the realistic execution model.

    Defaults are calibrated for liquid spot crypto (BTC/ETH/major alts).
    For low-liquidity altcoins or other asset classes, override:
      - half_spread_bps: 1 bp ≈ 0.01%; raise to 5-10 for thinly traded coins
      - impact_coefficient: higher = more impact per $ traded
      - reference_adv_usd: typical daily $ volume the calibration assumes
    """
    half_spread_bps: float = 1.5         # ~3 bps round-trip spread on BTC/ETH
    impact_coefficient: float = 10.0     # bps at 100% of reference_adv_usd traded
    reference_adv_usd: float = 1_000_000.0  # 1M USD reference daily volume
    min_slippage_bps: float = 0.5        # floor — never assume a free fill
    max_slippage_bps: float = 200.0      # ceiling — anything above is a halt
    apply_gap_penalty: bool = True
    # Stress-test only: random component in bps. 0 = deterministic.
    volatility_jitter_bps: float = 0.0


@dataclass(frozen=True)
class FillResult:
    """Outcome of one simulated fill."""
    fill_price: float
    slippage_bps: float
    reason: FillReason
    notes: str = ""


def _bps_to_factor(bps: float) -> float:
    return bps / 10_000.0


class RealisticExecutionModel:
    """Composes spread + impact + gap into a single fill price.

    Stateless. Inject the same instance into a ``BacktestEngine`` and reuse
    across runs; each call is independent.
    """

    def __init__(self, config: Optional[ExecutionConfig] = None) -> None:
        self._cfg = config or ExecutionConfig()

    # ──────────────────────────────────────────────
    # Slippage components
    # ──────────────────────────────────────────────

    def _impact_bps(self, notional_usd: float, adv_usd: Optional[float]) -> float:
        """Square-root market-impact in basis points.

        impact_bps = k · sqrt(notional / ADV)

        With defaults: a $100k trade in a $1M ADV pair → ~3.16 bps;
                       a $10k trade in the same pair → ~1.0 bps.
        """
        adv = adv_usd if (adv_usd is not None and adv_usd > 0) else self._cfg.reference_adv_usd
        if notional_usd <= 0:
            return 0.0
        ratio = notional_usd / adv
        return self._cfg.impact_coefficient * math.sqrt(ratio)

    def _total_slippage_bps(
        self,
        notional_usd: float,
        adv_usd: Optional[float],
    ) -> float:
        slip = self._cfg.half_spread_bps + self._impact_bps(notional_usd, adv_usd)
        if self._cfg.volatility_jitter_bps > 0:
            # Pseudo-random but seedable jitter: hash of notional gives stable but varied result.
            seed = int(notional_usd * 1000) & 0xffff
            rand = ((seed * 2654435761) & 0xffffffff) / 0xffffffff  # in [0, 1)
            jitter = (rand - 0.5) * 2 * self._cfg.volatility_jitter_bps
            slip += jitter
        return max(self._cfg.min_slippage_bps, min(self._cfg.max_slippage_bps, slip))

    # ──────────────────────────────────────────────
    # Public fill simulation
    # ──────────────────────────────────────────────

    def fill_market_buy(
        self,
        reference_price: float,
        notional_usd: float,
        adv_usd: Optional[float] = None,
    ) -> FillResult:
        """Simulate a market buy. Slippage moves price UP (worse for buyer)."""
        slip_bps = self._total_slippage_bps(notional_usd, adv_usd)
        fill_price = reference_price * (1.0 + _bps_to_factor(slip_bps))
        return FillResult(
            fill_price=fill_price,
            slippage_bps=slip_bps,
            reason=FillReason.SIGNAL,
        )

    def fill_market_sell(
        self,
        reference_price: float,
        notional_usd: float,
        adv_usd: Optional[float] = None,
    ) -> FillResult:
        """Simulate a market sell. Slippage moves price DOWN (worse for seller)."""
        slip_bps = self._total_slippage_bps(notional_usd, adv_usd)
        fill_price = reference_price * (1.0 - _bps_to_factor(slip_bps))
        return FillResult(
            fill_price=fill_price,
            slippage_bps=slip_bps,
            reason=FillReason.SIGNAL,
        )

    def fill_stop_loss(
        self,
        stop_price: float,
        candle_open: float,
        candle_low: float,
        notional_usd: float,
        adv_usd: Optional[float] = None,
    ) -> Optional[FillResult]:
        """Stop-loss exit with gap awareness.

        Returns None if the candle never reached the stop. Returns a fill
        otherwise. Gap-through stops fill at the OPEN price (worse than
        stop_price), never at stop_price itself — that's the realistic
        outcome for any market-protected stop.
        """
        if candle_low > stop_price:
            return None  # candle never reached stop

        slip_bps = self._total_slippage_bps(notional_usd, adv_usd)

        # Gap case: open is already below the stop. Fill at open with extra slip.
        if self._cfg.apply_gap_penalty and candle_open <= stop_price:
            fill_price = candle_open * (1.0 - _bps_to_factor(slip_bps))
            gap_pct = abs(candle_open - stop_price) / stop_price * 100
            return FillResult(
                fill_price=fill_price,
                slippage_bps=slip_bps,
                reason=FillReason.STOP_LOSS_GAP,
                notes=f"gap of {gap_pct:.2f}% through stop",
            )

        # Normal touch: fill at stop_price - slip.
        fill_price = stop_price * (1.0 - _bps_to_factor(slip_bps))
        return FillResult(
            fill_price=fill_price,
            slippage_bps=slip_bps,
            reason=FillReason.STOP_LOSS,
        )

    def fill_take_profit(
        self,
        tp_price: float,
        candle_open: float,
        candle_high: float,
        notional_usd: float,
        adv_usd: Optional[float] = None,
    ) -> Optional[FillResult]:
        """Take-profit exit with gap awareness.

        Gap-up through TP fills at the OPEN price (better than tp_price), still
        with slippage cost since we're selling.
        """
        if candle_high < tp_price:
            return None

        slip_bps = self._total_slippage_bps(notional_usd, adv_usd)

        if self._cfg.apply_gap_penalty and candle_open >= tp_price:
            fill_price = candle_open * (1.0 - _bps_to_factor(slip_bps))
            return FillResult(
                fill_price=fill_price,
                slippage_bps=slip_bps,
                reason=FillReason.TAKE_PROFIT_GAP,
                notes=f"gap above TP",
            )

        fill_price = tp_price * (1.0 - _bps_to_factor(slip_bps))
        return FillResult(
            fill_price=fill_price,
            slippage_bps=slip_bps,
            reason=FillReason.TAKE_PROFIT,
        )
