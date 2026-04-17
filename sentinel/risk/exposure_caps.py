"""
Asset-class exposure caps — limit gross USD exposure per crypto category.

Pairs and their classes are configured externally so the mapping can be
edited without code changes. The default mapping covers the majors traded
by Sentinel; unknown symbols fall into the ``UNKNOWN`` bucket which has
its own conservative cap.

Why a class cap on top of the per-position and total-exposure caps already
in ``risk/sentinel.py``: when L1 majors are correlated (BTC, ETH, SOL,
BNB all moved together in March 2024), three small positions in
"different" coins are still one bet on L1 risk. A 25% portfolio-wide cap
on L1 forces actual diversification rather than the appearance of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# Default asset-class taxonomy. Tradable crypto majors only — extend in
# config or constructor for new symbols. Keys are Binance USDT pairs.
DEFAULT_ASSET_CLASS_MAP: dict[str, str] = {
    # Layer-1 majors
    "BTCUSDT": "L1",
    "ETHUSDT": "L1",
    "SOLUSDT": "L1",
    "BNBUSDT": "L1",
    "ADAUSDT": "L1",
    "AVAXUSDT": "L1",
    "DOTUSDT": "L1",
    "NEARUSDT": "L1",
    "TRXUSDT": "L1",
    # Layer-2 / scaling
    "ARBUSDT": "L2",
    "OPUSDT": "L2",
    "MATICUSDT": "L2",
    # DeFi
    "UNIUSDT": "DEFI",
    "AAVEUSDT": "DEFI",
    "LINKUSDT": "DEFI",
    "MKRUSDT": "DEFI",
    # Memes
    "DOGEUSDT": "MEME",
    "SHIBUSDT": "MEME",
    "PEPEUSDT": "MEME",
    # Stables (rare to trade actively but here for completeness)
    "USDCUSDT": "STABLE",
}


# Default per-class caps as % of total equity. Sum can exceed 100 because
# positions are not held in every class simultaneously; the gross cap is
# enforced separately by ``risk/sentinel.RiskLimits.max_total_exposure_pct``.
DEFAULT_CLASS_CAPS_PCT: dict[str, float] = {
    "L1": 35.0,        # majors — bulk of liquidity, allow biggest bucket
    "L2": 15.0,        # scaling layers — high beta to L1
    "DEFI": 15.0,      # DeFi — sector-specific tail risk
    "MEME": 5.0,       # memes — extreme volatility, hard cap
    "STABLE": 50.0,    # not really risk — allow most of book
    "UNKNOWN": 10.0,   # everything not in the map
}


@dataclass(frozen=True)
class ExposureCapConfig:
    asset_class_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ASSET_CLASS_MAP))
    class_caps_pct: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CLASS_CAPS_PCT))
    unknown_class: str = "UNKNOWN"


@dataclass
class ExposureDecision:
    approved: bool
    reason: str = ""
    asset_class: str = ""
    class_exposure_pct_after: float = 0.0
    class_cap_pct: float = 0.0


@dataclass(frozen=True)
class OpenPositionExposure:
    """Minimal projection of a position needed for exposure accounting."""
    symbol: str
    notional_usd: float


class ExposureCapGuard:
    """Enforce per-asset-class gross exposure caps."""

    def __init__(self, config: Optional[ExposureCapConfig] = None) -> None:
        self._cfg = config or ExposureCapConfig()

    def asset_class(self, symbol: str) -> str:
        """Return class for a symbol; unknown symbols map to ``unknown_class``."""
        return self._cfg.asset_class_map.get(symbol, self._cfg.unknown_class)

    def cap_for_class(self, asset_class: str) -> float:
        return self._cfg.class_caps_pct.get(asset_class, self._cfg.class_caps_pct.get("UNKNOWN", 10.0))

    def check(
        self,
        candidate_symbol: str,
        candidate_notional_usd: float,
        equity_usd: float,
        open_positions: Iterable[OpenPositionExposure],
    ) -> ExposureDecision:
        """Approve or reject a candidate trade based on post-trade class exposure.

        ``equity_usd`` is the total account value (cash + open notional). The
        guard does not double-count: the candidate adds notional_usd, and the
        post-trade class exposure is compared against the cap.
        """
        if equity_usd <= 0 or candidate_notional_usd <= 0:
            return ExposureDecision(
                approved=False,
                reason=f"Invalid sizes: equity={equity_usd}, notional={candidate_notional_usd}",
                asset_class=self.asset_class(candidate_symbol),
            )

        klass = self.asset_class(candidate_symbol)
        cap_pct = self.cap_for_class(klass)

        # Existing exposure in the same class. Replace any pre-existing position
        # in the candidate symbol — otherwise we'd be double-counting if this is
        # an add-on, but the trading layer here is single-position-per-symbol.
        current_class_notional = sum(
            p.notional_usd for p in open_positions
            if p.symbol != candidate_symbol and self.asset_class(p.symbol) == klass
        )
        post_trade_class_notional = current_class_notional + candidate_notional_usd
        post_trade_class_pct = post_trade_class_notional / equity_usd * 100

        if post_trade_class_pct > cap_pct:
            return ExposureDecision(
                approved=False,
                reason=(
                    f"{klass} exposure cap exceeded: {post_trade_class_pct:.1f}% > "
                    f"{cap_pct:.1f}% (current {current_class_notional / equity_usd * 100:.1f}%, "
                    f"adding ${candidate_notional_usd:.2f})"
                ),
                asset_class=klass,
                class_exposure_pct_after=post_trade_class_pct,
                class_cap_pct=cap_pct,
            )

        return ExposureDecision(
            approved=True,
            reason="Within class cap",
            asset_class=klass,
            class_exposure_pct_after=post_trade_class_pct,
            class_cap_pct=cap_pct,
        )

    def snapshot(
        self,
        equity_usd: float,
        open_positions: Iterable[OpenPositionExposure],
    ) -> dict:
        """Aggregate exposure by class for dashboards."""
        per_class: dict[str, float] = {}
        for p in open_positions:
            k = self.asset_class(p.symbol)
            per_class[k] = per_class.get(k, 0.0) + p.notional_usd
        return {
            klass: {
                "notional_usd": round(notional, 2),
                "pct_of_equity": round(notional / equity_usd * 100, 2) if equity_usd > 0 else 0.0,
                "cap_pct": self.cap_for_class(klass),
                "headroom_pct": max(0.0, self.cap_for_class(klass) - (notional / equity_usd * 100 if equity_usd > 0 else 0.0)),
            }
            for klass, notional in sorted(per_class.items())
        }
