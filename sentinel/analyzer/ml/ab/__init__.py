"""Champion-challenger A/B framework with statistical promotion gates."""
from analyzer.ml.ab.champion_challenger import (
    DEFAULT_ALPHA,
    DEFAULT_MIN_LIFT,
    DEFAULT_MIN_SAMPLES,
    ChampionChallengerComparator,
    PredictionPair,
    PromotionDecision,
    mcnemars_test,
    wilson_lift_lower_bound,
)

__all__ = [
    "ChampionChallengerComparator",
    "PromotionDecision",
    "PredictionPair",
    "mcnemars_test",
    "wilson_lift_lower_bound",
    "DEFAULT_ALPHA",
    "DEFAULT_MIN_LIFT",
    "DEFAULT_MIN_SAMPLES",
]
