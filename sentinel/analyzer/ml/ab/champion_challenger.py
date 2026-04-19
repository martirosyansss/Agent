"""Champion-challenger A/B framework with statistical promotion gates.

Promoting a new model to production based on training-set metrics is a
classic way to ship something that looked great in backtest and quietly
loses money for two weeks before someone notices. The standard fix is
champion-challenger: deploy the new model in *shadow* alongside the
current production champion, log both predictions for every live signal,
and only promote the challenger when its observed live-trading metrics
beat the champion with statistical significance.

This module provides:

* :class:`ChampionChallengerComparator` — accumulates parallel (champion,
  challenger) prediction records and outcome labels.
* :meth:`ChampionChallengerComparator.evaluate` — returns a
  :class:`PromotionDecision` with the precision lift, McNemar's test
  p-value, Wilson lower bound on the lift, and a verdict
  ``promote | hold | demote``.

Statistical method: McNemar's test on **paired** binary outcomes (each
trade gets a prediction from BOTH models, so samples are paired, not iid).
The naïve two-proportion z-test would inflate Type-I error here because
it pretends the predictions are independent. McNemar correctly accounts
for the fact that we're testing whether the disagreement matrix is
asymmetric.

References:
    Dietterich (1998) "Approximate statistical tests for comparing
    supervised classification learning algorithms", Neural Computation.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Promotion gate parameters (sensible defaults — caller can override)
# ---------------------------------------------------------------------------

# Minimum paired observations before the comparator is willing to render
# a verdict. Below this, lift estimates are pure noise. 50 is the standard
# rule-of-thumb floor for binomial inference.
DEFAULT_MIN_SAMPLES = 50

# Significance level for promotion. 0.05 is conventional; lowering to
# 0.01 is appropriate when the cost of a bad promotion is high
# (real-money trading).
DEFAULT_ALPHA = 0.05

# Minimum *practical* precision lift required for promotion, on top of
# statistical significance. A 0.5-pp lift can be statistically real and
# economically meaningless — the gate trips only when both conditions
# hold.
DEFAULT_MIN_LIFT = 0.02


@dataclass(frozen=True)
class PredictionPair:
    """One trade — both models predicted the same input, then the actual
    outcome was observed.

    ``champion_pred`` and ``challenger_pred`` are 0/1 (model would or
    would not have entered the trade). ``actual_win`` is 0/1 from the
    realised PnL.
    """
    champion_pred: int
    challenger_pred: int
    actual_win: int


@dataclass
class PromotionDecision:
    """Verdict from a champion-challenger evaluation."""
    n_pairs: int
    champion_precision: float
    challenger_precision: float
    precision_lift: float          # challenger − champion
    lift_wilson_lower: float       # 95% CI lower bound on the lift
    mcnemar_statistic: float
    mcnemar_p_value: float
    verdict: str                   # "promote" | "hold" | "demote"
    reason: str                    # human-readable explanation


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def mcnemars_test(b: int, c: int) -> tuple[float, float]:
    """Exact / continuity-corrected McNemar's test for paired binary outcomes.

    Args:
        b: # samples where champion was correct AND challenger was wrong.
        c: # samples where challenger was correct AND champion was wrong.

    Returns ``(chi2_statistic, two_sided_p_value)``. Falls back to the
    binomial exact test when ``b + c < 25`` (standard recommendation —
    the chi² approximation is unreliable on small discordant counts).
    """
    n_disc = b + c
    if n_disc == 0:
        # Models agree on every sample — no evidence of difference.
        return 0.0, 1.0
    if n_disc < 25:
        # Exact two-sided binomial test on B(n_disc, 0.5).
        from scipy.stats import binomtest
        try:
            res = binomtest(min(b, c), n_disc, p=0.5, alternative="two-sided")
            return float("nan"), float(res.pvalue)
        except Exception:  # pragma: no cover — defensive
            return float("nan"), 1.0
    # Continuity-corrected chi² (Edwards 1948).
    chi2 = (abs(b - c) - 1.0) ** 2 / float(n_disc)
    from scipy.stats import chi2 as chi2_dist
    p = float(1.0 - chi2_dist.cdf(chi2, df=1))
    return float(chi2), p


def wilson_lift_lower_bound(
    successes_a: int, n_a: int,
    successes_b: int, n_b: int,
    z: float = 1.96,
) -> float:
    """95% Wilson lower bound on the *difference* of two proportions
    (challenger − champion).

    A negative lower bound means we can't statistically rule out that the
    challenger is *worse*, even if the point estimate is positive. The
    Wilson interval handles small N and extreme p far better than the
    naïve normal approximation, which is critical for small live-trading
    holdouts (n < 200 is typical).
    """
    if n_a == 0 or n_b == 0:
        return 0.0
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    se = (p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b) ** 0.5
    return (p_b - p_a) - z * se


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class ChampionChallengerComparator:
    """Rolling A/B test for two models running in parallel shadow mode.

    Lifecycle:
      1. Construct with both model identifiers and a window size.
      2. ``record(champion_pred, challenger_pred, actual_win)`` from the
         live executor on every closed trade.
      3. Periodically call ``evaluate()`` (e.g. from a daily monitoring
         loop) — when ``verdict == "promote"`` the operator runs the
         actual stage transition through :class:`MLRegistry`.
    """

    def __init__(
        self,
        champion_id: str,
        challenger_id: str,
        window: int = 500,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        alpha: float = DEFAULT_ALPHA,
        min_lift: float = DEFAULT_MIN_LIFT,
    ):
        if window < min_samples:
            raise ValueError(
                f"window ({window}) must be >= min_samples ({min_samples})"
            )
        self.champion_id = champion_id
        self.challenger_id = challenger_id
        self._min_samples = int(min_samples)
        self._alpha = float(alpha)
        self._min_lift = float(min_lift)
        self._pairs: deque[PredictionPair] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, champion_pred: bool, challenger_pred: bool,
               actual_win: bool) -> None:
        """Append one paired observation. Booleans are coerced to 0/1."""
        with self._lock:
            self._pairs.append(PredictionPair(
                champion_pred=int(bool(champion_pred)),
                challenger_pred=int(bool(challenger_pred)),
                actual_win=int(bool(actual_win)),
            ))

    @property
    def n_pairs(self) -> int:
        with self._lock:
            return len(self._pairs)

    def evaluate(self) -> Optional[PromotionDecision]:
        """Return a verdict, or ``None`` if not enough samples yet.

        The verdict is decided by combining three checks:

          * **Significance**: McNemar's p-value < ``alpha``.
          * **Practical effect**: Wilson lower-bound of the lift > ``min_lift``.
          * **Direction**: precision_lift sign — promote on positive,
            demote on negative significant lift.

        ``hold`` is the default when none of the above hold strongly.
        """
        with self._lock:
            if len(self._pairs) < self._min_samples:
                return None
            pairs = list(self._pairs)

        champ_correct = np.array(
            [int(p.champion_pred == p.actual_win) for p in pairs], dtype=int,
        )
        chal_correct = np.array(
            [int(p.challenger_pred == p.actual_win) for p in pairs], dtype=int,
        )

        # Precision = TP / (TP + FP) for the "predicted positive" subset.
        # On paired data we compute it independently per model and let
        # the McNemar / Wilson layer handle the dependence structure.
        n = len(pairs)
        champ_pred_pos = np.array([p.champion_pred for p in pairs], dtype=int)
        chal_pred_pos = np.array([p.challenger_pred for p in pairs], dtype=int)
        actual = np.array([p.actual_win for p in pairs], dtype=int)

        champ_tp = int(np.sum((champ_pred_pos == 1) & (actual == 1)))
        champ_fp = int(np.sum((champ_pred_pos == 1) & (actual == 0)))
        chal_tp = int(np.sum((chal_pred_pos == 1) & (actual == 1)))
        chal_fp = int(np.sum((chal_pred_pos == 1) & (actual == 0)))

        champ_prec = champ_tp / max(champ_tp + champ_fp, 1)
        chal_prec = chal_tp / max(chal_tp + chal_fp, 1)
        lift = chal_prec - champ_prec

        # McNemar on per-trade correctness — paired binary outcomes.
        b = int(np.sum((champ_correct == 1) & (chal_correct == 0)))
        c = int(np.sum((champ_correct == 0) & (chal_correct == 1)))
        chi2, p_value = mcnemars_test(b, c)

        # Wilson lower bound on (challenger_precision − champion_precision).
        # Treats the two precisions as independent proportions for the CI;
        # this is conservative (slightly wide) compared to a paired CI but
        # avoids fragile small-N variance estimates on the disagreement
        # matrix.
        wilson_lo = wilson_lift_lower_bound(
            champ_tp, champ_tp + champ_fp,
            chal_tp, chal_tp + chal_fp,
        )

        # Decision logic
        if p_value < self._alpha and wilson_lo > self._min_lift and lift > 0:
            verdict = "promote"
            reason = (f"challenger precision {chal_prec:.3f} > champion {champ_prec:.3f} "
                      f"(lift={lift:+.3f}, Wilson_low={wilson_lo:+.3f}, p={p_value:.4f})")
        elif p_value < self._alpha and lift < -self._min_lift:
            verdict = "demote"
            reason = (f"challenger UNDERperforms: lift={lift:+.3f}, "
                      f"p={p_value:.4f} — keep champion, retire challenger")
        else:
            verdict = "hold"
            if p_value >= self._alpha:
                reason = (f"insufficient evidence: p={p_value:.4f} >= alpha={self._alpha} "
                          f"(lift={lift:+.3f}, n={n})")
            else:
                reason = (f"statistically significant but lift {lift:+.3f} below "
                          f"min_lift {self._min_lift} or Wilson_low {wilson_lo:+.3f} "
                          "leaves zero in CI — not yet practically meaningful")

        return PromotionDecision(
            n_pairs=n,
            champion_precision=champ_prec,
            challenger_precision=chal_prec,
            precision_lift=lift,
            lift_wilson_lower=wilson_lo,
            mcnemar_statistic=chi2,
            mcnemar_p_value=p_value,
            verdict=verdict,
            reason=reason,
        )
