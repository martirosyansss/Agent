"""``MLMetrics``, ``MLPrediction``, ``LivePerformanceTracker``.

These three types form the observable surface of a trained model:

* ``MLMetrics`` — what training produced (precision, AUC, calibration
  diagnostics, bootstrap CI, ...). Persisted verbatim in the pickle.
* ``MLPrediction`` — what predict() returns to callers: probability +
  decision + rollout mode.
* ``LivePerformanceTracker`` — rolling window of live prediction vs
  realised outcome pairs, used for drift detection.

Moved out of ``analyzer.ml_predictor`` during the round-10 refactor.
Re-exported from the old module path for backwards compatibility,
including unpickle of pre-refactor saves (the unpickler whitelist now
includes ``analyzer.ml`` so both old and new module-paths resolve).
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MLMetrics:
    """Trained-model metrics. All numeric fields default to neutral
    values so a freshly-constructed MLMetrics can round-trip through
    save/load without needing every field to be set first."""
    precision: float = 0.0
    recall: float = 0.0
    roc_auc: float = 0.0
    accuracy: float = 0.0
    skill_score: float = 0.0
    train_samples: int = 0
    test_samples: int = 0
    feature_importances: dict[str, float] = field(default_factory=dict)
    # Statistical confidence (bootstrap 95% CI)
    precision_ci_95: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    auc_ci_95: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    # Baseline comparison
    baseline_win_rate: float = 0.0    # precision of "predict always win" naive model
    precision_lift: float = 0.0       # model precision − baseline_win_rate
    auc_lift: float = 0.0             # model AUC − 0.5 (random baseline)
    # Out-of-time robustness
    oot_auc: Optional[float] = None   # AUC on most-recent 20% (independent OOT set)
    # Calibration diagnostics — flag silent biases like "every prediction ≈ 0.9"
    # that hurt downstream interpretation even when AUC/precision look fine.
    brier_score: float = 0.0          # mean (proba - actual)² on holdout, lower is better
    ece: float = 0.0                  # Expected Calibration Error (10 bins) on holdout
    mean_proba: float = 0.5           # mean calibrated probability across holdout
    median_proba: float = 0.5         # median — flags one-sided distributions
    proba_p10: float = 0.0            # 10th percentile — should not collapse onto p90
    proba_p90: float = 1.0            # 90th percentile
    calibration_method: str = "none"  # "none" | "platt" | "isotonic"
    # ------------------------------------------------------------------
    # Risk-adjusted PnL metrics on the predicted-positive holdout slice.
    # Computed from the trades the model actually flagged (not the full
    # universe), so they answer "if we had traded only what the model
    # said, what would the risk profile have looked like?". All three are
    # ``None`` when there are too few predicted-positive samples to be
    # meaningful (n_pred_win < 10), to avoid showing noise as performance.
    sortino_ratio: Optional[float] = None     # mean / downside-σ × √252
    calmar_ratio: Optional[float] = None      # annualised return / |max DD|
    max_drawdown_pct: float = 0.0             # peak-to-trough on cumulative PnL
    max_drawdown_duration: int = 0            # longest underwater streak (in samples)
    # Tail-risk: characterises the *worst* trades, not the average. For crypto
    # (kurtosis ≈ 8–15), naïve Sharpe / Sortino under-state tail loss because
    # they assume Gaussian — so we report VaR alongside, plus the Cornish-
    # Fisher variant which adjusts the quantile for sample skew & kurtosis.
    # All three are dollar-denominated losses (positive numbers, e.g. 50 = $50
    # worst-1%-day loss). ``None`` when fewer than ``min_pred_wins`` samples.
    var_95: Optional[float] = None            # 95% historical VaR (loss in PnL units)
    var_99: Optional[float] = None            # 99% historical VaR
    cvar_95: Optional[float] = None           # mean of losses beyond VaR_95 (Expected Shortfall)
    var_95_cornish_fisher: Optional[float] = None  # parametric VaR with skew/kurt correction
    # Cohen's d effect size for the precision-vs-baseline comparison. Lets
    # the dashboard show "is precision lift big or small?" in standardised
    # σ-units rather than raw points; |d| ≥ 0.8 = large effect.
    precision_cohens_d: Optional[float] = None
    # Per-symbol risk attribution: {symbol: {sortino, calmar, var_95, ...}}.
    # Empty dict when the holdout has no symbol metadata or every symbol
    # had fewer than ``min_pred_wins`` predicted-positives. Lets dashboards
    # see whether a "good" aggregate Sortino is one strong symbol carrying
    # several weak ones, or genuine multi-asset edge.
    per_symbol_metrics: dict = field(default_factory=dict)
    # Population Stability Index summary, captured once after training as
    # a baseline (PSI vs itself = 0). Live drift is tracked separately by
    # ``FeatureDriftMonitor`` and surfaced through its own report API.
    feature_drift_status: str = "unmonitored"  # "ok"|"minor_drift"|"major_drift"|"unmonitored"
    feature_drift_max_psi: float = 0.0
    # Probabilistic Sharpe Ratio (López de Prado 2012) — the probability
    # that the *true* per-period Sharpe of the predicted-positive trades
    # exceeds zero once skew/kurtosis/sample-size are accounted for.
    # 0.95 is the textbook pass threshold. ``None`` when N is too small
    # to estimate the higher moments (< 3 predicted-positives).
    psr: Optional[float] = None
    # Deflated Sharpe Ratio — PSR evaluated against a benchmark inflated
    # for multiple-testing (the expected max Sharpe of ``psr_n_trials``
    # coin-flip strategies). Filled when the training pipeline reports
    # how many hyper-parameter / model variants were tried. ``None``
    # when n_trials ≤ 1 or the sample is too small.
    dsr: Optional[float] = None
    psr_n_trials: int = 1             # reported N used in DSR deflation
    psr_gate_passed: bool = False     # PSR ≥ 0.95 AND (DSR ≥ 0.95 or DSR is None)


@dataclass
class MLPrediction:
    """What ``MLPredictor.predict`` returns to callers."""
    probability: float = 0.5
    decision: str = "allow"  # allow, reduce, block
    model_version: str = ""
    rollout_mode: str = "shadow"  # off, shadow, block


class LivePerformanceTracker:
    """Tracks model predictions vs actual trade outcomes in live/paper trading.

    Detects concept drift: when live precision drops significantly below
    training precision, the model has likely overfit to a historical
    regime that no longer holds.

    Thread-safe: uses a lock for concurrent access from async code paths.
    Memory-bounded: uses ``collections.deque`` with fixed ``maxlen``.
    """

    def __init__(self, window: int = 50, drift_threshold: Optional[float] = None) -> None:
        self._window = window
        # When None, drift threshold is computed adaptively from sample
        # size via the Wilson-score width (see is_drifting). A fixed
        # threshold like 0.12 fires spuriously at small N and misses
        # real drift at large N.
        self._drift_threshold = drift_threshold
        self._history: deque[tuple[float, int]] = deque(maxlen=window * 3)
        self._lock = threading.Lock()

    def record(self, predicted_prob: float, actual_win: bool) -> None:
        """Record one live prediction + its realised outcome (thread-safe)."""
        with self._lock:
            self._history.append((predicted_prob, int(actual_win)))

    def live_metrics(self) -> dict:
        """Compute rolling precision, win rate, and calibration on the recent window."""
        with self._lock:
            snapshot = list(self._history)

        n = len(snapshot)
        if n < 10:
            return {"status": "insufficient_data", "n": n}

        recent = snapshot[-self._window:]
        probs = np.array([p for p, _ in recent], dtype=np.float64)
        actuals = np.array([a for _, a in recent], dtype=np.float64)
        preds = (probs >= 0.5).astype(int)

        win_rate = float(actuals.mean())
        n_pred_win = int(preds.sum())
        live_prec = float(np.sum((preds == 1) & (actuals == 1)) / n_pred_win) if n_pred_win > 0 else 0.0
        calib_err = float(abs(probs.mean() - win_rate))

        try:
            from sklearn.metrics import roc_auc_score as _auc
            live_auc = float(_auc(actuals, probs)) if len(set(actuals)) > 1 else 0.5
        except Exception:
            live_auc = 0.5

        return {
            "n": len(recent),
            "n_pred_win": n_pred_win,   # denominator of precision — needed for Wilson CI
            "live_precision": live_prec,
            "live_win_rate": win_rate,
            "live_auc": live_auc,
            "calibration_error": calib_err,
        }

    def is_drifting(self, training_precision: float) -> bool:
        """Detect concept drift using a sample-size-aware threshold.

        Uses a two-proportion z-test-style margin: drift is flagged only
        when the gap between training and live precision exceeds what
        sampling noise alone would produce at 95% confidence (z=1.96)::

            margin = z * sqrt(p * (1 - p) / n_pred_win)

        where p is training precision. At n=30, p=0.70 this gives
        margin ≈ 0.164, so a 12-point drop is noise; at n=200 it gives
        0.064, so even small drops are meaningful. A fixed fallback is
        used when the caller explicitly set ``drift_threshold``.
        """
        m = self.live_metrics()
        if "live_precision" not in m or m.get("n", 0) < 10:
            return False
        n_pred_win = max(int(m.get("n_pred_win", 0)), 1)

        if self._drift_threshold is not None:
            margin = self._drift_threshold
        else:
            p = max(min(training_precision, 0.99), 0.01)
            margin = 1.96 * np.sqrt(p * (1.0 - p) / n_pred_win)

        return (training_precision - m["live_precision"]) > margin

    @property
    def n_recorded(self) -> int:
        with self._lock:
            return len(self._history)


# ---------------------------------------------------------------------------
# Risk-adjusted PnL helpers
# ---------------------------------------------------------------------------

# Trading days per year — used to annualise per-trade PnL into an "effective
# yearly" return for Sortino/Calmar. A trade ≠ a day, but at the per-trade
# level √N scaling stays statistically valid as long as trades are weakly
# autocorrelated (block bootstrap covers what scaling cannot).
_ANN_FACTOR = 252


def _max_drawdown_from_pnl(pnl: np.ndarray) -> tuple[float, int, float]:
    """Compute drawdown stats directly from a per-trade PnL series.

    Returns ``(max_dd_pct, longest_underwater_streak, max_dd_abs)``:
      * ``max_dd_abs`` — worst peak-to-trough drop in PnL units. Independent
        of any chosen initial capital, used as the Calmar denominator.
      * ``max_dd_pct`` — same drop expressed as a fraction of an *inferred*
        starting capital sized to comfortably absorb the worst observed
        cumulative loss (10× buffer, $1000 floor). This keeps the percentage
        meaningful — never collapses to −100% — without forcing the caller
        to pick a number.
      * ``longest_underwater_streak`` — longest run of consecutive samples
        where equity sits strictly below the running peak.
    """
    if pnl.size == 0:
        return 0.0, 0, 0.0
    cum = np.cumsum(pnl)
    # Capital large enough that even the worst cumulative drawdown leaves
    # equity well above zero — keeps DD% in the natural [0, 1] range
    # instead of collapsing to −100% as happens when the floor touches 0.
    worst_cum_loss = float(max(0.0, -cum.min())) if cum.size else 0.0
    inferred_capital = max(worst_cum_loss * 10.0, 1000.0)

    equity = inferred_capital + cum
    peaks = np.maximum.accumulate(equity)
    drops = peaks - equity                          # absolute drop in PnL units
    max_dd_abs = float(drops.max()) if drops.size else 0.0
    # Fraction of the running peak — well-defined because peaks ≥ inferred_capital.
    pct_drops = drops / peaks
    max_dd_pct = float(pct_drops.max()) if pct_drops.size else 0.0

    underwater = equity < peaks
    longest = run = 0
    for u in underwater:
        if u:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    # Sign convention: drawdown reported as a negative fraction so the
    # downstream label "max_drawdown_pct" reads naturally as a loss.
    return -max_dd_pct, int(longest), max_dd_abs


def _historical_var(losses_pos: np.ndarray, confidence: float) -> float:
    """Historical VaR at the given confidence level.

    ``losses_pos`` must be positive numbers representing losses (e.g.
    obtained from ``-pnl[pnl < 0]``). Returns the loss magnitude such
    that ``confidence`` fraction of losses are smaller. NumPy's
    ``percentile`` uses linear interpolation between order statistics —
    standard and well-defined for small samples.
    """
    if losses_pos.size == 0:
        return 0.0
    return float(np.percentile(losses_pos, confidence * 100.0))


def _cornish_fisher_var(returns: np.ndarray, confidence: float) -> Optional[float]:
    """Cornish-Fisher VaR — parametric VaR with skew/kurtosis correction.

    Standard parametric VaR assumes Gaussian returns; for crypto returns
    with skew γ₃ and excess kurtosis γ₄ - 3, the Cornish-Fisher expansion
    adjusts the z-quantile so the resulting VaR captures the actual fat
    tail. Returns ``None`` when the sample is too small (n < 20) for
    higher-moment estimates to be stable.

    Formula (loss-side, positive number):
        z_cf = z + (z²−1)γ₃/6 + (z³−3z)(γ₄−3)/24 − (2z³−5z)γ₃²/36
        VaR  = −(μ + σ·z_cf)            for left-tail (loss) confidence
    """
    n = returns.size
    if n < 20:
        return None
    try:
        from scipy.stats import norm, skew, kurtosis
    except ImportError:
        return None
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1)) if n > 1 else 0.0
    if sigma <= 0:
        return None
    # Left-tail z (e.g. -1.645 for 95%) — Cornish-Fisher is symmetric so
    # we take the negative side directly.
    z = float(norm.ppf(1.0 - confidence))
    g3 = float(skew(returns, bias=False))
    g4 = float(kurtosis(returns, fisher=True, bias=False))  # excess kurtosis
    z_cf = (z
            + (z * z - 1.0) * g3 / 6.0
            + (z ** 3 - 3.0 * z) * g4 / 24.0
            - (2.0 * z ** 3 - 5.0 * z) * (g3 ** 2) / 36.0)
    var = -(mu + sigma * z_cf)
    # Cornish-Fisher can produce negative VaR on degenerate samples
    # (very small N, near-zero variance) — clip to 0 so the dashboard
    # doesn't display a "negative loss".
    return max(var, 0.0)


def compute_pnl_risk_metrics_per_symbol(
    pnl_values: list[float] | np.ndarray,
    y_pred: np.ndarray,
    symbols: list[str],
    *,
    min_pred_wins: int = 10,
) -> dict[str, dict]:
    """Per-symbol breakdown of risk metrics.

    The aggregate ``compute_pnl_risk_metrics`` collapses everything into one
    Sortino / Calmar / VaR / MaxDD figure — which hides the failure mode
    where the "ensemble" is really one strong-edge symbol carrying four
    weak-edge symbols. By re-running the same metric pipeline per-symbol
    and side-by-side, the dashboard can show:

      * "BTC: Sortino 4.0, Calmar 12.0"   (real edge)
      * "ETH: Sortino 0.2, Calmar 1.0"    (no edge — coin-flip)

    rather than the blended "Sortino 2.1" figure that masks both signals.
    Returns a mapping ``{symbol: metrics_dict}``; symbols with fewer than
    ``min_pred_wins`` predicted-positives are omitted from the result
    rather than reported as ``None`` everywhere — keeps the dashboard
    quiet about symbols the model hasn't observed enough.
    """
    pnl = np.asarray(pnl_values, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if not (pnl.size == y_pred.size == len(symbols)):
        return {}
    out: dict[str, dict] = {}
    syms = np.asarray(symbols)
    for sym in np.unique(syms):
        mask = syms == sym
        sub = compute_pnl_risk_metrics(
            pnl[mask], y_pred[mask], min_pred_wins=min_pred_wins,
        )
        if sub:
            sub["n_predicted_positive"] = int(y_pred[mask].sum())
            out[str(sym)] = sub
    return out


def compute_pnl_risk_metrics(
    pnl_values: list[float] | np.ndarray,
    y_pred: np.ndarray,
    *,
    min_pred_wins: int = 10,
) -> dict:
    """Compute Sortino / Calmar / max-DD on the model's predicted-positive set.

    The ML pipeline flags a *subset* of trades to take. To know whether the
    flagged subset is profitable AND survivable (not just often-correct),
    we need risk-adjusted metrics on that subset:

    * ``sortino_ratio`` — mean return ÷ downside-σ × √252. Scale-invariant:
      multiplying every PnL by k leaves the ratio unchanged, so it can be
      computed directly on raw USD without picking an initial capital.
    * ``calmar_ratio``  — total PnL ÷ worst peak-to-trough drop, both in
      the same PnL units. Avoids the "what's the starting equity?" question
      by working with the strategy's own dollar-cumulative trajectory.
    * ``max_drawdown_pct`` and ``max_drawdown_duration`` — the worst
      historical underwater period the model would have signed up for.

    Returns ``{}`` if fewer than ``min_pred_wins`` predicted-positives exist
    (any computed metric on tiny N is noise and would mislead the dashboard).
    """
    pnl = np.asarray(pnl_values, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if pnl.size != y_pred.size:
        return {}
    selected = pnl[y_pred == 1]
    n = selected.size
    if n < min_pred_wins:
        return {}

    # Sortino on raw PnL: scale-invariant, no initial-capital choice needed.
    mean_pnl = float(selected.mean())
    downside = selected[selected < 0.0]
    if downside.size > 1:
        downside_std = float(downside.std(ddof=1))
    elif downside.size == 1:
        downside_std = float(abs(downside[0]))  # single loss → its magnitude
    else:
        downside_std = 0.0

    sortino: Optional[float]
    if downside_std > 0:
        sortino = mean_pnl / downside_std * float(np.sqrt(_ANN_FACTOR))
    else:
        sortino = None  # no losing trade observed yet — undefined, not infinite

    max_dd_pct, dd_dur, max_dd_abs = _max_drawdown_from_pnl(selected)

    # Calmar = annualised PnL / worst dollar drawdown. Both are in PnL
    # units, so the ratio is unit-free and independent of initial capital.
    # Annualisation: per-trade mean × 252 (linear, not geometric — geometric
    # compounding would require a base equity that we deliberately avoid).
    ann_pnl = mean_pnl * _ANN_FACTOR
    calmar: Optional[float]
    if max_dd_abs > 0:
        calmar = ann_pnl / max_dd_abs
    else:
        calmar = None  # monotonic equity — Calmar is undefined, not infinite

    # Tail-risk: VaR (historical) + CVaR (Expected Shortfall) + Cornish-Fisher.
    # Computed on the loss-side of selected PnL: positive = dollar loss size.
    losses = -selected[selected < 0.0]   # loss magnitudes, all ≥ 0
    if losses.size >= 5:
        var_95 = _historical_var(losses, 0.95)
        var_99 = _historical_var(losses, 0.99)
        # CVaR (Expected Shortfall) = mean of losses at or beyond VaR_95.
        # Standard ES definition: 𝔼[L | L ≥ VaR]. Falls back to VaR_95
        # itself when only the worst single loss qualifies.
        tail = losses[losses >= var_95]
        cvar_95: Optional[float] = float(tail.mean()) if tail.size > 0 else float(var_95)
    else:
        var_95 = var_99 = None  # type: ignore[assignment]
        cvar_95 = None
    var_cf = _cornish_fisher_var(selected, 0.95)

    return {
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown_pct": float(max_dd_pct) * 100.0,
        "max_drawdown_duration": dd_dur,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "var_95_cornish_fisher": var_cf,
    }
