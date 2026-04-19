"""Calibration + diagnostics — pure functions used during training.

* ``calibrate_threshold`` — pick the decision threshold that maximises
  profit factor (with precision/recall constraints) or F-beta when PnL
  isn't available.
* ``compute_profit_factor_score`` — normalised PF in [0, 1] for the
  skill-score formula.
* ``overfit_noise_margin`` — z-σ margin on the train-vs-val precision
  gap used by the overfit guard.
* ``expected_calibration_error`` — ECE over equal-width bins.
* ``compute_temporal_weights`` — exponential recency weights.

All five were static methods on ``MLPredictor`` that took no ``self``
state. They moved here during the round-10 refactor to free the main
class from pure-math utilities that have no reason to be bound to it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain.constants import _TEMPORAL_DECAY


def calibrate_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.55,
    pnl: Optional[np.ndarray] = None,
    min_recall: float = 0.30,
) -> float:
    """Find the decision threshold that maximises profit factor under
    precision + recall constraints.

    When ``pnl`` is provided, optimise for realised profit factor (gross
    winning PnL / gross losing PnL among predicted-win trades) subject
    to ``precision >= min_precision`` and ``recall >= min_recall``. When
    ``pnl`` is absent, fall back to precision-weighted F-beta (β = 0.5,
    so precision is weighted 2× recall).
    """
    from sklearn.metrics import precision_recall_curve, precision_score, recall_score

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_threshold = 0.5

    if pnl is not None and len(pnl) == len(y_true):
        best_pf = 0.0
        for thr in np.arange(0.30, 0.75, 0.01):
            y_pred = (y_proba >= thr).astype(int)
            n_pred_pos = int(y_pred.sum())
            if n_pred_pos < 5:
                continue
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            if prec < min_precision or rec < min_recall:
                continue
            wins_pnl = float(sum(p for p, pred in zip(pnl, y_pred) if pred == 1 and p > 0))
            loss_pnl = abs(float(sum(p for p, pred in zip(pnl, y_pred) if pred == 1 and p <= 0)))
            pf = wins_pnl / loss_pnl if loss_pnl > 0 else (3.0 if wins_pnl > 0 else 0.0)
            if pf > best_pf:
                best_pf = pf
                best_threshold = float(thr)
    else:
        # F-beta fallback (β = 0.5 → β² = 0.25, precision weighted 2× recall)
        best_fb = 0.0
        beta_sq = 0.25
        for prec, rec, thr in zip(precisions[:-1], recalls[:-1], thresholds):
            if prec < min_precision:
                continue
            fb = (1 + beta_sq) * prec * rec / (beta_sq * prec + rec) if (beta_sq * prec + rec) > 0 else 0
            if fb > best_fb:
                best_fb = fb
                best_threshold = float(thr)

    return best_threshold


def compute_profit_factor_score(y_pred, pnl_values) -> float:
    """Normalised profit factor from predicted wins vs actual PnL (0..1).

    Measures how well the model's "win" predictions align with actual
    profitable trades. Returns a value in [0, 1] where 1.0 = perfect PF
    (≥ 3.0). Used by the skill-score composite.
    """
    if pnl_values is None or len(pnl_values) == 0:
        return 0.5
    pred_wins_pnl = sum(p for p, pred in zip(pnl_values, y_pred) if pred == 1 and p > 0)
    pred_wins_loss = abs(sum(p for p, pred in zip(pnl_values, y_pred) if pred == 1 and p <= 0))
    if pred_wins_loss <= 0:
        pf = 3.0 if pred_wins_pnl > 0 else 0.0
    else:
        pf = min(pred_wins_pnl / pred_wins_loss, 3.0)
    return pf / 3.0


def bonferroni_z(alpha: float = 0.05, n_tests: int = 1) -> float:
    """Two-sided z critical value with Bonferroni family-wise correction.

    The overfit guard runs the same test on K ≥ 1 candidate models and
    accepts the best one. Without correction, the family-wise error rate
    inflates roughly linearly in K — at K=4 you have ≈ 18.5% chance of
    accepting an overfit candidate at a nominal 5% per-test level. Splitting
    alpha across the K tests (Bonferroni) restores the family-wise level
    at the cost of a wider per-test margin.

    Uses ``scipy.stats.norm.ppf`` (already a transitive dep via scikit-learn)
    so we don't carry a hand-rolled rational approximation that needs its own
    accuracy proof. For K=1 returns ≈ 1.96; K=4 returns ≈ 2.498; K=10 ≈ 2.807.
    """
    from scipy.stats import norm
    n = max(int(n_tests), 1)
    a = max(min(float(alpha), 0.5), 1e-6)
    per_test_tail = a / (2.0 * n)  # two-sided
    return float(norm.ppf(1.0 - per_test_tail))


def overfit_noise_margin(
    p_train: float,
    p_val: float,
    n_train: int,
    n_val: int,
    z: float = 1.96,
    n_tests: int = 1,
) -> float:
    """Statistical z-σ margin for the train-vs-val precision gap.

    Replaces the earlier ``0.5 / sqrt(n_val)`` heuristic, which ignored
    training-side variance and the actual proportion ``p``. Returns the
    real z-σ margin for the difference of two binomial proportions::

        SE(p_train) = sqrt(p_train * (1 - p_train) / n_train)
        SE(p_val)   = sqrt(p_val   * (1 - p_val)   / n_val)
        margin      = z * sqrt(SE_train² + SE_val²)

    When ``n_tests > 1`` (e.g. the guard runs the same test on K candidate
    models and we accept the best one) the supplied ``z`` is widened via
    Bonferroni so the family-wise false-acceptance rate stays at α=0.05.
    Without this, the "best-of-K" selection biases the holdout estimate
    upward by 5–15% on K=4.

    Returns 0 when either sample is empty — caller treats that as "not
    enough data to call overfitting either way".
    """
    if n_train <= 0 or n_val <= 0:
        return 0.0
    # Clamp p into (0, 1) so SE doesn't collapse to zero when a model
    # trivially predicts a single class on either split.
    p_t = min(max(p_train, 1e-3), 1.0 - 1e-3)
    p_v = min(max(p_val, 1e-3), 1.0 - 1e-3)
    var_t = p_t * (1.0 - p_t) / n_train
    var_v = p_v * (1.0 - p_v) / n_val
    z_eff = bonferroni_z(alpha=0.05, n_tests=n_tests) if n_tests > 1 else z
    return float(z_eff * (var_t + var_v) ** 0.5)


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error: mean |confidence − accuracy| across
    equal-width bins.

    Complements Brier score — a model with ECE > ~0.10 is meaningfully
    miscalibrated even if AUC / precision look fine. Empty bins are
    skipped, so this is robust on the small holdout sets we work with
    (~50–300 samples).

    Returns ECE in [0, 1]: 0 = perfectly calibrated, 1 = maximally
    miscalibrated.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Clip the right edge so 1.0 lands in the last bin (not one past it).
    bin_idx = np.clip(np.digitize(y_proba, bin_edges[1:-1]), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(y_proba[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += (count / n) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_temporal_weights(n: int, decay: float = _TEMPORAL_DECAY) -> np.ndarray:
    """Exponential temporal weights — recent trades weighted more.

    ``w(i) = exp(decay * i)`` so the last trade has weight ≈ 1.0 and the
    first trade has weight ``exp(-decay * (n - 1))``. The returned
    vector is normalised to ``mean = 1.0`` so it can be fed to sklearn's
    ``sample_weight`` without shifting the total loss magnitude.
    """
    indices = np.arange(n, dtype=np.float64)
    weights = np.exp(decay * indices)
    return weights / weights.mean()
