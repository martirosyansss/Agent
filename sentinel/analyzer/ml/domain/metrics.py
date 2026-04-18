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
