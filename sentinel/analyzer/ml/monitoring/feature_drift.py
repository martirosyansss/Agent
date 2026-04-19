"""Population Stability Index (PSI) feature drift detector.

The trained model assumes that live-time feature distributions match the
training-time distributions. When that assumption breaks — a regime shift,
a broken upstream pipeline, a new exchange data feed — predictions still
go out, but they come from a model that is statistically out of domain
for the inputs it sees. This is the "model rotting in production" failure
mode that ``LivePerformanceTracker`` only catches AFTER live precision
has already dropped, by which point trades are already lost.

PSI compares the *distribution* of each feature between a fixed reference
window (sampled at training time) and a rolling live window (filled by
runtime ``predict`` calls). The standard reading is::

    PSI <  0.10   no significant shift
    PSI <  0.20   small shift, monitor
    PSI >= 0.20   significant shift, retrain or investigate

Implementation choices that matter:

* **Quantile-based binning.** Equal-width bins on skewed financial features
  (e.g. ATR, RSI clusters) collapse most observations into one bin and
  exaggerate PSI on tails. Quantile bins from the reference distribution
  put the same number of training samples in each bin, so the PSI reflects
  *actual* divergence rather than a binning artefact.
* **Single source of truth for bins.** Both the reference histogram and
  every live histogram use the same bin edges, fixed at training time.
  Re-deriving bins from live data each time would make PSI = 0 by
  definition.
* **Epsilon smoothing.** ``log(0)`` blows up; we add ``eps`` to every bin
  count before normalisation so empty bins contribute a finite (large)
  divergence instead of ``±inf``.
* **Rolling live window with a minimum.** PSI on n=5 samples is noise; we
  refuse to score until at least ``min_live_samples`` (default 50) have
  accumulated, then keep a fixed-size deque so old data ages out.

Reference: Yurdakul (2018), "Statistical Properties of Population
Stability Index". The cube-root rule for n_bins comes from Sturges's
generalisation; we cap at 10 because PSI loses power above that on
financial-grade sample sizes.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard PSI thresholds and configuration
# ---------------------------------------------------------------------------

# Industry-standard cutoffs (Siddiqi 2005, Yurdakul 2018). Treat anything
# above MAJOR_DRIFT as "stop trusting the model until investigated".
PSI_NO_SHIFT = 0.10
PSI_MINOR_SHIFT = 0.20

# Default live-window size (number of recent prediction feature vectors
# kept for drift scoring). 200 balances "enough to be statistically
# meaningful" against "small enough to actually catch a recent regime
# change before averaging it away".
DEFAULT_WINDOW = 200

# Refuse to compute PSI below this many live samples — anything less is
# noise and would generate spurious drift alarms during the first hour
# of a freshly-deployed model.
DEFAULT_MIN_LIVE = 50

# Epsilon added to bin counts before normalisation. 1e-6 is small enough
# not to bias PSI on healthy distributions, large enough to keep log(0)
# at bay for empty bins.
_EPS = 1e-6


@dataclass
class FeatureDriftReport:
    """Drift verdict for one feature at a moment in time."""
    feature: str
    psi: float
    severity: str            # "stable" | "minor" | "major"
    n_reference: int
    n_live: int

    def is_drifting(self) -> bool:
        return self.severity == "major"

    @property
    def is_minor(self) -> bool:
        return self.severity == "minor"


def _categorise_psi(psi: float) -> str:
    if psi < PSI_NO_SHIFT:
        return "stable"
    if psi < PSI_MINOR_SHIFT:
        return "minor"
    return "major"


def population_stability_index(
    reference: np.ndarray,
    live: np.ndarray,
    bin_edges: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> tuple[float, np.ndarray]:
    """Compute PSI between a reference and a live sample.

    Returns ``(psi, bin_edges)``. When ``bin_edges`` is ``None`` the bins
    are derived from quantiles of ``reference`` and returned for re-use on
    subsequent live windows — the caller MUST persist these edges so every
    later PSI call uses the same partitioning.

    Both samples are flattened, NaN-stripped, and clipped to the edges so
    out-of-range live values land in the extreme bins (rather than being
    silently dropped, which would mask the very drift we're looking for).
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    lv = np.asarray(live, dtype=np.float64).ravel()
    ref = ref[np.isfinite(ref)]
    lv = lv[np.isfinite(lv)]
    if ref.size == 0 or lv.size == 0:
        return 0.0, np.array([], dtype=np.float64)

    if bin_edges is None:
        # Quantile bins from REFERENCE only. Using live data here would
        # make PSI ≡ 0 by construction.
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(ref, quantiles))
        if edges.size < 2:
            # Reference is degenerate (all same value) — PSI is undefined;
            # report 0 and a singleton edge array so callers know to skip.
            return 0.0, edges
        # Widen the outermost bins to ±inf so live values outside the
        # reference range still get counted, in the bin closest to them.
        edges[0] = -np.inf
        edges[-1] = np.inf
    else:
        edges = bin_edges

    ref_counts, _ = np.histogram(ref, bins=edges)
    live_counts, _ = np.histogram(lv, bins=edges)

    # Convert counts to proportions with epsilon smoothing.
    ref_props = (ref_counts + _EPS) / (ref_counts.sum() + _EPS * len(ref_counts))
    live_props = (live_counts + _EPS) / (live_counts.sum() + _EPS * len(live_counts))

    # PSI = sum_i (p_live_i − p_ref_i) · ln(p_live_i / p_ref_i)
    #     ≥ 0 with equality iff distributions are identical.
    psi = float(np.sum((live_props - ref_props) * np.log(live_props / ref_props)))
    return psi, edges


# ---------------------------------------------------------------------------
# Stateful monitor
# ---------------------------------------------------------------------------


@dataclass
class _ReferenceSnapshot:
    """Per-feature reference distribution captured at training time.

    We persist:
      * ``bin_edges`` — quantile-derived edges so live histograms use the
        same partitioning (otherwise PSI ≡ 0 by construction).
      * ``ref_props`` — the reference proportions per bin, smoothed with
        the same epsilon used for live proportions. This is the actual
        comparand for PSI.
      * ``n_reference`` — sample count for diagnostic display.

    Storing proportions (a k-element array, k=n_bins) instead of the raw
    reference column keeps the pickle small (~80 bytes per feature for
    k=10) while making PSI exact rather than approximated through some
    midpoint reconstruction.
    """
    bin_edges: np.ndarray
    ref_props: np.ndarray
    n_reference: int


class FeatureDriftMonitor:
    """Rolling PSI tracker across all features the model consumes.

    Lifecycle:
      1. ``fit_reference(X_train, feature_names)`` after the model is
         trained — captures quantile bin edges per feature.
      2. ``record(feature_vector)`` on every live predict — appends to a
         bounded deque (oldest-out).
      3. ``report()`` — returns a list of :class:`FeatureDriftReport`
         sorted by PSI descending. Caller decides what to do with major
         drifts (warn, halt trading, force retrain).

    Thread-safe: a lock guards the deques because record() can fire from
    async predict paths while report() is invoked from the dashboard /
    monitoring loop.
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        min_live_samples: int = DEFAULT_MIN_LIVE,
        n_bins: int = 10,
    ) -> None:
        if window < min_live_samples:
            raise ValueError(
                f"window ({window}) must be >= min_live_samples ({min_live_samples})"
            )
        self._window = int(window)
        self._min_live = int(min_live_samples)
        self._n_bins = int(n_bins)
        # feature_name -> bin edges + reference size
        self._reference: dict[str, _ReferenceSnapshot] = {}
        # feature_name -> deque of live values
        self._live: dict[str, deque[float]] = {}
        self._feature_names: list[str] = []
        self._lock = self._fresh_lock()

    @staticmethod
    def _fresh_lock() -> "threading.RLock":
        # RLock so report() can call helpers like n_live_samples() while
        # already holding the lock — a plain Lock deadlocks the moment
        # any internal method needs the lock too. RLock is the standard
        # Python idiom when public methods compose other public methods.
        return threading.RLock()

    # ----- Pickle support -----------------------------------------------
    # threading.RLock is not picklable, so we strip the lock on dump and
    # reinstantiate it on load. Reference + live state survive the round-
    # trip; in-flight critical sections do not (acceptable: pickling
    # happens on quiet snapshots, never under contention).

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.__dict__.get("_lock") is None:
            self._lock = self._fresh_lock()

    # -------------------- lifecycle --------------------

    def fit_reference(
        self,
        X_train: np.ndarray,
        feature_names: Iterable[str],
    ) -> None:
        """Capture per-feature quantile bin edges from the training matrix.

        After this call, :meth:`record` and :meth:`report` are well-defined
        for the listed features. Calling this a second time replaces all
        previous reference state — typical use is once after each
        successful retrain.
        """
        names = list(feature_names)
        if X_train.ndim != 2 or X_train.shape[1] != len(names):
            raise ValueError(
                f"X_train shape {X_train.shape} doesn't match {len(names)} names"
            )
        with self._lock:
            self._reference.clear()
            self._live.clear()
            self._feature_names = names
            for i, name in enumerate(names):
                col = X_train[:, i]
                col_finite = col[np.isfinite(col)]
                if col_finite.size == 0:
                    continue
                # Derive quantile bins from the reference column itself.
                quantiles = np.linspace(0.0, 1.0, self._n_bins + 1)
                edges = np.unique(np.quantile(col_finite, quantiles))
                if edges.size < 2:
                    # Degenerate column (constant) — skip drift tracking.
                    continue
                edges[0] = -np.inf
                edges[-1] = np.inf
                ref_counts, _ = np.histogram(col_finite, bins=edges)
                ref_props = (ref_counts + _EPS) / (
                    ref_counts.sum() + _EPS * len(ref_counts)
                )
                self._reference[name] = _ReferenceSnapshot(
                    bin_edges=edges,
                    ref_props=ref_props,
                    n_reference=int(col_finite.size),
                )
                self._live[name] = deque(maxlen=self._window)

    def record(self, feature_vector: Iterable[float]) -> None:
        """Append a single live feature vector to the rolling window.

        Silently no-ops if :meth:`fit_reference` hasn't been called yet —
        a freshly-deployed model can drop into prediction mode before any
        retrain has populated the reference, and we don't want a
        prediction call to crash on missing monitor state.
        """
        with self._lock:
            if not self._feature_names:
                return
            vec = list(feature_vector)
            if len(vec) != len(self._feature_names):
                # Feature schema drift — different column count than training.
                # Don't silently truncate or pad: that would mask the real
                # bug (caller using a stale feature extractor against a
                # newer model).
                logger.warning(
                    "FeatureDriftMonitor: vector length %d != %d trained features — skipping",
                    len(vec), len(self._feature_names),
                )
                return
            for name, val in zip(self._feature_names, vec):
                if val is not None and np.isfinite(val):
                    self._live[name].append(float(val))

    # -------------------- reporting --------------------

    def has_reference(self) -> bool:
        with self._lock:
            return bool(self._reference)

    def n_live_samples(self) -> int:
        """Smallest live window across all features — bounded by the
        quietest feature, not the median."""
        with self._lock:
            return min((len(d) for d in self._live.values()), default=0)

    def report(self) -> list[FeatureDriftReport]:
        """Return one PSI verdict per feature, sorted by drift severity
        descending. Returns ``[]`` if either the reference is unset or
        not enough live samples have accumulated yet (< min_live_samples).
        """
        with self._lock:
            if not self._reference:
                return []
            if self.n_live_samples() < self._min_live:
                return []

            reports: list[FeatureDriftReport] = []
            for name in self._feature_names:
                snapshot = self._reference.get(name)
                live = self._live.get(name)
                if snapshot is None or live is None or snapshot.bin_edges.size < 2:
                    continue
                live_arr = np.fromiter(live, dtype=np.float64)
                # PSI directly from stored reference proportions vs the
                # live histogram on the same bin edges. No reconstruction
                # tricks — exact arithmetic.
                live_counts, _ = np.histogram(live_arr, bins=snapshot.bin_edges)
                live_props = (live_counts + _EPS) / (
                    live_counts.sum() + _EPS * len(live_counts)
                )
                psi = float(np.sum(
                    (live_props - snapshot.ref_props)
                    * np.log(live_props / snapshot.ref_props)
                ))
                reports.append(FeatureDriftReport(
                    feature=name,
                    psi=psi,
                    severity=_categorise_psi(psi),
                    n_reference=snapshot.n_reference,
                    n_live=live_arr.size,
                ))
            reports.sort(key=lambda r: r.psi, reverse=True)
            return reports

    def summary(self) -> dict:
        """Compact dict summary suitable for dashboards / JSON serialisation.

        Returns ``{"status": "<state>", "max_psi": <float>, "drifting":
        [...features...], "minor": [...features...], "n_live": int}``.
        ``status`` is one of: ``"ok"``, ``"insufficient_data"``,
        ``"minor_drift"``, ``"major_drift"``.
        """
        reps = self.report()
        if not reps:
            return {
                "status": "insufficient_data",
                "max_psi": 0.0,
                "drifting": [],
                "minor": [],
                "n_live": self.n_live_samples(),
            }
        major = [r.feature for r in reps if r.severity == "major"]
        minor = [r.feature for r in reps if r.severity == "minor"]
        if major:
            status = "major_drift"
        elif minor:
            status = "minor_drift"
        else:
            status = "ok"
        return {
            "status": status,
            "max_psi": float(reps[0].psi),
            "drifting": major,
            "minor": minor,
            "n_live": self.n_live_samples(),
        }
