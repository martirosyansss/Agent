"""
Feature-importance stability across multiple fits.

A point estimate of feature importance says "feature X contributes 8% to
this model". It does *not* say whether the same feature would top the
chart on a different sample or a different seed — and in practice
tree-based importance is notoriously unstable: a feature that ranks #1
on fold A can drop to #15 on fold B with the same data.

This module takes a list of importance dictionaries (one per member of
an ensemble, per walk-forward fold, or per bootstrap resample) and
reports:

* **Per-feature mean + 5th/95th percentile band** — so the dashboard can
  draw an error bar instead of a single number.
* **Top-K persistence** — fraction of runs where each feature was in the
  top K. A feature with mean importance 0.04 and top-5 persistence 0.9
  is more trustworthy than one with mean 0.08 and persistence 0.3.
* **Rank-correlation stability** — mean pairwise Spearman correlation
  between ranking lists across runs. ≥ 0.7 ≈ stable, ≤ 0.4 is a red flag
  meaning the model's interpretation shifts fold-to-fold.

All outputs are JSON-serialisable floats / dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class FeatureImportanceStats:
    """Summary statistics for one feature across multiple fits."""
    name: str
    mean: float
    p05: float
    p95: float
    top3_rate: float
    top5_rate: float
    n_observations: int


@dataclass(slots=True)
class ImportanceStabilityReport:
    """Aggregated result across all features + rank-correlation summary."""
    per_feature: dict[str, FeatureImportanceStats] = field(default_factory=dict)
    # Mean Spearman correlation between ranking lists across all pairs of
    # runs. 1.0 = identical rankings, 0.0 = uncorrelated, < 0 = reversed.
    mean_rank_spearman: float = 0.0
    # Pairs of runs that actually had ≥ 5 ranked features (denominator);
    # small N hints the metric is noisy.
    n_pairs: int = 0
    # How many runs contributed to the summary (folds / members / bootstraps).
    n_runs: int = 0

    def sorted_by_mean(self) -> list[FeatureImportanceStats]:
        return sorted(
            self.per_feature.values(), key=lambda s: s.mean, reverse=True,
        )

    def to_dashboard(self) -> dict:
        """Flat dict for JSON / dashboard consumption."""
        return {
            "n_runs": self.n_runs,
            "mean_rank_spearman": round(self.mean_rank_spearman, 4),
            "n_pairs": self.n_pairs,
            "features": [
                {
                    "name": s.name,
                    "mean": round(s.mean, 6),
                    "p05": round(s.p05, 6),
                    "p95": round(s.p95, 6),
                    "top3_rate": round(s.top3_rate, 3),
                    "top5_rate": round(s.top5_rate, 3),
                    "n": s.n_observations,
                }
                for s in self.sorted_by_mean()
            ],
        }


def _rank_importances(imp: dict[str, float]) -> dict[str, int]:
    """Return a ``{name: rank}`` dict, rank 1 = largest importance."""
    if not imp:
        return {}
    items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
    return {name: rank for rank, (name, _) in enumerate(items, start=1)}


def _spearman_on_shared(a: dict[str, int], b: dict[str, int]) -> float:
    """Spearman correlation between two rank dicts on their intersection.

    Implemented manually rather than pulling scipy — Spearman reduces to
    Pearson on the rank values, which is one-liner with numpy. Returns
    0.0 when fewer than 2 shared features (correlation undefined).
    """
    shared = sorted(set(a.keys()) & set(b.keys()))
    if len(shared) < 2:
        return 0.0
    x = np.array([a[k] for k in shared], dtype=np.float64)
    y = np.array([b[k] for k in shared], dtype=np.float64)
    x_std = x.std()
    y_std = y.std()
    if x_std == 0 or y_std == 0:
        # One of the rankings has no variation (all tied) — correlation
        # is mathematically undefined; 0 is a conservative default.
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_stability(
    importances: list[dict[str, float]],
    *,
    top_k_sets: tuple[int, ...] = (3, 5),
) -> ImportanceStabilityReport:
    """Aggregate stability diagnostics across a list of importance dicts.

    Each dict is one "run" — a walk-forward fold, an ensemble member,
    a bootstrap resample, or any other independent re-fit.

    Args:
        importances: List of ``{feature_name: importance_value}``. Runs
            can have overlapping-but-not-identical feature sets; each
            feature's mean/percentiles use only the runs where it
            actually appeared.
        top_k_sets: Top-K persistence thresholds to report. Default
            ``(3, 5)`` → ``top3_rate`` and ``top5_rate``.

    Returns:
        ``ImportanceStabilityReport`` with per-feature summary and
        pairwise rank-correlation aggregate.
    """
    runs = [r for r in importances if r]  # drop empty dicts
    n_runs = len(runs)
    if n_runs == 0:
        return ImportanceStabilityReport()

    feature_names: set[str] = set()
    for run in runs:
        feature_names.update(run.keys())

    # Gather per-feature value sequences + top-K presence flags.
    per_feature: dict[str, FeatureImportanceStats] = {}
    top_ks = sorted(set(top_k_sets))
    for name in feature_names:
        values: list[float] = []
        top_flags: dict[int, list[int]] = {k: [] for k in top_ks}
        for run in runs:
            if name in run:
                values.append(float(run[name]))
                # Sort once per run; small overhead given typical feature count.
                ranked = sorted(run.items(), key=lambda kv: kv[1], reverse=True)
                rank_positions = {n: i + 1 for i, (n, _) in enumerate(ranked)}
                rank = rank_positions.get(name, 10**9)
                for k in top_ks:
                    top_flags[k].append(1 if rank <= k else 0)
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        per_feature[name] = FeatureImportanceStats(
            name=name,
            mean=float(arr.mean()),
            p05=float(np.percentile(arr, 5)),
            p95=float(np.percentile(arr, 95)),
            top3_rate=float(np.mean(top_flags[3])) if 3 in top_flags else 0.0,
            top5_rate=float(np.mean(top_flags[5])) if 5 in top_flags else 0.0,
            n_observations=arr.size,
        )

    # Pairwise rank-correlation across runs. For K runs that's K·(K-1)/2
    # pairs; fine for K ≤ 20 (ensemble + folds ≈ 4 × 5 = 20).
    ranks_per_run = [_rank_importances(run) for run in runs]
    correlations: list[float] = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            rho = _spearman_on_shared(ranks_per_run[i], ranks_per_run[j])
            correlations.append(rho)
    mean_rank = float(np.mean(correlations)) if correlations else 0.0

    return ImportanceStabilityReport(
        per_feature=per_feature,
        mean_rank_spearman=mean_rank,
        n_pairs=len(correlations),
        n_runs=n_runs,
    )
