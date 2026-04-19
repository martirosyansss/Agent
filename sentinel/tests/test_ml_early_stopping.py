"""Unit tests for ``_fit_with_early_stopping`` in the ML trainer.

Locks in the contracts:
* ES is *applied* (not silently skipped) when LGBM/XGB are available and
  the validation set is meaningful (≥ 10 samples, both classes present).
* ES is *gracefully skipped* on degenerate validation sets.
* Sample weights propagate in both ES and no-ES paths.
* The XGBoost early_stopping_rounds probe in factories.py correctly
  detects support on the installed version.

These tests were the gap that the post-audit review flagged: the original
ES integration shipped with a silent bug (XGBoost ES never engaged because
of an inspect.signature probe that didn't see kwargs-collected params).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from analyzer.ml.training.trainer import _fit_with_early_stopping, _EARLY_STOPPING_ROUNDS

# Suppress sklearn FutureWarnings about feature names — they're noise here.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


@pytest.fixture
def synthetic_split():
    """200-row train / 60-row val with a learnable signal."""
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(200, 8))
    y_tr = (X_tr[:, 0] + rng.normal(0, 0.5, 200) > 0).astype(int)
    X_va = rng.normal(size=(60, 8))
    y_va = (X_va[:, 0] + rng.normal(0, 0.5, 60) > 0).astype(int)
    sw = np.ones(200)
    return X_tr, y_tr, X_va, y_va, sw


def _build_lgbm():
    """Construct a fresh LightGBM with a large n_estimators to make ES visible."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        pytest.skip("lightgbm not installed")
    return LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=4, min_child_samples=20,
        random_state=0, n_jobs=1, verbose=-1,
    )


def _build_xgb():
    """Construct a fresh XGBoost with the same probed early_stopping_rounds
    kwarg the trainer uses, so the ES path inside fit() actually engages."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        pytest.skip("xgboost not installed")
    from analyzer.ml.models.factories import _xgb_es_kwargs
    return XGBClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=4, random_state=0, n_jobs=1,
        eval_metric="logloss", verbosity=0,
        **_xgb_es_kwargs(),
    )


class TestEarlyStoppingEngages:
    def test_lgbm_stops_early_when_val_set_is_meaningful(self, synthetic_split):
        X_tr, y_tr, X_va, y_va, sw = synthetic_split
        m = _build_lgbm()
        _fit_with_early_stopping(m, "lgbm", X_tr, y_tr, X_va, y_va, sw)
        # Either n_iter_ < n_estimators OR best_iteration_ is set — at least
        # one signals that ES actually engaged.
        n_iter = getattr(m, "n_iter_", m.n_estimators)
        best = getattr(m, "best_iteration_", None)
        assert n_iter < m.n_estimators or (best is not None and best < m.n_estimators), (
            "LGBM ES did not engage — model trained all rounds"
        )

    def test_xgb_records_best_iteration(self, synthetic_split):
        X_tr, y_tr, X_va, y_va, sw = synthetic_split
        m = _build_xgb()
        _fit_with_early_stopping(m, "xgb", X_tr, y_tr, X_va, y_va, sw)
        # XGBoost exposes best_iteration when ES is active.
        best = getattr(m, "best_iteration", None)
        assert best is not None, "XGBoost did not register best_iteration — ES not active"
        assert best < m.n_estimators, "XGBoost trained all rounds — ES gate broken"


class TestEarlyStoppingFallback:
    def test_skips_when_val_too_small(self, synthetic_split):
        X_tr, y_tr, _, _, sw = synthetic_split
        # Validation of size 5 — below the 10-row threshold.
        X_va_tiny = X_tr[:5]
        y_va_tiny = y_tr[:5]
        m = _build_lgbm()
        _fit_with_early_stopping(m, "lgbm", X_tr, y_tr, X_va_tiny, y_va_tiny, sw)
        # No ES → trains all rounds; best_iteration_ should be unset / equal to n_estimators.
        n_iter = getattr(m, "n_iter_", m.n_estimators)
        assert n_iter == m.n_estimators, (
            "Expected full-rounds training when val set < 10, got early stop"
        )

    def test_skips_when_val_is_single_class(self, synthetic_split):
        X_tr, y_tr, X_va, _, sw = synthetic_split
        # Force val labels to be all 1 — degenerate, no ES signal.
        y_va_single = np.ones(len(X_va), dtype=np.int64)
        m = _build_lgbm()
        _fit_with_early_stopping(m, "lgbm", X_tr, y_tr, X_va, y_va_single, sw)
        n_iter = getattr(m, "n_iter_", m.n_estimators)
        assert n_iter == m.n_estimators

    def test_unknown_kind_falls_back_to_plain_fit(self, synthetic_split):
        # Random Forest doesn't go through this helper in production, but
        # the fallback path must not raise on an unknown kind.
        X_tr, y_tr, X_va, y_va, sw = synthetic_split
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
        _fit_with_early_stopping(m, "unknown_kind", X_tr, y_tr, X_va, y_va, sw)
        # Just verify the model is fitted (predict works).
        preds = m.predict(X_va)
        assert preds.shape == (len(X_va),)


class TestSampleWeightPropagation:
    def test_sample_weights_actually_change_fit(self, synthetic_split):
        """Two trainings with vastly different sample weights must produce
        different models — proving sample_weight reaches LightGBM."""
        X_tr, y_tr, X_va, y_va, _ = synthetic_split
        sw_uniform = np.ones(len(y_tr))
        sw_skewed = np.where(y_tr == 1, 10.0, 1.0)  # 10× weight on positives

        m_a = _build_lgbm()
        m_b = _build_lgbm()
        _fit_with_early_stopping(m_a, "lgbm", X_tr, y_tr, X_va, y_va, sw_uniform)
        _fit_with_early_stopping(m_b, "lgbm", X_tr, y_tr, X_va, y_va, sw_skewed)

        p_a = m_a.predict_proba(X_va)[:, 1].mean()
        p_b = m_b.predict_proba(X_va)[:, 1].mean()
        # Strongly up-weighting positives must shift the average predicted
        # probability of the positive class upward.
        assert p_b > p_a, (
            f"Sample weight had no observable effect: uniform p̄={p_a:.3f} vs skewed p̄={p_b:.3f}"
        )


class TestXgbEsProbe:
    def test_probe_returns_kwargs_on_modern_xgboost(self):
        """On any xgboost ≥ 1.6 the probe must return the kwarg dict, not
        an empty {}. Catches the regression where inspect.signature missed
        the param on xgboost ≥ 3.x."""
        try:
            import xgboost  # noqa: F401
        except ImportError:
            pytest.skip("xgboost not installed")
        from analyzer.ml.models.factories import _xgb_es_kwargs
        probed = _xgb_es_kwargs()
        assert probed == {"early_stopping_rounds": _EARLY_STOPPING_ROUNDS}, (
            f"XGBoost ES probe returned {probed!r}; expected the rounds kwarg"
        )

    def test_probe_is_cached(self):
        """The probe should run at most once per process — verified by
        checking that two consecutive calls return the same dict object."""
        from analyzer.ml.models.factories import _xgb_es_kwargs
        first = _xgb_es_kwargs()
        second = _xgb_es_kwargs()
        assert first is second, "Probe is not cached — it ran twice"
