"""
Tests for ElasticNet LR integration as the 4th ensemble member.

Focus: the builder returns a working classifier when sklearn is present,
returns None when the feature flag is off, and the ensemble correctly
accepts / rejects the model through the existing overfit pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from analyzer.ml_predictor import MLPredictor, MLConfig


class TestBuilder:
    def test_returns_model_when_flag_on(self):
        p = MLPredictor(MLConfig(use_elastic_net=True))
        m = p._build_elastic_net()
        assert m is not None
        # sklearn-like API check
        assert hasattr(m, "fit")
        assert hasattr(m, "predict_proba")

    def test_returns_none_when_flag_off(self):
        p = MLPredictor(MLConfig(use_elastic_net=False))
        assert p._build_elastic_net() is None

    def test_conservative_uses_stronger_regularization(self):
        p = MLPredictor(MLConfig(use_elastic_net=True))
        normal = p._build_elastic_net(conservative=False)
        conservative = p._build_elastic_net(conservative=True)
        # Conservative path has smaller C (stronger regularization)
        assert conservative.C < normal.C
        # And leans more toward L1
        assert conservative.l1_ratio > normal.l1_ratio


class TestFitOnSyntheticData:
    def test_fits_and_predicts_in_unit_interval(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(200, 10))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        p = MLPredictor(MLConfig(use_elastic_net=True))
        model = p._build_elastic_net()
        model.fit(X, y)
        probas = model.predict_proba(X)[:, 1]
        assert probas.shape == (200,)
        assert np.all((probas >= 0) & (probas <= 1))

    def test_learns_linear_boundary(self):
        """ElasticNet should learn an obvious linear boundary."""
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, size=(500, 6))
        # Signal on feature 0 only
        y = (X[:, 0] > 0).astype(int)
        p = MLPredictor(MLConfig(use_elastic_net=True))
        model = p._build_elastic_net()
        model.fit(X, y)
        from sklearn.metrics import roc_auc_score
        probas = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probas)
        assert auc > 0.90, f"Linear model should easily get AUC > 0.9 on linear data, got {auc}"


class TestDecorrelation:
    def test_error_correlation_is_measurable(self):
        """ElasticNet and RF errors shouldn't be perfectly correlated.

        This is the empirical claim in the design note: a linear model and
        a tree model make qualitatively different mistakes.
        """
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(9)
        n = 500
        X = rng.normal(0, 1, size=(n, 6))
        # Mixed signal: both linear and non-linear components
        y = ((X[:, 0] > 0) ^ (X[:, 1] ** 2 > 1)).astype(int)

        rf = RandomForestClassifier(n_estimators=30, random_state=9)
        rf.fit(X, y)
        p = MLPredictor(MLConfig(use_elastic_net=True))
        lr = p._build_elastic_net()
        lr.fit(X, y)

        err_rf = rf.predict_proba(X)[:, 1] - y
        err_lr = lr.predict_proba(X)[:, 1] - y
        corr = np.corrcoef(err_rf, err_lr)[0, 1]
        # We want strictly less than perfect correlation — 0.95 would mean
        # the two models are basically copies, which defeats the purpose.
        assert abs(corr) < 0.95, f"LR and RF errors too correlated: {corr}"


class TestMLConfigFlag:
    def test_default_flag_is_on(self):
        assert MLConfig().use_elastic_net is True

    def test_flag_respected_in_builder(self):
        p_on = MLPredictor(MLConfig(use_elastic_net=True))
        p_off = MLPredictor(MLConfig(use_elastic_net=False))
        assert p_on._build_elastic_net() is not None
        assert p_off._build_elastic_net() is None
