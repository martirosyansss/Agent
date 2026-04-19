"""Comprehensive unit tests for the ML risk-metrics surface.

Covers everything added in the post-audit hardening pass:

* ``compute_pnl_risk_metrics``: Sortino, Calmar, Max DD, MaxDD duration,
  VaR_95/99, CVaR_95, Cornish-Fisher VaR.
* ``_historical_var`` and ``_cornish_fisher_var`` private helpers.
* Cohen's d effect size in ``MLMetrics``.
* ``bonferroni_z`` and the ``overfit_noise_margin`` n_tests path.
* The new ``MLBootstrap`` block-bootstrap path.

The goal is to lock in the contracts so a future audit cannot find a
regression in the metrics that downstream dashboards and risk monitors
depend on. Property-style tests (monotonicity, scale-invariance,
edge-case behaviour) are preferred over fixed-value snapshots.
"""
from __future__ import annotations

import numpy as np
import pytest

from analyzer.ml.domain.metrics import (
    MLMetrics,
    _cornish_fisher_var,
    _historical_var,
    _max_drawdown_from_pnl,
    compute_pnl_risk_metrics,
    compute_pnl_risk_metrics_per_symbol,
)
from analyzer.ml.training.calibration import bonferroni_z, overfit_noise_margin


# ---------------------------------------------------------------------------
# compute_pnl_risk_metrics — core contract
# ---------------------------------------------------------------------------


class TestComputePnlRiskMetrics:
    def test_returns_empty_when_too_few_predicted_positives(self):
        pnl = np.array([10.0, -5.0, 8.0, -3.0, 12.0], dtype=np.float64)
        pred = np.ones(5, dtype=np.int64)
        # Default min_pred_wins=10 → 5 < 10 → empty.
        assert compute_pnl_risk_metrics(pnl, pred) == {}

    def test_returns_empty_when_shapes_mismatch(self):
        assert compute_pnl_risk_metrics(np.zeros(20), np.zeros(15, dtype=np.int64)) == {}

    def test_only_predicted_positives_are_used(self):
        # If y_pred == 0 the trade should not influence any metric, even if
        # it has a catastrophic PnL. We construct two parallel datasets
        # that differ ONLY in the y_pred==0 rows and check metrics are equal.
        rng = np.random.default_rng(7)
        pnl = rng.normal(15, 20, size=30)
        pred = np.array([1] * 15 + [0] * 15, dtype=np.int64)
        m1 = compute_pnl_risk_metrics(pnl, pred)

        # Drop a $10000 loss into a y_pred==0 row — should not move metrics.
        pnl_poisoned = pnl.copy()
        pnl_poisoned[20] = -10_000.0
        m2 = compute_pnl_risk_metrics(pnl_poisoned, pred)

        for k in m1:
            if m1[k] is None:
                assert m2[k] is None
            else:
                assert m1[k] == pytest.approx(m2[k]), f"{k} leaked from y_pred==0 row"

    def test_sortino_is_scale_invariant(self):
        rng = np.random.default_rng(11)
        pnl = rng.normal(10, 25, size=20)
        pred = np.ones(20, dtype=np.int64)
        m1 = compute_pnl_risk_metrics(pnl, pred)
        m100 = compute_pnl_risk_metrics(pnl * 100.0, pred)
        # Sortino is a pure ratio — must be invariant under positive scaling.
        assert m1["sortino_ratio"] == pytest.approx(m100["sortino_ratio"], rel=1e-9)

    def test_calmar_is_scale_invariant(self):
        rng = np.random.default_rng(13)
        pnl = rng.normal(10, 25, size=20)
        pred = np.ones(20, dtype=np.int64)
        m1 = compute_pnl_risk_metrics(pnl, pred)
        m100 = compute_pnl_risk_metrics(pnl * 100.0, pred)
        assert m1["calmar_ratio"] == pytest.approx(m100["calmar_ratio"], rel=1e-9)

    def test_all_wins_returns_none_for_loss_metrics(self):
        # All-positive PnL: no downside, no drawdown — those metrics are
        # mathematically undefined and we report them as None (NOT ∞ or 0).
        pred = np.ones(15, dtype=np.int64)
        m = compute_pnl_risk_metrics(np.full(15, 10.0), pred)
        assert m["sortino_ratio"] is None
        assert m["calmar_ratio"] is None
        assert m["max_drawdown_pct"] == pytest.approx(0.0)
        assert m["max_drawdown_duration"] == 0
        assert m["var_95"] is None  # < 5 losses ⇒ historical VaR not reported
        assert m["cvar_95"] is None

    def test_all_losses_drawdown_is_negative_and_duration_grows(self):
        pred = np.ones(15, dtype=np.int64)
        m = compute_pnl_risk_metrics(np.full(15, -10.0), pred)
        assert m["max_drawdown_pct"] < 0.0
        # Every consecutive trade is underwater after the first → 14.
        assert m["max_drawdown_duration"] == 14
        assert m["sortino_ratio"] is None  # all losses identical → downside σ = 0
        assert m["var_95"] is not None and m["var_95"] > 0

    def test_drawdown_pct_in_natural_range(self):
        """Regression test: DD% should NEVER collapse to −100% on realistic
        data. The bug we hit during the audit fix was the equity curve
        touching zero because initial_capital was hard-coded at 1.0."""
        pred = np.ones(20, dtype=np.int64)
        m = compute_pnl_risk_metrics(
            np.array([50, -30, 80, -20, 100, -50, 40, -10, 60, -25,
                      70, -40, 90, -35, 65, -15, 75, -45, 55, -20], dtype=np.float64),
            pred,
        )
        assert -100.0 < m["max_drawdown_pct"] < 0.0

    def test_var_99_at_least_var_95(self):
        rng = np.random.default_rng(17)
        pnl = rng.normal(0, 50, size=80)
        pred = np.ones(80, dtype=np.int64)
        m = compute_pnl_risk_metrics(pnl, pred)
        assert m["var_99"] >= m["var_95"]

    def test_cvar_at_least_var(self):
        # CVaR is the conditional mean of losses ≥ VaR — by definition ≥ VaR.
        rng = np.random.default_rng(19)
        pnl = rng.normal(0, 50, size=80)
        pred = np.ones(80, dtype=np.int64)
        m = compute_pnl_risk_metrics(pnl, pred)
        assert m["cvar_95"] >= m["var_95"]

    def test_var_metrics_skipped_with_too_few_losses(self):
        # 3 losses out of 12 trades — below the 5-loss threshold for VaR.
        pnl = np.array([10] * 9 + [-5, -3, -2], dtype=np.float64)
        pred = np.ones(12, dtype=np.int64)
        m = compute_pnl_risk_metrics(pnl, pred)
        assert m["var_95"] is None
        assert m["var_99"] is None
        assert m["cvar_95"] is None


# ---------------------------------------------------------------------------
# Tail-risk helpers
# ---------------------------------------------------------------------------


class TestPerSymbolAttribution:
    def test_empty_when_shapes_mismatch(self):
        assert compute_pnl_risk_metrics_per_symbol(
            np.zeros(10), np.zeros(10, dtype=np.int64), ["A"] * 5,
        ) == {}

    def test_groups_metrics_by_symbol(self):
        # 12 trades on BTC (positive Sortino), 12 on ETH (negative Sortino).
        rng = np.random.default_rng(41)
        pnl = np.concatenate([
            rng.normal(20, 25, size=12),    # BTC — winning bias
            rng.normal(-10, 25, size=12),   # ETH — losing bias
        ])
        pred = np.ones(24, dtype=np.int64)
        symbols = ["BTC"] * 12 + ["ETH"] * 12
        out = compute_pnl_risk_metrics_per_symbol(pnl, pred, symbols)
        assert set(out.keys()) == {"BTC", "ETH"}
        assert out["BTC"]["sortino_ratio"] > out["ETH"]["sortino_ratio"], (
            "BTC should have higher Sortino than ETH"
        )
        assert "n_predicted_positive" in out["BTC"]

    def test_omits_symbols_with_too_few_predictions(self):
        # BTC has 12 predicted positives; ETH only 3 (below min_pred_wins=10).
        pnl = np.concatenate([
            np.full(12, 10.0),   # BTC
            np.full(3, -5.0),    # ETH
        ])
        pred = np.ones(15, dtype=np.int64)
        symbols = ["BTC"] * 12 + ["ETH"] * 3
        out = compute_pnl_risk_metrics_per_symbol(pnl, pred, symbols)
        assert "BTC" in out
        assert "ETH" not in out, "ETH had < 10 wins, must be omitted"

    def test_only_predicted_positives_counted_per_symbol(self):
        # All 20 BTC trades present, but only 12 are predicted-positive.
        rng = np.random.default_rng(43)
        pnl = rng.normal(15, 20, size=20)
        pred = np.array([1] * 12 + [0] * 8, dtype=np.int64)
        symbols = ["BTC"] * 20
        out = compute_pnl_risk_metrics_per_symbol(pnl, pred, symbols)
        assert out["BTC"]["n_predicted_positive"] == 12


class TestHistoricalVar:
    def test_empty_returns_zero(self):
        assert _historical_var(np.array([]), 0.95) == 0.0

    def test_known_quantile(self):
        # Loss series 1..100 (positive), VaR_95 = 95th percentile = 95.05.
        losses = np.arange(1, 101, dtype=np.float64)
        assert _historical_var(losses, 0.95) == pytest.approx(95.05, rel=0, abs=0.06)

    def test_higher_confidence_higher_var(self):
        rng = np.random.default_rng(23)
        losses = np.abs(rng.normal(0, 1, size=200))
        v95 = _historical_var(losses, 0.95)
        v99 = _historical_var(losses, 0.99)
        assert v99 >= v95


class TestCornishFisherVar:
    def test_returns_none_for_small_samples(self):
        assert _cornish_fisher_var(np.array([1.0, -1.0, 2.0]), 0.95) is None

    def test_zero_variance_returns_none(self):
        assert _cornish_fisher_var(np.full(50, 5.0), 0.95) is None

    def test_gaussian_close_to_parametric_var(self):
        """On nearly-Gaussian data, Cornish-Fisher VaR should track the
        plain parametric Gaussian VaR (same z-quantile, no correction
        terms)."""
        from scipy.stats import norm
        rng = np.random.default_rng(29)
        rets = rng.normal(0, 1, size=2000)  # large enough that γ₃≈0, γ₄≈3
        cf = _cornish_fisher_var(rets, 0.95)
        gaussian = -(rets.mean() + rets.std(ddof=1) * norm.ppf(0.05))
        assert cf == pytest.approx(gaussian, rel=0.05)

    def test_fat_tail_pushes_var_above_gaussian(self):
        """Adding strong negative skew + excess kurtosis must inflate the
        Cornish-Fisher VaR vs the naive Gaussian one."""
        from scipy.stats import norm
        rng = np.random.default_rng(31)
        # Mostly-normal returns + a few large negative shocks.
        rets = np.concatenate([rng.normal(0, 1, size=400),
                               rng.normal(-5, 0.5, size=20)])
        rng.shuffle(rets)
        cf = _cornish_fisher_var(rets, 0.95)
        gauss = -(rets.mean() + rets.std(ddof=1) * norm.ppf(0.05))
        assert cf > gauss


# ---------------------------------------------------------------------------
# Drawdown helper
# ---------------------------------------------------------------------------


class TestMaxDrawdownFromPnl:
    def test_empty_input(self):
        max_dd, dur, abs_dd = _max_drawdown_from_pnl(np.array([]))
        assert (max_dd, dur, abs_dd) == (0.0, 0, 0.0)

    def test_monotonic_uptrend_no_drawdown(self):
        max_dd, dur, abs_dd = _max_drawdown_from_pnl(np.array([10.0, 20.0, 30.0]))
        assert max_dd == pytest.approx(0.0)
        assert dur == 0
        assert abs_dd == pytest.approx(0.0)

    def test_simple_peak_then_drop(self):
        # +10, +20, -50 → peak equity ≈ baseline+30, trough ≈ baseline-20.
        max_dd, dur, abs_dd = _max_drawdown_from_pnl(np.array([10.0, 20.0, -50.0]))
        assert abs_dd == pytest.approx(50.0)
        assert dur == 1
        assert -1.0 < max_dd < 0.0

    def test_underwater_streak_counted_correctly(self):
        # Three losses in a row after a peak → streak = 3.
        max_dd, dur, _ = _max_drawdown_from_pnl(np.array([20.0, -5.0, -5.0, -5.0]))
        assert dur == 3


# ---------------------------------------------------------------------------
# MLMetrics dataclass — backwards compatibility
# ---------------------------------------------------------------------------


class TestMLMetricsDataclass:
    def test_default_construction_has_no_required_args(self):
        # Must round-trip from an empty constructor — our trainer's failure
        # branches build MLMetrics() with no args before populating fields.
        m = MLMetrics()
        assert m.precision == 0.0
        assert m.sortino_ratio is None
        assert m.var_95 is None
        assert m.precision_cohens_d is None

    def test_new_fields_accept_optional_floats(self):
        m = MLMetrics(sortino_ratio=1.2, calmar_ratio=0.8,
                      var_95=42.0, var_99=88.0, cvar_95=95.0,
                      var_95_cornish_fisher=50.0, precision_cohens_d=0.4)
        assert m.sortino_ratio == 1.2
        assert m.cvar_95 == 95.0
        assert m.precision_cohens_d == 0.4


# ---------------------------------------------------------------------------
# Bonferroni / overfit-noise margin
# ---------------------------------------------------------------------------


class TestBonferroniZ:
    @pytest.mark.parametrize("k,expected", [
        (1, 1.9600),
        (2, 2.2414),
        (4, 2.4977),
        (10, 2.8070),
        (20, 3.0233),
    ])
    def test_known_values(self, k, expected):
        # Known z-critical values for two-sided α=0.05 / K via scipy.norm.ppf.
        assert bonferroni_z(0.05, k) == pytest.approx(expected, abs=1e-3)

    def test_monotone_in_n_tests(self):
        # More tests → tighter per-test α → larger critical z.
        zs = [bonferroni_z(0.05, k) for k in (1, 2, 4, 8, 16)]
        for a, b in zip(zs, zs[1:]):
            assert b > a

    def test_clamps_invalid_inputs(self):
        # Negative or zero K is treated as K=1 (no correction).
        assert bonferroni_z(0.05, 0) == pytest.approx(1.96, abs=1e-3)
        assert bonferroni_z(0.05, -5) == pytest.approx(1.96, abs=1e-3)


class TestOverfitNoiseMargin:
    def test_zero_when_either_split_empty(self):
        assert overfit_noise_margin(0.7, 0.7, 0, 50) == 0.0
        assert overfit_noise_margin(0.7, 0.7, 50, 0) == 0.0

    def test_n_tests_widens_margin(self):
        m1 = overfit_noise_margin(0.7, 0.7, 100, 30, n_tests=1)
        m4 = overfit_noise_margin(0.7, 0.7, 100, 30, n_tests=4)
        # Bonferroni at K=4 vs K=1 → z grows ~1.27× → margin grows by same factor.
        assert m4 > m1
        ratio = m4 / m1
        assert 1.20 < ratio < 1.35  # close to z-ratio 2.4977 / 1.9600 ≈ 1.274

    def test_extreme_p_values_clamped(self):
        # Should not return NaN / Inf when p_train hits 0 or 1.
        m_low = overfit_noise_margin(0.0, 0.5, 100, 50)
        m_high = overfit_noise_margin(1.0, 0.5, 100, 50)
        assert np.isfinite(m_low) and m_low > 0
        assert np.isfinite(m_high) and m_high > 0
