"""Tests for draci.baselines module."""

import numpy as np
import pytest

from draci.baselines import (
    CIMetrics,
    dml_wald_ci,
    block_bootstrap_ci,
    causal_forest_ci,
    compute_ci_metrics,
    _local_nn_variance,
    _newey_west_neff,
    _circular_block_bootstrap,
)
from draci.core import ConformalResult


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_local_nn_variance_shape(self, rng):
        """Local NN variance should return correct shape."""
        T = 100
        residuals = rng.normal(0, 1, T)
        X = rng.normal(0, 1, (T, 5))

        V_hat = _local_nn_variance(residuals, X, k=20)

        assert V_hat.shape == (T,)

    def test_local_nn_variance_positive(self, rng):
        """Local NN variance should be positive."""
        T = 100
        residuals = rng.normal(0, 1, T)
        X = rng.normal(0, 1, (T, 5))

        V_hat = _local_nn_variance(residuals, X, k=20)

        assert np.all(V_hat > 0)

    def test_local_nn_variance_1d_input(self, rng):
        """Should handle 1D covariate input."""
        T = 100
        residuals = rng.normal(0, 1, T)
        X = rng.normal(0, 1, T)  # 1D

        V_hat = _local_nn_variance(residuals, X, k=20)

        assert V_hat.shape == (T,)

    def test_newey_west_neff_iid(self, rng):
        """Newey-West n_eff should be ~T for iid data."""
        T = 500
        residuals = rng.normal(0, 1, T)

        n_eff = _newey_west_neff(residuals)

        # For iid, n_eff should be close to T
        assert 0.8 * T < n_eff < 1.2 * T

    def test_newey_west_neff_dependent(self, rng):
        """Newey-West n_eff should be < T for dependent data."""
        T = 500
        # Generate AR(1) process
        rho = 0.8
        residuals = np.zeros(T)
        residuals[0] = rng.normal()
        for t in range(1, T):
            residuals[t] = rho * residuals[t-1] + np.sqrt(1 - rho**2) * rng.normal()

        n_eff = _newey_west_neff(residuals)

        # For dependent data, n_eff should be smaller than T
        assert n_eff < T
        assert n_eff > 0

    def test_circular_block_bootstrap_length(self, rng):
        """Circular block bootstrap should return T indices."""
        T = 100
        block_size = 10

        indices = _circular_block_bootstrap(T, block_size, rng)

        assert len(indices) == T

    def test_circular_block_bootstrap_valid_indices(self, rng):
        """All bootstrap indices should be in [0, T-1]."""
        T = 100
        block_size = 10

        indices = _circular_block_bootstrap(T, block_size, rng)

        assert np.all(indices >= 0)
        assert np.all(indices < T)

    def test_circular_block_bootstrap_wraps(self, rng):
        """Circular bootstrap should handle wraparound."""
        T = 20
        block_size = 15  # Large blocks to force wrapping

        # Run multiple times to increase chance of hitting edge
        for _ in range(10):
            indices = _circular_block_bootstrap(T, block_size, rng)
            assert len(indices) == T


class TestDMLWaldCI:
    """Tests for DML-Wald CI method."""

    def test_returns_conformal_result(self, rng):
        """Should return ConformalResult."""
        T = 200
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)
        X_cal = rng.normal(0, 1, (T, 5))

        result = dml_wald_ci(psi_dr, tau_hat, true_residuals, X_cal, alpha=0.1)

        assert isinstance(result, ConformalResult)

    def test_coverage_in_valid_range(self, rng):
        """Coverage should be in [0, 1]."""
        T = 200
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)
        X_cal = rng.normal(0, 1, (T, 5))

        result = dml_wald_ci(psi_dr, tau_hat, true_residuals, X_cal, alpha=0.1)

        assert 0 <= result.coverage <= 1

    def test_width_positive(self, rng):
        """Interval widths should be positive."""
        T = 200
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)
        X_cal = rng.normal(0, 1, (T, 5))

        result = dml_wald_ci(psi_dr, tau_hat, true_residuals, X_cal, alpha=0.1)

        assert result.avg_width > 0
        assert np.all(result.widths_t > 0)

    def test_coverages_t_shape(self, rng):
        """Per-timestep coverages should have correct shape."""
        T = 200
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)
        X_cal = rng.normal(0, 1, (T, 5))

        result = dml_wald_ci(psi_dr, tau_hat, true_residuals, X_cal, alpha=0.1)

        assert result.coverages_t.shape == (T,)
        assert result.widths_t.shape == (T,)


class TestBlockBootstrapCI:
    """Tests for Block Bootstrap CI method."""

    def test_returns_conformal_result(self, rng):
        """Should return ConformalResult."""
        T = 200
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)

        result = block_bootstrap_ci(psi_dr, tau_hat, true_residuals, alpha=0.1, B=50, rng=rng)

        assert isinstance(result, ConformalResult)

    def test_coverage_in_valid_range(self, rng):
        """Coverage should be in [0, 1]."""
        T = 200
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)

        result = block_bootstrap_ci(psi_dr, tau_hat, true_residuals, alpha=0.1, B=50, rng=rng)

        assert 0 <= result.coverage <= 1

    def test_width_positive(self, rng):
        """Interval widths should be positive."""
        T = 200
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)

        result = block_bootstrap_ci(psi_dr, tau_hat, true_residuals, alpha=0.1, B=50, rng=rng)

        assert result.avg_width > 0

    def test_default_block_size(self, rng):
        """Should use floor(T^{1/3}) as default block size."""
        T = 1000
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)

        # Just verify it runs without error with default block_size
        result = block_bootstrap_ci(psi_dr, tau_hat, true_residuals, alpha=0.1, B=20, rng=rng)

        assert isinstance(result, ConformalResult)

    def test_custom_block_size(self, rng):
        """Should respect custom block size."""
        T = 200
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)

        result = block_bootstrap_ci(
            psi_dr, tau_hat, true_residuals, alpha=0.1, B=20, block_size=20, rng=rng
        )

        assert isinstance(result, ConformalResult)

    def test_reproducibility(self):
        """Same rng state should produce same results."""
        T = 100
        psi_dr = np.random.default_rng(42).normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)

        rng1 = np.random.default_rng(123)
        result1 = block_bootstrap_ci(psi_dr, tau_hat, true_residuals, alpha=0.1, B=50, rng=rng1)

        rng2 = np.random.default_rng(123)
        result2 = block_bootstrap_ci(psi_dr, tau_hat, true_residuals, alpha=0.1, B=50, rng=rng2)

        assert result1.coverage == result2.coverage


class TestCausalForestCI:
    """Tests for Causal Forest CI method."""

    def test_returns_conformal_result(self, rng):
        """Should return ConformalResult (possibly with NaN if econml not installed)."""
        T = 200
        Y = rng.normal(0, 1, T)
        W = rng.binomial(1, 0.5, T).astype(float)
        X = rng.normal(0, 1, (T, 5))
        tau_true = np.sin(X[:, 0])

        result = causal_forest_ci(Y, W, X, tau_true, alpha=0.1, random_state=42)

        assert isinstance(result, ConformalResult)

    def test_handles_missing_econml(self, rng):
        """Should return NaN gracefully if econml not installed."""
        # This test may pass with NaN or actual values depending on environment
        T = 100
        Y = rng.normal(0, 1, T)
        W = rng.binomial(1, 0.5, T).astype(float)
        X = rng.normal(0, 1, (T, 5))
        tau_true = np.sin(X[:, 0])

        result = causal_forest_ci(Y, W, X, tau_true, alpha=0.1, random_state=42)

        # Should not raise, may return NaN
        assert isinstance(result, ConformalResult)


class TestCIMetrics:
    """Tests for CIMetrics and compute_ci_metrics."""

    def test_cimetrics_named_tuple(self):
        """CIMetrics should be a NamedTuple."""
        metrics = CIMetrics(
            miscoverage_gap=0.05,
            piaw=1.0,
            rolling_coverage=np.array([0.9]),
            undercoverage_rate=0.1,
        )

        assert metrics.miscoverage_gap == 0.05
        assert metrics.piaw == 1.0
        assert len(metrics.rolling_coverage) == 1
        assert metrics.undercoverage_rate == 0.1

    def test_compute_ci_metrics_basic(self, rng):
        """compute_ci_metrics should return valid metrics."""
        T = 500
        coverages_t = rng.binomial(1, 0.9, T).astype(float)
        widths_t = rng.uniform(0.5, 1.5, T)

        metrics = compute_ci_metrics(coverages_t, widths_t, alpha=0.1, window=50)

        assert isinstance(metrics, CIMetrics)
        assert np.isfinite(metrics.miscoverage_gap)
        assert np.isfinite(metrics.piaw)
        assert len(metrics.rolling_coverage) > 0
        assert np.isfinite(metrics.undercoverage_rate)

    def test_miscoverage_gap_formula(self, rng):
        """Miscoverage gap should be (1 - coverage) - alpha."""
        T = 1000
        # 85% coverage
        coverages_t = np.zeros(T)
        coverages_t[:850] = 1.0

        metrics = compute_ci_metrics(coverages_t, 1.0, alpha=0.1)

        # Gap = (1 - 0.85) - 0.1 = 0.05
        assert abs(metrics.miscoverage_gap - 0.05) < 0.01

    def test_piaw_is_mean_width(self, rng):
        """PIAW should be mean of widths."""
        T = 100
        coverages_t = rng.binomial(1, 0.9, T).astype(float)
        widths_t = np.ones(T) * 2.0  # All widths = 2.0

        metrics = compute_ci_metrics(coverages_t, widths_t, alpha=0.1)

        assert abs(metrics.piaw - 2.0) < 0.01

    def test_handles_scalar_width(self, rng):
        """Should handle scalar width input."""
        T = 100
        coverages_t = rng.binomial(1, 0.9, T).astype(float)

        metrics = compute_ci_metrics(coverages_t, 1.5, alpha=0.1)  # Scalar width

        assert metrics.piaw == 1.5

    def test_handles_nan_coverages(self, rng):
        """Should handle NaN in coverages."""
        T = 100
        coverages_t = np.full(T, np.nan)

        metrics = compute_ci_metrics(coverages_t, 1.0, alpha=0.1)

        assert np.isnan(metrics.miscoverage_gap)
        assert np.isnan(metrics.piaw)

    def test_rolling_coverage_length(self, rng):
        """Rolling coverage should have correct length."""
        T = 500
        window = 50
        coverages_t = rng.binomial(1, 0.9, T).astype(float)

        metrics = compute_ci_metrics(coverages_t, 1.0, alpha=0.1, window=window)

        # Convolution with mode='valid' gives T - window + 1 elements
        assert len(metrics.rolling_coverage) == T - window + 1

    def test_undercoverage_rate_range(self, rng):
        """Undercoverage rate should be in [0, 1]."""
        T = 500
        coverages_t = rng.binomial(1, 0.9, T).astype(float)

        metrics = compute_ci_metrics(coverages_t, 1.0, alpha=0.1, window=50)

        assert 0 <= metrics.undercoverage_rate <= 1
