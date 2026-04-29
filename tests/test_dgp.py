"""Tests for draci.dgp module."""

import numpy as np
import pytest

from draci.dgp import (
    AR1DGP,
    DGPData,
    generate_data,
    generate_covariates,
    propensity_true,
    cate_true,
    outcome_mean,
    generate_garch_errors,
    DEFAULT_DIM,
)


class TestDGPData:
    """Tests for DGPData dataclass."""

    def test_properties(self):
        """DGPData properties should work correctly."""
        T, dim = 100, 5
        data = DGPData(
            X=np.zeros((T, dim)),
            W=np.zeros(T),
            Y=np.zeros(T),
            e_true=np.zeros(T),
            mu0_true=np.zeros(T),
            mu1_true=np.zeros(T),
            tau_true=np.zeros(T),
        )
        assert data.T == T
        assert data.dim == dim


class TestAR1DGP:
    """Tests for AR1DGP class."""

    def test_output_shapes(self, rng):
        """All outputs should have correct dimensions."""
        dgp = AR1DGP(rho=0.5)
        T = 100

        data = dgp.generate(T, rng=rng)

        assert isinstance(data, DGPData)
        assert data.X.shape == (T, DEFAULT_DIM)
        assert data.W.shape == (T,)
        assert data.Y.shape == (T,)
        assert data.e_true.shape == (T,)
        assert data.mu0_true.shape == (T,)
        assert data.mu1_true.shape == (T,)
        assert data.tau_true.shape == (T,)

    def test_custom_dim(self, rng):
        """Should respect custom covariate dimension."""
        dgp = AR1DGP(rho=0.5, dim=3)
        data = dgp.generate(100, rng=rng)

        assert data.X.shape == (100, 3)
        assert data.dim == 3

    def test_treatment_binary(self, rng):
        """Treatment W should be binary (0 or 1)."""
        dgp = AR1DGP(rho=0.5)
        data = dgp.generate(500, rng=rng)

        assert set(np.unique(data.W)).issubset({0.0, 1.0})

    def test_treatment_rate_approximately_correct(self, rng):
        """Treatment rate should be approximately 50%."""
        dgp = AR1DGP(rho=0.0)
        # Large sample for stable estimate
        data = dgp.generate(5000, rng=rng)
        treatment_rate = data.W.mean()

        assert 0.35 < treatment_rate < 0.65

    def test_propensity_bounds(self, rng):
        """Propensity scores should be clipped to [0.05, 0.95]."""
        dgp = AR1DGP(rho=0.9)
        data = dgp.generate(1000, rng=rng)

        assert np.all(data.e_true >= 0.05)
        assert np.all(data.e_true <= 0.95)

    def test_custom_propensity_bounds(self, rng):
        """Should respect custom propensity bounds."""
        dgp = AR1DGP(rho=0.5, propensity_clip=(0.1, 0.9))
        data = dgp.generate(1000, rng=rng)

        assert np.all(data.e_true >= 0.1)
        assert np.all(data.e_true <= 0.9)

    def test_ar1_stationarity(self, rng):
        """Variance should stabilize for |rho| < 1."""
        dgp = AR1DGP(rho=0.8)
        data = dgp.generate(2000, rng=rng)

        # Compare variance in first half vs second half (first covariate)
        var_first = np.var(data.X[:1000, 0])
        var_second = np.var(data.X[1000:, 0])

        # Should be similar (ratio close to 1)
        ratio = var_first / var_second
        assert 0.5 < ratio < 2.0

    def test_iid_case(self, rng):
        """When rho=0, X should be approximately IID N(0,1)."""
        dgp = AR1DGP(rho=0.0)
        data = dgp.generate(5000, rng=rng)

        # Each covariate should have mean ~0 and variance ~1
        assert abs(np.mean(data.X[:, 0])) < 0.1
        assert abs(np.var(data.X[:, 0]) - 1.0) < 0.2

    def test_cate_formula(self, rng):
        """tau_true should match: tau(x) = sin(2*pi*x1) + 0.5*x2."""
        dgp = AR1DGP(rho=0.5)
        data = dgp.generate(100, rng=rng)

        expected_tau = np.sin(2 * np.pi * data.X[:, 0]) + 0.5 * data.X[:, 1]
        np.testing.assert_array_almost_equal(data.tau_true, expected_tau)

    def test_mu1_equals_mu0_plus_tau(self, rng):
        """mu1_true should equal mu0_true + tau_true."""
        dgp = AR1DGP(rho=0.5)
        data = dgp.generate(100, rng=rng)

        np.testing.assert_array_almost_equal(
            data.mu1_true, data.mu0_true + data.tau_true
        )

    def test_outcome_formula(self, rng):
        """mu0 should match: 2 + x1 + 0.5*x2^2."""
        dgp = AR1DGP(rho=0.5)
        data = dgp.generate(100, rng=rng)

        expected_mu0 = 2.0 + data.X[:, 0] + 0.5 * data.X[:, 1] ** 2
        np.testing.assert_array_almost_equal(data.mu0_true, expected_mu0)

    def test_reproducibility_with_seed(self):
        """Same seed should produce same data."""
        dgp = AR1DGP(rho=0.5)

        data1 = dgp.generate(100, seed=42)
        data2 = dgp.generate(100, seed=42)

        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.W, data2.W)
        np.testing.assert_array_equal(data1.Y, data2.Y)

    def test_reproducibility_with_rng(self, rng):
        """Same rng state should produce same data."""
        dgp = AR1DGP(rho=0.5)

        rng1 = np.random.default_rng(42)
        data1 = dgp.generate(100, rng=rng1)

        rng2 = np.random.default_rng(42)
        data2 = dgp.generate(100, rng=rng2)

        np.testing.assert_array_equal(data1.X, data2.X)


class TestGARCHErrors:
    """Tests for GARCH error generation."""

    def test_error_shape(self, rng):
        """GARCH errors should have correct shape."""
        dgp = AR1DGP()
        errors = dgp.generate_garch_errors(500, rng)

        assert errors.shape == (500,)

    def test_error_finite(self, rng):
        """GARCH errors should be finite."""
        dgp = AR1DGP()
        errors = dgp.generate_garch_errors(1000, rng)

        assert np.all(np.isfinite(errors))

    def test_error_volatility_clustering(self, rng):
        """GARCH errors should exhibit volatility clustering."""
        dgp = AR1DGP()
        errors = dgp.generate_garch_errors(2000, rng)

        # Squared errors should have positive autocorrelation
        squared = errors**2
        acf1 = np.corrcoef(squared[1:], squared[:-1])[0, 1]

        # With GARCH parameters (alpha=0.3, beta=0.5), expect positive acf
        assert acf1 > 0


class TestFunctionalInterface:
    """Tests for functional interface functions."""

    def test_generate_covariates(self, rng):
        """generate_covariates should work."""
        X = generate_covariates(100, rho=0.5, rng=rng)
        assert X.shape == (100, DEFAULT_DIM)

    def test_propensity_true(self, rng):
        """propensity_true should return valid propensities."""
        X = np.random.default_rng(42).normal(0, 1, (100, DEFAULT_DIM))
        e = propensity_true(X)

        assert e.shape == (100,)
        assert np.all(e >= 0.05)
        assert np.all(e <= 0.95)

    def test_cate_true(self, rng):
        """cate_true should return CATE values."""
        X = np.random.default_rng(42).normal(0, 1, (100, DEFAULT_DIM))
        tau = cate_true(X)

        assert tau.shape == (100,)
        assert np.all(np.isfinite(tau))

    def test_outcome_mean(self, rng):
        """outcome_mean should work for both arms."""
        X = np.random.default_rng(42).normal(0, 1, (100, DEFAULT_DIM))

        mu0 = outcome_mean(X, w=0)
        mu1 = outcome_mean(X, w=1)

        assert mu0.shape == (100,)
        assert mu1.shape == (100,)
        # mu1 should be different from mu0 (treatment effect)
        assert not np.allclose(mu0, mu1)

    def test_generate_data_dict(self, rng):
        """generate_data should return dict with correct keys."""
        data = generate_data(100, rho=0.5, rng=rng)

        assert isinstance(data, dict)
        expected_keys = {'X', 'W', 'Y', 'e_true', 'mu0_true', 'mu1_true', 'tau_true'}
        assert set(data.keys()) == expected_keys

    def test_generate_garch_errors_func(self, rng):
        """generate_garch_errors function should work."""
        errors = generate_garch_errors(500, rng)
        assert errors.shape == (500,)
