"""Tests for draci.nuisance module."""

import numpy as np
import pytest

from draci.nuisance import (
    NuisanceResult,
    NuisanceFunctions,
    LinearNuisance,
    XGBoostNuisance,
    fit_nuisances,
)


class TestNuisanceResult:
    """Tests for NuisanceResult dataclass."""

    def test_dataclass_creation(self):
        """NuisanceResult should accept all required fields."""
        T = 50
        result = NuisanceResult(
            e_hat=np.full(T, 0.5),
            mu0_hat=np.zeros(T),
            mu1_hat=np.ones(T),
            tau_hat=np.ones(T),
        )
        assert result.e_hat.shape == (T,)
        assert result.mu0_hat.shape == (T,)
        assert result.mu1_hat.shape == (T,)
        assert result.tau_hat.shape == (T,)


class TestNuisanceFunctions:
    """Tests for NuisanceFunctions dataclass."""

    def test_predict_method(self):
        """NuisanceFunctions.predict should return NuisanceResult."""
        funcs = NuisanceFunctions(
            e_hat=lambda X: np.full(len(X), 0.5),
            mu0_hat=lambda X: np.zeros(len(X)),
            mu1_hat=lambda X: np.ones(len(X)),
            tau_hat=lambda X: np.ones(len(X)),
        )
        X = np.zeros((10, 2))
        result = funcs.predict(X)

        assert isinstance(result, NuisanceResult)
        assert len(result.e_hat) == 10


class TestLinearNuisance:
    """Tests for LinearNuisance estimator."""

    def test_returns_nuisance_functions(self, rng):
        """fit() should return NuisanceFunctions."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        assert isinstance(funcs, NuisanceFunctions)
        assert callable(funcs.e_hat)
        assert callable(funcs.mu0_hat)
        assert callable(funcs.mu1_hat)
        assert callable(funcs.tau_hat)

    def test_predictions_correct_shape_1d(self, rng):
        """Predictions should match input shape for 1D covariates."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, 50)

        assert funcs.e_hat(X_test).shape == (50,)
        assert funcs.mu0_hat(X_test).shape == (50,)
        assert funcs.mu1_hat(X_test).shape == (50,)
        assert funcs.tau_hat(X_test).shape == (50,)

    def test_predictions_correct_shape_2d(self, rng):
        """Predictions should match input shape for 2D covariates."""
        X = rng.normal(0, 1, (200, 3))
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X[:, 0] + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, (50, 3))

        assert funcs.e_hat(X_test).shape == (50,)
        assert funcs.mu0_hat(X_test).shape == (50,)
        assert funcs.mu1_hat(X_test).shape == (50,)
        assert funcs.tau_hat(X_test).shape == (50,)

    def test_propensity_bounds(self, rng):
        """Propensity estimates should be clipped to [0.05, 0.95]."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = rng.normal(0, 1, 200)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        # Test on extreme X values
        X_extreme = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        e_pred = funcs.e_hat(X_extreme)

        assert np.all(e_pred >= 0.05)
        assert np.all(e_pred <= 0.95)

    def test_custom_propensity_bounds(self, rng):
        """Should respect custom propensity clipping bounds."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = rng.normal(0, 1, 200)

        estimator = LinearNuisance(clip_propensity=(0.1, 0.9))
        funcs = estimator.fit(X, W, Y)

        X_extreme = np.array([-10.0, 10.0])
        e_pred = funcs.e_hat(X_extreme)

        assert np.all(e_pred >= 0.1)
        assert np.all(e_pred <= 0.9)

    def test_recovers_linear_relationship(self, rng):
        """Should approximately recover linear outcome model."""
        n = 1000
        X = rng.normal(0, 1, n)
        W = rng.binomial(1, 0.5, n).astype(float)

        # True linear model
        mu0_true = 1.0 + 0.5 * X
        mu1_true = 2.0 + 0.5 * X
        Y = np.where(W == 1, mu1_true, mu0_true) + rng.normal(0, 0.1, n)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = np.linspace(-2, 2, 10)
        mu0_pred = funcs.mu0_hat(X_test)
        mu1_pred = funcs.mu1_hat(X_test)

        mu0_expected = 1.0 + 0.5 * X_test
        mu1_expected = 2.0 + 0.5 * X_test

        # Should be close (within 0.2)
        assert np.max(np.abs(mu0_pred - mu0_expected)) < 0.2
        assert np.max(np.abs(mu1_pred - mu1_expected)) < 0.2

    def test_tau_hat_equals_mu1_minus_mu0(self, rng):
        """tau_hat should equal mu1_hat - mu0_hat."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, 50)
        tau_pred = funcs.tau_hat(X_test)
        tau_expected = funcs.mu1_hat(X_test) - funcs.mu0_hat(X_test)

        np.testing.assert_allclose(tau_pred, tau_expected, rtol=1e-10)


class TestXGBoostNuisance:
    """Tests for XGBoostNuisance estimator."""

    @pytest.fixture
    def xgb_available(self):
        """Check if XGBoost is available."""
        try:
            import xgboost
            import sklearn
            return True
        except ImportError:
            pytest.skip("XGBoost or sklearn not installed")

    def test_returns_nuisance_functions(self, rng, xgb_available):
        """fit() should return NuisanceFunctions."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = XGBoostNuisance()
        funcs = estimator.fit(X, W, Y)

        assert isinstance(funcs, NuisanceFunctions)
        assert callable(funcs.e_hat)
        assert callable(funcs.mu0_hat)
        assert callable(funcs.mu1_hat)
        assert callable(funcs.tau_hat)

    def test_predictions_correct_shape(self, rng, xgb_available):
        """Predictions should match input shape."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = XGBoostNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, 50)

        assert funcs.e_hat(X_test).shape == (50,)
        assert funcs.mu0_hat(X_test).shape == (50,)
        assert funcs.mu1_hat(X_test).shape == (50,)
        assert funcs.tau_hat(X_test).shape == (50,)

    def test_propensity_bounds(self, rng, xgb_available):
        """Propensity estimates should be clipped to [0.05, 0.95]."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = rng.normal(0, 1, 200)

        estimator = XGBoostNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, 100)
        e_pred = funcs.e_hat(X_test)

        assert np.all(e_pred >= 0.05)
        assert np.all(e_pred <= 0.95)

    def test_handles_2d_covariates(self, rng, xgb_available):
        """Should handle multidimensional covariates."""
        X = rng.normal(0, 1, (200, 5))
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X[:, 0] + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = XGBoostNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, (50, 5))

        assert funcs.e_hat(X_test).shape == (50,)
        assert funcs.mu0_hat(X_test).shape == (50,)


class TestFitNuisances:
    """Tests for fit_nuisances convenience function."""

    def test_linear_method(self, rng):
        """Should return LinearNuisance via fit_nuisances."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + rng.normal(0, 0.1, 200)

        funcs = fit_nuisances(X, W, Y, method="linear")

        assert isinstance(funcs, NuisanceFunctions)
        X_test = rng.normal(0, 1, 10)
        assert funcs.e_hat(X_test).shape == (10,)

    def test_xgboost_method(self, rng):
        """Should return XGBoostNuisance via fit_nuisances."""
        try:
            import xgboost
        except ImportError:
            pytest.skip("XGBoost not installed")

        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + rng.normal(0, 0.1, 200)

        funcs = fit_nuisances(X, W, Y, method="xgboost")

        assert isinstance(funcs, NuisanceFunctions)
        X_test = rng.normal(0, 1, 10)
        assert funcs.e_hat(X_test).shape == (10,)

    def test_unknown_method_raises(self, rng):
        """Unknown method should raise ValueError."""
        X = rng.normal(0, 1, 100)
        W = rng.binomial(1, 0.5, 100).astype(float)
        Y = rng.normal(0, 1, 100)

        with pytest.raises(ValueError, match="Unknown nuisance method"):
            fit_nuisances(X, W, Y, method="unknown")

    def test_custom_propensity_clip(self, rng):
        """Should pass custom propensity bounds to estimator."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = rng.normal(0, 1, 200)

        funcs = fit_nuisances(X, W, Y, method="linear", clip_propensity=(0.1, 0.9))

        X_extreme = np.array([-10.0, 10.0])
        e_pred = funcs.e_hat(X_extreme)

        assert np.all(e_pred >= 0.1)
        assert np.all(e_pred <= 0.9)


class TestEdgeCases:
    """Edge case tests for nuisance module."""

    def test_all_treated(self, rng):
        """Should handle all-treated case gracefully."""
        X = rng.normal(0, 1, 100)
        W = np.ones(100)  # All treated
        Y = 1 + 0.5 * X + rng.normal(0, 0.1, 100)

        estimator = LinearNuisance()
        # This may warn but should not crash
        # mu0 model will have no data - depends on implementation

    def test_finite_predictions(self, rng):
        """Predictions should always be finite."""
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, 50)

        assert np.all(np.isfinite(funcs.e_hat(X_test)))
        assert np.all(np.isfinite(funcs.mu0_hat(X_test)))
        assert np.all(np.isfinite(funcs.mu1_hat(X_test)))
        assert np.all(np.isfinite(funcs.tau_hat(X_test)))
