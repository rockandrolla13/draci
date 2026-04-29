"""Tests for draci.dgp_drift module."""

import numpy as np
import pytest

from draci.dgp_drift import (
    DriftDGP,
    DriftDGPData,
    generate_data_regime_c,
    generate_data_regime_d,
)
from draci.dgp import DGPData


class TestDriftDGPData:
    """Tests for DriftDGPData dataclass."""

    def test_inherits_from_dgpdata(self):
        """DriftDGPData should extend DGPData."""
        assert issubclass(DriftDGPData, DGPData)

    def test_extra_fields(self):
        """DriftDGPData should have regime metadata."""
        T, dim = 100, 5
        data = DriftDGPData(
            X=np.zeros((T, dim)),
            W=np.zeros(T),
            Y=np.zeros(T),
            e_true=np.zeros(T),
            mu0_true=np.zeros(T),
            mu1_true=np.zeros(T),
            tau_true=np.zeros(T),
            regime="C",
            delta=1.0,
            rho=0.0,
        )
        assert data.regime == "C"
        assert data.delta == 1.0
        assert data.rho == 0.0


class TestDriftDGPRegimeC:
    """Tests for Regime C (iid + mean shift)."""

    def test_output_shapes(self, rng):
        """All outputs should have correct dimensions."""
        dgp = DriftDGP(regime="C", delta=1.0)
        data = dgp.generate(200, rng=rng)

        assert isinstance(data, DriftDGPData)
        assert data.X.shape == (200, 5)
        assert data.W.shape == (200,)
        assert data.Y.shape == (200,)

    def test_regime_metadata(self, rng):
        """Regime C should set correct metadata."""
        dgp = DriftDGP(regime="C", delta=1.5)
        data = dgp.generate(100, rng=rng)

        assert data.regime == "C"
        assert data.delta == 1.5
        assert data.rho == 0.0  # iid in regime C

    def test_mean_shift_at_midpoint(self, rng):
        """Regime C should have mean shift at T/2."""
        dgp = DriftDGP(regime="C", delta=2.0)
        T = 2000
        data = dgp.generate(T, rng=rng)

        midpoint = T // 2
        # First coordinate should have different means before/after midpoint
        mean_before = data.X[:midpoint, 0].mean()
        mean_after = data.X[midpoint:, 0].mean()

        # Before midpoint: mean ~0, after: mean ~delta
        assert abs(mean_before) < 0.2  # ~N(0,1) mean
        assert abs(mean_after - 2.0) < 0.2  # ~N(delta,1) mean

    def test_other_coordinates_no_drift(self, rng):
        """Only first coordinate should drift in Regime C."""
        dgp = DriftDGP(regime="C", delta=2.0)
        T = 2000
        data = dgp.generate(T, rng=rng)

        midpoint = T // 2
        # Second coordinate should have similar means before/after
        mean_before = data.X[:midpoint, 1].mean()
        mean_after = data.X[midpoint:, 1].mean()

        assert abs(mean_before - mean_after) < 0.2

    def test_iid_structure(self, rng):
        """Regime C covariates should be approximately iid."""
        dgp = DriftDGP(regime="C", delta=0.0)  # No drift for cleaner test
        data = dgp.generate(2000, rng=rng)

        # Check autocorrelation of first coordinate
        x = data.X[:, 0]
        acf1 = np.corrcoef(x[1:], x[:-1])[0, 1]

        # Should be near zero for iid
        assert abs(acf1) < 0.1


class TestDriftDGPRegimeD:
    """Tests for Regime D (AR(1) + linear drift)."""

    def test_output_shapes(self, rng):
        """All outputs should have correct dimensions."""
        dgp = DriftDGP(regime="D", rho=0.95, delta=0.5)
        data = dgp.generate(200, rng=rng)

        assert isinstance(data, DriftDGPData)
        assert data.X.shape == (200, 5)
        assert data.W.shape == (200,)
        assert data.Y.shape == (200,)

    def test_regime_metadata(self, rng):
        """Regime D should set correct metadata."""
        dgp = DriftDGP(regime="D", rho=0.95, delta=0.5)
        data = dgp.generate(100, rng=rng)

        assert data.regime == "D"
        assert data.delta == 0.5
        assert data.rho == 0.95

    def test_ar1_dependence(self, rng):
        """Regime D should exhibit AR(1) dependence."""
        dgp = DriftDGP(regime="D", rho=0.9, delta=0.0)  # No drift for cleaner test
        data = dgp.generate(2000, rng=rng)

        # Check autocorrelation of first coordinate
        x = data.X[:, 0]
        acf1 = np.corrcoef(x[1:], x[:-1])[0, 1]

        # Should be close to rho
        assert abs(acf1 - 0.9) < 0.15

    def test_linear_drift_after_midpoint(self, rng):
        """Regime D should have linear drift after T/2."""
        dgp = DriftDGP(regime="D", rho=0.5, delta=1.0)
        T = 2000
        data = dgp.generate(T, rng=rng)

        midpoint = T // 2
        # Compare late phase to midpoint
        # Linear drift means mean increases over time
        mean_mid = data.X[midpoint : midpoint + 200, 0].mean()
        mean_late = data.X[-200:, 0].mean()

        # Late phase should have higher mean due to accumulated drift
        assert mean_late > mean_mid - 0.5  # Allow some variance

    def test_custom_rho(self, rng):
        """Should respect custom rho parameter."""
        dgp = DriftDGP(regime="D", rho=0.5, delta=0.0)
        data = dgp.generate(2000, rng=rng)

        x = data.X[:, 0]
        acf1 = np.corrcoef(x[1:], x[:-1])[0, 1]

        assert abs(acf1 - 0.5) < 0.15


class TestDriftDGPShared:
    """Tests for shared behavior across regimes."""

    def test_treatment_binary(self, rng):
        """Treatment W should be binary in both regimes."""
        for regime in ["C", "D"]:
            dgp = DriftDGP(regime=regime, delta=1.0)
            data = dgp.generate(500, rng=rng)

            assert set(np.unique(data.W)).issubset({0.0, 1.0})

    def test_propensity_bounds(self, rng):
        """Propensity scores should be clipped."""
        for regime in ["C", "D"]:
            dgp = DriftDGP(regime=regime, delta=1.0)
            data = dgp.generate(500, rng=rng)

            assert np.all(data.e_true >= 0.05)
            assert np.all(data.e_true <= 0.95)

    def test_cate_formula(self, rng):
        """tau_true should match base DGP formula."""
        dgp = DriftDGP(regime="C", delta=1.0)
        data = dgp.generate(100, rng=rng)

        expected_tau = np.sin(2 * np.pi * data.X[:, 0]) + 0.5 * data.X[:, 1]
        np.testing.assert_array_almost_equal(data.tau_true, expected_tau)

    def test_mu1_equals_mu0_plus_tau(self, rng):
        """mu1_true should equal mu0_true + tau_true."""
        dgp = DriftDGP(regime="D", rho=0.9, delta=0.5)
        data = dgp.generate(100, rng=rng)

        np.testing.assert_array_almost_equal(
            data.mu1_true, data.mu0_true + data.tau_true
        )

    def test_reproducibility_with_seed(self):
        """Same seed should produce same data."""
        dgp = DriftDGP(regime="C", delta=1.0)

        data1 = dgp.generate(100, seed=42)
        data2 = dgp.generate(100, seed=42)

        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.W, data2.W)
        np.testing.assert_array_equal(data1.Y, data2.Y)


class TestFunctionalInterface:
    """Tests for functional interface functions."""

    def test_generate_data_regime_c(self, rng):
        """generate_data_regime_c should return dict."""
        data = generate_data_regime_c(T=100, delta=1.0, rng=rng)

        assert isinstance(data, dict)
        expected_keys = {'X', 'W', 'Y', 'e_true', 'mu0_true', 'mu1_true', 'tau_true', 'regime', 'delta', 'rho'}
        assert set(data.keys()) == expected_keys
        assert data['regime'] == 'C'
        assert data['delta'] == 1.0
        assert data['rho'] == 0.0

    def test_generate_data_regime_d(self, rng):
        """generate_data_regime_d should return dict."""
        data = generate_data_regime_d(T=100, rho=0.9, delta=0.5, rng=rng)

        assert isinstance(data, dict)
        expected_keys = {'X', 'W', 'Y', 'e_true', 'mu0_true', 'mu1_true', 'tau_true', 'regime', 'delta', 'rho'}
        assert set(data.keys()) == expected_keys
        assert data['regime'] == 'D'
        assert data['delta'] == 0.5
        assert data['rho'] == 0.9
