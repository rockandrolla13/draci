"""Tests for draci.mixing module."""

import numpy as np
import pytest

from draci.mixing import (
    MixingDiagnostics,
    compute_acf,
    estimate_beta_mixing,
    fit_exponential_decay,
    compute_optimal_mixing_gap,
    mixing_diagnostics,
    mixing_diagnostics_panel,
)


class TestComputeACF:
    """Tests for compute_acf function."""

    def test_output_shape(self, rng):
        """ACF should return max_lag + 1 values."""
        T = 200
        series = rng.normal(0, 1, T)
        max_lag = 20

        acf = compute_acf(series, max_lag)

        assert acf.shape == (max_lag + 1,)

    def test_lag_zero_is_one(self, rng):
        """ACF at lag 0 should be 1."""
        series = rng.normal(0, 1, 200)

        acf = compute_acf(series, max_lag=10)

        assert np.isclose(acf[0], 1.0)

    def test_iid_acf_near_zero(self, rng):
        """ACF of iid series should be near zero for lag > 0."""
        T = 2000
        series = rng.normal(0, 1, T)

        acf = compute_acf(series, max_lag=10)

        # For iid, ACF should be small (within ~2/sqrt(T))
        threshold = 3 / np.sqrt(T)
        assert np.all(np.abs(acf[1:]) < threshold)

    def test_ar1_acf_decays(self, rng):
        """ACF of AR(1) should decay exponentially."""
        T = 2000
        rho = 0.8
        series = np.zeros(T)
        series[0] = rng.normal()
        for t in range(1, T):
            series[t] = rho * series[t-1] + np.sqrt(1 - rho**2) * rng.normal()

        acf = compute_acf(series, max_lag=20)

        # ACF at lag 1 should be close to rho
        assert abs(acf[1] - rho) < 0.1

        # ACF should decay
        assert acf[1] > acf[5] > acf[10]

    def test_short_series_returns_nan(self, rng):
        """Series shorter than max_lag + 10 should return NaN."""
        series = rng.normal(0, 1, 15)

        acf = compute_acf(series, max_lag=20)

        assert np.all(np.isnan(acf))

    def test_constant_series_returns_nan(self, rng):
        """Constant series (zero variance) should return NaN."""
        series = np.ones(100)

        acf = compute_acf(series, max_lag=10)

        # Lag 0 might be NaN or 1 depending on implementation
        # But other lags should be NaN due to zero variance
        assert np.all(np.isnan(acf[1:]))


class TestEstimateBetaMixing:
    """Tests for estimate_beta_mixing function."""

    def test_output_dict_keys(self, rng):
        """Should return dict with specified lags as keys."""
        series = rng.normal(0, 1, 500)
        lags = [1, 5, 10, 20]

        beta = estimate_beta_mixing(series, lags=lags)

        assert set(beta.keys()) == set(lags)

    def test_default_lags(self, rng):
        """Should use default lags if not specified."""
        series = rng.normal(0, 1, 500)

        beta = estimate_beta_mixing(series)

        default_lags = [1, 5, 10, 20, 50, 100]
        assert set(beta.keys()) == set(default_lags)

    def test_values_nonnegative(self, rng):
        """Beta values should be non-negative."""
        series = rng.normal(0, 1, 500)

        beta = estimate_beta_mixing(series, lags=[1, 5, 10])

        for val in beta.values():
            if not np.isnan(val):
                assert val >= 0

    def test_values_bounded_by_one(self, rng):
        """Beta values (TV distance) should be at most 1."""
        series = rng.normal(0, 1, 500)

        beta = estimate_beta_mixing(series, lags=[1, 5, 10])

        for val in beta.values():
            if not np.isnan(val):
                assert val <= 1.0

    def test_iid_series_small_beta(self, rng):
        """IID series should have small beta values."""
        T = 2000
        series = rng.normal(0, 1, T)

        beta = estimate_beta_mixing(series, lags=[1, 10, 50])

        # For truly independent data, beta should be small
        for val in beta.values():
            if not np.isnan(val):
                assert val < 0.3  # Allow for histogram estimation noise

    def test_dependent_series_larger_beta_at_short_lag(self, rng):
        """Dependent series should have larger beta at short lags."""
        T = 2000
        rho = 0.9
        series = np.zeros(T)
        series[0] = rng.normal()
        for t in range(1, T):
            series[t] = rho * series[t-1] + np.sqrt(1 - rho**2) * rng.normal()

        beta = estimate_beta_mixing(series, lags=[1, 50])

        # For AR(1), beta at lag 1 should be larger than at lag 50
        if not np.isnan(beta[1]) and not np.isnan(beta[50]):
            assert beta[1] >= beta[50]

    def test_large_lag_returns_nan(self, rng):
        """Lag larger than T-10 should return NaN."""
        series = rng.normal(0, 1, 50)

        beta = estimate_beta_mixing(series, lags=[1, 5, 100])

        assert np.isnan(beta[100])


class TestFitExponentialDecay:
    """Tests for fit_exponential_decay function."""

    def test_output_keys(self, rng):
        """Should return dict with a, b, r2 keys."""
        beta_values = {1: 0.4, 5: 0.2, 10: 0.1, 20: 0.05}

        fit = fit_exponential_decay(beta_values)

        assert "a" in fit
        assert "b" in fit
        assert "r2" in fit

    def test_perfect_exponential(self):
        """Should fit perfect exponential exactly."""
        # beta(tau) = 0.5 * exp(-0.1 * tau)
        a_true, b_true = 0.5, 0.1
        beta_values = {tau: a_true * np.exp(-b_true * tau) for tau in [1, 5, 10, 20, 50]}

        fit = fit_exponential_decay(beta_values)

        assert abs(fit["a"] - a_true) < 0.1
        assert abs(fit["b"] - b_true) < 0.05
        assert fit["r2"] > 0.99

    def test_too_few_values_returns_nan(self):
        """Should return NaN if fewer than 3 valid values."""
        beta_values = {1: 0.4, 5: 0.2}

        fit = fit_exponential_decay(beta_values)

        assert np.isnan(fit["a"])
        assert np.isnan(fit["b"])
        assert np.isnan(fit["r2"])

    def test_handles_nan_values(self):
        """Should filter out NaN values."""
        beta_values = {1: 0.4, 5: 0.2, 10: np.nan, 20: 0.1, 50: 0.05}

        fit = fit_exponential_decay(beta_values)

        # Should still fit using the 4 valid values
        assert np.isfinite(fit["a"])
        assert np.isfinite(fit["b"])


class TestComputeOptimalMixingGap:
    """Tests for compute_optimal_mixing_gap function."""

    def test_output_types(self, rng):
        """Should return (int, float) tuple."""
        beta_values = {1: 0.4, 5: 0.2, 10: 0.1}
        T = 1000

        opt_tau, opt_gap = compute_optimal_mixing_gap(beta_values, T)

        assert isinstance(opt_tau, (int, np.integer))
        assert isinstance(opt_gap, (float, np.floating))

    def test_gap_formula(self):
        """Gap should equal tau/T + 2*beta(tau)."""
        beta_values = {10: 0.05}
        T = 1000

        opt_tau, opt_gap = compute_optimal_mixing_gap(beta_values, T)

        expected_gap = 10 / 1000 + 2 * 0.05
        assert np.isclose(opt_gap, expected_gap)

    def test_selects_minimum(self):
        """Should select tau that minimizes the gap."""
        beta_values = {
            1: 0.4,   # gap = 1/100 + 0.8 = 0.81
            10: 0.05, # gap = 10/100 + 0.1 = 0.20
            50: 0.01, # gap = 50/100 + 0.02 = 0.52
        }
        T = 100

        opt_tau, opt_gap = compute_optimal_mixing_gap(beta_values, T)

        assert opt_tau == 10
        assert np.isclose(opt_gap, 0.20)

    def test_handles_all_nan(self):
        """Should return (0, NaN) if all beta values are NaN."""
        beta_values = {1: np.nan, 5: np.nan}
        T = 100

        opt_tau, opt_gap = compute_optimal_mixing_gap(beta_values, T)

        assert opt_tau == 0
        assert np.isnan(opt_gap)


class TestMixingDiagnostics:
    """Tests for mixing_diagnostics function."""

    def test_returns_mixing_diagnostics(self, rng):
        """Should return MixingDiagnostics instance."""
        series = rng.normal(0, 1, 500)

        result = mixing_diagnostics(series)

        assert isinstance(result, MixingDiagnostics)

    def test_all_fields_populated(self, rng):
        """All fields should be populated."""
        series = rng.normal(0, 1, 500)

        result = mixing_diagnostics(series, lags=[1, 5, 10])

        assert result.beta_values is not None
        assert result.exp_fit is not None
        assert result.acf_lag1_median is not None or np.isnan(result.acf_lag1_median)
        assert result.optimal_tau is not None
        assert result.optimal_gap is not None or np.isnan(result.optimal_gap)

    def test_beta_values_structure(self, rng):
        """Beta values should have correct structure."""
        series = rng.normal(0, 1, 500)
        lags = [1, 5, 10]

        result = mixing_diagnostics(series, lags=lags)

        assert set(result.beta_values.keys()) == set(lags)
        for tau, stats in result.beta_values.items():
            assert "median" in stats
            assert "mean" in stats
            assert "std" in stats

    def test_ar1_series(self, rng):
        """Should detect dependence in AR(1) series."""
        T = 1000
        rho = 0.8
        series = np.zeros(T)
        series[0] = rng.normal()
        for t in range(1, T):
            series[t] = rho * series[t-1] + np.sqrt(1 - rho**2) * rng.normal()

        result = mixing_diagnostics(series)

        # ACF lag-1 should be positive
        assert result.acf_lag1_median > 0.3


class TestMixingDiagnosticsPanel:
    """Tests for mixing_diagnostics_panel function."""

    def test_returns_mixing_diagnostics(self, rng):
        """Should return MixingDiagnostics instance."""
        panel = [rng.normal(0, 1, 200) for _ in range(5)]

        result = mixing_diagnostics_panel(panel)

        assert isinstance(result, MixingDiagnostics)

    def test_aggregates_across_series(self, rng):
        """Should aggregate estimates across multiple series."""
        # Create panel with different characteristics
        panel = [
            rng.normal(0, 1, 200) for _ in range(10)
        ]

        result = mixing_diagnostics_panel(panel, lags=[1, 5, 10])

        # Should have estimates
        for tau in [1, 5, 10]:
            assert tau in result.beta_values
            # Stats should reflect aggregation
            assert "median" in result.beta_values[tau]
            assert "std" in result.beta_values[tau]

    def test_filters_short_series(self, rng):
        """Should filter out series shorter than min_obs."""
        panel = [
            rng.normal(0, 1, 200),  # Include
            rng.normal(0, 1, 50),   # Exclude (< 100)
            rng.normal(0, 1, 200),  # Include
        ]

        result = mixing_diagnostics_panel(panel, min_obs=100)

        # Should still produce results from the 2 long series
        assert isinstance(result, MixingDiagnostics)

    def test_handles_empty_panel(self, rng):
        """Should handle empty or all-filtered panel."""
        panel = [rng.normal(0, 1, 20) for _ in range(3)]  # All too short

        result = mixing_diagnostics_panel(panel, min_obs=100)

        # Should return with NaN values
        assert np.isnan(result.acf_lag1_median)

    def test_std_reflects_variability(self, rng):
        """Standard deviation should reflect cross-sectional variability."""
        # Create panel with varying dependence
        panel = []
        for rho in [0.2, 0.4, 0.6, 0.8]:
            T = 300
            series = np.zeros(T)
            series[0] = rng.normal()
            for t in range(1, T):
                series[t] = rho * series[t-1] + np.sqrt(1 - rho**2) * rng.normal()
            panel.append(series)

        result = mixing_diagnostics_panel(panel, lags=[1])

        # With varying rho, std should be positive
        assert result.beta_values[1]["std"] > 0


class TestMixingDiagnosticsDataclass:
    """Tests for MixingDiagnostics dataclass."""

    def test_fields_accessible(self):
        """Should have all expected fields."""
        diag = MixingDiagnostics(
            beta_values={1: {"median": 0.3, "mean": 0.3, "std": 0.1, "q25": 0.2, "q75": 0.4}},
            exp_fit={"a": 0.5, "b": 0.1, "r2": 0.9},
            acf_lag1_median=0.5,
            optimal_tau=10,
            optimal_gap=0.2,
        )

        assert diag.beta_values[1]["median"] == 0.3
        assert diag.exp_fit["b"] == 0.1
        assert diag.acf_lag1_median == 0.5
        assert diag.optimal_tau == 10
        assert diag.optimal_gap == 0.2
