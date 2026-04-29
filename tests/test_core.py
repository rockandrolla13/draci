"""Tests for draci.core module."""

import numpy as np
import pytest

from draci.core import (
    ConformalResult,
    IntervalResult,
    aci,
    dr_aci,
    vs_dr_aci,
    split_conformal,
    nexcp,
    eci,
    block_cp,
    hac,
    DRACI,
)
from draci.scores import dr_score, vs_dr_score


class TestConformalResult:
    """Tests for ConformalResult dataclass."""

    def test_dataclass_creation(self):
        """ConformalResult should accept required fields."""
        result = ConformalResult(
            coverage=0.9,
            avg_width=2.0,
            coverages_t=np.array([1, 1, 0, 1]),
        )
        assert result.coverage == 0.9
        assert result.avg_width == 2.0
        assert len(result.coverages_t) == 4

    def test_optional_fields(self):
        """ConformalResult optional fields should default to None."""
        result = ConformalResult(
            coverage=0.9,
            avg_width=2.0,
            coverages_t=np.array([1, 1, 0, 1]),
        )
        assert result.widths_t is None
        assert result.alpha_trajectory is None


class TestIntervalResult:
    """Tests for IntervalResult dataclass."""

    def test_dataclass_creation(self):
        """IntervalResult should accept required fields."""
        T = 10
        result = IntervalResult(
            point=np.ones(T),
            lower=np.zeros(T),
            upper=np.ones(T) * 2,
            width=np.ones(T) * 2,
        )
        assert len(result.point) == T
        assert len(result.lower) == T
        assert len(result.upper) == T

    def test_interval_consistency(self):
        """IntervalResult width should equal upper - lower."""
        T = 10
        lower = np.zeros(T)
        upper = np.ones(T) * 2
        result = IntervalResult(
            point=np.ones(T),
            lower=lower,
            upper=upper,
            width=upper - lower,
        )
        np.testing.assert_array_equal(result.width, result.upper - result.lower)


class TestACI:
    """Tests for aci function."""

    def test_returns_conformal_result(self, iid_scores):
        """aci should return ConformalResult."""
        result = aci(
            iid_scores["scores"],
            iid_scores["true_residuals"],
            alpha=0.1,
            gamma=0.01,
            n_warmup=20,
        )
        assert isinstance(result, ConformalResult)
        assert 0 <= result.coverage <= 1
        assert result.avg_width > 0

    def test_coverages_t_length(self, iid_scores):
        """Per-timestep coverages should match T - n_warmup."""
        n_warmup = 50
        result = aci(
            iid_scores["scores"],
            iid_scores["true_residuals"],
            alpha=0.1,
            gamma=0.01,
            n_warmup=n_warmup,
        )
        expected_len = iid_scores["T"] - n_warmup
        assert len(result.coverages_t) == expected_len

    def test_converges_to_nominal_iid(self, rng):
        """ACI should converge toward nominal coverage on IID data."""
        T = 1000
        alpha = 0.1

        # IID scores from same distribution
        scores = rng.exponential(1, T)
        true_residuals = rng.exponential(1, T)

        result = aci(scores, true_residuals, alpha=alpha, gamma=0.02, n_warmup=50)

        # Should be reasonably close to nominal
        assert abs(result.coverage - (1 - alpha)) < 0.15

    def test_alpha_trajectory_bounded(self, iid_scores):
        """Alpha trajectory should stay bounded in [0.01, 0.99]."""
        result = aci(
            iid_scores["scores"],
            iid_scores["true_residuals"],
            alpha=0.1,
            gamma=0.05,  # Larger gamma for faster adaptation
            n_warmup=20,
        )
        assert np.all(result.alpha_trajectory >= 0.01)
        assert np.all(result.alpha_trajectory <= 0.99)


class TestDRACI:
    """Tests for dr_aci function."""

    def test_same_as_aci(self, iid_scores):
        """dr_aci should produce same results as aci (it's a wrapper)."""
        result_aci = aci(
            iid_scores["scores"],
            iid_scores["true_residuals"],
            alpha=0.1,
            gamma=0.01,
            n_warmup=20,
        )
        result_draci = dr_aci(
            iid_scores["scores"],
            iid_scores["true_residuals"],
            alpha=0.1,
            gamma=0.01,
            n_warmup=20,
        )
        assert result_aci.coverage == result_draci.coverage
        assert result_aci.avg_width == result_draci.avg_width


class TestVSDRACI:
    """Tests for vs_dr_aci function."""

    def test_returns_conformal_result(self, rng):
        """vs_dr_aci should return ConformalResult."""
        T = 300
        vs_scores = rng.exponential(1, T)
        true_residuals_std = rng.exponential(1, T)

        result = vs_dr_aci(
            vs_scores,
            true_residuals_std,
            alpha=0.1,
            gamma=0.01,
            n_warmup=50,
        )
        assert isinstance(result, ConformalResult)
        assert 0 <= result.coverage <= 1

    def test_widths_in_standardized_units(self, rng):
        """VS-DR-ACI widths should be in standardized units."""
        T = 300
        vs_scores = rng.exponential(1, T)  # ~mean 1
        true_residuals_std = rng.exponential(1, T)

        result = vs_dr_aci(
            vs_scores,
            true_residuals_std,
            alpha=0.1,
            gamma=0.01,
            n_warmup=50,
        )
        # Average width should be in range of exponential quantiles
        assert result.avg_width > 0
        assert result.avg_width < 10  # Reasonable bound


class TestDRACIClass:
    """Tests for DRACI class."""

    def test_init_defaults(self):
        """DRACI should initialize with default parameters."""
        model = DRACI()
        assert model.alpha == 0.10
        assert model.gamma == 0.005
        assert model.n_warmup == 50
        assert model.score_type == "dr"

    def test_init_custom_params(self):
        """DRACI should accept custom parameters."""
        model = DRACI(alpha=0.05, gamma=0.01, n_warmup=100, score_type="vs_dr")
        assert model.alpha == 0.05
        assert model.gamma == 0.01
        assert model.n_warmup == 100
        assert model.score_type == "vs_dr"

    def test_fit_predict_returns_interval_result(self, sample_data, nuisance_estimates):
        """DRACI.fit_predict should return IntervalResult."""
        model = DRACI(alpha=0.1, n_warmup=20)
        result = model.fit_predict(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        assert isinstance(result, IntervalResult)
        assert len(result.point) == sample_data["T"]
        assert len(result.lower) == sample_data["T"]
        assert len(result.upper) == sample_data["T"]

    def test_fit_predict_with_tau_true(self, sample_data, nuisance_estimates):
        """DRACI should compute coverage when tau_true provided."""
        model = DRACI(alpha=0.1, n_warmup=20)
        result = model.fit_predict(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
            tau_true=sample_data["tau_true"],
        )
        assert result.coverage_t is not None
        # Coverage should be computed for T - n_warmup points
        assert len(result.coverage_t) == sample_data["T"] - 20

    def test_score_type_dr(self, sample_data, nuisance_estimates):
        """DRACI with score_type='dr' should use DR scores."""
        model = DRACI(alpha=0.1, score_type="dr", n_warmup=20)
        result = model.fit_predict(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        assert isinstance(result, IntervalResult)

    def test_score_type_vs_dr(self, sample_data, nuisance_estimates):
        """DRACI with score_type='vs_dr' should use VS-DR scores."""
        model = DRACI(alpha=0.1, score_type="vs_dr", n_warmup=20)
        result = model.fit_predict(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        assert isinstance(result, IntervalResult)

    def test_score_type_naive(self, sample_data, nuisance_estimates):
        """DRACI with score_type='naive' should use naive scores."""
        model = DRACI(alpha=0.1, score_type="naive", n_warmup=20)
        result = model.fit_predict(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        assert isinstance(result, IntervalResult)

    def test_fractional_warmup(self, sample_data, nuisance_estimates):
        """DRACI should interpret fractional n_warmup as proportion."""
        model = DRACI(alpha=0.1, n_warmup=0.2)  # 20% warmup
        result = model.fit_predict(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
            tau_true=sample_data["tau_true"],
        )
        # With T=200 and 20% warmup, n_warmup = 40
        expected_warmup = int(0.2 * sample_data["T"])
        assert len(result.coverage_t) == sample_data["T"] - expected_warmup


class TestEdgeCases:
    """Edge case tests for core module."""

    def test_small_sample_aci(self, rng):
        """ACI should handle small samples without crashing."""
        T = 25
        scores = rng.exponential(1, T)
        true_residuals = rng.exponential(1, T)

        # Should not raise
        result = aci(scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=5)
        assert isinstance(result, ConformalResult)

    def test_interval_bounds_order(self, sample_data, nuisance_estimates):
        """Lower bounds should be less than upper bounds."""
        model = DRACI(alpha=0.1, n_warmup=20)
        result = model.fit_predict(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        # After warmup, lower < upper
        valid = result.lower[20:] != 0
        assert np.all(result.lower[20:][valid] < result.upper[20:][valid])

    def test_point_estimate_in_interval(self, sample_data, nuisance_estimates):
        """Point estimate should be centered in interval."""
        model = DRACI(alpha=0.1, n_warmup=20)
        result = model.fit_predict(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        # After warmup
        valid_idx = slice(20, None)
        centers = (result.lower[valid_idx] + result.upper[valid_idx]) / 2
        np.testing.assert_allclose(
            result.point[valid_idx], centers, rtol=1e-10
        )


class TestSplitConformal:
    """Tests for split_conformal function."""

    def test_returns_conformal_result(self, rng):
        """split_conformal should return ConformalResult."""
        n_cal, n_test = 200, 100
        scores_cal = rng.exponential(1, n_cal)
        scores_test = rng.exponential(1, n_test)
        true_residuals_test = rng.exponential(1, n_test)

        result = split_conformal(scores_cal, scores_test, true_residuals_test, alpha=0.1)

        assert isinstance(result, ConformalResult)
        assert 0 <= result.coverage <= 1
        assert result.avg_width > 0

    def test_coverage_t_length(self, rng):
        """Per-timestep coverages should match test set length."""
        n_cal, n_test = 200, 100
        scores_cal = rng.exponential(1, n_cal)
        scores_test = rng.exponential(1, n_test)
        true_residuals_test = rng.exponential(1, n_test)

        result = split_conformal(scores_cal, scores_test, true_residuals_test)

        assert len(result.coverages_t) == n_test

    def test_fixed_quantile(self, rng):
        """Split conformal should use a fixed quantile from calibration."""
        n_cal, n_test = 500, 100
        scores_cal = rng.exponential(1, n_cal)
        scores_test = rng.exponential(1, n_test)
        true_residuals_test = rng.exponential(1, n_test)

        result = split_conformal(scores_cal, scores_test, true_residuals_test, alpha=0.1)

        # Width should be constant (fixed quantile)
        widths_unique = np.unique(result.widths_t)
        assert len(widths_unique) == 1

    def test_coverage_near_nominal_iid(self, rng):
        """Coverage should be near nominal on IID data."""
        n_cal, n_test = 500, 1000
        scores_cal = rng.exponential(1, n_cal)
        # True residuals from same distribution
        true_residuals_test = rng.exponential(1, n_test)

        result = split_conformal(scores_cal, scores_cal[:n_test], true_residuals_test, alpha=0.1)

        # Should be close to 90% on well-calibrated data
        assert abs(result.coverage - 0.9) < 0.1


class TestNexCP:
    """Tests for nexcp function."""

    def test_returns_conformal_result(self, iid_scores):
        """nexcp should return ConformalResult."""
        result = nexcp(
            iid_scores["scores"],
            iid_scores["true_residuals"],
            alpha=0.1,
            lam=0.05,
            n_warmup=50,
        )

        assert isinstance(result, ConformalResult)
        assert 0 <= result.coverage <= 1
        assert result.avg_width > 0

    def test_coverages_t_length(self, iid_scores):
        """Per-timestep coverages should match T - n_warmup."""
        n_warmup = 50
        result = nexcp(
            iid_scores["scores"],
            iid_scores["true_residuals"],
            alpha=0.1,
            lam=0.05,
            n_warmup=n_warmup,
        )

        expected_len = iid_scores["T"] - n_warmup
        assert len(result.coverages_t) == expected_len

    def test_larger_lambda_more_recent_weights(self, rng):
        """Larger lambda should give more weight to recent observations."""
        T = 300
        # Scores that increase over time
        scores = np.linspace(0.5, 2.0, T)
        true_residuals = rng.exponential(1, T)

        # Small lambda: more uniform weights -> larger quantile
        result_small = nexcp(scores, true_residuals, lam=0.01, n_warmup=50)

        # Large lambda: recent weights dominate -> smaller quantile from recent scores
        result_large = nexcp(scores, true_residuals, lam=0.2, n_warmup=50)

        # With increasing scores, larger lambda should track recent (higher) scores better
        # This affects coverage and width differently - key is they differ
        assert result_small.avg_width != result_large.avg_width

    def test_widths_positive(self, iid_scores):
        """All interval widths should be positive."""
        result = nexcp(iid_scores["scores"], iid_scores["true_residuals"])

        assert np.all(result.widths_t > 0)


class TestECI:
    """Tests for eci function."""

    def test_returns_conformal_result(self, iid_scores):
        """eci should return ConformalResult."""
        result = eci(
            iid_scores["scores"],
            iid_scores["true_residuals"],
            alpha=0.1,
            gamma=0.005,
            n_warmup=50,
        )

        assert isinstance(result, ConformalResult)
        assert 0 <= result.coverage <= 1
        assert result.avg_width > 0

    def test_coverages_t_length(self, iid_scores):
        """Per-timestep coverages should match T - n_warmup."""
        n_warmup = 50
        result = eci(
            iid_scores["scores"],
            iid_scores["true_residuals"],
            alpha=0.1,
            gamma=0.005,
            n_warmup=n_warmup,
        )

        expected_len = iid_scores["T"] - n_warmup
        assert len(result.coverages_t) == expected_len

    def test_alpha_trajectory_bounded(self, iid_scores):
        """Alpha trajectory should stay bounded in [0.01, 0.99]."""
        result = eci(
            iid_scores["scores"],
            iid_scores["true_residuals"],
            alpha=0.1,
            gamma=0.05,  # Larger gamma for faster adaptation
            n_warmup=20,
        )

        assert np.all(result.alpha_trajectory >= 0.01)
        assert np.all(result.alpha_trajectory <= 0.99)

    def test_temperature_affects_smoothness(self, rng):
        """Higher temperature should produce smoother updates."""
        T = 500
        scores = rng.exponential(1, T)
        true_residuals = rng.exponential(1, T)

        # Low temperature: closer to hard indicator (like standard ACI)
        result_low = eci(scores, true_residuals, temperature=0.1, n_warmup=50)

        # High temperature: smoother sigmoid
        result_high = eci(scores, true_residuals, temperature=5.0, n_warmup=50)

        # Both should produce valid results
        assert isinstance(result_low, ConformalResult)
        assert isinstance(result_high, ConformalResult)

    def test_converges_to_nominal_iid(self, rng):
        """ECI should converge toward nominal coverage on IID data."""
        T = 1000
        alpha = 0.1
        scores = rng.exponential(1, T)
        true_residuals = rng.exponential(1, T)

        result = eci(scores, true_residuals, alpha=alpha, gamma=0.02, n_warmup=50)

        # Should be reasonably close to nominal
        assert abs(result.coverage - (1 - alpha)) < 0.15


class TestBlockCP:
    """Tests for block_cp function."""

    def test_returns_conformal_result(self, rng):
        """block_cp should return ConformalResult."""
        T = 400
        scores = rng.exponential(1, T)
        true_residuals = rng.exponential(1, T)

        result = block_cp(scores, true_residuals, alpha=0.1, block_size=10, rng=rng)

        assert isinstance(result, ConformalResult)
        assert 0 <= result.coverage <= 1
        assert result.avg_width > 0

    def test_coverages_t_length(self, rng):
        """Per-timestep coverages should match test set (second half)."""
        T = 400
        scores = rng.exponential(1, T)
        true_residuals = rng.exponential(1, T)

        result = block_cp(scores, true_residuals, block_size=10, rng=rng)

        # Test set is second half
        n_test = T // 2
        assert len(result.coverages_t) == n_test

    def test_fixed_width(self, rng):
        """Block CP should produce fixed width (same for all test points)."""
        T = 400
        scores = rng.exponential(1, T)
        true_residuals = rng.exponential(1, T)

        result = block_cp(scores, true_residuals, block_size=10, rng=rng)

        # Width should be constant
        widths_unique = np.unique(result.widths_t)
        assert len(widths_unique) == 1

    def test_reproducibility(self):
        """Same rng state should produce same results."""
        T = 400
        scores = np.random.default_rng(42).exponential(1, T)
        true_residuals = np.random.default_rng(43).exponential(1, T)

        rng1 = np.random.default_rng(123)
        result1 = block_cp(scores, true_residuals, block_size=10, rng=rng1)

        rng2 = np.random.default_rng(123)
        result2 = block_cp(scores, true_residuals, block_size=10, rng=rng2)

        assert result1.coverage == result2.coverage
        assert result1.avg_width == result2.avg_width

    def test_fallback_with_few_blocks(self, rng):
        """Should fall back to simple split when too few blocks."""
        T = 30  # Small sample
        scores = rng.exponential(1, T)
        true_residuals = rng.exponential(1, T)

        # Should not raise, falls back to standard split
        result = block_cp(scores, true_residuals, block_size=20, rng=rng)

        assert isinstance(result, ConformalResult)


class TestHAC:
    """Tests for hac function."""

    def test_returns_conformal_result(self, rng):
        """hac should return ConformalResult."""
        T = 500
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)

        result = hac(psi_dr, tau_hat, true_residuals, alpha=0.1, bandwidth=10)

        assert isinstance(result, ConformalResult)
        assert 0 <= result.coverage <= 1
        assert result.avg_width > 0

    def test_coverages_t_length(self, rng):
        """Per-timestep coverages should match T."""
        T = 500
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)

        result = hac(psi_dr, tau_hat, true_residuals)

        assert len(result.coverages_t) == T

    def test_fixed_width(self, rng):
        """HAC should produce fixed width (constant SE across all points)."""
        T = 500
        psi_dr = rng.normal(0, 1, T)
        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)

        result = hac(psi_dr, tau_hat, true_residuals)

        # Width should be constant
        widths_unique = np.unique(result.widths_t)
        assert len(widths_unique) == 1

    def test_bandwidth_affects_variance(self, rng):
        """Different bandwidths should produce different standard errors."""
        T = 500
        # Create dependent residuals (AR(1))
        rho = 0.7
        psi_dr = np.zeros(T)
        psi_dr[0] = rng.normal()
        for t in range(1, T):
            psi_dr[t] = rho * psi_dr[t-1] + np.sqrt(1 - rho**2) * rng.normal()

        tau_hat = np.zeros(T)
        true_residuals = np.abs(psi_dr - tau_hat)

        result_small_bw = hac(psi_dr, tau_hat, true_residuals, bandwidth=2)
        result_large_bw = hac(psi_dr, tau_hat, true_residuals, bandwidth=20)

        # With dependence, larger bandwidth should account for more autocorrelation
        # Widths should differ
        assert result_small_bw.avg_width != result_large_bw.avg_width

    def test_coverage_on_centered_data(self, rng):
        """HAC should achieve reasonable coverage on well-centered data."""
        T = 1000
        alpha = 0.1

        # psi^DR should be unbiased: E[psi^DR] = tau_true
        # So residuals = psi^DR - tau_hat should be mean-zero if tau_hat = tau_true
        psi_dr = rng.normal(0, 1, T)  # Centered at 0
        tau_hat = np.zeros(T)  # tau_hat = 0
        true_residuals = np.abs(rng.normal(0, 1, T))  # Independent true deviations

        result = hac(psi_dr, tau_hat, true_residuals, alpha=alpha)

        # Coverage should be reasonable (not too far from nominal)
        assert result.coverage > 0.5
