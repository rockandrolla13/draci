"""
Integration tests: end-to-end DR-ACI pipeline.

Tests the full workflow from data generation through nuisance estimation,
cross-fitting, and conformal inference. Based on simulation patterns from
the paper repo (sim_coverage.py, sim_ci_comparison.py, sim_drift.py).
"""

import numpy as np
import pytest

from draci import (
    # Core
    DRACI,
    ConformalResult,
    IntervalResult,
    aci,
    dr_aci,
    vs_dr_aci,
    # Scores
    dr_score,
    vs_dr_score,
    dr_pseudo_outcome,
    # Nuisance
    fit_nuisances,
    XGBoostNuisance,
    LinearNuisance,
    NuisanceResult,
    NuisanceFunctions,
    # Cross-fitting
    TemporalCrossFitter,
    temporal_block_crossfit,
    CrossFitResult,
    # DGP
    generate_data,
    generate_data_regime_c,
    generate_data_regime_d,
    # Baselines
    dml_wald_ci,
    block_bootstrap_ci,
    compute_ci_metrics,
    # Mixing
    mixing_diagnostics,
    compute_acf,
)


class TestEndToEndPipeline:
    """End-to-end tests for the full DR-ACI pipeline."""

    @pytest.fixture
    def ar1_data(self, rng):
        """Generate AR(1) data for testing."""
        return generate_data(T=500, rho=0.5, rng=rng)

    def test_basic_pipeline_with_oracle_nuisance(self, ar1_data):
        """Basic pipeline using true nuisance functions."""
        data = ar1_data
        X, W, Y = data["X"], data["W"], data["Y"]
        e_true = data["e_true"]
        mu0_true, mu1_true = data["mu0_true"], data["mu1_true"]
        tau_true = data["tau_true"]

        # Use oracle nuisance (true values) + estimated CATE
        # Fit nuisance to get tau_hat
        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        tau_hat = nuisance_fns.tau_hat(X)

        # Compute DR scores
        dr_scores = dr_score(Y, W, e_true, mu0_true, mu1_true, tau_hat)
        true_residuals = np.abs(tau_true - tau_hat)

        # Run DR-ACI
        result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=50)

        assert isinstance(result, ConformalResult)
        assert 0 <= result.coverage <= 1
        assert result.avg_width > 0
        assert len(result.coverages_t) == 500 - 50

    def test_full_pipeline_with_fitted_nuisance(self, ar1_data):
        """Full pipeline with nuisance estimation."""
        data = ar1_data
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        # Split into train and calibration
        T = len(Y)
        n_train = int(T * 0.5)
        n_gap = int(T * 0.05)
        cal_start = n_train + n_gap

        X_train, W_train, Y_train = X[:n_train], W[:n_train], Y[:n_train]
        X_cal, W_cal, Y_cal = X[cal_start:], W[cal_start:], Y[cal_start:]
        tau_cal = tau_true[cal_start:]

        # Fit nuisances on training block
        nuisance_fns = fit_nuisances(X_train, W_train, Y_train, method="linear")

        # Predict on calibration block
        e_hat_cal = nuisance_fns.e_hat(X_cal)
        mu0_hat_cal = nuisance_fns.mu0_hat(X_cal)
        mu1_hat_cal = nuisance_fns.mu1_hat(X_cal)
        tau_hat_cal = nuisance_fns.tau_hat(X_cal)

        # Compute scores
        dr_scores = dr_score(Y_cal, W_cal, e_hat_cal, mu0_hat_cal, mu1_hat_cal, tau_hat_cal)
        true_residuals = np.abs(tau_cal - tau_hat_cal)

        # Run DR-ACI
        n_warmup = max(int(len(X_cal) * 0.1), 20)
        result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=n_warmup)

        assert isinstance(result, ConformalResult)
        assert result.coverage > 0.5  # Should have decent coverage
        assert result.avg_width > 0

    def test_draci_class_interface(self, ar1_data):
        """Test high-level DRACI class."""
        data = ar1_data
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        # Fit nuisances
        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        # Use DRACI class
        draci = DRACI(alpha=0.1, gamma=0.01, n_warmup=50, score_type="dr")
        result = draci.fit_predict(
            Y, W, e_hat, mu0_hat, mu1_hat, tau_hat, tau_true=tau_true
        )

        assert isinstance(result, IntervalResult)
        assert len(result.point) == len(Y)
        assert len(result.lower) == len(Y)
        assert len(result.upper) == len(Y)
        assert np.all(result.upper >= result.lower)
        assert result.coverage_t is not None

    def test_draci_vs_dr_mode(self, ar1_data):
        """Test VS-DR-ACI score mode."""
        data = ar1_data
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        draci = DRACI(alpha=0.1, gamma=0.01, n_warmup=50, score_type="vs_dr")
        result = draci.fit_predict(
            Y, W, e_hat, mu0_hat, mu1_hat, tau_hat, tau_true=tau_true
        )

        assert isinstance(result, IntervalResult)
        # VS-DR should typically have tighter intervals
        assert result.width.mean() > 0


class TestCrossFittingIntegration:
    """Integration tests for temporal cross-fitting."""

    def test_crossfit_to_draci_pipeline(self, rng):
        """Test cross-fitting feeds correctly into DR-ACI."""
        data = generate_data(T=500, rho=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        # Cross-fit nuisances
        result = temporal_block_crossfit(X, W, Y, n_blocks=5, method="linear")

        assert isinstance(result, CrossFitResult)
        assert len(result.e_hat) == len(Y)
        assert len(result.tau_hat) == len(Y)

        # Use cross-fit predictions for DR-ACI
        dr_scores = dr_score(
            Y, W, result.e_hat, result.mu0_hat, result.mu1_hat, result.tau_hat
        )
        true_residuals = np.abs(tau_true - result.tau_hat)

        aci_result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=50)
        assert isinstance(aci_result, ConformalResult)

    def test_temporal_crossfitter_class(self, rng):
        """Test TemporalCrossFitter class interface."""
        data = generate_data(T=500, rho=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]

        fitter = TemporalCrossFitter(n_blocks=5, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y)

        assert isinstance(result, CrossFitResult)
        assert len(result.block_indices) == 5
        assert sum(len(idx) for idx in result.block_indices) == 500


class TestBaselineComparison:
    """Integration tests comparing DR-ACI with CI baselines."""

    def test_draci_vs_dml_wald(self, rng):
        """Compare DR-ACI with DML-Wald CI."""
        data = generate_data(T=500, rho=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        # Fit nuisances
        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        # Compute pseudo-outcomes and scores
        psi_dr = dr_pseudo_outcome(Y, W, e_hat, mu0_hat, mu1_hat)
        dr_scores = np.abs(psi_dr - tau_hat)
        true_residuals = np.abs(tau_true - tau_hat)

        # DR-ACI
        draci_result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=50)

        # DML-Wald
        dml_result = dml_wald_ci(psi_dr, tau_hat, true_residuals, X, alpha=0.1)

        # Both should produce valid results
        assert isinstance(draci_result, ConformalResult)
        assert isinstance(dml_result, ConformalResult)
        assert draci_result.coverage > 0
        assert dml_result.coverage > 0

    def test_draci_vs_block_bootstrap(self, rng):
        """Compare DR-ACI with block bootstrap."""
        data = generate_data(T=300, rho=0.3, rng=rng)  # Smaller for bootstrap speed
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        psi_dr = dr_pseudo_outcome(Y, W, e_hat, mu0_hat, mu1_hat)
        dr_scores = np.abs(psi_dr - tau_hat)
        true_residuals = np.abs(tau_true - tau_hat)

        # DR-ACI
        draci_result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=30)

        # Block bootstrap (fewer bootstrap samples for speed)
        boot_result = block_bootstrap_ci(psi_dr, tau_hat, true_residuals, alpha=0.1, B=99, rng=rng)

        assert isinstance(draci_result, ConformalResult)
        assert isinstance(boot_result, ConformalResult)


class TestDriftRegimes:
    """Integration tests for drift regime DGPs."""

    def test_regime_c_pipeline(self, rng):
        """Test full pipeline under regime C (drift only)."""
        data = generate_data_regime_c(T=500, delta=1.0, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        # Fit on first half (pre-drift)
        n_train = int(len(Y) * 0.4)
        X_train, W_train, Y_train = X[:n_train], W[:n_train], Y[:n_train]

        nuisance_fns = fit_nuisances(X_train, W_train, Y_train, method="linear")

        # Predict on full series
        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        dr_scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
        true_residuals = np.abs(tau_true - tau_hat)

        result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=50)
        assert isinstance(result, ConformalResult)
        # Coverage may be lower due to drift, but should still be reasonable
        assert result.coverage > 0.3

    def test_regime_d_pipeline(self, rng):
        """Test full pipeline under regime D (drift + dependence)."""
        data = generate_data_regime_d(T=500, rho=0.7, delta=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        n_train = int(len(Y) * 0.4)
        X_train, W_train, Y_train = X[:n_train], W[:n_train], Y[:n_train]

        nuisance_fns = fit_nuisances(X_train, W_train, Y_train, method="linear")

        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        dr_scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
        true_residuals = np.abs(tau_true - tau_hat)

        result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=50)
        assert isinstance(result, ConformalResult)


class TestMixingDiagnosticsIntegration:
    """Integration tests for mixing diagnostics."""

    def test_diagnostics_on_generated_data(self, rng):
        """Test mixing diagnostics on AR(1) data."""
        data = generate_data(T=500, rho=0.7, rng=rng)
        X = data["X"]

        # Compute diagnostics on first covariate
        diag = mixing_diagnostics(X[:, 0])

        assert diag.beta_values is not None
        assert len(diag.beta_values) > 0
        # High rho should produce high lag-1 ACF
        assert abs(diag.acf_lag1_median) > 0.3

    def test_diagnostics_on_residuals(self, rng):
        """Test mixing diagnostics on DR residuals."""
        data = generate_data(T=500, rho=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]

        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        psi_dr = dr_pseudo_outcome(Y, W, e_hat, mu0_hat, mu1_hat)
        residuals = psi_dr - tau_hat

        diag = mixing_diagnostics(residuals)
        assert diag.beta_values is not None

    def test_acf_computation(self, rng):
        """Test ACF computation."""
        data = generate_data(T=500, rho=0.7, rng=rng)
        X = data["X"]

        acf = compute_acf(X[:, 0], max_lag=20)
        assert len(acf) == 21  # lag 0 to 20
        assert acf[0] == 1.0  # lag-0 is always 1
        assert abs(acf[1]) > 0.5  # lag-1 should be high for rho=0.7


class TestCoverageProperties:
    """Tests for coverage validity properties."""

    def test_coverage_near_nominal_iid(self, rng):
        """Coverage should be near 90% for i.i.d. data (rho=0)."""
        # Multiple trials to reduce variance
        coverages = []
        for seed in range(5):
            trial_rng = np.random.default_rng(seed + 100)
            data = generate_data(T=1000, rho=0.0, rng=trial_rng)
            X, W, Y = data["X"], data["W"], data["Y"]
            tau_true = data["tau_true"]

            nuisance_fns = fit_nuisances(X, W, Y, method="linear")
            e_hat = nuisance_fns.e_hat(X)
            mu0_hat = nuisance_fns.mu0_hat(X)
            mu1_hat = nuisance_fns.mu1_hat(X)
            tau_hat = nuisance_fns.tau_hat(X)

            dr_scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
            true_residuals = np.abs(tau_true - tau_hat)

            result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.005, n_warmup=100)
            coverages.append(result.coverage)

        mean_coverage = np.mean(coverages)
        # Should be within 0.15 of nominal 0.9 (conservative due to estimation error)
        assert 0.75 <= mean_coverage <= 1.0

    def test_width_positive(self, rng):
        """All interval widths should be positive."""
        data = generate_data(T=500, rho=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        nuisance_fns = fit_nuisances(X, W, Y, method="linear")

        draci = DRACI(alpha=0.1, score_type="dr")
        result = draci.fit_predict(
            Y, W, nuisance_fns.e_hat(X), nuisance_fns.mu0_hat(X),
            nuisance_fns.mu1_hat(X), nuisance_fns.tau_hat(X), tau_true=tau_true
        )

        # Width should be positive everywhere (after warmup)
        positive_widths = result.width[result.width > 0]
        assert len(positive_widths) > 400  # Most widths should be positive

    def test_intervals_contain_point_estimate(self, rng):
        """Intervals should always contain the point estimate."""
        data = generate_data(T=500, rho=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]

        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        tau_hat = nuisance_fns.tau_hat(X)

        draci = DRACI(alpha=0.1, score_type="dr", n_warmup=50)
        result = draci.fit_predict(
            Y, W, nuisance_fns.e_hat(X), nuisance_fns.mu0_hat(X),
            nuisance_fns.mu1_hat(X), tau_hat
        )

        # After warmup, point should be within bounds
        valid_idx = result.width > 0  # Where intervals are computed
        assert np.all(result.lower[valid_idx] <= result.point[valid_idx])
        assert np.all(result.point[valid_idx] <= result.upper[valid_idx])


class TestNuisanceEstimatorIntegration:
    """Integration tests for nuisance estimators."""

    def test_xgboost_nuisance_in_pipeline(self, rng):
        """Test XGBoost nuisance in full pipeline."""
        data = generate_data(T=500, rho=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]

        # Train/cal split
        n_train = 300
        X_train, W_train, Y_train = X[:n_train], W[:n_train], Y[:n_train]
        X_cal, W_cal, Y_cal = X[n_train:], W[n_train:], Y[n_train:]

        # Use XGBoost
        estimator = XGBoostNuisance()
        nuisance_fns = estimator.fit(X_train, W_train, Y_train)

        e_hat = nuisance_fns.e_hat(X_cal)
        mu0_hat = nuisance_fns.mu0_hat(X_cal)
        mu1_hat = nuisance_fns.mu1_hat(X_cal)
        tau_hat = nuisance_fns.tau_hat(X_cal)

        # Predictions should be valid
        assert np.all((e_hat > 0) & (e_hat < 1))
        assert np.all(np.isfinite(tau_hat))

    def test_linear_nuisance_in_pipeline(self, rng):
        """Test linear nuisance in full pipeline."""
        data = generate_data(T=500, rho=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]

        n_train = 300
        X_train, W_train, Y_train = X[:n_train], W[:n_train], Y[:n_train]
        X_cal = X[n_train:]

        estimator = LinearNuisance()
        nuisance_fns = estimator.fit(X_train, W_train, Y_train)

        e_hat = nuisance_fns.e_hat(X_cal)
        tau_hat = nuisance_fns.tau_hat(X_cal)

        assert np.all((e_hat > 0) & (e_hat < 1))
        assert np.all(np.isfinite(tau_hat))


class TestCIMetricsIntegration:
    """Integration tests for CI metrics computation."""

    def test_compute_metrics_from_draci(self, rng):
        """Test CI metrics computation from DR-ACI result."""
        data = generate_data(T=500, rho=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        dr_scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
        true_residuals = np.abs(tau_true - tau_hat)

        result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=50)

        # Compute metrics
        metrics = compute_ci_metrics(result.coverages_t, result.avg_width, alpha=0.1)

        assert hasattr(metrics, "miscoverage_gap")
        assert hasattr(metrics, "piaw")
        assert np.isfinite(metrics.miscoverage_gap)
        assert np.isfinite(metrics.piaw)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_sample(self, rng):
        """Test with small sample size."""
        data = generate_data(T=100, rho=0.3, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        dr_scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
        true_residuals = np.abs(tau_true - tau_hat)

        # Should work with small warmup
        result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=10)
        assert isinstance(result, ConformalResult)

    def test_high_dependence(self, rng):
        """Test with high AR(1) coefficient."""
        data = generate_data(T=500, rho=0.95, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        dr_scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
        true_residuals = np.abs(tau_true - tau_hat)

        result = dr_aci(dr_scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=50)
        assert isinstance(result, ConformalResult)
        # Coverage may degrade but should still be positive
        assert result.coverage > 0

    def test_extreme_alpha(self, rng):
        """Test with extreme alpha values."""
        data = generate_data(T=500, rho=0.5, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        nuisance_fns = fit_nuisances(X, W, Y, method="linear")
        e_hat = nuisance_fns.e_hat(X)
        mu0_hat = nuisance_fns.mu0_hat(X)
        mu1_hat = nuisance_fns.mu1_hat(X)
        tau_hat = nuisance_fns.tau_hat(X)

        dr_scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
        true_residuals = np.abs(tau_true - tau_hat)

        # Very conservative (alpha=0.01 -> 99% coverage target)
        result = dr_aci(dr_scores, true_residuals, alpha=0.01, gamma=0.005, n_warmup=50)
        assert isinstance(result, ConformalResult)
        assert result.avg_width > 0

        # Very aggressive (alpha=0.5 -> 50% coverage target)
        result = dr_aci(dr_scores, true_residuals, alpha=0.5, gamma=0.02, n_warmup=50)
        assert isinstance(result, ConformalResult)
