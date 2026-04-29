"""Tests for draci.scores module."""

import numpy as np
import pytest

from draci.scores import dr_pseudo_outcome, dr_score, vs_dr_score, naive_score


class TestDRPseudoOutcome:
    """Tests for dr_pseudo_outcome function."""

    def test_output_shape(self, sample_data, nuisance_estimates):
        """DR pseudo-outcome should match input shape."""
        psi_dr = dr_pseudo_outcome(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
        )
        assert psi_dr.shape == (sample_data["T"],)

    def test_perfect_nuisance_centers_at_tau(self, sample_data, nuisance_estimates):
        """With perfect nuisances, psi^DR should center at true tau."""
        psi_dr = dr_pseudo_outcome(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
        )
        # Mean of psi_dr should be close to mean of tau_true
        mean_psi = np.mean(psi_dr)
        mean_tau = np.mean(sample_data["tau_true"])
        # Allow for residual noise (sigma=0.1)
        assert abs(mean_psi - mean_tau) < 0.3

    def test_dr_formula_correctness(self, rng):
        """Verify DR formula manually on simple case."""
        T = 10
        Y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        W = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        e_hat = np.full(T, 0.5)
        mu0_hat = np.arange(1, T + 1, dtype=float)
        mu1_hat = np.arange(2, T + 2, dtype=float)

        psi_dr = dr_pseudo_outcome(Y, W, e_hat, mu0_hat, mu1_hat)

        # Manual computation for t=0: W=1, Y=1, e=0.5, mu0=1, mu1=2
        # psi_dr = (1/0.5)*(1-2) - (0/0.5)*(1-1) + (2-1) = -2 + 1 = -1
        expected_0 = 1 / 0.5 * (1 - 2) + (2 - 1)  # = -2 + 1 = -1
        assert abs(psi_dr[0] - expected_0) < 1e-10


class TestDRScore:
    """Tests for dr_score function."""

    def test_output_shape(self, sample_data, nuisance_estimates):
        """DR score output should match input shape."""
        scores = dr_score(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        assert scores.shape == (sample_data["T"],)
        assert np.all(scores >= 0)  # Absolute values

    def test_perfect_nuisance_small_scores(self, sample_data, nuisance_estimates):
        """With perfect nuisances, DR scores should be small (just noise)."""
        scores = dr_score(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        # With perfect nuisances, scores should be O(sigma_eta)
        assert np.mean(scores) < 0.5

    def test_scores_nonnegative(self, sample_data, noisy_nuisance_estimates):
        """DR scores are absolute values, hence non-negative."""
        scores = dr_score(
            sample_data["Y"],
            sample_data["W"],
            noisy_nuisance_estimates["e_hat"],
            noisy_nuisance_estimates["mu0_hat"],
            noisy_nuisance_estimates["mu1_hat"],
            noisy_nuisance_estimates["tau_hat"],
        )
        assert np.all(scores >= 0)

    def test_scores_finite(self, sample_data, nuisance_estimates):
        """DR scores should be finite with valid inputs."""
        scores = dr_score(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        assert np.all(np.isfinite(scores))


class TestVSDRScore:
    """Tests for vs_dr_score function."""

    def test_output_shape(self, sample_data, nuisance_estimates):
        """VS-DR score output should match input shape."""
        vs_scores = vs_dr_score(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        assert vs_scores.shape == (sample_data["T"],)
        assert np.all(vs_scores >= 0)

    def test_return_sigma_option(self, sample_data, nuisance_estimates):
        """VS-DR score should optionally return sigma_hat."""
        result = vs_dr_score(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
            return_sigma=True,
        )
        assert isinstance(result, tuple)
        vs_scores, sigma_hat = result
        assert vs_scores.shape == (sample_data["T"],)
        assert sigma_hat.shape == (sample_data["T"],)
        assert np.all(sigma_hat > 0)

    def test_standardization_effect(self, sample_data, nuisance_estimates):
        """VS scores should be smaller than raw scores when sigma > 1."""
        raw_scores = dr_score(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        vs_scores, sigma_hat = vs_dr_score(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["e_hat"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
            return_sigma=True,
        )
        # Verify relation: vs_scores ≈ raw_scores / sigma_hat
        expected = raw_scores / sigma_hat
        np.testing.assert_allclose(vs_scores, expected, rtol=1e-10)


class TestNaiveScore:
    """Tests for naive_score function."""

    def test_output_shape(self, sample_data, nuisance_estimates):
        """Naive score output should match input shape."""
        scores = naive_score(
            sample_data["Y"],
            sample_data["W"],
            nuisance_estimates["mu0_hat"],
            nuisance_estimates["mu1_hat"],
            nuisance_estimates["tau_hat"],
        )
        assert scores.shape == (sample_data["T"],)
        assert np.all(scores >= 0)

    def test_scores_nonnegative(self, sample_data, noisy_nuisance_estimates):
        """Naive scores are absolute values, hence non-negative."""
        scores = naive_score(
            sample_data["Y"],
            sample_data["W"],
            noisy_nuisance_estimates["mu0_hat"],
            noisy_nuisance_estimates["mu1_hat"],
            noisy_nuisance_estimates["tau_hat"],
        )
        assert np.all(scores >= 0)


class TestEdgeCases:
    """Edge case tests for score functions."""

    def test_extreme_propensities_clipped(self, rng):
        """DR score should handle propensities near boundaries."""
        T = 50
        Y = rng.normal(0, 1, T)
        W = rng.binomial(1, 0.5, T).astype(float)

        # Propensities clipped away from 0 and 1
        e_hat = np.clip(rng.uniform(0.01, 0.99, T), 0.05, 0.95)
        mu0_hat = np.zeros(T)
        mu1_hat = np.ones(T)
        tau_hat = np.ones(T)

        scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)

        # Should not have NaN or Inf
        assert np.all(np.isfinite(scores))

    def test_small_sample(self, rng):
        """Score functions should work on small samples."""
        T = 5
        Y = rng.normal(0, 1, T)
        W = np.array([1, 0, 1, 0, 1], dtype=float)
        e_hat = np.full(T, 0.5)
        mu0_hat = np.zeros(T)
        mu1_hat = np.ones(T)
        tau_hat = np.ones(T)

        # All score functions should work
        dr_scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
        vs_scores = vs_dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
        n_scores = naive_score(Y, W, mu0_hat, mu1_hat, tau_hat)

        assert len(dr_scores) == T
        assert len(vs_scores) == T
        assert len(n_scores) == T

    def test_2d_covariates(self, sample_data_2d, rng):
        """Scores should work with true nuisances from 2D data."""
        d = sample_data_2d
        scores = dr_score(
            d["Y"],
            d["W"],
            d["e_true"],
            d["mu0_true"],
            d["mu1_true"],
            d["tau_true"],
        )
        assert scores.shape == (d["T"],)
        assert np.all(np.isfinite(scores))
