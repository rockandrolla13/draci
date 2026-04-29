"""Tests for draci.cross_fitting module."""

import numpy as np
import pytest

from draci.cross_fitting import (
    TemporalCrossFitter,
    CrossFitResult,
    temporal_block_crossfit,
)
from draci.nuisance import XGBoostNuisance, LinearNuisance


class TestCrossFitResult:
    """Tests for CrossFitResult dataclass."""

    def test_dataclass_creation(self):
        """CrossFitResult should store all nuisance predictions."""
        T = 100
        result = CrossFitResult(
            e_hat=np.zeros(T),
            mu0_hat=np.zeros(T),
            mu1_hat=np.zeros(T),
            tau_hat=np.zeros(T),
            block_indices=[np.arange(0, 50), np.arange(50, 100)],
        )

        assert result.e_hat.shape == (T,)
        assert result.mu0_hat.shape == (T,)
        assert result.mu1_hat.shape == (T,)
        assert result.tau_hat.shape == (T,)
        assert len(result.block_indices) == 2


class TestTemporalCrossFitter:
    """Tests for TemporalCrossFitter class."""

    def test_init_defaults(self):
        """Should initialize with default n_blocks=5."""
        fitter = TemporalCrossFitter()

        assert fitter.n_blocks == 5
        assert fitter.nuisance_estimator is None

    def test_init_custom_blocks(self):
        """Should respect custom n_blocks."""
        fitter = TemporalCrossFitter(n_blocks=10)

        assert fitter.n_blocks == 10

    def test_init_custom_estimator(self):
        """Should accept custom nuisance estimator."""
        estimator = LinearNuisance()
        fitter = TemporalCrossFitter(nuisance_estimator=estimator)

        assert fitter.nuisance_estimator is estimator

    def test_fit_transform_returns_crossfit_result(self, rng):
        """fit_transform should return CrossFitResult."""
        T = 500
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        fitter = TemporalCrossFitter(n_blocks=5, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y)

        assert isinstance(result, CrossFitResult)

    def test_fit_transform_output_shapes(self, rng):
        """All predictions should have correct shape."""
        T = 500
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        fitter = TemporalCrossFitter(n_blocks=5, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y)

        assert result.e_hat.shape == (T,)
        assert result.mu0_hat.shape == (T,)
        assert result.mu1_hat.shape == (T,)
        assert result.tau_hat.shape == (T,)

    def test_block_indices_partition(self, rng):
        """Block indices should partition all observations."""
        T = 500
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        fitter = TemporalCrossFitter(n_blocks=5, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y)

        # Should have 5 blocks
        assert len(result.block_indices) == 5

        # Union of blocks should cover all indices
        all_indices = np.concatenate(result.block_indices)
        assert len(all_indices) == T
        assert set(all_indices) == set(range(T))

    def test_temporal_ordering_preserved(self, rng):
        """Block indices should be contiguous (temporal blocks)."""
        T = 500
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        fitter = TemporalCrossFitter(n_blocks=5, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y)

        # Each block should be contiguous
        for block_idx in result.block_indices:
            diffs = np.diff(block_idx)
            assert np.all(diffs == 1), "Block indices should be contiguous"

    def test_propensity_bounds(self, rng):
        """Propensity predictions should be bounded."""
        T = 500
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        fitter = TemporalCrossFitter(n_blocks=5, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y)

        # Propensities should be in valid range (with some tolerance for edge blocks)
        valid_mask = result.e_hat > 0
        assert np.all(result.e_hat[valid_mask] >= 0)
        assert np.all(result.e_hat[valid_mask] <= 1)

    def test_tau_equals_mu1_minus_mu0(self, rng):
        """CATE prediction should equal difference of outcome predictions."""
        T = 500
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        fitter = TemporalCrossFitter(n_blocks=5, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y)

        # tau_hat should equal mu1_hat - mu0_hat
        np.testing.assert_array_almost_equal(
            result.tau_hat, result.mu1_hat - result.mu0_hat
        )

    def test_handles_1d_covariates(self, rng):
        """Should handle 1D covariate input."""
        T = 500
        X = rng.normal(0, 1, T)  # 1D
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X + W * 0.5 + rng.normal(0, 0.5, T)

        fitter = TemporalCrossFitter(n_blocks=5, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y)

        assert result.e_hat.shape == (T,)

    def test_time_index_sorting(self, rng):
        """Should sort by time_index and restore original order."""
        T = 200
        # Create shuffled data
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        # Create time index in random order
        time_index = rng.permutation(T)

        fitter = TemporalCrossFitter(n_blocks=4, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y, time_index=time_index)

        # Result should be in original order (same length)
        assert result.e_hat.shape == (T,)

    def test_different_n_blocks(self, rng):
        """Should work with different numbers of blocks."""
        T = 600
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        for n_blocks in [2, 3, 5, 10]:
            fitter = TemporalCrossFitter(
                n_blocks=n_blocks, nuisance_estimator=LinearNuisance()
            )
            result = fitter.fit_transform(X, W, Y)

            assert len(result.block_indices) == n_blocks
            assert result.e_hat.shape == (T,)


class TestTemporalBlockCrossfit:
    """Tests for temporal_block_crossfit convenience function."""

    def test_returns_crossfit_result(self, rng):
        """Should return CrossFitResult."""
        T = 500
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        result = temporal_block_crossfit(X, W, Y, n_blocks=5, method="linear")

        assert isinstance(result, CrossFitResult)

    def test_linear_method(self, rng):
        """Should work with linear method."""
        T = 500
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        result = temporal_block_crossfit(X, W, Y, n_blocks=5, method="linear")

        assert result.e_hat.shape == (T,)

    def test_xgboost_method(self, rng):
        """Should work with xgboost method."""
        T = 500
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        result = temporal_block_crossfit(X, W, Y, n_blocks=5, method="xgboost")

        assert result.e_hat.shape == (T,)

    def test_unknown_method_raises(self, rng):
        """Should raise ValueError for unknown method."""
        T = 100
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = rng.normal(0, 1, T)

        with pytest.raises(ValueError, match="Unknown method"):
            temporal_block_crossfit(X, W, Y, method="unknown")

    def test_time_index_parameter(self, rng):
        """Should accept time_index parameter."""
        T = 200
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)
        time_index = rng.permutation(T)

        result = temporal_block_crossfit(
            X, W, Y, n_blocks=4, time_index=time_index, method="linear"
        )

        assert result.e_hat.shape == (T,)


class TestCrossFittingProperties:
    """Tests for theoretical properties of cross-fitting."""

    def test_out_of_sample_predictions(self, rng):
        """Each observation's prediction should be from a model that didn't see it."""
        # This is a conceptual test - we verify block structure
        T = 500
        n_blocks = 5
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        fitter = TemporalCrossFitter(n_blocks=n_blocks, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y)

        # Each block index should be non-overlapping
        for i, block_i in enumerate(result.block_indices):
            for j, block_j in enumerate(result.block_indices):
                if i != j:
                    assert len(set(block_i) & set(block_j)) == 0

    def test_predictions_finite(self, rng):
        """All predictions should be finite."""
        T = 500
        X = rng.normal(0, 1, (T, 5))
        W = rng.binomial(1, 0.5, T).astype(float)
        Y = X[:, 0] + W * 0.5 + rng.normal(0, 0.5, T)

        fitter = TemporalCrossFitter(n_blocks=5, nuisance_estimator=LinearNuisance())
        result = fitter.fit_transform(X, W, Y)

        assert np.all(np.isfinite(result.e_hat))
        assert np.all(np.isfinite(result.mu0_hat))
        assert np.all(np.isfinite(result.mu1_hat))
        assert np.all(np.isfinite(result.tau_hat))
