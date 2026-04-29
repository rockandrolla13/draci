"""
Temporal block cross-fitting for dependent data.

Implements K-fold temporal block cross-fitting that preserves time ordering
and avoids data leakage across time. This is the time-series analogue of
standard cross-fitting from DML.

References:
- Chernozhukov et al. (2018): Double/Debiased Machine Learning
- Cao et al. (2025): Neighborhood Stability for DML under Mixing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .nuisance import NuisanceEstimator, NuisanceResult


@dataclass
class CrossFitResult:
    """Result from temporal block cross-fitting."""

    e_hat: np.ndarray  # Out-of-block propensity predictions
    mu0_hat: np.ndarray  # Out-of-block control outcome predictions
    mu1_hat: np.ndarray  # Out-of-block treated outcome predictions
    tau_hat: np.ndarray  # Out-of-block CATE predictions
    block_indices: list[np.ndarray]  # Indices for each temporal block


class TemporalCrossFitter:
    """K-fold temporal block cross-fitting for dependent data.

    Splits the time series into K contiguous blocks. For each block,
    trains nuisance models on all OTHER blocks, then predicts on the
    held-out block. This preserves temporal ordering and avoids data
    leakage across time.

    Parameters
    ----------
    n_blocks : int
        Number of temporal blocks (default 5)
    nuisance_estimator : NuisanceEstimator, optional
        Estimator to use for each fold. If None, uses XGBoostNuisance.

    Example
    -------
    >>> from draci import TemporalCrossFitter, XGBoostNuisance
    >>> fitter = TemporalCrossFitter(n_blocks=5)
    >>> result = fitter.fit_transform(X, W, Y)
    >>> # result contains out-of-block predictions for all observations
    """

    def __init__(
        self,
        n_blocks: int = 5,
        nuisance_estimator: "NuisanceEstimator | None" = None,
    ):
        self.n_blocks = n_blocks
        self.nuisance_estimator = nuisance_estimator

    def fit_transform(
        self,
        X: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
        time_index: np.ndarray | None = None,
    ) -> CrossFitResult:
        """Fit nuisance models via temporal cross-fitting.

        Parameters
        ----------
        X : array (T, p)
            Covariates (assumed to be in temporal order if time_index=None)
        W : array (T,)
            Treatment indicators
        Y : array (T,)
            Observed outcomes
        time_index : array (T,), optional
            Time indices for sorting. If None, assumes X is already sorted.

        Returns
        -------
        CrossFitResult with out-of-block predictions for all observations
        """
        from .nuisance import XGBoostNuisance

        T = len(Y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Sort by time if index provided
        if time_index is not None:
            sort_idx = np.argsort(time_index)
            X = X[sort_idx]
            W = W[sort_idx]
            Y = Y[sort_idx]
            unsort_idx = np.argsort(sort_idx)  # To restore original order
        else:
            unsort_idx = None

        # Create temporal blocks
        block_size = T // self.n_blocks
        block_indices = []
        for k in range(self.n_blocks):
            start_idx = k * block_size
            end_idx = (k + 1) * block_size if k < self.n_blocks - 1 else T
            block_indices.append(np.arange(start_idx, end_idx))

        # Initialize output arrays
        e_hat = np.zeros(T)
        mu0_hat = np.zeros(T)
        mu1_hat = np.zeros(T)
        tau_hat = np.zeros(T)

        # Get or create nuisance estimator
        estimator = self.nuisance_estimator
        if estimator is None:
            estimator = XGBoostNuisance()

        # Cross-fit: train on K-1 blocks, predict on held-out block
        for k, test_idx in enumerate(block_indices):
            # Training indices: all blocks except k
            train_idx = np.concatenate(
                [block_indices[j] for j in range(self.n_blocks) if j != k]
            )

            if len(train_idx) < 100 or len(test_idx) < 10:
                # Skip if insufficient data
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            W_train, Y_train = W[train_idx], Y[train_idx]

            # Fit on training blocks
            nuisance_fns = estimator.fit(X_train, W_train, Y_train)

            # Predict on held-out block
            e_hat[test_idx] = nuisance_fns.e_hat(X_test)
            mu0_hat[test_idx] = nuisance_fns.mu0_hat(X_test)
            mu1_hat[test_idx] = nuisance_fns.mu1_hat(X_test)
            tau_hat[test_idx] = nuisance_fns.tau_hat(X_test)

        # Restore original order if needed
        if unsort_idx is not None:
            e_hat = e_hat[unsort_idx]
            mu0_hat = mu0_hat[unsort_idx]
            mu1_hat = mu1_hat[unsort_idx]
            tau_hat = tau_hat[unsort_idx]
            block_indices = [unsort_idx[idx] for idx in block_indices]

        return CrossFitResult(
            e_hat=e_hat,
            mu0_hat=mu0_hat,
            mu1_hat=mu1_hat,
            tau_hat=tau_hat,
            block_indices=block_indices,
        )


def temporal_block_crossfit(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    n_blocks: int = 5,
    time_index: np.ndarray | None = None,
    method: str = "xgboost",
) -> CrossFitResult:
    """Convenience function for temporal block cross-fitting.

    Parameters
    ----------
    X, W, Y : arrays
        Data (assumed in temporal order if time_index=None)
    n_blocks : int
        Number of temporal blocks
    time_index : array, optional
        Time indices for sorting
    method : {"xgboost", "linear"}
        Nuisance estimation method

    Returns
    -------
    CrossFitResult with out-of-block predictions
    """
    from .nuisance import XGBoostNuisance, LinearNuisance

    if method == "xgboost":
        estimator = XGBoostNuisance()
    elif method == "linear":
        estimator = LinearNuisance()
    else:
        raise ValueError(f"Unknown method: {method}")

    fitter = TemporalCrossFitter(n_blocks=n_blocks, nuisance_estimator=estimator)
    return fitter.fit_transform(X, W, Y, time_index=time_index)
