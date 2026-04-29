#!/usr/bin/env python
"""
Temporal Block Cross-Fitting
=============================

This example demonstrates temporal block cross-fitting for dependent data.

Standard K-fold cross-validation is not appropriate for time series because
it breaks the temporal structure and can cause data leakage. Temporal block
cross-fitting addresses this by:
1. Splitting time series into K contiguous temporal blocks
2. Training on K-1 blocks, predicting on the held-out block
3. Repeating for each block

This produces out-of-block predictions for the entire series while preserving
time ordering and minimizing temporal leakage.
"""

import numpy as np

from draci import (
    # Cross-fitting
    TemporalCrossFitter,
    temporal_block_crossfit,
    CrossFitResult,
    # Nuisance estimators
    XGBoostNuisance,
    LinearNuisance,
    # Data generation
    generate_data,
    # DR-ACI
    dr_score,
    dr_aci,
)


def main():
    # Generate data with temporal dependence
    rng = np.random.default_rng(42)
    data = generate_data(T=1000, rho=0.7, rng=rng)
    X, W, Y = data["X"], data["W"], data["Y"]
    tau_true = data["tau_true"]

    print("=" * 60)
    print("Temporal Block Cross-Fitting")
    print("=" * 60)
    print()

    # =========================================================================
    # Method 1: Using convenience function
    # =========================================================================
    print("Method 1: temporal_block_crossfit() function")
    print("-" * 40)

    result = temporal_block_crossfit(
        X, W, Y,
        n_blocks=5,       # Number of temporal blocks
        method="linear",  # "xgboost" or "linear"
    )

    print(f"  Number of blocks: {len(result.block_indices)}")
    print(f"  Block sizes: {[len(idx) for idx in result.block_indices]}")
    print(f"  Propensity range: [{result.e_hat.min():.3f}, {result.e_hat.max():.3f}]")
    print(f"  CATE range: [{result.tau_hat.min():.3f}, {result.tau_hat.max():.3f}]")
    print()

    # =========================================================================
    # Method 2: Using TemporalCrossFitter class
    # =========================================================================
    print("Method 2: TemporalCrossFitter class")
    print("-" * 40)

    # Can specify custom nuisance estimator
    fitter = TemporalCrossFitter(
        n_blocks=5,
        nuisance_estimator=LinearNuisance(),  # or XGBoostNuisance()
    )

    result = fitter.fit_transform(X, W, Y)

    print(f"  CrossFitResult fields: {list(result.__dataclass_fields__.keys())}")
    print()

    # =========================================================================
    # Using cross-fit results with DR-ACI
    # =========================================================================
    print("Using Cross-Fit Results with DR-ACI")
    print("-" * 40)

    # Compute scores using out-of-block predictions
    dr_scores = dr_score(
        Y, W,
        result.e_hat,
        result.mu0_hat,
        result.mu1_hat,
        result.tau_hat,
    )

    true_residuals = np.abs(tau_true - result.tau_hat)

    # Run DR-ACI
    aci_result = dr_aci(
        dr_scores,
        true_residuals,
        alpha=0.1,
        gamma=0.005,
        n_warmup=100,
    )

    print(f"  Coverage: {aci_result.coverage:.1%}")
    print(f"  Average width: {aci_result.avg_width:.4f}")
    print()

    # =========================================================================
    # Comparison: Cross-fit vs. Single Split
    # =========================================================================
    print("Comparison: Cross-Fit vs. Single Split")
    print("-" * 40)

    # Single split: train on first 50%, predict on rest
    n_train = 500
    from draci import fit_nuisances

    # fit_nuisances returns a NuisanceFunctions object
    nuisance_fns = fit_nuisances(
        X[:n_train], W[:n_train], Y[:n_train], method="linear"
    )

    e_hat_split = nuisance_fns.e_hat(X[n_train:])
    mu0_hat_split = nuisance_fns.mu0_hat(X[n_train:])
    mu1_hat_split = nuisance_fns.mu1_hat(X[n_train:])
    tau_hat_split = nuisance_fns.tau_hat(X[n_train:])

    dr_scores_split = dr_score(
        Y[n_train:], W[n_train:],
        e_hat_split, mu0_hat_split, mu1_hat_split, tau_hat_split
    )
    true_res_split = np.abs(tau_true[n_train:] - tau_hat_split)

    aci_split = dr_aci(
        dr_scores_split, true_res_split, alpha=0.1, gamma=0.005, n_warmup=50
    )

    print(f"  Cross-Fit:    Coverage={aci_result.coverage:.1%}, Width={aci_result.avg_width:.4f}")
    print(f"  Single Split: Coverage={aci_split.coverage:.1%}, Width={aci_split.avg_width:.4f}")
    print()
    print("Cross-fitting uses all data for both training and evaluation,")
    print("which can improve efficiency especially for smaller samples.")


if __name__ == "__main__":
    main()
