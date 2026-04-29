#!/usr/bin/env python
"""
Basic DR-ACI Usage
==================

This example demonstrates the basic DR-ACI workflow:
1. Generate synthetic data with temporal dependence
2. Estimate nuisance functions
3. Compute doubly robust scores
4. Run adaptive conformal inference
5. Evaluate coverage and interval width

For real data, replace step 1 with your data loading.
"""

import numpy as np

from draci import (
    # Data generation (for simulation)
    generate_data,
    # Nuisance estimation
    fit_nuisances,
    # Score computation
    dr_score,
    # Adaptive conformal inference
    dr_aci,
)


def main():
    # =========================================================================
    # 1. Generate synthetic data
    # =========================================================================
    # AR(1) covariate process with rho=0.7 (moderate dependence)
    rng = np.random.default_rng(42)
    data = generate_data(T=1000, rho=0.7, rng=rng)

    X = data["X"]        # Covariates (T x 5)
    W = data["W"]        # Treatment indicators (T,)
    Y = data["Y"]        # Observed outcomes (T,)
    tau_true = data["tau_true"]  # True CATE (for evaluation only)

    print("Data generated:")
    print(f"  Sample size T = {len(Y)}")
    print(f"  Covariate dimension p = {X.shape[1]}")
    print(f"  Treatment rate = {W.mean():.2%}")
    print()

    # =========================================================================
    # 2. Split into training and calibration
    # =========================================================================
    # Train nuisance models on first 50%, calibrate on remaining 45%
    # (5% gap to reduce temporal leakage)
    train_frac, gap_frac = 0.50, 0.05
    n_train = int(len(Y) * train_frac)
    n_gap = int(len(Y) * gap_frac)
    cal_start = n_train + n_gap

    X_train, W_train, Y_train = X[:n_train], W[:n_train], Y[:n_train]
    X_cal, W_cal, Y_cal = X[cal_start:], W[cal_start:], Y[cal_start:]
    tau_cal = tau_true[cal_start:]

    print("Data split:")
    print(f"  Training: {n_train} observations")
    print(f"  Gap: {n_gap} observations")
    print(f"  Calibration: {len(X_cal)} observations")
    print()

    # =========================================================================
    # 3. Estimate nuisance functions
    # =========================================================================
    # fit_nuisances returns a NuisanceFunctions object with callable predictors
    nuisance_fns = fit_nuisances(
        X_train, W_train, Y_train,
        method="xgboost",  # or "linear" for faster fitting
    )

    # Predict on calibration set
    e_hat = nuisance_fns.e_hat(X_cal)
    mu0_hat = nuisance_fns.mu0_hat(X_cal)
    mu1_hat = nuisance_fns.mu1_hat(X_cal)
    tau_hat = nuisance_fns.tau_hat(X_cal)

    print("Nuisance estimation complete")
    print(f"  Propensity range: [{e_hat.min():.3f}, {e_hat.max():.3f}]")
    print(f"  CATE estimate range: [{tau_hat.min():.3f}, {tau_hat.max():.3f}]")
    print()

    # =========================================================================
    # 4. Compute doubly robust scores
    # =========================================================================
    # DR score: |psi^DR - tau_hat| where psi^DR is the AIPW pseudo-outcome
    dr_scores = dr_score(Y_cal, W_cal, e_hat, mu0_hat, mu1_hat, tau_hat)

    # True residuals for coverage evaluation (oracle - not available in practice)
    true_residuals = np.abs(tau_cal - tau_hat)

    print("Score computation:")
    print(f"  Mean DR score: {dr_scores.mean():.4f}")
    print(f"  Mean true residual: {true_residuals.mean():.4f}")
    print()

    # =========================================================================
    # 5. Run DR-ACI
    # =========================================================================
    # Parameters:
    #   alpha=0.1: Target 90% coverage
    #   gamma=0.005: ACI learning rate (smaller = slower adaptation)
    #   n_warmup=50: Initial observations before starting ACI
    n_warmup = max(int(len(X_cal) * 0.1), 50)

    result = dr_aci(
        dr_scores,
        true_residuals,
        alpha=0.10,
        gamma=0.005,
        n_warmup=n_warmup,
    )

    # =========================================================================
    # 6. Evaluate results
    # =========================================================================
    print("=" * 50)
    print("DR-ACI Results")
    print("=" * 50)
    print(f"  Target coverage: 90%")
    print(f"  Empirical coverage: {result.coverage:.1%}")
    print(f"  Average interval width: {result.avg_width:.4f}")
    print(f"  Coverage per timestep stored: {len(result.coverages_t)} values")
    print()

    # Coverage over time
    window = 100
    if len(result.coverages_t) >= window:
        rolling_cov = np.convolve(
            result.coverages_t,
            np.ones(window) / window,
            mode="valid"
        )
        print(f"  Rolling coverage (window={window}):")
        print(f"    Min: {rolling_cov.min():.1%}")
        print(f"    Max: {rolling_cov.max():.1%}")


if __name__ == "__main__":
    main()
