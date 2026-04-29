#!/usr/bin/env python
"""
Mixing Diagnostics
==================

This example demonstrates how to use mixing diagnostics to assess
the temporal dependence structure in your data.

Understanding the mixing properties is important because:
1. Strong dependence (high rho) can slow down ACI adaptation
2. The effective sample size is reduced under dependence
3. Nuisance estimation may require larger training samples

The mixing diagnostics provide:
- Autocorrelation function (ACF) coefficients
- Beta-mixing coefficient estimates
- Optimal mixing gap for coverage bounds
"""

import numpy as np

from draci import (
    # Mixing diagnostics
    mixing_diagnostics,
    estimate_beta_mixing,
    compute_acf,
    MixingDiagnostics,
    # Data generation
    generate_data,
    # For residuals
    fit_nuisances,
    dr_pseudo_outcome,
)


def main():
    print("=" * 60)
    print("Mixing Diagnostics for Temporal Dependence")
    print("=" * 60)
    print()

    # =========================================================================
    # Basic ACF computation
    # =========================================================================
    print("1. Autocorrelation Function (ACF)")
    print("-" * 40)

    rng = np.random.default_rng(42)

    # Compare ACF for different rho values
    for rho in [0.0, 0.5, 0.9]:
        data = generate_data(T=500, rho=rho, rng=rng)
        X = data["X"]

        # Compute ACF on first covariate
        acf = compute_acf(X[:, 0], max_lag=5)

        # acf[0] is lag 0 (always 1.0), acf[1] is lag 1, etc.
        print(f"  rho={rho}: ACF lags 1-5 = [{', '.join(f'{a:.2f}' for a in acf[1:6])}]")

    print()

    # =========================================================================
    # Full mixing diagnostics
    # =========================================================================
    print("2. Full Mixing Diagnostics")
    print("-" * 40)

    data = generate_data(T=1000, rho=0.7, rng=rng)
    X = data["X"]

    # Run full mixing diagnostics
    # lags: which lags to estimate beta-mixing coefficients
    # max_acf_lag: maximum lag for ACF computation
    diag = mixing_diagnostics(X[:, 0], lags=[1, 5, 10, 20, 50], max_acf_lag=50)

    # Compute ACF separately for detailed lag analysis
    acf = compute_acf(X[:, 0], max_lag=20)

    print(f"  ACF at lag 1: {acf[1]:.3f}")
    print(f"  ACF at lag 5: {acf[5]:.3f}")
    print(f"  ACF at lag 10: {acf[10]:.3f}")
    print(f"  ACF at lag 20: {acf[20]:.3f}")
    print(f"  Median lag-1 ACF: {diag.acf_lag1_median:.3f}")

    # Beta-mixing coefficients
    print("  Beta-mixing estimates:")
    for tau, stats in diag.beta_values.items():
        print(f"    tau={tau}: beta={stats['median']:.4f}")

    # Optimal mixing gap (from Theorem 1)
    print(f"  Optimal tau: {diag.optimal_tau}")
    print(f"  Optimal mixing gap: {diag.optimal_gap:.4f}")

    # Exponential fit quality
    print(f"  Exp fit R^2: {diag.exp_fit['r2']:.3f}")
    print()

    # =========================================================================
    # Diagnostics on DR pseudo-outcomes
    # =========================================================================
    print("3. Diagnostics on DR Pseudo-Outcomes")
    print("-" * 40)

    data = generate_data(T=1000, rho=0.7, rng=rng)
    X, W, Y = data["X"], data["W"], data["Y"]

    # Fit nuisances - returns NuisanceFunctions object
    nuisance_fns = fit_nuisances(X, W, Y, method="linear")
    e_hat = nuisance_fns.e_hat(X)
    mu0_hat = nuisance_fns.mu0_hat(X)
    mu1_hat = nuisance_fns.mu1_hat(X)
    tau_hat = nuisance_fns.tau_hat(X)

    # Compute DR pseudo-outcomes
    psi_dr = dr_pseudo_outcome(Y, W, e_hat, mu0_hat, mu1_hat)

    # ACF on pseudo-outcomes
    acf_psi = compute_acf(psi_dr, max_lag=10)

    print("  DR pseudo-outcome series:")
    print(f"    ACF at lag 1: {acf_psi[1]:.3f}")
    print(f"    ACF at lag 5: {acf_psi[5]:.3f}")

    # Diagnostics on residuals
    residuals = psi_dr - tau_hat
    acf_resid = compute_acf(residuals, max_lag=10)

    print("  DR residual series (psi - tau_hat):")
    print(f"    ACF at lag 1: {acf_resid[1]:.3f}")
    print(f"    ACF at lag 5: {acf_resid[5]:.3f}")
    print()

    # =========================================================================
    # Using diagnostics to inform ACI parameters
    # =========================================================================
    print("4. Using Diagnostics to Inform ACI Parameters")
    print("-" * 40)

    # Strong dependence suggests:
    # - Larger n_warmup (more data to stabilize)
    # - Smaller gamma (slower adaptation to avoid oscillation)

    lag1_acf = abs(acf_resid[1])

    if lag1_acf > 0.7:
        recommended_gamma = 0.001
        recommended_warmup_frac = 0.15
        dependence_level = "Strong"
    elif lag1_acf > 0.3:
        recommended_gamma = 0.005
        recommended_warmup_frac = 0.10
        dependence_level = "Moderate"
    else:
        recommended_gamma = 0.01
        recommended_warmup_frac = 0.05
        dependence_level = "Weak"

    print(f"  Lag-1 ACF of residuals: {lag1_acf:.3f}")
    print(f"  Dependence level: {dependence_level}")
    print(f"  Recommended gamma: {recommended_gamma}")
    print(f"  Recommended warmup fraction: {recommended_warmup_frac:.0%}")
    print()

    print("Notes:")
    print("- High ACF means observations are correlated over time")
    print("- This reduces effective sample size for quantile estimation")
    print("- Use smaller gamma to smooth ACI updates under strong dependence")
    print("- Beta-mixing decays exponentially for geometrically mixing processes")


if __name__ == "__main__":
    main()
