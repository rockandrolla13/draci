#!/usr/bin/env python
"""
Using the DRACI Class
=====================

This example demonstrates the high-level DRACI class interface,
which wraps score computation and ACI into a single object.

The DRACI class supports three score types:
- "dr": Standard doubly robust scores (default, wider but more robust)
- "vs_dr": Variance-standardized DR scores (tighter intervals)
- "naive": Non-DR plug-in scores (baseline, not recommended)
"""

import numpy as np

from draci import (
    DRACI,
    generate_data,
    fit_nuisances,
)


def main():
    # Generate data
    rng = np.random.default_rng(42)
    data = generate_data(T=1000, rho=0.5, rng=rng)
    X, W, Y = data["X"], data["W"], data["Y"]
    tau_true = data["tau_true"]

    # Fit nuisances on full data (for simplicity - use cross-fitting in practice)
    # fit_nuisances returns a NuisanceFunctions object
    nuisance_fns = fit_nuisances(X, W, Y, method="linear")
    e_hat = nuisance_fns.e_hat(X)
    mu0_hat = nuisance_fns.mu0_hat(X)
    mu1_hat = nuisance_fns.mu1_hat(X)
    tau_hat = nuisance_fns.tau_hat(X)

    print("=" * 60)
    print("Comparing DR-ACI Score Types")
    print("=" * 60)
    print()

    # =========================================================================
    # Standard DR scores
    # =========================================================================
    draci_dr = DRACI(
        alpha=0.1,           # 90% coverage target
        gamma=0.005,         # ACI learning rate
        n_warmup=50,         # Warmup period
        score_type="dr",     # Standard DR scores
    )

    result_dr = draci_dr.fit_predict(
        Y, W, e_hat, mu0_hat, mu1_hat, tau_hat,
        tau_true=tau_true,  # Optional: for coverage evaluation
    )

    # Result contains interval bounds
    print("Standard DR Scores (score_type='dr'):")
    print(f"  Coverage: {result_dr.coverage_t.mean():.1%}")
    print(f"  Mean width: {result_dr.width[50:].mean():.4f}")
    print(f"  Median width: {np.median(result_dr.width[50:]):.4f}")
    print()

    # =========================================================================
    # Variance-standardized DR scores
    # =========================================================================
    draci_vs = DRACI(
        alpha=0.1,
        gamma=0.005,
        n_warmup=50,
        score_type="vs_dr",  # Variance-standardized
    )

    result_vs = draci_vs.fit_predict(
        Y, W, e_hat, mu0_hat, mu1_hat, tau_hat,
        tau_true=tau_true,
    )

    print("Variance-Standardized DR Scores (score_type='vs_dr'):")
    print(f"  Coverage: {result_vs.coverage_t.mean():.1%}")
    print(f"  Mean width: {result_vs.width[50:].mean():.4f}")
    print(f"  Median width: {np.median(result_vs.width[50:]):.4f}")
    print()

    # =========================================================================
    # Naive scores (baseline)
    # =========================================================================
    draci_naive = DRACI(
        alpha=0.1,
        gamma=0.005,
        n_warmup=50,
        score_type="naive",
    )

    result_naive = draci_naive.fit_predict(
        Y, W, e_hat, mu0_hat, mu1_hat, tau_hat,
        tau_true=tau_true,
    )

    print("Naive Scores (score_type='naive'):")
    print(f"  Coverage: {result_naive.coverage_t.mean():.1%}")
    print(f"  Mean width: {result_naive.width[50:].mean():.4f}")
    print()

    # =========================================================================
    # Using intervals
    # =========================================================================
    print("=" * 60)
    print("Working with Intervals")
    print("=" * 60)
    print()

    # Access interval bounds
    lower = result_dr.lower  # Lower bounds (T,)
    upper = result_dr.upper  # Upper bounds (T,)
    point = result_dr.point  # Point estimates (tau_hat)

    # Example: intervals for first 5 observations after warmup
    print("First 5 intervals (after warmup):")
    for i in range(50, 55):
        print(f"  t={i}: tau_hat={point[i]:.3f}, "
              f"CI=[{lower[i]:.3f}, {upper[i]:.3f}], "
              f"width={result_dr.width[i]:.3f}")
    print()

    # Example: coverage for specific observations
    if result_dr.coverage_t is not None:
        print("Coverage indicators (1=covered, 0=not):")
        print(f"  First 20 after warmup: {result_dr.coverage_t[:20].astype(int).tolist()}")


if __name__ == "__main__":
    main()
