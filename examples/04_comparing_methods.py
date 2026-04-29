#!/usr/bin/env python
"""
Comparing DR-ACI with CI Baselines
==================================

This example compares DR-ACI with traditional CI methods:
1. DML-Wald: Asymptotic normal CI with Newey-West adjustment
2. Block Bootstrap: Circular block bootstrap percentile CI

These methods are commonly used for causal inference under dependence,
but they rely on asymptotic approximations that may break down in
finite samples or under strong dependence.

DR-ACI provides finite-sample coverage guarantees that are more robust
to model misspecification and temporal dependence.
"""

import numpy as np

from draci import (
    # Core methods
    dr_aci,
    dr_pseudo_outcome,
    dr_score,
    # Baselines
    dml_wald_ci,
    block_bootstrap_ci,
    compute_ci_metrics,
    # Data and nuisance
    generate_data,
    fit_nuisances,
)


def main():
    print("=" * 60)
    print("Comparing DR-ACI with CI Baselines")
    print("=" * 60)
    print()

    # =========================================================================
    # Experiment across different dependence levels
    # =========================================================================
    rhos = [0.0, 0.5, 0.9]  # IID, moderate, strong dependence
    results = {}

    for rho in rhos:
        print(f"rho = {rho} (AR(1) coefficient)")
        print("-" * 40)

        # Generate data
        rng = np.random.default_rng(42)
        data = generate_data(T=500, rho=rho, rng=rng)
        X, W, Y = data["X"], data["W"], data["Y"]
        tau_true = data["tau_true"]

        # Train/calibration split
        n_train = 250
        X_train, W_train, Y_train = X[:n_train], W[:n_train], Y[:n_train]
        X_cal, W_cal, Y_cal = X[n_train:], W[n_train:], Y[n_train:]
        tau_cal = tau_true[n_train:]

        # Fit nuisances - returns NuisanceFunctions object
        nuisance_fns = fit_nuisances(
            X_train, W_train, Y_train, method="linear"
        )
        e_hat = nuisance_fns.e_hat(X_cal)
        mu0_hat = nuisance_fns.mu0_hat(X_cal)
        mu1_hat = nuisance_fns.mu1_hat(X_cal)
        tau_hat = nuisance_fns.tau_hat(X_cal)

        # Compute DR pseudo-outcomes and scores
        psi_dr = dr_pseudo_outcome(Y_cal, W_cal, e_hat, mu0_hat, mu1_hat)
        dr_scores = dr_score(Y_cal, W_cal, e_hat, mu0_hat, mu1_hat, tau_hat)
        true_residuals = np.abs(tau_cal - tau_hat)

        # ---------------------------------------------------------------------
        # DR-ACI
        # ---------------------------------------------------------------------
        draci_result = dr_aci(
            dr_scores, true_residuals,
            alpha=0.1, gamma=0.005, n_warmup=25
        )
        draci_metrics = compute_ci_metrics(
            draci_result.coverages_t, draci_result.avg_width, alpha=0.1
        )

        print(f"  DR-ACI:         Coverage={draci_result.coverage:.1%}, "
              f"Width={draci_result.avg_width:.3f}, "
              f"Gap={draci_metrics.miscoverage_gap:+.3f}")

        # ---------------------------------------------------------------------
        # DML-Wald CI
        # ---------------------------------------------------------------------
        dml_result = dml_wald_ci(
            psi_dr, tau_hat, true_residuals, X_cal, alpha=0.1
        )
        dml_metrics = compute_ci_metrics(
            dml_result.coverages_t, dml_result.avg_width, alpha=0.1
        )

        print(f"  DML-Wald:       Coverage={dml_result.coverage:.1%}, "
              f"Width={dml_result.avg_width:.3f}, "
              f"Gap={dml_metrics.miscoverage_gap:+.3f}")

        # ---------------------------------------------------------------------
        # Block Bootstrap CI
        # ---------------------------------------------------------------------
        boot_result = block_bootstrap_ci(
            psi_dr, tau_hat, true_residuals,
            alpha=0.1, B=199, rng=rng
        )
        boot_metrics = compute_ci_metrics(
            boot_result.coverages_t, boot_result.avg_width, alpha=0.1
        )

        print(f"  Block Bootstrap: Coverage={boot_result.coverage:.1%}, "
              f"Width={boot_result.avg_width:.3f}, "
              f"Gap={boot_metrics.miscoverage_gap:+.3f}")

        print()

        results[rho] = {
            "draci": (draci_result.coverage, draci_result.avg_width),
            "dml_wald": (dml_result.coverage, dml_result.avg_width),
            "block_boot": (boot_result.coverage, boot_result.avg_width),
        }

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("Method comparison across dependence levels:")
    print()
    print("                   rho=0.0      rho=0.5      rho=0.9")
    print("                   Cov   Wid    Cov   Wid    Cov   Wid")
    print("-" * 60)

    for method in ["draci", "dml_wald", "block_boot"]:
        label = {
            "draci": "DR-ACI",
            "dml_wald": "DML-Wald",
            "block_boot": "Block Boot",
        }[method]
        row = f"{label:14s}"
        for rho in rhos:
            cov, wid = results[rho][method]
            row += f"  {cov:.2f}  {wid:.2f} "
        print(row)

    print()
    print("Key observations:")
    print("- DR-ACI maintains coverage closer to nominal (90%) across rho values")
    print("- DML-Wald can undercover for high dependence (large positive gap)")
    print("- Block bootstrap is more robust but computationally expensive")
    print("- Gap = (1 - coverage) - alpha: positive = undercoverage")


if __name__ == "__main__":
    main()
