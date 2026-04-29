#!/usr/bin/env python
"""
Empirical Panel Analysis with DR-ACI
====================================

This example shows how to use draci for empirical panel data analysis,
similar to the Dynamic M-ELO experiments in the paper.

The workflow:
1. Load your panel data (ticker x date)
2. Define treatment and outcome
3. Temporal block cross-fitting for nuisance estimation
4. Run DR-ACI and compare methods
5. Compute coverage metrics

Replace the synthetic data generation with your actual data loading.
"""

import numpy as np
import pandas as pd

from draci import (
    # Cross-fitting
    TemporalCrossFitter,
    LinearNuisance,
    # Scores
    dr_score,
    vs_dr_score,
    dr_pseudo_outcome,
    # Methods
    dr_aci,
    vs_dr_aci,
    aci,
    hac,
    # Mixing
    mixing_diagnostics_panel,
    # Baselines
    dml_wald_ci,
    block_bootstrap_ci,
)


def generate_synthetic_panel(
    n_tickers: int = 100,
    n_dates: int = 200,
    treatment_effect: float = 0.05,
    rho: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic panel data for demonstration.

    In practice, replace this with your data loading code.
    """
    rng = np.random.default_rng(seed)

    records = []
    for ticker in range(n_tickers):
        # Staggered treatment adoption (random date in middle third)
        adoption_date = rng.integers(n_dates // 3, 2 * n_dates // 3)

        # Ticker-specific baseline
        baseline = rng.normal(0, 0.5)

        # Generate AR(1) outcomes
        y_prev = baseline
        for t in range(n_dates):
            # Treatment indicator
            W = 1 if t >= adoption_date else 0

            # Covariates (could be richer in practice)
            x1 = rng.normal(0, 1)
            x2 = rng.normal(0, 1)

            # AR(1) error
            eps = rng.normal(0, 0.3)

            # Outcome: baseline + treatment effect + AR(1) noise
            Y = baseline + treatment_effect * W + rho * (y_prev - baseline) + eps
            y_prev = Y

            records.append({
                'ticker': f'TKR_{ticker:03d}',
                'date': t,
                'Y': Y,
                'W': W,
                'x1': x1,
                'x2': x2,
                'adoption_date': adoption_date,
            })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("Empirical Panel Analysis with DR-ACI")
    print("=" * 60)

    # =========================================================================
    # 1. Load/generate panel data
    # =========================================================================
    print("\n1. Loading panel data...")
    panel = generate_synthetic_panel(
        n_tickers=100,
        n_dates=200,
        treatment_effect=0.05,
        rho=0.7,
        seed=42,
    )

    n_obs = len(panel)
    n_tickers = panel['ticker'].nunique()
    n_dates = panel['date'].nunique()
    treatment_rate = panel['W'].mean()

    print(f"   Observations: {n_obs:,}")
    print(f"   Tickers: {n_tickers}")
    print(f"   Dates: {n_dates}")
    print(f"   Treatment rate: {treatment_rate:.1%}")

    # =========================================================================
    # 2. Prepare arrays
    # =========================================================================
    print("\n2. Preparing arrays...")

    # Sort by date for temporal structure
    panel = panel.sort_values(['date', 'ticker']).reset_index(drop=True)

    X = panel[['x1', 'x2']].values
    W = panel['W'].values.astype(float)
    Y = panel['Y'].values.astype(float)

    # Time index for cross-fitting (use date as integer)
    time_idx = panel['date'].values

    print(f"   X shape: {X.shape}")
    print(f"   Y range: [{Y.min():.3f}, {Y.max():.3f}]")

    # =========================================================================
    # 3. Temporal block cross-fitting
    # =========================================================================
    print("\n3. Temporal block cross-fitting (K=5 blocks)...")

    fitter = TemporalCrossFitter(
        n_blocks=5,
        nuisance_estimator=LinearNuisance(clip_propensity=(0.05, 0.95)),
    )
    cf_result = fitter.fit_transform(X, W, Y, time_index=time_idx)

    e_hat = cf_result.e_hat
    mu0_hat = cf_result.mu0_hat
    mu1_hat = cf_result.mu1_hat
    tau_hat = cf_result.tau_hat

    print(f"   Propensity range: [{e_hat.min():.3f}, {e_hat.max():.3f}]")
    print(f"   CATE estimate mean: {tau_hat.mean():.4f}")

    # =========================================================================
    # 4. Compute scores
    # =========================================================================
    print("\n4. Computing DR scores...")

    dr_scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
    vs_scores, sigma_hat = vs_dr_score(
        Y, W, e_hat, mu0_hat, mu1_hat, tau_hat, return_sigma=True
    )
    psi_dr = dr_pseudo_outcome(Y, W, e_hat, mu0_hat, mu1_hat)

    # For empirical work, we don't have true tau, so we use ATT as reference
    ATT_estimate = psi_dr[W == 1].mean()  # Simple ATT
    print(f"   ATT estimate: {ATT_estimate:.4f}")

    # Pseudo-residuals for coverage (distance from ATT)
    pseudo_residuals = np.abs(psi_dr - ATT_estimate)
    pseudo_residuals_std = pseudo_residuals / sigma_hat

    # =========================================================================
    # 5. Run conformal methods
    # =========================================================================
    print("\n5. Running conformal methods...")

    alpha = 0.10
    gamma = 0.005
    n_warmup = max(int(len(Y) * 0.1), 100)

    results = {}

    # DR-ACI
    res = dr_aci(dr_scores, pseudo_residuals, alpha=alpha, gamma=gamma, n_warmup=n_warmup)
    results['DR-ACI'] = res

    # VS-DR-ACI
    res = vs_dr_aci(vs_scores, pseudo_residuals_std, pseudo_residuals,
                    alpha=alpha, gamma=gamma, n_warmup=n_warmup)
    results['VS-DR-ACI'] = res

    # ACI (non-DR)
    res = aci(dr_scores, pseudo_residuals, alpha=alpha, gamma=gamma, n_warmup=n_warmup)
    results['ACI'] = res

    # HAC
    res = hac(psi_dr, tau_hat, pseudo_residuals, alpha=alpha, bandwidth=10)
    results['HAC'] = res

    print(f"\n   {'Method':<12} {'Self-Cov':>10} {'Width':>10}")
    print("   " + "-" * 34)
    for method, res in results.items():
        print(f"   {method:<12} {res.coverage:>10.1%} {res.avg_width:>10.4f}")

    # =========================================================================
    # 6. CI Baselines
    # =========================================================================
    print("\n6. CI Baselines...")

    wald_res = dml_wald_ci(psi_dr, tau_hat, pseudo_residuals, X, alpha=alpha)
    boot_res = block_bootstrap_ci(psi_dr, tau_hat, pseudo_residuals, alpha=alpha)

    print(f"   DML-Wald:        Cov={wald_res.coverage:.1%}, Width={wald_res.avg_width:.4f}")
    print(f"   Block Bootstrap: Cov={boot_res.coverage:.1%}, Width={boot_res.avg_width:.4f}")

    # =========================================================================
    # 7. Mixing diagnostics
    # =========================================================================
    print("\n7. Mixing diagnostics...")

    # Reshape to panel for mixing analysis
    residuals_by_ticker = {}
    for ticker in panel['ticker'].unique()[:20]:  # Sample of tickers
        mask = panel['ticker'] == ticker
        residuals_by_ticker[ticker] = psi_dr[mask]

    diag = mixing_diagnostics_panel(residuals_by_ticker, lags=[1, 5, 10, 20])
    print(f"   ACF lag-1 median: {diag.acf_lag1_median:.3f}")
    print(f"   Optimal mixing gap: {diag.optimal_gap:.3f}")

    print("\n" + "=" * 60)
    print("Empirical analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
