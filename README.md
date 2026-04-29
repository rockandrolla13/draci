# draci

**Doubly Robust Adaptive Conformal Inference for CATEs**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for constructing prediction intervals on Conditional Average Treatment Effects (CATEs) with finite-sample coverage guarantees under temporal dependence.

## Features

- **Doubly robust pseudo-outcomes** for consistent CATE estimation even with model misspecification
- **Adaptive conformal inference** for finite-sample coverage that adapts online to distribution shift
- **Temporal block cross-fitting** for handling dependent data (beta-mixing time series)
- **Variance-standardized scores** for tighter intervals under heterogeneous variance
- **9 conformal methods** for comprehensive benchmarking

## Installation

```bash
pip install draci
```

With ML dependencies (recommended):
```bash
pip install draci[ml]
```

For development:
```bash
git clone https://github.com/rockandrolla13/draci.git
cd draci
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from draci import DRACI, fit_nuisances, generate_data

# Generate synthetic data with temporal dependence
data = generate_data(T=1000, rho=0.7, seed=42)
X, W, Y = data["X"], data["W"], data["Y"]

# Split into train/calibration
n_train = 500
X_train, W_train, Y_train = X[:n_train], W[:n_train], Y[:n_train]
X_cal, W_cal, Y_cal = X[n_train:], W[n_train:], Y[n_train:]

# Fit nuisance models
nuisance = fit_nuisances(X_train, W_train, Y_train, method="xgboost")

# Get predictions on calibration set
e_hat = nuisance.e_hat(X_cal)
mu0_hat = nuisance.mu0_hat(X_cal)
mu1_hat = nuisance.mu1_hat(X_cal)
tau_hat = nuisance.tau_hat(X_cal)

# Run DR-ACI
model = DRACI(alpha=0.1, score_type="vs_dr")
result = model.fit_predict(Y_cal, W_cal, e_hat, mu0_hat, mu1_hat, tau_hat)

print(f"Average interval width: {result.width.mean():.3f}")
```

## Implemented Methods

| Method | Function | Description |
|--------|----------|-------------|
| **DR-ACI** | `dr_aci()` | Doubly robust adaptive conformal inference |
| **VS-DR-ACI** | `vs_dr_aci()` | Variance-standardized DR-ACI for tighter intervals |
| **ACI** | `aci()` | Standard adaptive conformal inference |
| **Split Conformal** | `split_conformal()` | Fixed quantile from calibration set |
| **NexCP** | `nexcp()` | Weighted conformal with exponential decay |
| **ECI** | `eci()` | Smooth sigmoid feedback (Wu et al., 2025) |
| **Block CP** | `block_cp()` | Block-permutation conformal |
| **HAC** | `hac()` | Newey-West asymptotic confidence intervals |

## Core Components

### Nuisance Estimation

```python
from draci import fit_nuisances, XGBoostNuisance, LinearNuisance

# Using convenience function
nuisance = fit_nuisances(X, W, Y, method="xgboost")

# Or with custom estimator
estimator = XGBoostNuisance(propensity_clip=(0.05, 0.95))
nuisance = estimator.fit(X, W, Y)
```

### Temporal Cross-Fitting

For dependent data, use temporal block cross-fitting to avoid data leakage:

```python
from draci import TemporalCrossFitter, LinearNuisance

fitter = TemporalCrossFitter(n_blocks=5, nuisance_estimator=LinearNuisance())
result = fitter.fit_transform(X, W, Y)

# Out-of-block predictions
e_hat = result.e_hat
tau_hat = result.tau_hat
```

### Mixing Diagnostics

Estimate beta-mixing coefficients to assess temporal dependence:

```python
from draci import mixing_diagnostics

diag = mixing_diagnostics(time_series, lags=[1, 5, 10, 20])
print(f"ACF at lag 1: {diag.acf_lag1_median:.3f}")
print(f"Optimal mixing gap: {diag.optimal_gap:.3f}")
```

### CI Baselines

Compare against standard confidence interval methods:

```python
from draci import dml_wald_ci, block_bootstrap_ci

# DML-Wald intervals
wald_result = dml_wald_ci(psi_dr, tau_hat, true_residuals, X_cal, alpha=0.1)

# Block bootstrap
boot_result = block_bootstrap_ci(psi_dr, tau_hat, true_residuals, alpha=0.1)
```

## Data Generating Processes

For simulation studies, the package includes DGPs from the paper:

```python
from draci import AR1DGP, DriftDGP

# AR(1) process with GARCH errors
dgp = AR1DGP(rho=0.7)
data = dgp.generate(T=1000, seed=42)

# Drift regimes (covariate shift)
drift_dgp = DriftDGP(regime="C", delta=1.0)  # Mean shift at T/2
data = drift_dgp.generate(T=1000, seed=42)
```

## Package Structure

```
draci/
    __init__.py         # Public API
    core.py             # DRACI class and conformal methods
    scores.py           # DR score functions
    nuisance.py         # Nuisance estimators
    cross_fitting.py    # Temporal block cross-fitting
    mixing.py           # Beta-mixing diagnostics
    baselines.py        # CI comparison methods
    dgp.py              # AR(1) data generating process
    dgp_drift.py        # Drift regime DGPs
```

## References

- Gibbs & Candes (2021): [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170)
- Chernozhukov et al. (2018): [Double/Debiased Machine Learning](https://arxiv.org/abs/1608.00060)
- Lei & Candes (2021): [Conformal Inference of Counterfactuals](https://arxiv.org/abs/2006.06138)
- Barber et al. (2023): [Conformal Prediction Beyond Exchangeability](https://arxiv.org/abs/2202.13415)
- Wu et al. (2025): Error-Quantified Conformal Inference

## License

MIT License
