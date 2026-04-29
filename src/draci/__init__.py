"""
draci: Doubly Robust Adaptive Conformal Inference for CATEs

A package for constructing prediction intervals on Conditional Average
Treatment Effects (CATEs) with finite-sample coverage guarantees under
temporal dependence.

Key features:
- Doubly robust pseudo-outcomes for consistent CATE estimation
- Adaptive conformal inference for finite-sample coverage
- Temporal block cross-fitting for dependent data (beta-mixing)
- Variance-standardized scores for tighter intervals

Quick Start:
-----------
>>> from draci import DRACI, TemporalCrossFitter, XGBoostNuisance
>>>
>>> # 1. Cross-fit nuisance models
>>> fitter = TemporalCrossFitter(n_blocks=5)
>>> nuisance = fitter.fit_transform(X, W, Y)
>>>
>>> # 2. Run DR-ACI
>>> draci = DRACI(alpha=0.1, score_type="vs_dr")
>>> result = draci.fit_predict(
...     Y, W, nuisance.e_hat, nuisance.mu0_hat,
...     nuisance.mu1_hat, nuisance.tau_hat
... )
>>>
>>> # 3. Use intervals
>>> print(f"Mean width: {result.width.mean():.3f}")

References:
----------
- Gibbs & Candes (2021): Adaptive Conformal Inference Under Distribution Shift
- Chernozhukov et al. (2018): Double/Debiased Machine Learning
- Lei & Candes (2021): Conformal Inference of Counterfactuals
- Barber et al. (2023): Conformal Prediction Beyond Exchangeability
"""

__version__ = "0.1.0"

# Core classes and functions
from .core import (
    DRACI,
    ConformalResult,
    IntervalResult,
    aci,
    dr_aci,
    vs_dr_aci,
    split_conformal,
    nexcp,
    eci,
    block_cp,
    hac,
)

# Score functions
from .scores import (
    dr_score,
    vs_dr_score,
    naive_score,
    dr_pseudo_outcome,
)

# Nuisance estimation
from .nuisance import (
    NuisanceResult,
    NuisanceFunctions,
    NuisanceEstimator,
    XGBoostNuisance,
    LinearNuisance,
    fit_nuisances,
)

# Temporal cross-fitting
from .cross_fitting import (
    TemporalCrossFitter,
    CrossFitResult,
    temporal_block_crossfit,
)

# Mixing diagnostics
from .mixing import (
    MixingDiagnostics,
    mixing_diagnostics,
    mixing_diagnostics_panel,
    estimate_beta_mixing,
    compute_acf,
)

# Data generating processes
from .dgp import (
    AR1DGP,
    DGPData,
    generate_data,
    generate_covariates,
    propensity_true,
    cate_true,
    outcome_mean,
    generate_garch_errors,
)

from .dgp_drift import (
    DriftDGP,
    DriftDGPData,
    generate_data_regime_c,
    generate_data_regime_d,
)

# CI baselines
from .baselines import (
    CIMetrics,
    dml_wald_ci,
    block_bootstrap_ci,
    causal_forest_ci,
    compute_ci_metrics,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "DRACI",
    "ConformalResult",
    "IntervalResult",
    "aci",
    "dr_aci",
    "vs_dr_aci",
    "split_conformal",
    "nexcp",
    "eci",
    "block_cp",
    "hac",
    # Scores
    "dr_score",
    "vs_dr_score",
    "naive_score",
    "dr_pseudo_outcome",
    # Nuisance
    "NuisanceResult",
    "NuisanceFunctions",
    "NuisanceEstimator",
    "XGBoostNuisance",
    "LinearNuisance",
    "fit_nuisances",
    # Cross-fitting
    "TemporalCrossFitter",
    "CrossFitResult",
    "temporal_block_crossfit",
    # Mixing
    "MixingDiagnostics",
    "mixing_diagnostics",
    "mixing_diagnostics_panel",
    "estimate_beta_mixing",
    "compute_acf",
    # DGP
    "AR1DGP",
    "DGPData",
    "generate_data",
    "generate_covariates",
    "propensity_true",
    "cate_true",
    "outcome_mean",
    "generate_garch_errors",
    "DriftDGP",
    "DriftDGPData",
    "generate_data_regime_c",
    "generate_data_regime_d",
    # Baselines
    "CIMetrics",
    "dml_wald_ci",
    "block_bootstrap_ci",
    "causal_forest_ci",
    "compute_ci_metrics",
]
