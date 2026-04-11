"""draci: Doubly Robust Adaptive Conformal Inference for CATEs.

This package provides prediction intervals for Conditional Average Treatment
Effects (CATEs) with finite-sample coverage guarantees, designed for temporal
dependence (beta-mixing time series).

Main classes:
    DRACI: Core adaptive conformal inference algorithm
    TemporalCrossFitter: K-fold temporal block cross-fitting

Score functions:
    dr_score: Doubly robust conformity score
    vs_dr_score: Variance-standardized DR score
    naive_score: Plug-in score (baseline)

Example:
    >>> from draci import DRACI, XGBoostNuisance
    >>> nuisance = XGBoostNuisance().fit(X_train, W_train, Y_train)
    >>> draci = DRACI(alpha=0.1)
    >>> result = draci.fit_predict(Y, W, X, nuisance)
    >>> print(f"Coverage: {result.coverage:.2%}")
"""

__version__ = "0.1.0"

# TODO: Implement and export these
# from .core import DRACI, IntervalResult
# from .scores import dr_score, vs_dr_score, naive_score
# from .nuisance import XGBoostNuisance, LinearNuisance, NuisanceResult
# from .cross_fitting import TemporalCrossFitter

__all__ = [
    "__version__",
    # "DRACI",
    # "IntervalResult",
    # "dr_score",
    # "vs_dr_score",
    # "naive_score",
    # "XGBoostNuisance",
    # "LinearNuisance",
    # "NuisanceResult",
    # "TemporalCrossFitter",
]
