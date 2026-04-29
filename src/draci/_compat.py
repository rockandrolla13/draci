"""
Optional integrations with external packages.

This module provides compatibility layers for:
- econml: CausalForestDML integration
- Other causal inference packages (future)

These integrations are optional and require additional dependencies.
Install with: pip install draci[econml]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class EconMLDRACIResult:
    """Result from DR-ACI with econml CausalForestDML baseline."""

    ticker: np.ndarray
    cate_point: np.ndarray
    cate_lower_draci: np.ndarray
    cate_upper_draci: np.ndarray
    cate_lower_baseline: np.ndarray  # From CausalForestDML
    cate_upper_baseline: np.ndarray
    conformity_scores: np.ndarray
    calibration_quantile: float
    alpha: float
    draci_interval_width: np.ndarray
    baseline_interval_width: np.ndarray


def conformal_calibration(
    residuals: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute conformal calibration quantile.

    For coverage 1-alpha, returns the (1-alpha)(1 + 1/n) quantile
    of the absolute residuals.

    Parameters
    ----------
    residuals : array
        Conformity scores (typically |AIPW - CATE_hat|)
    alpha : float
        Miscoverage rate

    Returns
    -------
    q : float
        Calibration quantile
    """
    n = len(residuals)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(np.abs(residuals), q_level)


def cross_conformal_with_econml(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    alpha: float = 0.05,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Cross-conformal (CV+) inference using econml for nuisance estimation.

    This is more data-efficient than split conformal:
    1. Use K-fold cross-validation
    2. For each fold, train on K-1 folds, calibrate on held-out fold
    3. Aggregate conformity scores across all folds

    Parameters
    ----------
    Y : array (n,)
        Observed outcomes
    T : array (n,)
        Treatment indicators (0 or 1)
    X : array (n, p)
        Covariates
    alpha : float
        Miscoverage rate
    n_folds : int
        Number of CV folds
    random_state : int
        Random seed

    Returns
    -------
    cate_point : CATE point estimates
    cate_lower : Lower conformal bound
    cate_upper : Upper conformal bound
    q : Calibration quantile
    conformity_scores : Conformity scores
    """
    try:
        from sklearn.model_selection import KFold
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    except ImportError as e:
        raise ImportError("This function requires scikit-learn") from e

    from .scores import dr_pseudo_outcome

    n = len(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cate_point = np.zeros(n)
    conformity_scores = np.zeros(n)
    aipw_scores = np.zeros(n)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]

        # Fit nuisance models on training fold
        mu0_train, mu1_train, e_train = _fit_nuisance_models(
            Y_train, T_train, X_train, random_state=random_state
        )

        # AIPW scores for training
        aipw_train = dr_pseudo_outcome(
            Y_train, T_train, e_train, mu0_train, mu1_train
        )

        # Fit CATE model (simple RF on AIPW)
        from sklearn.ensemble import RandomForestRegressor
        cate_model = RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=random_state, n_jobs=-1
        )
        cate_model.fit(X_train, aipw_train)

        # Predict on test fold
        cate_test = cate_model.predict(X_test)
        cate_point[test_idx] = cate_test

        # Compute test fold nuisance and AIPW
        mu0_test, mu1_test, e_test = _fit_nuisance_models(
            Y_test, T_test, X_test, random_state=random_state
        )
        aipw_test = dr_pseudo_outcome(
            Y_test, T_test, e_test, mu0_test, mu1_test
        )

        aipw_scores[test_idx] = aipw_test
        conformity_scores[test_idx] = np.abs(aipw_test - cate_test)

    # Calibration quantile from all conformity scores
    q = conformal_calibration(conformity_scores, alpha)

    cate_lower = cate_point - q
    cate_upper = cate_point + q

    return cate_point, cate_lower, cate_upper, q, conformity_scores


def _fit_nuisance_models(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit nuisance models with cross-fitting.

    Returns out-of-fold predictions for mu0, mu1, e.
    """
    from sklearn.model_selection import KFold
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

    n = len(Y)
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    e_hat = np.zeros(n)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, T_train = Y[train_idx], T[train_idx]

        # Propensity
        prop_model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        prop_model.fit(X_train, T_train)
        e_hat[test_idx] = np.clip(prop_model.predict_proba(X_test)[:, 1], 0.01, 0.99)

        # Outcome models
        treated_mask = T_train == 1
        control_mask = T_train == 0

        if treated_mask.sum() > 10:
            mu1_model = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
            )
            mu1_model.fit(X_train[treated_mask], Y_train[treated_mask])
            mu1_hat[test_idx] = mu1_model.predict(X_test)
        else:
            mu1_hat[test_idx] = Y_train[treated_mask].mean() if treated_mask.sum() > 0 else 0

        if control_mask.sum() > 10:
            mu0_model = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
            )
            mu0_model.fit(X_train[control_mask], Y_train[control_mask])
            mu0_hat[test_idx] = mu0_model.predict(X_test)
        else:
            mu0_hat[test_idx] = Y_train[control_mask].mean() if control_mask.sum() > 0 else 0

    return mu0_hat, mu1_hat, e_hat


def compute_baseline_intervals_econml(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute baseline CATE intervals using econml's CausalForestDML.

    These intervals rely on asymptotic normality and may have
    undercoverage in finite samples.

    Parameters
    ----------
    Y, T, X : arrays
        Outcome, treatment, covariates
    alpha : float
        Miscoverage rate
    random_state : int
        Random seed

    Returns
    -------
    cate_point, cate_lower, cate_upper : arrays
        Point estimates and asymptotic confidence bounds
    """
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    except ImportError as e:
        raise ImportError(
            "This function requires econml. Install with: pip install draci[econml]"
        ) from e

    est = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=random_state),
        model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_state),
        discrete_treatment=True,
        n_estimators=200,
        random_state=random_state,
        max_depth=5,
    )
    est.fit(Y, T, X=X)

    cate_point = est.effect(X).flatten()
    intervals = est.effect_interval(X, alpha=alpha)
    cate_lower = intervals[0].flatten()
    cate_upper = intervals[1].flatten()

    return cate_point, cate_lower, cate_upper
