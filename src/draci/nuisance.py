"""
Nuisance estimator framework for DR-ACI.

Provides a Protocol for nuisance estimation and concrete implementations
using XGBoost/RandomForest (production) and linear models (fast testing).

The nuisance models estimate:
- e(X): Propensity score P(W=1|X)
- mu0(X): Outcome regression E[Y|X, W=0]
- mu1(X): Outcome regression E[Y|X, W=1]
- tau(X): CATE estimate (optional, can be derived from mu1 - mu0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable, runtime_checkable

import numpy as np


@dataclass
class NuisanceResult:
    """Container for fitted nuisance predictions."""

    e_hat: np.ndarray  # P(W=1|X) propensity scores
    mu0_hat: np.ndarray  # E[Y|X, W=0] control outcome
    mu1_hat: np.ndarray  # E[Y|X, W=1] treated outcome
    tau_hat: np.ndarray  # CATE estimate


@runtime_checkable
class NuisanceEstimator(Protocol):
    """Protocol for nuisance function estimators.

    Implementations must provide a fit() method that returns
    callable prediction functions for each nuisance component.
    """

    def fit(
        self,
        X: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
    ) -> NuisanceFunctions:
        """Fit nuisance models on training data.

        Parameters
        ----------
        X : array (n, p)
            Covariates
        W : array (n,)
            Treatment indicators (0 or 1)
        Y : array (n,)
            Observed outcomes

        Returns
        -------
        NuisanceFunctions with callable predictors
        """
        ...


@dataclass
class NuisanceFunctions:
    """Callable nuisance functions for out-of-sample prediction."""

    e_hat: Callable[[np.ndarray], np.ndarray]
    mu0_hat: Callable[[np.ndarray], np.ndarray]
    mu1_hat: Callable[[np.ndarray], np.ndarray]
    tau_hat: Callable[[np.ndarray], np.ndarray]

    def predict(self, X: np.ndarray) -> NuisanceResult:
        """Generate predictions for new data."""
        return NuisanceResult(
            e_hat=self.e_hat(X),
            mu0_hat=self.mu0_hat(X),
            mu1_hat=self.mu1_hat(X),
            tau_hat=self.tau_hat(X),
        )


class XGBoostNuisance:
    """XGBoost-based nuisance estimator.

    Uses XGBClassifier for propensity, XGBRegressor for outcomes,
    and RandomForest for CATE.

    Parameters
    ----------
    propensity_params : dict, optional
        XGBClassifier parameters
    outcome_params : dict, optional
        XGBRegressor parameters
    cate_params : dict, optional
        RandomForestRegressor parameters
    clip_propensity : tuple
        (min, max) bounds for propensity clipping (overlap assumption)
    """

    def __init__(
        self,
        propensity_params: dict | None = None,
        outcome_params: dict | None = None,
        cate_params: dict | None = None,
        clip_propensity: tuple[float, float] = (0.05, 0.95),
    ):
        self.propensity_params = propensity_params or {
            "max_depth": 3,
            "n_estimators": 100,
            "learning_rate": 0.1,
        }
        self.outcome_params = outcome_params or {
            "max_depth": 3,
            "n_estimators": 100,
            "learning_rate": 0.1,
        }
        self.cate_params = cate_params or {
            "n_estimators": 100,
            "max_depth": 5,
        }
        self.clip_propensity = clip_propensity

    def fit(
        self,
        X: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
    ) -> NuisanceFunctions:
        """Fit XGBoost nuisance models."""
        try:
            from xgboost import XGBClassifier, XGBRegressor
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as e:
            raise ImportError(
                "XGBoostNuisance requires xgboost and scikit-learn. "
                "Install with: pip install draci[ml]"
            ) from e

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        lo, hi = self.clip_propensity

        # Propensity model
        prop_model = XGBClassifier(
            **self.propensity_params,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
        )
        prop_model.fit(X, W)

        def e_hat(X_new: np.ndarray) -> np.ndarray:
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            return np.clip(prop_model.predict_proba(X_new)[:, 1], lo, hi)

        # Outcome models (separate per arm)
        idx1 = W == 1
        idx0 = W == 0

        mu1_model = XGBRegressor(**self.outcome_params, verbosity=0, n_jobs=1)
        mu1_model.fit(X[idx1], Y[idx1])

        mu0_model = XGBRegressor(**self.outcome_params, verbosity=0, n_jobs=1)
        mu0_model.fit(X[idx0], Y[idx0])

        def mu1_hat(X_new: np.ndarray) -> np.ndarray:
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            return mu1_model.predict(X_new)

        def mu0_hat(X_new: np.ndarray) -> np.ndarray:
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            return mu0_model.predict(X_new)

        # CATE: RF on plug-in pseudo-outcome
        tau_pseudo = mu1_hat(X) - mu0_hat(X)
        tau_model = RandomForestRegressor(
            **self.cate_params,
            n_jobs=1,
            random_state=0,
        )
        tau_model.fit(X, tau_pseudo)

        def tau_hat(X_new: np.ndarray) -> np.ndarray:
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            return tau_model.predict(X_new)

        return NuisanceFunctions(
            e_hat=e_hat,
            mu0_hat=mu0_hat,
            mu1_hat=mu1_hat,
            tau_hat=tau_hat,
        )


class LinearNuisance:
    """Linear nuisance estimators (fast baseline for testing).

    Uses logistic regression for propensity and OLS for outcomes.

    Parameters
    ----------
    clip_propensity : tuple
        (min, max) bounds for propensity clipping
    """

    def __init__(
        self,
        clip_propensity: tuple[float, float] = (0.05, 0.95),
    ):
        self.clip_propensity = clip_propensity

    def fit(
        self,
        X: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
    ) -> NuisanceFunctions:
        """Fit linear nuisance models."""
        from scipy.optimize import minimize

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        lo, hi = self.clip_propensity
        d = X.shape[1]

        # Propensity: logistic regression
        X_aug = np.column_stack([np.ones(len(X)), X])

        def neg_loglik(beta):
            logits = X_aug @ beta
            logits = np.clip(logits, -30, 30)
            return -np.mean(W * logits - np.log(1 + np.exp(logits)))

        res = minimize(neg_loglik, np.zeros(d + 1), method="L-BFGS-B")
        beta_e = res.x

        def e_hat(X_new: np.ndarray) -> np.ndarray:
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            X_a = np.column_stack([np.ones(len(X_new)), X_new])
            logits = np.clip(X_a @ beta_e, -30, 30)
            return np.clip(1.0 / (1.0 + np.exp(-logits)), lo, hi)

        # OLS per arm
        idx1 = W == 1
        idx0 = W == 0

        X1_aug = np.column_stack([np.ones(idx1.sum()), X[idx1]])
        X0_aug = np.column_stack([np.ones(idx0.sum()), X[idx0]])
        beta1 = np.linalg.lstsq(X1_aug, Y[idx1], rcond=None)[0]
        beta0 = np.linalg.lstsq(X0_aug, Y[idx0], rcond=None)[0]

        def mu1_hat(X_new: np.ndarray) -> np.ndarray:
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            return np.column_stack([np.ones(len(X_new)), X_new]) @ beta1

        def mu0_hat(X_new: np.ndarray) -> np.ndarray:
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            return np.column_stack([np.ones(len(X_new)), X_new]) @ beta0

        def tau_hat(X_new: np.ndarray) -> np.ndarray:
            return mu1_hat(X_new) - mu0_hat(X_new)

        return NuisanceFunctions(
            e_hat=e_hat,
            mu0_hat=mu0_hat,
            mu1_hat=mu1_hat,
            tau_hat=tau_hat,
        )


def fit_nuisances(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    method: str = "xgboost",
    clip_propensity: tuple[float, float] = (0.05, 0.95),
) -> NuisanceFunctions:
    """Convenience function to fit nuisance estimators.

    Parameters
    ----------
    X, W, Y : arrays
        Training data
    method : {"xgboost", "linear"}
        Estimator type
    clip_propensity : tuple
        Propensity bounds

    Returns
    -------
    NuisanceFunctions with callable predictors
    """
    if method == "xgboost":
        estimator = XGBoostNuisance(clip_propensity=clip_propensity)
    elif method == "linear":
        estimator = LinearNuisance(clip_propensity=clip_propensity)
    else:
        raise ValueError(f"Unknown nuisance method: {method}")

    return estimator.fit(X, W, Y)
