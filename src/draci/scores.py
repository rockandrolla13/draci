"""
Score functions for DR-ACI conformal inference.

Implements three score types:
- dr_score: Doubly robust conformity score |psi^DR - tau_hat|
- vs_dr_score: Variance-standardized DR score (tighter intervals)
- naive_score: Non-DR plug-in score (baseline)

References:
- Chernozhukov et al. (2018): Double/Debiased Machine Learning
- Lei & Candes (2021): Conformal Inference of Counterfactuals
"""

from __future__ import annotations

import numpy as np


def dr_pseudo_outcome(
    Y: np.ndarray,
    W: np.ndarray,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
) -> np.ndarray:
    """Compute doubly robust (AIPW) pseudo-outcome.

    psi^DR = W/e(X) * (Y - mu1(X)) - (1-W)/(1-e(X)) * (Y - mu0(X)) + mu1(X) - mu0(X)

    This is the influence-function-based pseudo-outcome that is consistent
    for the CATE if either the propensity OR outcome model is correct.

    Parameters
    ----------
    Y : array (T,)
        Observed outcomes
    W : array (T,)
        Treatment indicators (0 or 1)
    e_hat : array (T,)
        Estimated propensity scores P(W=1|X)
    mu0_hat : array (T,)
        Estimated E[Y|X, W=0]
    mu1_hat : array (T,)
        Estimated E[Y|X, W=1]

    Returns
    -------
    psi_dr : array (T,)
        Doubly robust pseudo-outcomes (signed, not absolute value)
    """
    return (
        W / e_hat * (Y - mu1_hat)
        - (1 - W) / (1 - e_hat) * (Y - mu0_hat)
        + mu1_hat - mu0_hat
    )


def dr_score(
    Y: np.ndarray,
    W: np.ndarray,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    tau_hat: np.ndarray,
) -> np.ndarray:
    """Doubly robust conformity score |psi^DR - tau_hat|.

    This is the primary conformity score for DR-ACI. It measures the
    absolute deviation of the DR pseudo-outcome from the CATE estimate.

    Parameters
    ----------
    Y : array (T,)
        Observed outcomes
    W : array (T,)
        Treatment indicators
    e_hat : array (T,)
        Estimated propensity scores
    mu0_hat, mu1_hat : array (T,)
        Estimated outcome means by treatment arm
    tau_hat : array (T,)
        CATE point predictions

    Returns
    -------
    scores : array (T,)
        Conformity scores |psi^DR - tau_hat|
    """
    psi_dr = dr_pseudo_outcome(Y, W, e_hat, mu0_hat, mu1_hat)
    return np.abs(psi_dr - tau_hat)


def vs_dr_score(
    Y: np.ndarray,
    W: np.ndarray,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    tau_hat: np.ndarray,
    return_sigma: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Variance-standardized DR conformity score.

    S_t = |psi^DR - tau_hat| / sigma_hat(X_t)

    sigma_hat^2 is the influence-function variance estimate:
        Var(psi^DR | X) ~ 1/e(X) * Var(Y(1)|X) + 1/(1-e(X)) * Var(Y(0)|X)

    Variance standardization produces tighter intervals when the DR score
    variance is heterogeneous across observations.

    Parameters
    ----------
    Y, W, e_hat, mu0_hat, mu1_hat, tau_hat : arrays
        Same as dr_score
    return_sigma : bool
        If True, also return sigma_hat for interval rescaling

    Returns
    -------
    vs_scores : array (T,)
        Variance-standardized conformity scores
    sigma_hat : array (T,), optional
        Standard deviation estimates (if return_sigma=True)
    """
    psi_dr = dr_pseudo_outcome(Y, W, e_hat, mu0_hat, mu1_hat)

    # Estimate conditional variance from squared residuals
    res1 = (Y - mu1_hat) ** 2  # Proxy for Var(Y(1)|X)
    res0 = (Y - mu0_hat) ** 2  # Proxy for Var(Y(0)|X)

    # Influence function variance: sigma^2(x) = res1/e + res0/(1-e)
    sigma2_hat = res1 / e_hat + res0 / (1 - e_hat)

    # Numerical stability
    sigma_hat = np.sqrt(np.maximum(sigma2_hat, 1e-6))

    vs_scores = np.abs(psi_dr - tau_hat) / sigma_hat

    if return_sigma:
        return vs_scores, sigma_hat
    return vs_scores


def naive_score(
    Y: np.ndarray,
    W: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    tau_hat: np.ndarray,
) -> np.ndarray:
    """Non-DR conformity score: plug-in residual without IPW.

    This baseline score does not use inverse propensity weighting,
    making it inconsistent under treatment effect heterogeneity
    but potentially less variable than the DR score.

    Parameters
    ----------
    Y : array (T,)
        Observed outcomes
    W : array (T,)
        Treatment indicators
    mu0_hat, mu1_hat : array (T,)
        Estimated outcome means
    tau_hat : array (T,)
        CATE point predictions

    Returns
    -------
    scores : array (T,)
        Non-DR conformity scores
    """
    residual = np.where(W == 1, Y - mu1_hat, -(Y - mu0_hat))
    return np.abs(residual + (mu1_hat - mu0_hat) - tau_hat)
