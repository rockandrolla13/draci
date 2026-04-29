"""
Data generating process for DR-ACI simulations.

Implements the DGP from Section 5 of the paper:
  - Covariates: 5D AR(1) process X_t = rho * X_{t-1} + sqrt(1-rho^2) * eps_t
  - Propensity: e(x) = expit(0.5 + 0.8*x1 - 0.3*x1^2 + 0.4*x2)
  - CATE: tau(x) = sin(2*pi*x1) + 0.5*x2
  - Outcome: mu_w(x) = 2 + x1 + 0.5*x2^2 + w*tau(x)
  - Error: u_t = phi*u_{t-1} + sigma_t*eta_t (AR(1) + GARCH(1,1))

Usage:
    from draci.dgp import AR1DGP

    dgp = AR1DGP(rho=0.9)
    data = dgp.generate(T=2000, seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# Default DGP parameters matching the paper
DEFAULT_DIM = 5
DEFAULT_GARCH_OMEGA = 0.1
DEFAULT_GARCH_ALPHA = 0.3
DEFAULT_GARCH_BETA = 0.5
DEFAULT_AR_ERROR_COEF = 0.5
DEFAULT_PROPENSITY_CLIP = (0.05, 0.95)


def _expit(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


@dataclass
class DGPData:
    """Container for generated DGP data."""

    X: np.ndarray  # (T, dim) covariates
    W: np.ndarray  # (T,) treatment indicators
    Y: np.ndarray  # (T,) observed outcomes
    e_true: np.ndarray  # (T,) true propensity scores
    mu0_true: np.ndarray  # (T,) true E[Y(0)|X]
    mu1_true: np.ndarray  # (T,) true E[Y(1)|X]
    tau_true: np.ndarray  # (T,) true CATE

    @property
    def T(self) -> int:
        """Sample size."""
        return len(self.Y)

    @property
    def dim(self) -> int:
        """Covariate dimension."""
        return self.X.shape[1]


@dataclass
class AR1DGP:
    """AR(1) covariate DGP with GARCH(1,1) errors.

    This is the main DGP used in the DR-ACI paper simulations.

    Parameters
    ----------
    rho : float
        AR(1) coefficient for covariates (0 = iid, 0.95 = strong dependence)
    dim : int
        Covariate dimension (default: 5)
    garch_omega : float
        GARCH unconditional variance intercept
    garch_alpha : float
        GARCH ARCH coefficient
    garch_beta : float
        GARCH persistence coefficient
    ar_error_coef : float
        AR(1) coefficient for error process
    propensity_clip : tuple
        (min, max) bounds for propensity clipping

    Example
    -------
    >>> dgp = AR1DGP(rho=0.9)
    >>> data = dgp.generate(T=2000, seed=42)
    >>> print(f"Coverage target: {data.tau_true.mean():.3f}")
    """

    rho: float = 0.0
    dim: int = DEFAULT_DIM
    garch_omega: float = DEFAULT_GARCH_OMEGA
    garch_alpha: float = DEFAULT_GARCH_ALPHA
    garch_beta: float = DEFAULT_GARCH_BETA
    ar_error_coef: float = DEFAULT_AR_ERROR_COEF
    propensity_clip: tuple[float, float] = DEFAULT_PROPENSITY_CLIP

    def generate_covariates(
        self, T: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate AR(1) covariates.

        X_t = rho * X_{t-1} + sqrt(1 - rho^2) * eps_t
        where eps_t ~ N(0, I_dim).

        Returns array of shape (T, dim).
        """
        innovation_sd = np.sqrt(max(1.0 - self.rho**2, 1e-8))
        X = np.zeros((T, self.dim))
        X[0] = rng.normal(0, 1, self.dim)
        for t in range(1, T):
            X[t] = self.rho * X[t - 1] + innovation_sd * rng.normal(0, 1, self.dim)
        return X

    def propensity(self, X: np.ndarray) -> np.ndarray:
        """True propensity: e(x) = expit(0.5 + 0.8*x1 - 0.3*x1^2 + 0.4*x2).

        Nonlinear in x1 (quadratic confounding), uses x2 for extra variation.
        """
        logit = 0.5 + 0.8 * X[:, 0] - 0.3 * X[:, 0]**2 + 0.4 * X[:, 1]
        lo, hi = self.propensity_clip
        return np.clip(_expit(logit), lo, hi)

    def cate(self, X: np.ndarray) -> np.ndarray:
        """True CATE: tau(x) = sin(2*pi*x1) + 0.5*x2.

        Strongly nonlinear in x1 via sine; linear in x2.
        """
        return np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 1]

    def outcome_mean(self, X: np.ndarray, w: int) -> np.ndarray:
        """Conditional mean: mu_w(x) = 2 + x1 + 0.5*x2^2 + w*tau(x).

        mu0 is nonlinear only through x2^2; mu1 adds the nonlinear CATE.
        """
        base = 2.0 + X[:, 0] + 0.5 * X[:, 1]**2
        if w == 1:
            return base + self.cate(X)
        return base

    def generate_garch_errors(
        self, T: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate GARCH(1,1) errors with AR(1) in level.

        u_t = phi * u_{t-1} + sigma_t * eta_t
        sigma_t^2 = omega + alpha * u_{t-1}^2 + beta * sigma_{t-1}^2
        """
        phi = self.ar_error_coef
        omega = self.garch_omega
        alpha = self.garch_alpha
        beta = self.garch_beta

        u = np.zeros(T)
        sigma2 = np.zeros(T)
        eta = rng.normal(0, 1, T)

        # Initialize at unconditional variance
        sigma2_unc = omega / max(1.0 - alpha - beta, 0.01)
        sigma2[0] = sigma2_unc
        u[0] = np.sqrt(sigma2[0]) * eta[0]

        for t in range(1, T):
            sigma2[t] = omega + alpha * u[t - 1]**2 + beta * sigma2[t - 1]
            u[t] = phi * u[t - 1] + np.sqrt(sigma2[t]) * eta[t]

        return u

    def generate(
        self, T: int, seed: int | None = None, rng: np.random.Generator | None = None
    ) -> DGPData:
        """Generate complete DGP data for one simulation trial.

        Parameters
        ----------
        T : int
            Sample size
        seed : int, optional
            Random seed (ignored if rng provided)
        rng : np.random.Generator, optional
            Random generator (created from seed if not provided)

        Returns
        -------
        DGPData with X, W, Y, and true nuisance functions
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        X = self.generate_covariates(T, rng)
        e_true = self.propensity(X)
        tau = self.cate(X)
        mu0 = self.outcome_mean(X, 0)
        mu1 = self.outcome_mean(X, 1)

        W = rng.binomial(1, e_true)
        u = self.generate_garch_errors(T, rng)

        Y0 = mu0 + u
        Y1 = mu1 + u
        Y = np.where(W == 1, Y1, Y0)

        return DGPData(
            X=X,
            W=W.astype(float),
            Y=Y,
            e_true=e_true,
            mu0_true=mu0,
            mu1_true=mu1,
            tau_true=tau,
        )


# Convenience functions matching paper's functional interface
def generate_covariates(
    T: int, rho: float, rng: np.random.Generator, dim: int = DEFAULT_DIM
) -> np.ndarray:
    """Generate AR(1) covariates (functional interface)."""
    dgp = AR1DGP(rho=rho, dim=dim)
    return dgp.generate_covariates(T, rng)


def propensity_true(X: np.ndarray) -> np.ndarray:
    """True propensity score (functional interface)."""
    dgp = AR1DGP()
    return dgp.propensity(X)


def cate_true(X: np.ndarray) -> np.ndarray:
    """True CATE (functional interface)."""
    dgp = AR1DGP()
    return dgp.cate(X)


def outcome_mean(X: np.ndarray, w: int) -> np.ndarray:
    """Conditional outcome mean (functional interface)."""
    dgp = AR1DGP()
    return dgp.outcome_mean(X, w)


def generate_garch_errors(T: int, rng: np.random.Generator) -> np.ndarray:
    """GARCH(1,1) errors (functional interface)."""
    dgp = AR1DGP()
    return dgp.generate_garch_errors(T, rng)


def generate_data(T: int, rho: float, rng: np.random.Generator) -> dict:
    """Generate complete DGP (functional interface returning dict)."""
    dgp = AR1DGP(rho=rho)
    data = dgp.generate(T, rng=rng)
    return {
        'X': data.X,
        'W': data.W,
        'Y': data.Y,
        'e_true': data.e_true,
        'mu0_true': data.mu0_true,
        'mu1_true': data.mu1_true,
        'tau_true': data.tau_true,
    }
