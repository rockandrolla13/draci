"""
Data generating processes for drift regime simulations.

Two regimes to stress-test DR-ACI under distribution shift:

Regime C (Covariate Drift Only):
    - iid covariates (rho=0) with mean shift at T/2
    - Phase 1: X ~ N(0, I_p)
    - Phase 2: X ~ N((delta, 0, ..., 0), I_p)

Regime D (Dependence + Drift):
    - AR(1) covariates (rho=0.95) with linear drift after T/2
    - X_t = rho * X_{t-1} + sqrt(1-rho^2) * eps_t + mu(t)
    - mu(t) = (t/T) * (delta, 0, ..., 0) * 1{t > T/2}

NOTE: Regime D is NOT covered by Theorem 1. The beta-mixing bound applies
to the dependence component. The drift component creates non-stationarity
that falls outside the theorem's assumptions.

Usage:
    from draci.dgp_drift import DriftDGP

    dgp = DriftDGP(regime="C", delta=1.0)
    data = dgp.generate(T=2000, seed=42)

    dgp = DriftDGP(regime="D", rho=0.95, delta=0.5)
    data = dgp.generate(T=2000, seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .dgp import (
    AR1DGP,
    DGPData,
    DEFAULT_DIM,
    DEFAULT_GARCH_OMEGA,
    DEFAULT_GARCH_ALPHA,
    DEFAULT_GARCH_BETA,
    DEFAULT_AR_ERROR_COEF,
    DEFAULT_PROPENSITY_CLIP,
)


@dataclass
class DriftDGPData(DGPData):
    """Extended DGP data container with drift metadata."""

    regime: str = "C"  # "C" or "D"
    delta: float = 0.0  # Drift magnitude
    rho: float = 0.0  # AR(1) coefficient


@dataclass
class DriftDGP:
    """DGP with distribution shift for stress-testing.

    Parameters
    ----------
    regime : {"C", "D"}
        C = iid + mean shift at T/2
        D = AR(1) + linear drift after T/2
    delta : float
        Drift magnitude in first coordinate
    rho : float
        AR(1) coefficient (only used in Regime D, default 0.95)
    dim : int
        Covariate dimension
    garch_omega, garch_alpha, garch_beta : float
        GARCH error parameters
    ar_error_coef : float
        AR(1) coefficient for error process
    propensity_clip : tuple
        Propensity clipping bounds

    Example
    -------
    >>> dgp = DriftDGP(regime="C", delta=1.0)
    >>> data = dgp.generate(T=2000, seed=42)
    >>> print(f"Regime: {data.regime}, Delta: {data.delta}")
    """

    regime: Literal["C", "D"] = "C"
    delta: float = 1.0
    rho: float = 0.95  # Only used for regime D
    dim: int = DEFAULT_DIM
    garch_omega: float = DEFAULT_GARCH_OMEGA
    garch_alpha: float = DEFAULT_GARCH_ALPHA
    garch_beta: float = DEFAULT_GARCH_BETA
    ar_error_coef: float = DEFAULT_AR_ERROR_COEF
    propensity_clip: tuple[float, float] = DEFAULT_PROPENSITY_CLIP

    def _base_dgp(self) -> AR1DGP:
        """Create base DGP for shared methods."""
        return AR1DGP(
            rho=0.0,  # Covariate generation handled separately
            dim=self.dim,
            garch_omega=self.garch_omega,
            garch_alpha=self.garch_alpha,
            garch_beta=self.garch_beta,
            ar_error_coef=self.ar_error_coef,
            propensity_clip=self.propensity_clip,
        )

    def generate_covariates_regime_c(
        self, T: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate iid covariates with mean shift at T/2 (Regime C).

        Phase 1 (t = 0 to T//2 - 1): X ~ N(0, I_dim)
        Phase 2 (t = T//2 to T - 1): X ~ N(mu_drift, I_dim)
        """
        X = rng.normal(0, 1, size=(T, self.dim))

        # Apply drift to Phase 2
        midpoint = T // 2
        mu_drift = np.zeros(self.dim)
        mu_drift[0] = self.delta
        X[midpoint:] += mu_drift

        return X

    def generate_covariates_regime_d(
        self, T: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate AR(1) covariates with linear drift after T/2 (Regime D).

        X_t = rho * X_{t-1} + sqrt(1-rho^2) * eps_t + mu(t)
        where mu(t) = (t/T) * (delta, 0, ..., 0) * 1{t > T/2}
        """
        innovation_sd = np.sqrt(max(1.0 - self.rho**2, 1e-8))
        X = np.zeros((T, self.dim))
        X[0] = rng.normal(0, 1, self.dim)

        midpoint = T // 2

        for t in range(1, T):
            # AR(1) dynamics
            eps = rng.normal(0, 1, self.dim)
            X[t] = self.rho * X[t - 1] + innovation_sd * eps

            # Add linear drift after midpoint
            if t > midpoint:
                drift_factor = (t / T) * self.delta
                X[t, 0] += drift_factor

        return X

    def generate(
        self, T: int, seed: int | None = None, rng: np.random.Generator | None = None
    ) -> DriftDGPData:
        """Generate complete DGP data with drift.

        Parameters
        ----------
        T : int
            Sample size
        seed : int, optional
            Random seed
        rng : np.random.Generator, optional
            Random generator

        Returns
        -------
        DriftDGPData with regime metadata
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        # Generate covariates based on regime
        if self.regime == "C":
            X = self.generate_covariates_regime_c(T, rng)
            effective_rho = 0.0
        else:  # Regime D
            X = self.generate_covariates_regime_d(T, rng)
            effective_rho = self.rho

        # Use base DGP for shared transformations
        base = self._base_dgp()
        e_true = base.propensity(X)
        tau = base.cate(X)
        mu0 = base.outcome_mean(X, 0)
        mu1 = base.outcome_mean(X, 1)

        W = rng.binomial(1, e_true)
        u = base.generate_garch_errors(T, rng)

        Y0 = mu0 + u
        Y1 = mu1 + u
        Y = np.where(W == 1, Y1, Y0)

        return DriftDGPData(
            X=X,
            W=W.astype(float),
            Y=Y,
            e_true=e_true,
            mu0_true=mu0,
            mu1_true=mu1,
            tau_true=tau,
            regime=self.regime,
            delta=self.delta,
            rho=effective_rho,
        )


# Convenience functions matching paper's interface
def generate_data_regime_c(
    T: int, delta: float, rng: np.random.Generator
) -> dict:
    """Generate Regime C data (functional interface)."""
    dgp = DriftDGP(regime="C", delta=delta)
    data = dgp.generate(T, rng=rng)
    return {
        'X': data.X,
        'W': data.W,
        'Y': data.Y,
        'e_true': data.e_true,
        'mu0_true': data.mu0_true,
        'mu1_true': data.mu1_true,
        'tau_true': data.tau_true,
        'regime': 'C',
        'delta': delta,
        'rho': 0.0,
    }


def generate_data_regime_d(
    T: int, rho: float, delta: float, rng: np.random.Generator
) -> dict:
    """Generate Regime D data (functional interface)."""
    dgp = DriftDGP(regime="D", rho=rho, delta=delta)
    data = dgp.generate(T, rng=rng)
    return {
        'X': data.X,
        'W': data.W,
        'Y': data.Y,
        'e_true': data.e_true,
        'mu0_true': data.mu0_true,
        'mu1_true': data.mu1_true,
        'tau_true': data.tau_true,
        'regime': 'D',
        'delta': delta,
        'rho': rho,
    }
