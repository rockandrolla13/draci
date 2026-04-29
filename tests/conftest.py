"""Shared pytest fixtures for draci tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_data(rng):
    """Generate simple synthetic data for testing.

    Returns dict with X, W, Y and true nuisance functions.
    """
    T = 200

    # Covariates (1D for simplicity)
    X = rng.normal(0, 1, T)

    # True propensity (simple logistic)
    logit_e = 0.3 * X
    e_true = 1 / (1 + np.exp(-logit_e))
    e_true = np.clip(e_true, 0.05, 0.95)

    # Treatment assignment
    W = rng.binomial(1, e_true).astype(float)

    # True outcome functions
    mu0_true = 1.0 + 0.5 * X
    tau_true = 0.5 + 0.3 * X
    mu1_true = mu0_true + tau_true

    # Observed outcome with noise
    sigma = 0.1
    Y = np.where(W == 1, mu1_true, mu0_true) + rng.normal(0, sigma, T)

    return {
        "X": X,
        "W": W,
        "Y": Y,
        "e_true": e_true,
        "mu0_true": mu0_true,
        "mu1_true": mu1_true,
        "tau_true": tau_true,
        "T": T,
    }


@pytest.fixture
def sample_data_2d(rng):
    """Generate synthetic data with 2D covariates."""
    T = 200
    p = 5

    X = rng.normal(0, 1, (T, p))

    # Propensity depends on first two covariates
    logit_e = 0.3 * X[:, 0] + 0.2 * X[:, 1]
    e_true = 1 / (1 + np.exp(-logit_e))
    e_true = np.clip(e_true, 0.05, 0.95)

    W = rng.binomial(1, e_true).astype(float)

    # Outcomes
    mu0_true = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1]
    tau_true = 0.5 + 0.3 * X[:, 0] + np.sin(X[:, 1])
    mu1_true = mu0_true + tau_true

    sigma = 0.2
    Y = np.where(W == 1, mu1_true, mu0_true) + rng.normal(0, sigma, T)

    return {
        "X": X,
        "W": W,
        "Y": Y,
        "e_true": e_true,
        "mu0_true": mu0_true,
        "mu1_true": mu1_true,
        "tau_true": tau_true,
        "T": T,
        "p": p,
    }


@pytest.fixture
def nuisance_estimates(sample_data):
    """Generate nuisance estimates (using true values as perfect estimates)."""
    return {
        "e_hat": sample_data["e_true"],
        "mu0_hat": sample_data["mu0_true"],
        "mu1_hat": sample_data["mu1_true"],
        "tau_hat": sample_data["tau_true"],
    }


@pytest.fixture
def noisy_nuisance_estimates(sample_data, rng):
    """Generate noisy nuisance estimates for testing robustness."""
    noise_scale = 0.1
    return {
        "e_hat": np.clip(
            sample_data["e_true"] + rng.normal(0, noise_scale, sample_data["T"]),
            0.05, 0.95
        ),
        "mu0_hat": sample_data["mu0_true"] + rng.normal(0, noise_scale, sample_data["T"]),
        "mu1_hat": sample_data["mu1_true"] + rng.normal(0, noise_scale, sample_data["T"]),
        "tau_hat": sample_data["tau_true"] + rng.normal(0, noise_scale, sample_data["T"]),
    }


@pytest.fixture
def iid_scores(rng):
    """Generate IID exponential scores for testing ACI."""
    T = 500
    return {
        "scores": rng.exponential(1, T),
        "true_residuals": rng.exponential(1, T),
        "T": T,
    }
