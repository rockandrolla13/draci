"""
Classical CI baselines for comparison against DR-ACI.

Three methods that construct confidence intervals for CATE:
1. DML-Wald: Asymptotic CI using local NN variance + Newey-West n_eff
2. Block Bootstrap: Circular block bootstrap percentile CI
3. Causal Forest: econml CausalForestDML built-in intervals

All methods return ConformalResult for interface consistency with core.py.

References:
- Newey & West (1987): HAC variance estimation
- Politis & Romano (1994): Circular block bootstrap
- Wager & Athey (2018): Causal forests with honest intervals
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy import stats
from scipy.spatial import KDTree

from .core import ConformalResult


# =============================================================================
# Extended metrics for CI comparison
# =============================================================================

class CIMetrics(NamedTuple):
    """Extended metrics for CI baseline comparison."""

    miscoverage_gap: float  # (1 - coverage) - alpha
    piaw: float  # mean interval width (Prediction Interval Average Width)
    rolling_coverage: np.ndarray  # (T-window,) rolling coverage
    undercoverage_rate: float  # fraction of windows with >5pp undercov


# =============================================================================
# Helper functions
# =============================================================================

def _local_nn_variance(
    residuals: np.ndarray, X: np.ndarray, k: int = 50
) -> np.ndarray:
    """Estimate local variance via k-NN on covariate space.

    For each point, find k nearest neighbors and compute variance of
    residuals among those neighbors.

    Parameters
    ----------
    residuals : array (T,)
        Residuals to compute variance of
    X : array (T, d)
        Covariates for neighbor search
    k : int
        Number of neighbors

    Returns
    -------
    V_hat : array (T,)
        Local variance estimates
    """
    T = len(residuals)
    k = min(k, T - 1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    tree = KDTree(X)
    V_hat = np.zeros(T)

    for i in range(T):
        _, idx = tree.query(X[i], k=k + 1)  # +1 because query includes self
        neighbor_residuals = residuals[idx]
        V_hat[i] = np.var(neighbor_residuals, ddof=1)

    # Floor variance for numerical stability
    V_hat = np.maximum(V_hat, 1e-8)
    return V_hat


def _newey_west_neff(residuals: np.ndarray) -> float:
    """Compute effective sample size with Newey-West adjustment.

    n_eff = T / (1 + 2 * sum_{j=1}^{bw} (1 - j/bw) * rho_j)
    where rho_j is the lag-j autocorrelation and bw = floor(T^{1/3}).

    Parameters
    ----------
    residuals : array (T,)
        Time series residuals

    Returns
    -------
    n_eff : float
        Effective sample size
    """
    T = len(residuals)
    bandwidth = max(int(T ** (1 / 3)), 1)

    # Compute autocorrelations
    residuals_centered = residuals - residuals.mean()
    var = np.var(residuals_centered)
    if var < 1e-10:
        return float(T)

    acf_sum = 0.0
    for j in range(1, bandwidth + 1):
        acf_j = np.mean(residuals_centered[j:] * residuals_centered[:-j]) / var
        weight = 1.0 - j / (bandwidth + 1)  # Bartlett kernel
        acf_sum += weight * acf_j

    n_eff = T / (1.0 + 2.0 * acf_sum)
    return max(n_eff, 1.0)


def _circular_block_bootstrap(
    T: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate indices for one circular block bootstrap resample.

    Circular: indices wrap around, so block starting at T-2 with size 5
    gives indices [T-2, T-1, 0, 1, 2].

    Parameters
    ----------
    T : int
        Sample size
    block_size : int
        Block length
    rng : np.random.Generator
        Random generator

    Returns
    -------
    indices : array (T,)
        Resampled indices
    """
    n_blocks = int(np.ceil(T / block_size))
    start_indices = rng.integers(0, T, size=n_blocks)

    indices = []
    for start in start_indices:
        block = [(start + j) % T for j in range(block_size)]
        indices.extend(block)

    return np.array(indices[:T])


# =============================================================================
# CI Baseline 1: DML-Wald CI
# =============================================================================

def dml_wald_ci(
    psi_dr: np.ndarray,
    tau_hat: np.ndarray,
    true_residuals: np.ndarray,
    X_cal: np.ndarray,
    alpha: float = 0.1,
    k_neighbors: int = 50,
) -> ConformalResult:
    """DML-Wald CI with local NN variance and Newey-West adjustment.

    Constructs interval: tau_hat(x) +/- z_{1-alpha/2} * sqrt(V_hat(x) / k_eff)
    where V_hat(x) is estimated via k-NN, and k_eff accounts for dependence.

    Parameters
    ----------
    psi_dr : array (T,)
        DR pseudo-outcomes
    tau_hat : array (T,)
        CATE point estimates
    true_residuals : array (T,)
        |tau_true - tau_hat| for coverage evaluation
    X_cal : array (T, d)
        Calibration covariates
    alpha : float
        Nominal miscoverage rate
    k_neighbors : int
        Number of neighbors for local variance

    Returns
    -------
    ConformalResult with coverage measured against true_residuals
    """
    T = len(psi_dr)
    residuals = psi_dr - tau_hat

    # Local variance estimate via k-NN
    V_hat = _local_nn_variance(residuals, X_cal, k=k_neighbors)

    # Effective sample size adjustment for dependence (global)
    global_neff_ratio = _newey_west_neff(residuals) / T
    k_eff = max(k_neighbors * global_neff_ratio, 1.0)

    # Wald interval: tau_hat +/- z * sqrt(V / k_eff)
    z = stats.norm.ppf(1 - alpha / 2)
    half_width = z * np.sqrt(V_hat / k_eff)

    # Coverage: |tau_true - tau_hat| <= half_width
    covered = (true_residuals <= half_width).astype(float)

    return ConformalResult(
        coverage=covered.mean(),
        avg_width=2 * half_width.mean(),
        coverages_t=covered,
        widths_t=2 * half_width,
    )


# =============================================================================
# CI Baseline 2: Block Bootstrap CI
# =============================================================================

def block_bootstrap_ci(
    psi_dr: np.ndarray,
    tau_hat: np.ndarray,
    true_residuals: np.ndarray,
    alpha: float = 0.1,
    B: int = 199,
    block_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> ConformalResult:
    """Circular block bootstrap percentile CI.

    Respects temporal dependence by resampling blocks of consecutive
    observations. Uses percentile method.

    Parameters
    ----------
    psi_dr : array (T,)
        DR pseudo-outcomes
    tau_hat : array (T,)
        CATE point estimates
    true_residuals : array (T,)
        |tau_true - tau_hat| for coverage evaluation
    alpha : float
        Nominal miscoverage rate
    B : int
        Number of bootstrap replicates
    block_size : int, optional
        Block length (default: floor(T^{1/3}))
    rng : np.random.Generator, optional
        Random generator

    Returns
    -------
    ConformalResult with coverage measured against true_residuals
    """
    T = len(psi_dr)
    if rng is None:
        rng = np.random.default_rng()
    if block_size is None:
        block_size = max(int(T ** (1 / 3)), 1)

    residuals = psi_dr - tau_hat

    # Bootstrap: resample residuals B times using block bootstrap
    boot_residuals = np.zeros((B, T))
    for b in range(B):
        idx = _circular_block_bootstrap(T, block_size, rng)
        boot_residuals[b] = residuals[idx]

    # Percentile CI for each observation
    lower_q = np.percentile(boot_residuals, 100 * alpha / 2, axis=0)
    upper_q = np.percentile(boot_residuals, 100 * (1 - alpha / 2), axis=0)

    # Half-width is max distance from center
    half_width = np.maximum(np.abs(lower_q), np.abs(upper_q))

    # Coverage: |tau_true - tau_hat| <= half_width
    covered = (true_residuals <= half_width).astype(float)

    return ConformalResult(
        coverage=covered.mean(),
        avg_width=2 * half_width.mean(),
        coverages_t=covered,
        widths_t=2 * half_width,
    )


# =============================================================================
# CI Baseline 3: Causal Forest CI
# =============================================================================

def causal_forest_ci(
    Y: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    tau_true: np.ndarray,
    alpha: float = 0.1,
    train_frac: float = 0.5,
    random_state: int | None = None,
) -> ConformalResult:
    """Causal forest CI via econml CausalForestDML.

    Uses built-in honest confidence intervals from Wager & Athey (2018).

    NOTE: This method fits its own nuisance models internally (does not
    reuse shared nuisance estimates). Uses temporal split, not random split.

    Parameters
    ----------
    Y : array (T,)
        Outcomes
    W : array (T,)
        Treatment indicators
    X : array (T, d)
        Covariates
    tau_true : array (T,)
        True CATE for coverage evaluation
    alpha : float
        Nominal miscoverage rate
    train_frac : float
        Fraction of data for training
    random_state : int, optional
        Random seed

    Returns
    -------
    ConformalResult (NaN if econml not installed)
    """
    T = len(Y)

    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            GradientBoostingClassifier,
        )
    except ImportError:
        warnings.warn("econml not installed; causal_forest_ci returning NaN")
        return ConformalResult(np.nan, np.nan, np.full(T, np.nan))

    # Temporal split (not random, to match DR-ACI methodology)
    n_train = int(T * train_frac)
    X_tr, W_tr, Y_tr = X[:n_train], W[:n_train], Y[:n_train]
    X_te, tau_te = X[n_train:], tau_true[n_train:]
    n_test = len(X_te)

    if n_train < 50 or n_test < 10:
        warnings.warn("Insufficient data for causal forest; returning NaN")
        return ConformalResult(np.nan, np.nan, np.full(T, np.nan))

    # Fit causal forest
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
        model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3),
        discrete_treatment=True,
        n_estimators=100,
        random_state=random_state,
    )

    try:
        cf.fit(Y_tr, W_tr, X=X_tr)
        lower, upper = cf.effect_interval(X_te, alpha=alpha)
        lower = lower.ravel()
        upper = upper.ravel()
    except Exception as e:
        warnings.warn(f"Causal forest fitting failed: {e}; returning NaN")
        return ConformalResult(np.nan, np.nan, np.full(T, np.nan))

    # Coverage on test set
    covered = ((tau_te >= lower) & (tau_te <= upper)).astype(float)
    width = upper - lower

    # Pad with NaN for training portion
    coverages_full = np.full(T, np.nan)
    coverages_full[n_train:] = covered
    widths_full = np.full(T, np.nan)
    widths_full[n_train:] = width

    return ConformalResult(
        coverage=covered.mean(),
        avg_width=width.mean(),
        coverages_t=coverages_full,
        widths_t=widths_full,
    )


# =============================================================================
# Metrics computation
# =============================================================================

def compute_ci_metrics(
    coverages_t: np.ndarray,
    widths_t: np.ndarray | float,
    alpha: float = 0.1,
    window: int = 100,
    undercov_threshold: float = 0.05,
) -> CIMetrics:
    """Compute extended CI comparison metrics.

    Parameters
    ----------
    coverages_t : array (T,)
        Per-timestep coverage indicators
    widths_t : array (T,) or float
        Per-timestep widths or scalar avg width
    alpha : float
        Nominal miscoverage rate
    window : int
        Rolling window size for coverage
    undercov_threshold : float
        Threshold for undercoverage (5pp default)

    Returns
    -------
    CIMetrics with miscoverage_gap, piaw, rolling_coverage, undercoverage_rate
    """
    # Handle NaN
    valid = ~np.isnan(coverages_t)
    if not valid.any():
        return CIMetrics(np.nan, np.nan, np.array([np.nan]), np.nan)

    coverages_valid = coverages_t[valid]
    T = len(coverages_valid)

    # Miscoverage gap: (1 - empirical_coverage) - alpha
    miscoverage_gap = (1 - coverages_valid.mean()) - alpha

    # PIAW (prediction interval average width)
    if isinstance(widths_t, (int, float)):
        piaw = float(widths_t)
    else:
        widths_valid = widths_t[valid] if hasattr(widths_t, "__len__") else widths_t
        piaw = float(np.mean(widths_valid))

    # Rolling coverage
    if T > window:
        kernel = np.ones(window) / window
        rolling = np.convolve(coverages_valid, kernel, mode="valid")
    else:
        rolling = np.array([coverages_valid.mean()])

    # Undercoverage rate: fraction of windows with coverage < (1-alpha) - threshold
    target = 1 - alpha - undercov_threshold
    undercov_rate = float((rolling < target).mean())

    return CIMetrics(miscoverage_gap, piaw, rolling, undercov_rate)
