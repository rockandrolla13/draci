"""
Core DR-ACI algorithm implementation.

Implements Adaptive Conformal Inference (ACI) with doubly robust scores
for constructing prediction intervals on CATEs under temporal dependence.

References:
- Gibbs & Candes (2021): Adaptive Conformal Inference Under Distribution Shift
- Barber et al. (2023): Conformal Prediction Beyond Exchangeability
- Wu et al. (2025): ECI - Error-quantified Conformal Inference
- Chernozhukov et al. (2018): Block-permutation conformal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from .scores import dr_score, vs_dr_score, naive_score, dr_pseudo_outcome


@dataclass
class ConformalResult:
    """Result from a conformal prediction method."""

    coverage: float  # Fraction of time target in interval
    avg_width: float  # Average prediction interval width (full width, 2*q)
    coverages_t: np.ndarray  # Per-timestep coverage indicators
    widths_t: np.ndarray | None = None  # Per-timestep widths
    alpha_trajectory: np.ndarray | None = None  # Adaptive alpha over time


@dataclass
class IntervalResult:
    """Result from DR-ACI interval construction."""

    point: np.ndarray  # CATE point estimates
    lower: np.ndarray  # Lower bounds
    upper: np.ndarray  # Upper bounds
    width: np.ndarray  # Interval widths
    coverage_t: np.ndarray | None = None  # Per-timestep coverage (if oracle)
    alpha_trajectory: np.ndarray | None = None  # Adaptive alpha over time


def aci(
    scores_seq: np.ndarray,
    true_residuals: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.005,
    n_warmup: int = 50,
) -> ConformalResult:
    """Adaptive Conformal Inference with online recalibration.

    Implements Algorithm 1 from Gibbs & Candes (2021):
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    where err_t = 1{S_t > q_hat_t} (self-coverage drives update).

    TRUE coverage is measured separately by checking if true_residuals[t] <= q_hat_t.

    Parameters
    ----------
    scores_seq : array (T,)
        Conformity scores for self-coverage (e.g., |psi^DR - tau_hat|)
    true_residuals : array (T,)
        True residuals for oracle coverage (e.g., |tau_true - tau_hat|)
    alpha : float
        Target miscoverage rate (e.g., 0.1 for 90% coverage)
    gamma : float
        ACI step size for alpha updates (smaller = slower adaptation)
    n_warmup : int
        Number of initial observations before starting ACI

    Returns
    -------
    ConformalResult with coverage, avg_width, and per-timestep indicators
    """
    T = len(scores_seq)
    alpha_t = alpha
    true_covered = np.zeros(T)
    widths = np.zeros(T)
    alphas = np.zeros(T)

    for t in range(n_warmup, T):
        cal_scores = scores_seq[:t]
        q_hat = np.quantile(cal_scores, min(max(1 - alpha_t, 0), 1))

        self_covered_t = float(scores_seq[t] <= q_hat)
        true_covered[t] = float(true_residuals[t] <= q_hat)
        widths[t] = 2 * q_hat
        alphas[t] = alpha_t

        err_t = 1 - self_covered_t
        alpha_t = alpha_t + gamma * (alpha - err_t)
        alpha_t = np.clip(alpha_t, 0.01, 0.99)

    valid = slice(n_warmup, T)
    return ConformalResult(
        coverage=true_covered[valid].mean(),
        avg_width=widths[valid].mean(),
        coverages_t=true_covered[valid],
        widths_t=widths[valid],
        alpha_trajectory=alphas[valid],
    )


def dr_aci(
    dr_scores_seq: np.ndarray,
    true_residuals: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.005,
    n_warmup: int = 50,
) -> ConformalResult:
    """DR-ACI: Adaptive conformal with doubly robust conformity scores.

    Wrapper around aci() that makes the DR nature explicit.

    Parameters
    ----------
    dr_scores_seq : array (T,)
        DR conformity scores from dr_score()
    true_residuals : array (T,)
        |tau_true - tau_hat| for oracle coverage
    alpha, gamma, n_warmup : see aci()

    Returns
    -------
    ConformalResult
    """
    return aci(dr_scores_seq, true_residuals, alpha=alpha, gamma=gamma,
               n_warmup=n_warmup)


def vs_dr_aci(
    vs_scores_seq: np.ndarray,
    true_residuals_standardized: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.005,
    n_warmup: int = 50,
) -> ConformalResult:
    """VS-DR-ACI: DR-ACI with variance-standardized scores.

    ACI runs on standardized scores. Coverage is measured on standardized
    scale. To get intervals on original scale, multiply q_hat by sigma_hat.

    Parameters
    ----------
    vs_scores_seq : array (T,)
        Standardized DR scores |psi^DR - tau_hat| / sigma_hat
    true_residuals_standardized : array (T,)
        |tau_true - tau_hat| / sigma_hat (standardized true residuals)
    alpha, gamma, n_warmup : see aci()

    Returns
    -------
    ConformalResult (widths are in standardized units)
    """
    T = len(vs_scores_seq)
    alpha_t = alpha
    true_covered = np.zeros(T)
    widths = np.zeros(T)
    alphas = np.zeros(T)

    for t in range(n_warmup, T):
        cal_scores = vs_scores_seq[:t]
        q_hat = np.quantile(cal_scores, min(max(1 - alpha_t, 0), 1))

        self_covered_t = float(vs_scores_seq[t] <= q_hat)
        true_covered[t] = float(true_residuals_standardized[t] <= q_hat)
        widths[t] = 2 * q_hat  # In standardized units
        alphas[t] = alpha_t

        err_t = 1 - self_covered_t
        alpha_t = alpha_t + gamma * (alpha - err_t)
        alpha_t = np.clip(alpha_t, 0.01, 0.99)

    valid = slice(n_warmup, T)
    return ConformalResult(
        coverage=true_covered[valid].mean(),
        avg_width=widths[valid].mean(),
        coverages_t=true_covered[valid],
        widths_t=widths[valid],
        alpha_trajectory=alphas[valid],
    )


def split_conformal(
    scores_cal: np.ndarray,
    scores_test: np.ndarray,
    true_residuals_test: np.ndarray,
    alpha: float = 0.1,
) -> ConformalResult:
    """Standard split conformal: fixed quantile from calibration set.

    No online adaptation. The calibration scores determine a fixed quantile
    that is applied to all test points. Ignores temporal dependence.

    Parameters
    ----------
    scores_cal : array (n_cal,)
        Conformity scores from calibration set (e.g., |psi^DR - tau_hat|)
    scores_test : array (n_test,)
        Conformity scores from test set (unused, kept for interface)
    true_residuals_test : array (n_test,)
        |tau_true - tau_hat| on test set for oracle coverage
    alpha : float
        Target miscoverage rate (default 0.1 for 90% coverage)

    Returns
    -------
    ConformalResult with coverage on test set
    """
    n_cal = len(scores_cal)
    q_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q_hat = np.quantile(scores_cal, min(q_level, 1.0))
    true_covered = (true_residuals_test <= q_hat).astype(float)
    return ConformalResult(
        coverage=true_covered.mean(),
        avg_width=2 * q_hat,
        coverages_t=true_covered,
        widths_t=np.full_like(true_covered, 2 * q_hat),
    )


def nexcp(
    scores_seq: np.ndarray,
    true_residuals: np.ndarray,
    alpha: float = 0.1,
    lam: float = 0.05,
    n_warmup: int = 50,
) -> ConformalResult:
    """NexCP: weighted quantile with exponential decay weights.

    At time t, computes a weighted quantile of past scores with weights
    w_s = exp(-lam * (t - s)) for s < t. More recent scores get higher weight.

    References:
    - Barber et al. (2023): Conformal Prediction Beyond Exchangeability

    Parameters
    ----------
    scores_seq : array (T,)
        Conformity scores for all timesteps
    true_residuals : array (T,)
        |tau_true - tau_hat| for oracle coverage
    alpha : float
        Target miscoverage rate
    lam : float
        Exponential decay rate. Larger = more weight on recent scores.
    n_warmup : int
        Number of initial observations before making predictions

    Returns
    -------
    ConformalResult with coverage and widths
    """
    T = len(scores_seq)
    true_covered = np.zeros(T)
    widths = np.zeros(T)

    for t in range(n_warmup, T):
        # Weights: exponential decay from most recent to oldest
        lags = np.arange(t, 0, -1, dtype=float)  # t, t-1, ..., 1
        weights = np.exp(-lam * lags)
        cal_scores = scores_seq[:t]

        # Weighted quantile via sorting
        sorted_idx = np.argsort(cal_scores)
        sorted_scores = cal_scores[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights)
        cum_weights /= cum_weights[-1]  # normalize to [0, 1]

        # Find the quantile level
        q_level = 1 - alpha
        idx = np.searchsorted(cum_weights, q_level)
        idx = min(idx, len(sorted_scores) - 1)
        q_hat = sorted_scores[idx]

        true_covered[t] = float(true_residuals[t] <= q_hat)
        widths[t] = 2 * q_hat

    valid = slice(n_warmup, T)
    return ConformalResult(
        coverage=true_covered[valid].mean(),
        avg_width=widths[valid].mean(),
        coverages_t=true_covered[valid],
        widths_t=widths[valid],
    )


def eci(
    scores_seq: np.ndarray,
    true_residuals: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.005,
    n_warmup: int = 50,
    temperature: float = 1.0,
) -> ConformalResult:
    """ECI: ACI with smooth sigmoid feedback instead of binary indicator.

    Replaces err_t = 1{S_t > q_t} with a smooth sigmoid:
        err_t = sigmoid((S_t - q_t) / temperature)

    This reduces oscillation in the quantile update, producing more
    stable interval widths than standard ACI.

    References:
    - Wu et al. (ICLR 2025): Error-quantified Conformal Inference

    Parameters
    ----------
    scores_seq : array (T,)
        Conformity scores for all timesteps
    true_residuals : array (T,)
        |tau_true - tau_hat| for oracle coverage
    alpha : float
        Target miscoverage rate
    gamma : float
        Step size for alpha updates
    n_warmup : int
        Number of initial observations before adaptation
    temperature : float
        Controls sigmoid sharpness. Lower = closer to hard indicator.

    Returns
    -------
    ConformalResult with coverage and widths
    """
    T = len(scores_seq)
    alpha_t = alpha
    true_covered = np.zeros(T)
    widths = np.zeros(T)
    alphas = np.zeros(T)

    for t in range(n_warmup, T):
        cal_scores = scores_seq[:t]
        q_hat = np.quantile(cal_scores, min(max(1 - alpha_t, 0), 1))

        # Smooth sigmoid feedback
        diff = (scores_seq[t] - q_hat) / max(temperature, 1e-6)
        # Numerically stable sigmoid
        smooth_err = 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))

        true_covered[t] = float(true_residuals[t] <= q_hat)
        widths[t] = 2 * q_hat
        alphas[t] = alpha_t

        # ACI update with smooth error
        alpha_t = alpha_t + gamma * (alpha - smooth_err)
        alpha_t = np.clip(alpha_t, 0.01, 0.99)

    valid = slice(n_warmup, T)
    return ConformalResult(
        coverage=true_covered[valid].mean(),
        avg_width=widths[valid].mean(),
        coverages_t=true_covered[valid],
        widths_t=widths[valid],
        alpha_trajectory=alphas[valid],
    )


def block_cp(
    scores_seq: np.ndarray,
    true_residuals: np.ndarray,
    alpha: float = 0.1,
    block_size: int = 10,
    n_reps: int = 200,
    rng: np.random.Generator | None = None,
) -> ConformalResult:
    """Block-permutation conformal prediction.

    Partitions calibration scores into non-overlapping blocks.
    Permutes entire blocks (preserving within-block dependence).
    Computes quantile from block-permuted distribution.

    References:
    - Chernozhukov et al. (2018): Valid Post-Selection Inference

    Parameters
    ----------
    scores_seq : array (T,)
        Conformity scores (first half used for calibration)
    true_residuals : array (T,)
        |tau_true - tau_hat| for oracle coverage
    alpha : float
        Target miscoverage rate
    block_size : int
        Size of each block for permutation
    n_reps : int
        Number of block-permutation replicates
    rng : Generator, optional
        Random number generator

    Returns
    -------
    ConformalResult with coverage on test set (second half)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T = len(scores_seq)
    n_cal = T // 2
    cal_scores = scores_seq[:n_cal]
    test_true_res = true_residuals[n_cal:]

    # Partition calibration into blocks
    n_blocks = n_cal // block_size
    if n_blocks < 2:
        # Fall back to standard split if too few blocks
        q_hat = np.quantile(cal_scores, 1 - alpha)
        true_covered = (test_true_res <= q_hat).astype(float)
        return ConformalResult(
            coverage=true_covered.mean(),
            avg_width=2 * q_hat,
            coverages_t=true_covered,
            widths_t=np.full_like(true_covered, 2 * q_hat),
        )

    # Trim calibration to fit exactly into blocks
    cal_trimmed = cal_scores[:n_blocks * block_size]
    blocks = cal_trimmed.reshape(n_blocks, block_size)

    # Block-permutation quantiles
    perm_quantiles = np.zeros(n_reps)
    for r in range(n_reps):
        perm_idx = rng.permutation(n_blocks)
        perm_scores = blocks[perm_idx].ravel()
        perm_quantiles[r] = np.quantile(perm_scores, 1 - alpha)

    q_hat = np.median(perm_quantiles)
    true_covered = (test_true_res <= q_hat).astype(float)

    return ConformalResult(
        coverage=true_covered.mean(),
        avg_width=2 * q_hat,
        coverages_t=true_covered,
        widths_t=np.full_like(true_covered, 2 * q_hat),
    )


def hac(
    psi_dr_seq: np.ndarray,
    tau_hat_seq: np.ndarray,
    true_residuals: np.ndarray,
    alpha: float = 0.1,
    bandwidth: int = 10,
) -> ConformalResult:
    """HAC (Newey-West) asymptotic confidence intervals.

    Constructs pointwise CIs for CATE using Newey-West standard errors
    with Bartlett kernel and fixed bandwidth.

    Parameters
    ----------
    psi_dr_seq : array (T,)
        DR pseudo-outcomes (signed, not absolute value)
    tau_hat_seq : array (T,)
        CATE point estimates
    true_residuals : array (T,)
        |tau_true - tau_hat| for oracle coverage
    alpha : float
        Target miscoverage rate
    bandwidth : int
        Lag truncation for Bartlett kernel

    Returns
    -------
    ConformalResult with coverage
    """
    T = len(psi_dr_seq)
    residuals = psi_dr_seq - tau_hat_seq  # centering: should be mean-zero

    # Newey-West variance estimator with Bartlett kernel
    gamma_0 = np.mean(residuals**2)
    nw_var = gamma_0
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1)  # Bartlett kernel
        gamma_lag = np.mean(residuals[lag:] * residuals[:-lag])
        nw_var += 2 * weight * gamma_lag

    # Use observation-level SD (not SE of the mean) for individual CATE coverage
    nw_se = np.sqrt(max(nw_var, 1e-10))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    half_width = z_crit * nw_se

    # Coverage: does true CATE fall within [tau_hat +/- half_width]?
    true_covered = (true_residuals <= half_width).astype(float)

    return ConformalResult(
        coverage=true_covered.mean(),
        avg_width=2 * half_width,
        coverages_t=true_covered,
        widths_t=np.full_like(true_covered, 2 * half_width),
    )


class DRACI:
    """Doubly Robust Adaptive Conformal Inference for CATEs.

    High-level interface that wraps the core ACI algorithm with
    automatic score computation and nuisance handling.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate (default 0.10 for 90% coverage)
    gamma : float
        ACI step size for alpha updates (default 0.005)
    n_warmup : int or float
        Warmup period before adaptation. If float < 1, interpreted as fraction.
    score_type : {"dr", "vs_dr", "naive"}
        Score function to use:
        - "dr": Standard DR score (robust, wider intervals)
        - "vs_dr": Variance-standardized DR (tighter intervals)
        - "naive": Non-DR plug-in score (baseline)

    Example
    -------
    >>> from draci import DRACI
    >>> model = DRACI(alpha=0.1, score_type="vs_dr")
    >>> result = model.fit_predict(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
    >>> print(f"Coverage: {result.coverage:.2%}")
    """

    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.005,
        n_warmup: int | float = 50,
        score_type: Literal["dr", "vs_dr", "naive"] = "dr",
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.n_warmup = n_warmup
        self.score_type = score_type

    def fit_predict(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        e_hat: np.ndarray,
        mu0_hat: np.ndarray,
        mu1_hat: np.ndarray,
        tau_hat: np.ndarray,
        tau_true: np.ndarray | None = None,
    ) -> IntervalResult:
        """Construct prediction intervals for CATEs.

        Runs ACI sequentially, updating alpha_t based on self-coverage.
        If tau_true provided, also computes true coverage.

        Parameters
        ----------
        Y : array (T,)
            Observed outcomes
        W : array (T,)
            Treatment indicators
        e_hat : array (T,)
            Propensity score estimates
        mu0_hat, mu1_hat : array (T,)
            Outcome model estimates
        tau_hat : array (T,)
            CATE point estimates
        tau_true : array (T,), optional
            True CATE for oracle coverage evaluation

        Returns
        -------
        IntervalResult with point estimates, bounds, and diagnostics
        """
        T = len(Y)
        n_warmup = (
            int(self.n_warmup * T) if isinstance(self.n_warmup, float) and self.n_warmup < 1
            else int(self.n_warmup)
        )

        # Compute scores
        if self.score_type == "vs_dr":
            scores, sigma_hat = vs_dr_score(
                Y, W, e_hat, mu0_hat, mu1_hat, tau_hat, return_sigma=True
            )
        elif self.score_type == "dr":
            scores = dr_score(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
            sigma_hat = None
        else:  # naive
            scores = naive_score(Y, W, mu0_hat, mu1_hat, tau_hat)
            sigma_hat = None

        # True residuals for coverage evaluation
        if tau_true is not None:
            true_residuals = np.abs(tau_true - tau_hat)
            if self.score_type == "vs_dr" and sigma_hat is not None:
                true_residuals = true_residuals / sigma_hat
        else:
            true_residuals = scores  # Self-coverage

        # Run ACI
        alpha_t = self.alpha
        lower = np.zeros(T)
        upper = np.zeros(T)
        widths = np.zeros(T)
        coverage_t = np.zeros(T) if tau_true is not None else None
        alphas = np.zeros(T)

        for t in range(n_warmup, T):
            cal_scores = scores[:t]
            q_hat = np.quantile(cal_scores, min(max(1 - alpha_t, 0), 1))

            # Construct interval
            if self.score_type == "vs_dr" and sigma_hat is not None:
                half_width = q_hat * sigma_hat[t]
            else:
                half_width = q_hat

            lower[t] = tau_hat[t] - half_width
            upper[t] = tau_hat[t] + half_width
            widths[t] = 2 * half_width
            alphas[t] = alpha_t

            if tau_true is not None:
                coverage_t[t] = float(lower[t] <= tau_true[t] <= upper[t])

            # ACI update
            self_covered = float(scores[t] <= q_hat)
            err_t = 1 - self_covered
            alpha_t = alpha_t + self.gamma * (self.alpha - err_t)
            alpha_t = np.clip(alpha_t, 0.01, 0.99)

        return IntervalResult(
            point=tau_hat,
            lower=lower,
            upper=upper,
            width=widths,
            coverage_t=coverage_t[n_warmup:] if coverage_t is not None else None,
            alpha_trajectory=alphas[n_warmup:],
        )
