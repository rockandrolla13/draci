"""
Beta-mixing coefficient estimation utilities.

Provides tools for estimating and validating the beta-mixing assumption
(Assumption A1 in the DR-ACI paper). This is useful for:
1. Validating that the mixing assumption holds for your data
2. Estimating the mixing gap term in the coverage bound
3. Choosing optimal block sizes for temporal cross-fitting

References:
- Oliveira et al. (2024): Split Conformal for Dependent Data
- Barber et al. (2025): Conformal Prediction Beyond Exchangeability
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class MixingDiagnostics:
    """Results from beta-mixing estimation."""

    beta_values: dict[int, dict]  # lag -> {median, mean, std, q25, q75}
    exp_fit: dict  # {a, b, r2} for beta(tau) ~ a * exp(-b * tau)
    acf_lag1_median: float  # Median lag-1 autocorrelation
    optimal_tau: int  # Optimal lag for mixing gap
    optimal_gap: float  # min_tau {tau/T + 2*beta(tau)}


def compute_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function up to max_lag.

    Uses biased estimator (dividing by T) which is positive semi-definite.

    Parameters
    ----------
    series : array (T,)
        Time series data
    max_lag : int
        Maximum lag to compute

    Returns
    -------
    acf : array (max_lag + 1,)
        Autocorrelation values from lag 0 to max_lag
    """
    series = series - np.mean(series)
    T = len(series)

    if T < max_lag + 10:
        return np.full(max_lag + 1, np.nan)

    var = np.sum(series ** 2) / T
    if var < 1e-12:
        return np.full(max_lag + 1, np.nan)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.sum(series[lag:] * series[:-lag]) / (T * var)

    return acf


def estimate_beta_mixing(
    series: np.ndarray,
    lags: list[int] | None = None,
    n_bins: int = 20,
) -> dict[int, float]:
    """Estimate beta-mixing coefficients for a single time series.

    Uses the histogram method from Oliveira et al. (2024):
    1. For each lag tau, bin (X_t, X_{t+tau}) pairs
    2. Estimate TV distance between joint and product marginal
    3. beta(tau) = 0.5 * sum |P_joint - P_product|

    Parameters
    ----------
    series : array (T,)
        Time series data
    lags : list of int, optional
        Lags at which to estimate beta. Default: [1, 5, 10, 20, 50, 100]
    n_bins : int
        Number of bins for histogram estimation

    Returns
    -------
    beta_values : dict
        lag -> estimated beta(tau)
    """
    if lags is None:
        lags = [1, 5, 10, 20, 50, 100]

    T = len(series)
    beta_values = {}

    for tau in lags:
        if tau >= T - 10:
            beta_values[tau] = np.nan
            continue

        past = series[:-tau]
        future = series[tau:]
        n = len(past)

        # Bin into histogram
        edges_p = np.linspace(np.min(past), np.max(past) + 1e-10, n_bins + 1)
        edges_f = np.linspace(np.min(future), np.max(future) + 1e-10, n_bins + 1)

        # Joint distribution
        hist_joint, _, _ = np.histogram2d(past, future, bins=[edges_p, edges_f])
        hist_joint = hist_joint / n

        # Marginals
        hist_past = np.histogram(past, bins=edges_p)[0] / n
        hist_future = np.histogram(future, bins=edges_f)[0] / n

        # Product of marginals
        hist_product = np.outer(hist_past, hist_future)

        # TV distance = 0.5 * sum |P_joint - P_product|
        tv = 0.5 * np.sum(np.abs(hist_joint - hist_product))
        beta_values[tau] = tv

    return beta_values


def fit_exponential_decay(
    beta_values: dict[int, float],
) -> dict:
    """Fit exponential decay model beta(tau) ~ a * exp(-b * tau).

    Parameters
    ----------
    beta_values : dict
        lag -> beta estimate

    Returns
    -------
    fit_params : dict
        {a, b, r2} where beta(tau) ~ a * exp(-b * tau)
    """
    # Filter valid values
    valid = [(tau, beta) for tau, beta in beta_values.items()
             if not np.isnan(beta) and beta > 0]

    if len(valid) < 3:
        return {"a": np.nan, "b": np.nan, "r2": np.nan}

    tau_arr = np.array([v[0] for v in valid])
    beta_arr = np.array([v[1] for v in valid])

    # Linear regression on log scale: log(beta) = log(a) - b*tau
    log_beta = np.log(np.maximum(beta_arr, 1e-10))
    slope, intercept, r_value, _, _ = stats.linregress(tau_arr, log_beta)

    return {
        "a": np.exp(intercept),
        "b": -slope,
        "r2": r_value ** 2,
    }


def compute_optimal_mixing_gap(
    beta_values: dict[int, float],
    T: int,
) -> tuple[int, float]:
    """Compute optimal mixing gap: min_tau {tau/T + 2*beta(tau)}.

    This is the first term in the DR-ACI coverage bound (Theorem 1).

    Parameters
    ----------
    beta_values : dict
        lag -> beta estimate
    T : int
        Sample size

    Returns
    -------
    optimal_tau : int
        Lag that minimizes the gap
    optimal_gap : float
        Value of the minimized gap
    """
    gaps = []
    for tau, beta in beta_values.items():
        if np.isnan(beta):
            continue
        gap = tau / T + 2 * beta
        gaps.append((tau, gap))

    if not gaps:
        return 0, np.nan

    optimal_tau, optimal_gap = min(gaps, key=lambda x: x[1])
    return optimal_tau, optimal_gap


def mixing_diagnostics(
    series: np.ndarray,
    lags: list[int] | None = None,
    max_acf_lag: int = 50,
    n_bins: int = 20,
) -> MixingDiagnostics:
    """Run full mixing diagnostics on a time series.

    Parameters
    ----------
    series : array (T,)
        Time series data
    lags : list of int, optional
        Lags for beta estimation
    max_acf_lag : int
        Maximum lag for ACF computation
    n_bins : int
        Number of bins for histogram method

    Returns
    -------
    MixingDiagnostics with all estimates and derived quantities
    """
    if lags is None:
        lags = [1, 5, 10, 20, 50, 100]

    T = len(series)

    # ACF
    acf = compute_acf(series, max_acf_lag)
    acf_lag1 = acf[1] if len(acf) > 1 else np.nan

    # Beta-mixing
    beta_raw = estimate_beta_mixing(series, lags=lags, n_bins=n_bins)

    # Format beta values with stats (for single series, just wrap the values)
    beta_values = {
        tau: {"median": beta, "mean": beta, "std": 0.0, "q25": beta, "q75": beta}
        for tau, beta in beta_raw.items()
    }

    # Exponential fit
    exp_fit = fit_exponential_decay(beta_raw)

    # Optimal gap
    optimal_tau, optimal_gap = compute_optimal_mixing_gap(beta_raw, T)

    return MixingDiagnostics(
        beta_values=beta_values,
        exp_fit=exp_fit,
        acf_lag1_median=acf_lag1,
        optimal_tau=optimal_tau,
        optimal_gap=optimal_gap,
    )


def mixing_diagnostics_panel(
    panel_series: list[np.ndarray],
    lags: list[int] | None = None,
    n_bins: int = 20,
    min_obs: int = 100,
) -> MixingDiagnostics:
    """Run mixing diagnostics across a panel of time series.

    Aggregates beta estimates across multiple series (e.g., tickers)
    to get a more robust estimate of the mixing structure.

    Parameters
    ----------
    panel_series : list of arrays
        List of time series, one per unit (e.g., ticker)
    lags : list of int, optional
        Lags for beta estimation
    n_bins : int
        Number of bins for histogram method
    min_obs : int
        Minimum observations per series to include

    Returns
    -------
    MixingDiagnostics with aggregated estimates
    """
    if lags is None:
        lags = [1, 5, 10, 20, 50, 100]

    # Collect estimates across series
    all_betas = {tau: [] for tau in lags}
    all_acf_lag1 = []
    total_T = 0

    for series in panel_series:
        if len(series) < min_obs:
            continue

        total_T += len(series)

        # ACF
        acf = compute_acf(series, max_lag=1)
        if not np.isnan(acf[1]):
            all_acf_lag1.append(acf[1])

        # Beta
        beta_raw = estimate_beta_mixing(series, lags=lags, n_bins=n_bins)
        for tau, beta in beta_raw.items():
            if not np.isnan(beta):
                all_betas[tau].append(beta)

    # Aggregate statistics
    beta_values = {}
    for tau in lags:
        if all_betas[tau]:
            arr = np.array(all_betas[tau])
            beta_values[tau] = {
                "median": np.median(arr),
                "mean": np.mean(arr),
                "std": np.std(arr),
                "q25": np.percentile(arr, 25),
                "q75": np.percentile(arr, 75),
            }
        else:
            beta_values[tau] = {
                "median": np.nan, "mean": np.nan, "std": np.nan,
                "q25": np.nan, "q75": np.nan
            }

    # Use median for exponential fit and gap
    beta_median = {tau: v["median"] for tau, v in beta_values.items()}
    exp_fit = fit_exponential_decay(beta_median)
    optimal_tau, optimal_gap = compute_optimal_mixing_gap(beta_median, total_T)

    acf_lag1_med = np.median(all_acf_lag1) if all_acf_lag1 else np.nan

    return MixingDiagnostics(
        beta_values=beta_values,
        exp_fit=exp_fit,
        acf_lag1_median=acf_lag1_med,
        optimal_tau=optimal_tau,
        optimal_gap=optimal_gap,
    )
