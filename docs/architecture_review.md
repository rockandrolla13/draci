# Architecture Review: DR-ACI Code Migration

**Date:** 2026-04-11
**Source:** `/media/ak/10E1026C4FA6006E/GitRepos/TrackF_Paper_1/src/`
**Target:** `/media/ak/10E1026C4FA6006E/GitRepos/draci-package/`

---

## 1. Executive Summary

The paper repository contains a well-structured but paper-specific implementation of DR-ACI across three main directories: `simulation/`, `empirical/`, and `econometrics/`. The core algorithm is clean and extractable, but is interleaved with paper-specific orchestration code, hardcoded paths, and DGP code that should not be ported.

**Key findings:**
1. The core ACI loop (`aci()`) is self-contained and portable
2. Score functions are cleanly separated and easily extractable
3. Nuisance estimators use a functional pattern that should become a protocol
4. Temporal cross-fitting logic is embedded in experiment code and needs extraction
5. The econometrics module has a parallel but incompatible implementation
6. Error handling and input validation are minimal

**Recommended priority for extraction:**
1. `conformal_methods.py` score functions and ACI loop (CRITICAL)
2. `nuisance.py` estimator framework (HIGH)
3. Temporal cross-fitting pattern from `panel_draci.py` (HIGH)
4. Beta-mixing diagnostics from `mixing_diagnostics.py` (MEDIUM)
5. econml integration from `econometrics/dr_aci.py` (LOW)

---

## 2. Current Architecture Analysis

### 2.1 Module Overview

```
src/
  simulation/                    # Monte Carlo coverage study (Section 5)
    config.py                    # Global constants, paths, CLI args
    dgp.py                       # AR(1)+GARCH DGP (paper-specific)
    nuisance.py                  # Nuisance estimators (XGBoost, Linear)
    conformal_methods.py         # 9 conformal methods + score functions
    sim_coverage.py              # MC orchestration + figures/tables
    tests/                       # pytest suite

  empirical/                     # M-ELO application (Section 6)
    config.py                    # Data paths, date windows
    data_prep.py                 # Parquet loading, panel construction
    panel_draci.py               # Temporal block cross-fitting + daily ACI
    mixing_diagnostics.py        # Beta-mixing estimation
    cross_sectional.py           # Cross-sectional experiment 2a
    sensitivity.py               # Robustness checks
    run_all.py                   # CLI entry point

  econometrics/                  # DiD framework (separate concern)
    dr_aci.py                    # Alternative implementation with econml
    ...                          # TWFE, event studies, etc.
```

### 2.2 Key Abstractions

**2.2.1 Score Functions (simulation/conformal_methods.py)**

The score functions are pure numpy functions with a consistent signature:

```python
def dr_score(Y, W, X, e_hat, mu0_hat, mu1_hat, tau_hat) -> np.ndarray
def vs_dr_score(Y, W, X, e_hat, mu0_hat, mu1_hat, tau_hat) -> np.ndarray
def naive_score(Y, W, X, mu0_hat, mu1_hat, tau_hat) -> np.ndarray
def dr_pseudo_outcome(Y, W, X, e_hat, mu0_hat, mu1_hat) -> np.ndarray
```

**Observation:** The `X` parameter is passed for interface consistency but unused in all functions. Consider removing it or documenting why it is there.

**2.2.2 Nuisance Estimators (simulation/nuisance.py)**

Uses a functional pattern returning a tuple of callables:

```python
NuisanceTuple = Tuple[
    Callable[[np.ndarray], np.ndarray],  # e_hat
    Callable[[np.ndarray], np.ndarray],  # mu0_hat
    Callable[[np.ndarray], np.ndarray],  # mu1_hat
    Callable[[np.ndarray], np.ndarray],  # tau_hat
]

def fit_nuisance_xgboost(X, W, Y, clip_propensity) -> NuisanceTuple
def fit_nuisance_linear(X, W, Y, clip_propensity) -> NuisanceTuple
def fit_nuisances(X, W, Y, method, clip_propensity) -> NuisanceTuple  # dispatcher
```

**Issue:** The tuple interface is fragile (order-dependent) and lacks metadata. Should become a dataclass or protocol.

**2.2.3 Conformal Methods (simulation/conformal_methods.py)**

Nine methods, all returning `ConformalResult`:

```python
class ConformalResult(NamedTuple):
    coverage: float          # Empirical coverage
    avg_width: float         # Mean prediction interval width
    coverages_t: np.ndarray  # Per-timestep coverage indicators
```

The methods are:
1. `dr_aci()` - Wrapper calling `aci()` with DR scores
2. `vs_dr_aci()` - Variance-standardized version (own implementation)
3. `split_conformal()` - Static split, no adaptation
4. `nexcp()` - Exponentially weighted quantile
5. `aci()` - Core ACI loop (Algorithm 1 from paper)
6. `eci()` - Smooth sigmoid feedback
7. `block_cp()` - Block-permutation conformal
8. `hac()` - Newey-West asymptotic CIs

**Core Algorithm (`aci()`):**
```python
def aci(scores_seq, true_residuals, alpha=0.1, gamma=0.005, n_warmup=50):
    """Sequential ACI with alpha_t update."""
    T = len(scores_seq)
    alpha_t = alpha
    true_covered = np.zeros(T)
    widths = np.zeros(T)

    for t in range(n_warmup, T):
        cal_scores = scores_seq[:t]
        q_hat = np.quantile(cal_scores, min(max(1 - alpha_t, 0), 1))

        self_covered_t = float(scores_seq[t] <= q_hat)
        true_covered[t] = float(true_residuals[t] <= q_hat)
        widths[t] = 2 * q_hat

        err_t = 1 - self_covered_t
        alpha_t = alpha_t + gamma * (alpha - err_t)
        alpha_t = np.clip(alpha_t, 0.01, 0.99)

    valid = slice(n_warmup, T)
    return ConformalResult(
        coverage=true_covered[valid].mean(),
        avg_width=widths[valid].mean(),
        coverages_t=true_covered[valid],
    )
```

**Issue:** The `true_residuals` parameter conflates oracle evaluation with the algorithm. The package should separate these concerns.

**2.2.4 Temporal Cross-Fitting (empirical/panel_draci.py)**

The `temporal_block_crossfit()` function implements K-fold temporal block cross-fitting:

```python
def temporal_block_crossfit(panel: pd.DataFrame, K: int = 5) -> list[dict]:
    dates = sorted(panel['date'].unique())
    block_size = n_dates // K
    block_results = []

    for k in range(K):
        # Split dates into block k vs rest
        block_dates = dates[start_idx:end_idx]
        train_mask = ~panel['date'].isin(block_dates)
        test_mask = panel['date'].isin(block_dates)

        # Fit nuisance on train, predict on test
        e_hat_fn, mu0_hat_fn, mu1_hat_fn, tau_hat_fn = fit_nuisances(...)
        # Store predictions for held-out block
        block_results.append({...})

    return block_results
```

**Issues:**
1. Tightly coupled to pandas DataFrames with specific column names
2. Returns list of dicts instead of structured object
3. No gap handling between training and calibration blocks
4. Hardcoded to M-ELO covariate extraction

**2.2.5 Beta-Mixing Estimation (empirical/mixing_diagnostics.py)**

Clean implementation of histogram-based beta-mixing estimation:

```python
def estimate_beta_mixing(panel, lags, n_bins, n_tickers_sample) -> dict:
    """Estimate beta(tau) = TV(P(X_t, X_{t+tau}) || P(X_t) x P(X_{t+tau}))"""
    for tau in lags:
        # Compute 2D histogram of (X_t, X_{t+tau})
        # Compute product of marginals
        # TV distance = 0.5 * sum |joint - product|
```

**Good:** Self-contained, algorithm is clearly implemented.
**Issue:** Tied to panel DataFrame with 'ticker' and 'hidden_share' columns.

**2.2.6 econml Integration (econometrics/dr_aci.py)**

Alternative implementation using econml's CausalForestDML:

```python
def run_draci_analysis(panel, outcome, alpha, method, ...) -> DRACIResult:
    # 1. Fit nuisance via sklearn cross-fitting (not temporal!)
    # 2. Compute AIPW scores
    # 3. Fit CATE model (RandomForest on pseudo-outcomes)
    # 4. Run split or cross-conformal calibration
    # 5. Compare to econml's asymptotic intervals
```

**Key differences from simulation code:**
1. Uses sklearn's `KFold` (shuffled, not temporal)
2. `DRACIResult` dataclass is richer (includes ticker labels, features)
3. Includes bootstrap fallback when econml unavailable
4. Has plotting functions embedded

**Issue:** This is a parallel implementation with different design decisions. Needs consolidation.

---

## 3. Data Flow Patterns

### 3.1 Simulation Pipeline

```
generate_data(T, rho, rng)
    |
    v
+-------------------+
| X, W, Y, tau_true |  <- Raw data + oracle
+-------------------+
    |
    v
fit_nuisances(X_train, W_train, Y_train)
    |
    v
+------------------------------+
| e_hat_fn, mu0_hat_fn, ...   |  <- Fitted callable functions
+------------------------------+
    |
    v (apply to calibration set)
+-------------------------------------+
| e_hat_cal, mu0_hat_cal, tau_hat_cal |
+-------------------------------------+
    |
    v
dr_score(Y_cal, W_cal, X_cal, e_hat_cal, ...)
    |
    v
+------------+
| scores_seq |
+------------+
    |
    v
aci(scores_seq, true_residuals, alpha, gamma, n_warmup)
    |
    v
+----------------+
| ConformalResult |
+----------------+
```

### 3.2 Empirical Pipeline (Panel)

```
load_daily_data()
    |
    v
prepare_panel(df_td, dev=False)
    |
    v
+----------------------------------+
| panel: DataFrame (ticker, date, W, Y, X) |
+----------------------------------+
    |
    v
temporal_block_crossfit(panel, K=5)
    |
    v
+-----------------------------------+
| block_results: list[dict]         |  <- K blocks of out-of-fold predictions
+-----------------------------------+
    |
    v
assemble_sequential(block_results)
    |
    v
+----------------------------------------+
| seq_df: DataFrame (date-sorted, scores) |
+----------------------------------------+
    |
    v
run_daily_aci(seq_df, score_col, method, ...)
    |
    v
+----------------------------+
| daily_results: DataFrame    |  <- Per-day coverage, width, alpha_t
+----------------------------+
```

### 3.3 Key Observation: Two ACI Implementations

The simulation uses `aci()` from `conformal_methods.py` which:
- Operates on numpy arrays
- Updates alpha_t per observation
- Returns `ConformalResult`

The empirical uses `run_daily_aci()` from `panel_draci.py` which:
- Operates on a DataFrame
- Updates alpha_t per DAY (all tickers pooled)
- Returns a DataFrame with daily metrics

**These need unification.** The package should provide a core ACI class that supports both per-observation and per-period modes.

---

## 4. Dependency Analysis

### 4.1 Module Dependencies (simulation/)

```
config.py
    |
    v
dgp.py <--- config.py (DGP parameters)
    |
    v
nuisance.py <--- scipy, xgboost, sklearn (lazy import)
    |
    v
conformal_methods.py <--- numpy, scipy.stats, nuisance (type alias)
    |
    v
sim_coverage.py <--- all of the above + matplotlib
```

### 4.2 Module Dependencies (empirical/)

```
config.py
    |
    v
data_prep.py <--- config.py, pandas, numpy
    |
    v
panel_draci.py <--- config.py, data_prep.py, nuisance.py (simulation/),
                    conformal_methods.py (simulation/)
    |
    v
mixing_diagnostics.py <--- config.py, data_prep.py, scipy.stats
```

### 4.3 External Dependencies

**Required (core functionality):**
- `numpy>=1.24`
- `scipy>=1.10` (stats.norm for HAC, stats.linregress for mixing)

**Optional (nuisance estimators):**
- `xgboost>=2.0` (XGBoostNuisance)
- `scikit-learn>=1.3` (RandomForestRegressor, KFold, GradientBoosting*)

**Optional (econml integration):**
- `econml>=0.14` (CausalForestDML)

**Optional (panel data support):**
- `pandas>=2.0` (temporal cross-fitting, mixing diagnostics)

---

## 5. Call Graph for Main Algorithm

```
DRACI.fit_predict(Y, W, X, nuisance, tau_true=None)
    |
    +---> score_fn(Y, W, e_hat, mu0_hat, mu1_hat, tau_hat)
    |         |
    |         +---> dr_score() OR vs_dr_score() OR naive_score()
    |
    +---> _run_aci_loop(scores, alpha, gamma, n_warmup)
              |
              +---> for t in range(n_warmup, T):
              |         |
              |         +---> q_hat = np.quantile(scores[:t], 1 - alpha_t)
              |         +---> covered = scores[t] <= q_hat
              |         +---> alpha_t += gamma * (alpha - err_t)
              |
              +---> return IntervalResult(point, lower, upper, coverage_t, alpha_trajectory)
```

For temporal cross-fitting:

```
TemporalCrossFitter.fit_transform(X, W, Y, dates)
    |
    +---> _split_into_blocks(dates, n_blocks)
    |
    +---> for k in range(n_blocks):
    |         |
    |         +---> train_mask, test_mask = _get_block_masks(k)
    |         +---> nuisance.fit(X[train], W[train], Y[train])
    |         +---> predictions[test] = nuisance.predict(X[test])
    |
    +---> return NuisanceResult(e_hat, mu0_hat, mu1_hat, tau_hat)
```

---

## 6. Refactoring Opportunities

### 6.1 Tight Coupling That Should Be Decoupled

| Current Coupling | Recommended Decoupling |
|------------------|------------------------|
| `aci()` receives `true_residuals` for oracle evaluation | Separate `aci()` (returns intervals) from `evaluate_coverage(intervals, tau_true)` |
| `NuisanceTuple` is a tuple | Use `NuisanceResult` dataclass with named fields |
| `temporal_block_crossfit` hardcodes column names | Pass column name parameters or use numpy arrays |
| Score functions receive unused `X` parameter | Remove or document; consider using kwargs |
| `vs_dr_aci` duplicates much of `aci` | Extract common logic into base function |
| `run_daily_aci` embeds pooling strategy | Make pooling strategy configurable |

### 6.2 Duplicated Logic Across Modules

| Duplication | Location 1 | Location 2 | Recommendation |
|-------------|------------|------------|----------------|
| AIPW computation | `conformal_methods.dr_pseudo_outcome()` | `econometrics/dr_aci.compute_aipw_scores()` | Single implementation in `scores.py` |
| Nuisance fitting | `simulation/nuisance.py` | `econometrics/dr_aci.fit_nuisance_models()` | Single `NuisanceEstimator` protocol |
| Propensity clipping | `nuisance.fit_nuisance_xgboost()` | `econometrics/dr_aci.compute_aipw_scores()` | Centralize in nuisance fitting |
| Cross-fitting loop | `empirical/panel_draci.temporal_block_crossfit()` | `econometrics/dr_aci.cross_conformal_cate()` | Abstract `CrossFitter` protocol |
| Conformity score to quantile | `aci()`, `vs_dr_aci()`, `nexcp()`, `eci()` | All share similar loop structure | Abstract `AdaptiveQuantile` strategy |

### 6.3 Missing Error Handling and Validation

| Missing Check | Current Behavior | Recommended |
|---------------|------------------|-------------|
| Propensity near 0 or 1 | Silent division issues | Validate and warn before clipping |
| Empty calibration set | Array index errors | Check `len(cal_scores) >= n_warmup` |
| NaN in inputs | Propagates silently | Validate with clear error messages |
| Mismatched array lengths | Undefined behavior | Assert Y, W, X have consistent shapes |
| Invalid alpha/gamma | No check | Assert `0 < alpha < 1`, `gamma > 0` |
| n_warmup > T | No valid predictions | Assert or warn |
| Single treatment arm in training | Model fitting fails | Check treatment/control sample sizes |

### 6.4 Paper-Specific Code Tangled with Reusable Code

| File | Paper-Specific | Reusable |
|------|----------------|----------|
| `dgp.py` | All (AR(1)+GARCH DGP) | None |
| `config.py` | Paths, RHOS, SAMPLE_SIZES | ALPHA, GAMMA defaults could be extractable |
| `sim_coverage.py` | All (MC orchestration) | None |
| `conformal_methods.py` | `ConformalResult.coverage` computation | Score functions, ACI loop |
| `panel_draci.py` | Column names, figure generation | Cross-fitting pattern, daily ACI loop |
| `mixing_diagnostics.py` | Column names, LaTeX table generation | Beta-mixing estimation algorithm |
| `data_prep.py` | M-ELO specific loading | `get_covariate_matrix` pattern |
| `econometrics/dr_aci.py` | Cutoff dates, feature engineering | AIPW, conformal calibration |

---

## 7. New API Design Recommendations

### 7.1 Public API Surface

The package should expose a minimal public API:

```python
# Core classes
from draci import DRACI, IntervalResult

# Score functions (for advanced users)
from draci.scores import dr_score, vs_dr_score, naive_score, dr_pseudo_outcome

# Nuisance estimation
from draci.nuisance import NuisanceResult, XGBoostNuisance, LinearNuisance

# Cross-fitting (optional)
from draci.cross_fitting import TemporalCrossFitter

# Alternative methods (optional, for benchmarking)
from draci.methods import NexCP, ECI, BlockCP, HAC, SplitConformal
```

### 7.2 Core DRACI Class Design

```python
@dataclass
class IntervalResult:
    """Result from DR-ACI interval construction."""
    point: np.ndarray           # tau_hat(X)
    lower: np.ndarray           # tau_hat - q_t
    upper: np.ndarray           # tau_hat + q_t
    width: np.ndarray           # 2 * q_t
    alpha_trajectory: np.ndarray  # alpha_t over time
    quantile_trajectory: np.ndarray  # q_t over time

    # Optional: only populated if tau_true provided
    coverage_t: np.ndarray | None = None
    empirical_coverage: float | None = None


class DRACI:
    """Doubly Robust Adaptive Conformal Inference.

    Parameters
    ----------
    alpha : float, default=0.10
        Target miscoverage rate (1 - coverage level)
    gamma : float, default=0.005
        Step size for alpha adaptation
    n_warmup : int | float, default=50
        Warmup period. If float < 1, interpreted as fraction of data.
    score : {"dr", "vs_dr", "naive"}, default="dr"
        Score function to use
    clip_alpha : tuple[float, float], default=(0.01, 0.99)
        Bounds for alpha_t during adaptation
    """

    def __init__(self, alpha=0.10, gamma=0.005, n_warmup=50,
                 score="dr", clip_alpha=(0.01, 0.99)):
        ...

    def fit_predict(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        nuisance: NuisanceResult,
        *,
        tau_true: np.ndarray | None = None,
    ) -> IntervalResult:
        """Construct prediction intervals for CATEs.

        Parameters
        ----------
        Y : array of shape (n,)
            Observed outcomes
        W : array of shape (n,)
            Treatment indicators (0 or 1)
        nuisance : NuisanceResult
            Pre-fitted nuisance function predictions
        tau_true : array of shape (n,), optional
            True CATEs for oracle coverage evaluation

        Returns
        -------
        IntervalResult
            Contains point estimates, bounds, and diagnostics
        """
        ...

    def evaluate(
        self,
        result: IntervalResult,
        tau_true: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate coverage and width metrics.

        Returns dict with: coverage, avg_width, coverage_by_quantile, etc.
        """
        ...
```

### 7.3 NuisanceResult Design

```python
@dataclass
class NuisanceResult:
    """Container for nuisance function predictions.

    Attributes
    ----------
    e_hat : array of shape (n,)
        Estimated propensity scores P(W=1|X), clipped to (0, 1)
    mu0_hat : array of shape (n,)
        Estimated E[Y|X, W=0]
    mu1_hat : array of shape (n,)
        Estimated E[Y|X, W=1]
    tau_hat : array of shape (n,)
        Estimated CATE = mu1 - mu0 (or more sophisticated estimator)

    Optional metadata:
    is_cross_fitted : bool
        Whether predictions are out-of-fold
    n_train : int
        Number of training observations
    """
    e_hat: np.ndarray
    mu0_hat: np.ndarray
    mu1_hat: np.ndarray
    tau_hat: np.ndarray
    is_cross_fitted: bool = False
    n_train: int | None = None

    def __post_init__(self):
        # Validate shapes match
        n = len(self.e_hat)
        assert len(self.mu0_hat) == n
        assert len(self.mu1_hat) == n
        assert len(self.tau_hat) == n
        # Validate propensity in (0, 1)
        assert np.all(self.e_hat > 0) and np.all(self.e_hat < 1)
```

### 7.4 Extensibility Points

**Custom Score Functions:**

```python
from typing import Protocol

class ScoreFunction(Protocol):
    def __call__(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        nuisance: NuisanceResult,
    ) -> np.ndarray:
        """Return conformity scores."""
        ...

# Users can implement custom scores
class MyCustomScore:
    def __call__(self, Y, W, nuisance):
        # Custom logic
        return np.abs(Y - nuisance.tau_hat)

draci = DRACI(score=MyCustomScore())
```

**Custom Nuisance Estimators:**

```python
class NuisanceEstimator(Protocol):
    def fit(
        self,
        X: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
    ) -> NuisanceResult:
        """Fit and return nuisance predictions on training data."""
        ...

    def predict(self, X: np.ndarray) -> NuisanceResult:
        """Predict on new data using fitted models."""
        ...

# Users can wrap any ML library
class MyNuisanceEstimator:
    def fit(self, X, W, Y):
        # Use PyTorch, JAX, whatever
        ...
        return NuisanceResult(e_hat, mu0_hat, mu1_hat, tau_hat)
```

### 7.5 Optional Dependencies Handling

```python
# draci/nuisance.py

def _check_xgboost():
    try:
        import xgboost
        return True
    except ImportError:
        return False

class XGBoostNuisance(NuisanceEstimator):
    """XGBoost-based nuisance estimator.

    Requires: pip install draci[ml]
    """

    def __init__(self, ...):
        if not _check_xgboost():
            raise ImportError(
                "XGBoostNuisance requires xgboost. "
                "Install with: pip install draci[ml]"
            )
        ...
```

---

## 8. Dependency Diagram

### 8.1 Current Structure (Paper Repo)

```
                     +----------+
                     | config.py |
                     +----------+
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
     +--------+     +----------+   +----------------+
     | dgp.py |     | nuisance |   | data_prep.py   |
     +--------+     +----------+   +----------------+
          |               |               |
          |               v               |
          |     +-----------------+       |
          +---> | conformal_      | <-----+
                | methods.py      |
                +-----------------+
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
    +-----------+  +-----------+  +-------------+
    | sim_cov.  |  | panel_    |  | mixing_     |
    | erage.py  |  | draci.py  |  | diag.py     |
    +-----------+  +-----------+  +-------------+
```

### 8.2 Proposed Structure (Package)

```
                     +----------+
                     | types.py |
                     +----------+
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
     +-----------+  +-----------+  +----------------+
     | scores.py |  | nuisance/ |  | cross_fitting/ |
     +-----------+  +-----------+  +----------------+
          |               |               |
          |               v               |
          |     +-----------------+       |
          +---> |    core.py      | <-----+
                |    (DRACI)      |
                +-----------------+
                        |
          +-------------+
          |             |
          v             v
    +-----------+  +-------------+
    | methods/  |  | mixing.py   |
    | (NexCP,   |  | (optional)  |
    | ECI, ...) |  +-------------+
    +-----------+
```

### 8.3 Import Graph (Proposed)

```
draci/
  __init__.py
      imports from: core, types, scores, nuisance

  types.py
      imports from: numpy, dataclasses
      imported by: all modules

  scores.py
      imports from: numpy, types
      imported by: core, methods

  nuisance/__init__.py
      imports from: types
      imported by: core, cross_fitting

  nuisance/xgboost.py
      imports from: xgboost (optional), sklearn, nuisance
      imported by: nuisance/__init__

  nuisance/linear.py
      imports from: scipy.optimize, nuisance
      imported by: nuisance/__init__

  core.py
      imports from: numpy, types, scores
      imported by: __init__

  cross_fitting.py
      imports from: numpy, pandas (optional), nuisance, types
      imported by: __init__ (optional)

  methods/__init__.py
      imports from: core (base class)
      imported by: __init__ (optional)

  mixing.py
      imports from: numpy, scipy.stats, pandas (optional)
      imported by: __init__ (optional)
```

---

## 9. Implementation Roadmap

### Phase 1: Core (Week 1)

1. **types.py** - IntervalResult, NuisanceResult dataclasses
2. **scores.py** - dr_score, vs_dr_score, naive_score, dr_pseudo_outcome
3. **core.py** - DRACI class with ACI loop
4. **tests/test_scores.py** - Unit tests for score functions
5. **tests/test_core.py** - Integration tests with synthetic data

### Phase 2: Nuisance Estimators (Week 1-2)

1. **nuisance/base.py** - NuisanceEstimator protocol
2. **nuisance/xgboost.py** - XGBoostNuisance (optional dependency)
3. **nuisance/linear.py** - LinearNuisance (no extra deps)
4. **tests/test_nuisance.py** - Tests with mock data

### Phase 3: Cross-Fitting (Week 2)

1. **cross_fitting.py** - TemporalCrossFitter
2. **tests/test_cross_fitting.py** - Tests for block splitting

### Phase 4: Alternative Methods (Week 2-3)

1. **methods/nexcp.py** - NexCP
2. **methods/eci.py** - ECI
3. **methods/block_cp.py** - BlockCP
4. **methods/hac.py** - HAC
5. **methods/split.py** - SplitConformal
6. **tests/test_methods.py** - Comparison tests

### Phase 5: Optional Extensions (Week 3-4)

1. **mixing.py** - Beta-mixing estimation
2. **_econml_compat.py** - econml CausalForestDML integration
3. **examples/** - Usage examples
4. **docs/** - API documentation

---

## 10. Testing Strategy

### 10.1 Unit Tests

- **Score functions:** Verify against hand-calculated examples
- **ACI loop:** Check alpha_t trajectory, quantile computation
- **Nuisance fitting:** Check predictions have correct shapes/types

### 10.2 Property-Based Tests

- **Coverage:** Given iid data, coverage should be close to 1-alpha
- **Monotonicity:** Wider intervals should have higher coverage
- **Adaptation:** alpha_t should converge under stationary conditions

### 10.3 Integration Tests

- **End-to-end:** Simulate AR(1) DGP, fit nuisance, run DRACI, check coverage
- **Cross-fitting:** Verify out-of-fold predictions are correct

### 10.4 Regression Tests

- **Reproduce paper results:** Use saved RNG state to match paper Table 1

---

## 11. Open Design Questions

1. **Should IntervalResult store all intermediate quantities (alpha_t, q_t, etc.)?**
   - Pro: Useful for diagnostics and plotting
   - Con: Memory overhead for large T

2. **How to handle panel data (multiple tickers)?**
   - Option A: First-class support with `PanelDRACI` class
   - Option B: Document pattern for users to handle grouping
   - Recommendation: Start with Option B, add A if demand

3. **Should alternative methods share a base class?**
   - Pro: Consistent interface, easier comparison
   - Con: Some methods are fundamentally different (HAC is asymptotic)
   - Recommendation: Use Protocol for interface, not inheritance

4. **How to handle streaming/online mode?**
   - Current code is batch (full history available)
   - True online would need `DRACI.update(new_obs)`
   - Recommendation: Defer to v0.2

5. **What validation strictness level?**
   - Option A: Strict (assert on all invalid inputs)
   - Option B: Lenient (warn and proceed where possible)
   - Recommendation: Strict for v0.1, can relax later

---

## 12. Migration Checklist

### Files to Extract (verbatim or with minor edits)

- [ ] `simulation/conformal_methods.py` lines 44-117 (score functions)
- [ ] `simulation/conformal_methods.py` lines 250-280 (`aci()` function)
- [ ] `simulation/nuisance.py` lines 19-93 (XGBoost nuisance)
- [ ] `simulation/nuisance.py` lines 96-158 (Linear nuisance)
- [ ] `empirical/panel_draci.py` lines 36-110 (temporal cross-fit pattern)
- [ ] `empirical/mixing_diagnostics.py` lines 101-189 (beta-mixing estimation)

### Files to Leave Behind

- [ ] All `config.py` files (paper-specific constants)
- [ ] `simulation/dgp.py` (paper-specific DGP)
- [ ] `simulation/sim_coverage.py` (MC orchestration)
- [ ] `empirical/data_prep.py` (M-ELO specific)
- [ ] All figure/table generation code
- [ ] `econometrics/*` except dr_aci.py patterns

### Tests to Port

- [ ] `simulation/tests/test_conformal.py` (if exists)
- [ ] `simulation/tests/test_nuisance.py` (if exists)

---

## 13. References

- Paper repo: `/media/ak/10E1026C4FA6006E/GitRepos/TrackF_Paper_1/src/`
- Target repo: `/media/ak/10E1026C4FA6006E/GitRepos/draci-package/`
- Existing README.md in target has API sketches
- Paper: DR-ACI for CATEs under beta-mixing (Section 5-6)
