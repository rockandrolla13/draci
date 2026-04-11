# draci: Doubly Robust Adaptive Conformal Inference

**Status: Planning / Design Phase**

This package aims to provide a clean, reusable implementation of Doubly Robust Adaptive Conformal Inference (DR-ACI) for constructing prediction intervals on Conditional Average Treatment Effects (CATEs) under temporal dependence.

---

## What This Package Will Provide

DR-ACI combines three ideas:
1. **Doubly robust pseudo-outcomes** for consistent CATE estimation even with model misspecification
2. **Adaptive conformal inference** for finite-sample coverage guarantees that adapt online
3. **Temporal block cross-fitting** for handling dependent data (beta-mixing time series)

The goal is a simple API:

```python
from draci import DRACI, TemporalCrossFitter

# Fit nuisance models with temporal block cross-fitting
fitter = TemporalCrossFitter(n_blocks=5)
nuisance = fitter.fit(X, W, Y, dates)

# Run DR-ACI
draci = DRACI(alpha=0.1, gamma=0.005)
intervals = draci.fit_predict(X, W, Y, nuisance)

# intervals.lower, intervals.upper, intervals.coverage
```

---

## Code to Port from Paper Repository

Source: `/media/ak/10E1026C4FA6006E/GitRepos/TrackF_Paper_1/src/`

### Core Algorithm (MUST port)

| Source File | Target Module | What to Extract |
|-------------|---------------|-----------------|
| `simulation/conformal_methods.py` | `draci/core.py` | `aci()` function (Algorithm 1), `dr_aci()` wrapper |
| `simulation/conformal_methods.py` | `draci/scores.py` | `dr_score()`, `dr_pseudo_outcome()`, `vs_dr_score()`, `naive_score()` |
| `simulation/nuisance.py` | `draci/nuisance.py` | `NuisanceTuple` type, `fit_nuisances()` dispatcher |
| `empirical/panel_draci.py` | `draci/cross_fitting.py` | `temporal_block_crossfit()` logic |

### Supporting Methods (SHOULD port)

| Source File | Target Module | What to Extract |
|-------------|---------------|-----------------|
| `simulation/conformal_methods.py` | `draci/methods.py` | `vs_dr_aci()`, `nexcp()`, `eci()`, `split_conformal()`, `hac()`, `block_cp()` |
| `empirical/mixing_diagnostics.py` | `draci/mixing.py` | `estimate_beta_mixing()`, `compute_ticker_acf()` |
| `simulation/conformal_methods.py` | `draci/types.py` | `ConformalResult` NamedTuple |

### Optional Extensions (MAY port)

| Source File | Target Module | Notes |
|-------------|---------------|-------|
| `econometrics/dr_aci.py` | `draci/econml_compat.py` | Integration with econml's CausalForestDML |
| `simulation/tests/` | `tests/` | Unit tests for score functions, ACI convergence |

---

## What NOT to Port (Paper-Specific)

These are tied to the paper's experiments and should stay in the paper repo:

- `simulation/dgp.py` - Simulation DGP (AR(1) + GARCH) specific to Section 5
- `simulation/sim_coverage.py` - Monte Carlo simulation orchestration
- `simulation/config.py` - Paper-specific configuration (rho values, sample sizes)
- `empirical/data_prep.py` - M-ELO dataset loading (proprietary data)
- `empirical/cross_sectional.py`, `panel_draci.py` - Paper experiments
- `empirical/sensitivity.py`, `misspecification.py` - Paper robustness checks
- `econometrics/*` - DiD framework (separate concern from conformal inference)
- All figure/table generation code

---

## Proposed Package Structure

```
draci/
    __init__.py           # Public API: DRACI, TemporalCrossFitter, scores
    core.py               # Main DRACI class wrapping ACI algorithm
    scores.py             # Score functions: dr_score, vs_dr_score, naive_score
    nuisance.py           # NuisanceEstimator protocol + implementations
    cross_fitting.py      # TemporalCrossFitter for dependent data
    types.py              # ConformalResult, IntervalResult dataclasses
    methods.py            # Alternative methods: NexCP, ECI, BlockCP, HAC
    mixing.py             # Optional: beta-mixing estimation utilities
    _compat.py            # Optional: econml integration

tests/
    test_scores.py
    test_core.py
    test_cross_fitting.py
    conftest.py           # Fixtures with synthetic data

examples/
    basic_usage.py
    temporal_crossfit.py
    comparison_methods.py
```

---

## Core Module Designs

### `draci/core.py`

```python
from dataclasses import dataclass
from typing import Protocol, Literal
import numpy as np

@dataclass
class IntervalResult:
    """Result from DR-ACI interval construction."""
    point: np.ndarray       # CATE point estimates
    lower: np.ndarray       # Lower bounds
    upper: np.ndarray       # Upper bounds
    width: np.ndarray       # Interval widths
    coverage_t: np.ndarray  # Per-timestep coverage indicators (if oracle available)
    alpha_trajectory: np.ndarray  # Adaptive alpha over time

class DRACI:
    """Doubly Robust Adaptive Conformal Inference for CATEs.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate (default 0.10 for 90% coverage)
    gamma : float
        ACI step size for alpha updates (default 0.005)
    n_warmup : int or float
        Warmup period before adaptation. If float < 1, interpreted as fraction.
    score : Literal["dr", "vs_dr", "naive"]
        Score function to use
    """

    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.005,
        n_warmup: int | float = 50,
        score: Literal["dr", "vs_dr", "naive"] = "dr",
    ):
        ...

    def fit_predict(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        X: np.ndarray,
        nuisance: "NuisanceResult",
        tau_true: np.ndarray | None = None,  # For oracle coverage
    ) -> IntervalResult:
        """Construct prediction intervals for CATEs.

        Runs ACI sequentially, updating alpha_t based on self-coverage.
        If tau_true provided, also computes true coverage.
        """
        ...
```

### `draci/scores.py`

```python
import numpy as np

def dr_score(
    Y: np.ndarray,
    W: np.ndarray,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    tau_hat: np.ndarray,
) -> np.ndarray:
    """Doubly robust conformity score: |psi^DR - tau_hat|.

    psi^DR = W/e*(Y-mu1) - (1-W)/(1-e)*(Y-mu0) + mu1 - mu0

    Returns absolute residuals for conformal calibration.
    """
    psi_dr = (
        W / e_hat * (Y - mu1_hat)
        - (1 - W) / (1 - e_hat) * (Y - mu0_hat)
        + mu1_hat - mu0_hat
    )
    return np.abs(psi_dr - tau_hat)


def vs_dr_score(
    Y: np.ndarray,
    W: np.ndarray,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    tau_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Variance-standardized DR score.

    Returns (standardized_scores, sigma_hat) for tighter intervals
    when variance is heterogeneous.
    """
    ...


def naive_score(
    Y: np.ndarray,
    W: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    tau_hat: np.ndarray,
) -> np.ndarray:
    """Non-DR plug-in score (baseline comparison)."""
    ...
```

### `draci/nuisance.py`

```python
from typing import Protocol, Callable
import numpy as np

class NuisanceEstimator(Protocol):
    """Protocol for nuisance function estimators."""

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> "NuisanceResult":
        """Fit propensity, outcome models, and CATE."""
        ...

@dataclass
class NuisanceResult:
    """Container for fitted nuisance predictions."""
    e_hat: np.ndarray       # P(W=1|X)
    mu0_hat: np.ndarray     # E[Y|X, W=0]
    mu1_hat: np.ndarray     # E[Y|X, W=1]
    tau_hat: np.ndarray     # CATE estimate


class XGBoostNuisance:
    """XGBoost-based nuisance estimator."""

    def __init__(
        self,
        propensity_params: dict | None = None,
        outcome_params: dict | None = None,
        cate_params: dict | None = None,
        clip_propensity: tuple[float, float] = (0.05, 0.95),
    ):
        ...


class LinearNuisance:
    """Linear nuisance estimator (fast baseline)."""
    ...
```

### `draci/cross_fitting.py`

```python
import numpy as np
import pandas as pd
from typing import Callable

class TemporalCrossFitter:
    """K-fold temporal block cross-fitting for dependent data.

    Splits time series into K contiguous blocks. For each block,
    trains nuisance models on all OTHER blocks, then predicts on
    the held-out block. This preserves temporal ordering and
    avoids data leakage across time.

    Parameters
    ----------
    n_blocks : int
        Number of temporal blocks (default 5)
    nuisance_estimator : NuisanceEstimator
        Estimator to use for each fold
    """

    def __init__(
        self,
        n_blocks: int = 5,
        nuisance_estimator: "NuisanceEstimator" = None,
    ):
        ...

    def fit_transform(
        self,
        X: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
        dates: np.ndarray | pd.Series,
    ) -> NuisanceResult:
        """Fit nuisance models via temporal cross-fitting.

        Returns out-of-block predictions for all observations.
        """
        ...
```

---

## Dependencies

Core (required):
```
numpy>=1.24
scipy>=1.10
```

Nuisance estimators (optional, but recommended):
```
scikit-learn>=1.3
xgboost>=2.0
```

econml integration (optional):
```
econml>=0.14
```

---

## Design Decisions to Make

1. **Score function interface**: Should scores be functions or classes?
   - Current: Functions (simpler, matches paper code)
   - Alternative: Protocol class for custom scores

2. **Nuisance estimator coupling**: How tightly to couple with sklearn/xgboost?
   - Current: Provide XGBoost impl, allow any sklearn-like estimator
   - Alternative: Pure protocol, no concrete implementations

3. **Panel data support**: First-class support for ticker-date panels?
   - Current: Focus on single time series, document panel usage
   - Alternative: `PanelDRACI` class with grouped operations

4. **Streaming vs batch**: Support true online operation?
   - Current: Batch mode (full history available)
   - Alternative: `DRACI.update(new_obs)` method

5. **Alternative methods**: Include NexCP, ECI, etc. or just DR-ACI?
   - Recommendation: Include in `draci.methods` for benchmarking
   - Keep core API focused on DR-ACI

---

## API Comparison with Existing Packages

| Feature | draci (proposed) | MAPIE | conformal-prediction |
|---------|------------------|-------|----------------------|
| Causal inference focus | Yes | No | No |
| DR scores | Yes | No | No |
| Temporal cross-fitting | Yes | No | No |
| Online adaptation (ACI) | Yes | No | Partial |
| Mixing diagnostics | Yes | No | No |
| econml integration | Planned | No | No |

---

## References

- Gibbs & Candes (2021): "Adaptive Conformal Inference Under Distribution Shift"
- Chernozhukov et al. (2018): "Double/Debiased Machine Learning"
- Lei & Candes (2021): "Conformal Inference of Counterfactuals"
- Barber et al. (2023): "Conformal Prediction Beyond Exchangeability"

---

## Next Steps

1. [ ] Finalize API design (this document)
2. [ ] Extract and refactor `core.py` from paper code
3. [ ] Extract and refactor `scores.py`
4. [ ] Add comprehensive tests with synthetic data
5. [ ] Write examples and documentation
6. [ ] Package and publish to PyPI
