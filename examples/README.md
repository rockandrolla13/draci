# DR-ACI Examples

This directory contains example scripts demonstrating typical usage patterns for the `draci` package.

## Examples

### 01_basic_usage.py
**Basic DR-ACI workflow**
- Generate synthetic data with temporal dependence
- Estimate nuisance functions (propensity, outcome models)
- Compute doubly robust scores
- Run adaptive conformal inference
- Evaluate coverage and interval width

### 02_draci_class.py
**Using the high-level DRACI class**
- Compare score types: "dr", "vs_dr", "naive"
- Access interval bounds (lower, upper)
- Work with coverage indicators

### 03_temporal_crossfitting.py
**Temporal block cross-fitting for dependent data**
- `temporal_block_crossfit()` convenience function
- `TemporalCrossFitter` class interface
- Compare cross-fitting vs single split

### 04_comparing_methods.py
**DR-ACI vs CI baselines**
- Compare with DML-Wald CI
- Compare with Block Bootstrap CI
- Evaluate across different dependence levels (rho)

### 05_mixing_diagnostics.py
**Assessing temporal dependence**
- Compute autocorrelation function (ACF)
- Estimate beta-mixing coefficients
- Use diagnostics to inform ACI parameters

## Running Examples

```bash
# Install draci first
pip install draci

# Run individual examples
python 01_basic_usage.py
python 02_draci_class.py
python 03_temporal_crossfitting.py
python 04_comparing_methods.py
python 05_mixing_diagnostics.py
```

## Key Concepts

### Doubly Robust Scores
The DR score combines propensity and outcome models:
```
|psi^DR - tau_hat| where psi^DR = mu1_hat + (W/e_hat)(Y - mu1_hat) - mu0_hat - ((1-W)/(1-e_hat))(Y - mu0_hat)
```

### Adaptive Conformal Inference
ACI updates the miscoverage target online:
```
alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
```
where `err_t = 1{score_t > q_hat_t}`.

### Temporal Cross-Fitting
Splits time series into K contiguous blocks, trains on K-1, predicts on held-out block. Preserves temporal ordering.
