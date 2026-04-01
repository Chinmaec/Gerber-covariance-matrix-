# Gerber Covariance Matrix (Python)

I built this project while trying to improve portfolio optimization with a covariance estimate that is more stable than the standard sample covariance matrix.

In practice, sample covariance can be noisy and very sensitive to outliers. To address that, I implemented the **Gerber covariance approach** in Python, which focuses on directional co-movement under thresholded return moves. The goal is to get cleaner risk estimates and more reliable portfolio weights.

## What’s in this repo
- `gerber_cov.py` — core Gerber covariance implementation
- `sample_data.csv` — example return data for testing
- supporting scripts for covariance experiments and portfolio workflows

## Why this matters
- Robust covariance estimation is a core problem in quantitative portfolio construction
- Better covariance inputs can improve optimization stability and reduce extreme allocations

## Next steps
- Compare Gerber covariance vs sample/shrinkage covariance on out-of-sample performance
- Add backtests across different market regimes
- Package as a reusable module with tests and documentation
