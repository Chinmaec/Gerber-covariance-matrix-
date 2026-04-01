# Gerber Covariance Matrix (Python)

I built this project while working on portfolio optimization and looking for a covariance estimate that is more stable than the plain sample covariance matrix.

In practice, sample covariance can be noisy and very sensitive to outliers. To address that, I implemented the **Gerber covariance approach** in Python, which focuses on significant directional moves (above a volatility-scaled threshold) to estimate dependence more robustly instead of using all return moves equally. 

## What’s in this repo
- `gerber_cov.py` — core Gerber covariance implementation
- `sample_data.csv` — example return data for testing
- `cookbook.ipynb` — usage + comparison vs raw sample covariance

## Method

Given returns $r_{i,t}$ for asset $i$ at time $t$:

1. Compute asset volatility:
$$
\sigma_i = \mathrm{std}(r_i, \mathrm{ddof}=1)
$$

2. Define thresholded direction indicators using parameter $c$:
$$
U_{i,t} = \mathbf{1}(r_{i,t} \ge c\sigma_i), \quad
D_{i,t} = \mathbf{1}(r_{i,t} \le -c\sigma_i)
$$

3. Count directional co-moves for each pair $(i,j)$:
$$
N_{UU}=U^\top U,\quad N_{DD}=D^\top D,\quad N_{UD}=U^\top D,\quad N_{DU}=D^\top U
$$

4. Compute Gerber correlation:
$$
g_{ij}=\frac{N_{UU}+N_{DD}-N_{UD}-N_{DU}}
{N_{UU}+N_{DD}+N_{UD}+N_{DU}}
$$
(with diagonal set to 1)

5. Convert to covariance:
$$
\Sigma^{(G)}_{ij}=g_{ij}\sigma_i\sigma_j
$$

6. Enforce positive semidefiniteness by eigenvalue clipping:
$$
\Sigma^{(G)} = Q\,\mathrm{diag}(\max(\lambda_k,\epsilon))\,Q^\top
$$


## Python API

```python
gerber_covariance(
    returns_df: pd.DataFrame,
    c: float = 0.5,
    eps: float = 1e-10,
    return_df: bool = True,
)

