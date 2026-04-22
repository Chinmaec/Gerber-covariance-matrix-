# Gerber Covariance Matrix Implementation(Python)

I built this project while working on portfolio optimization and looking for a covariance estimate that is more stable than the plain sample covariance matrix.

In practice, sample covariance can be noisy and very sensitive to outliers. To address that, I implemented the **Gerber covariance approach** in Python, which focuses on significant directional moves (above a volatility-scaled threshold) to estimate dependence more robustly instead of using all return moves equally. 

## What’s in this repo
- `gerber_cov.py` — core Gerber covariance implementation
- `sample_data.csv` — example return data for testing
- `cookbook.ipynb` — usage + comparison vs raw sample covariance

## Method

Given returns `r[i,t]` for asset `i` at time `t`:

1. Compute volatility for each asset  
   `sigma[i] = std(r[i], ddof=1)`

2. Mark large positive/negative moves using threshold `c`  
   `U[i,t] = 1 if r[i,t] >= c * sigma[i], else 0`  
   `D[i,t] = 1 if r[i,t] <= -c * sigma[i], else 0`

3. Count pairwise co-moves across time  
   `N_UU = U.T @ U`  
   `N_DD = D.T @ D`  
   `N_UD = U.T @ D`  
   `N_DU = D.T @ U`

4. Build Gerber correlation  
   `g[i,j] = (N_UU + N_DD - N_UD - N_DU) / (N_UU + N_DD + N_UD + N_DU)`  
   Diagonal is set to `1.0`.

5. Convert to covariance  
   `Cov_G[i,j] = g[i,j] * sigma[i] * sigma[j]`

6. Project to PSD (positive semidefinite)  
   Eigen-decompose covariance, clip negative eigenvalues to `eps`, and reconstruct.

