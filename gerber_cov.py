import numpy as np
import pandas as pd


def project_to_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project a symmetric matrix to PSD by clipping negative eigenvalues.
    """
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(vals) @ vecs.T


def gerber_covariance(
    returns_df: pd.DataFrame,
    c: float = 0.5,
    eps: float = 1e-10,
    return_df: bool = True,
) -> pd.DataFrame | np.ndarray:
    """
    Gerber covariance estimator.

    For each asset pair (i, j), only large moves are counted:
    - up if r_i,t >= c * sigma_i
    - down if r_i,t <= -c * sigma_i

    Gerber correlation:
        g_ij = (N_UU + N_DD - N_UD - N_DU) / (N_UU + N_DD + N_UD + N_DU)

    Covariance:
        cov_ij = g_ij * sigma_i * sigma_j
    """
    x = returns_df.values
    sig = returns_df.std(ddof=1).values
    sig = np.where(sig <= 0, eps, sig)

    up = (x >= (c * sig)).astype(np.int8)
    down = (x <= (-c * sig)).astype(np.int8)

    n_uu = up.T @ up
    n_dd = down.T @ down
    n_ud = up.T @ down
    n_du = down.T @ up

    denom = n_uu + n_dd + n_ud + n_du
    numer = n_uu + n_dd - n_ud - n_du

    g = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer, dtype=float),
        where=denom > 0,
    )
    np.fill_diagonal(g, 1.0)

    cov = g * np.outer(sig, sig)
    cov = project_to_psd(cov, eps=eps)

    if return_df:
        return pd.DataFrame(cov, index=returns_df.columns, columns=returns_df.columns)
    return cov
