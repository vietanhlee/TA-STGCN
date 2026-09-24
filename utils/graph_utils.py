import numpy as np
import pandas as pd


def load_adj_from_excel(path: str):
    """
    Loads spatial adjacency distance matrix from Excel (.xlsx) or CSV (.csv) file.
    
    Args:
        path (str): Path to Excel or CSV file containing distance matrix.

    Returns:
        tuple: (weight_matrix, list_of_node_ids)
    """
    if str(path).lower().endswith('.csv'):
        df = pd.read_csv(path, index_col=0)
    else:
        df = pd.read_excel(path, sheet_name=0, index_col=0)
    mat = df.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=float)
    nonzero = mat[mat > 0]
    sigma = nonzero.mean() if nonzero.size > 0 else 1.0
    weights = np.zeros_like(mat)
    mask = mat > 0
    weights[mask] = np.exp(-mat[mask] / (sigma + 1e-9))
    return weights, list(df.index)


def normalize_adj_sym(A: np.ndarray) -> np.ndarray:
    """
    Symmetric normalization of adjacency matrix A:
        A_tilde = D^{-1/2} (A + I) D^{-1/2}
    """
    A = A.astype(float)
    A = A + np.eye(A.shape[0])
    d = A.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d, dtype=float)
    np.power(d, -0.5, where=d > 0, out=d_inv_sqrt)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def compute_scaled_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Computes Scaled Chebyshev Laplacian L_tilde for Chebyshev Spectral Graph Convolutions.
    
    Mathematical Formulation:
        L_norm = I - D^{-1/2} A D^{-1/2}
        L_tilde = (2 / lambda_max) * L_norm - I
        where lambda_max is the largest eigenvalue of L_norm.
    """
    A = A.astype(float)
    n = A.shape[0]
    d = A.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d, dtype=float)
    np.power(d, -0.5, where=d > 0, out=d_inv_sqrt)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    try:
        eigenvalues = np.linalg.eigvalsh(L_norm)
        lambda_max = eigenvalues[-1]
    except Exception:
        lambda_max = 2.0

    if lambda_max < 1e-6:
        lambda_max = 2.0

    L_tilde = 2.0 * L_norm / lambda_max - np.eye(n)
    return L_tilde
