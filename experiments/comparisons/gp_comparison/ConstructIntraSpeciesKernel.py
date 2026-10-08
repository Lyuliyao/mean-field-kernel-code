# Original code from Charles Kulick for the paper "Learning particle swarming models from data with Gaussian processes" by J. Feng, C. Kulick, Y. Ren, S. Tang (2023)

import numpy as np
import torch


def _matern_kernel_torch(r_matrix, omega, sigma, nu):
    """
    Matern kernel on a torch tensor distance matrix.

    Parameters:
    r_matrix : Tensor, pairwise distance matrix
    omega : float, length scale
    sigma : float, signal variance
    n : int, total agent count (for 1/n^2 scaling)
    nu : float, smoothness (0.5, 1.5, 2.5, 3.5)
    """
    s = 1.0

    if nu == 0.5:
        return s * sigma**2 * torch.exp(-omega * r_matrix)

    elif nu == 1.5:
        t = 3**0.5 * omega * r_matrix
        return s * sigma**2 * (1 + t) * torch.exp(-t)

    elif nu == 2.5:
        t = 5**0.5 * omega * r_matrix
        return s * sigma**2 * (1 + t + 5/3 * omega**2 * r_matrix**2) * torch.exp(-t)

    elif nu == 3.5:
        t = 7**0.5 * omega * r_matrix
        return s * sigma**2 * (1 + t + 14/5 * omega**2 * r_matrix**2
                                + 7/15 * 7**0.5 * omega**3 * r_matrix**3) * torch.exp(-t)

    else:
        raise ValueError(f"Unsupported Matern nu={nu}")


def _matern_kernel_omega_partial_torch(r_matrix, omega, sigma, nu):
    """
    dK/domega on a torch tensor distance matrix.

    Same parameters as _matern_kernel_torch.
    """
    s = 1.0

    if nu == 0.5:
        return s * sigma**2 * (-r_matrix) * torch.exp(-omega * r_matrix)

    elif nu == 1.5:
        t = 3**0.5 * omega * r_matrix
        return -s * sigma**2 * 3 * omega * r_matrix**2 * torch.exp(-t)

    elif nu == 2.5:
        t = 5**0.5 * omega * r_matrix
        exp_t = torch.exp(-t)
        f  = 1 + t + 5 * omega**2 * r_matrix**2 / 3
        df = 5**0.5 * r_matrix + 10 * omega * r_matrix**2 / 3
        dg = -5**0.5 * r_matrix * exp_t
        return s * sigma**2 * (df * exp_t + f * dg)

    elif nu == 3.5:
        t = 7**0.5 * omega * r_matrix
        exp_t = torch.exp(-t)
        f  = (1 + t + 14/5 * omega**2 * r_matrix**2
              + 7/15 * 7**0.5 * omega**3 * r_matrix**3)
        df = (7**0.5 * r_matrix + 28/5 * omega * r_matrix**2
              + 7/5 * 7**0.5 * omega**2 * r_matrix**3)
        dg = -7**0.5 * r_matrix * exp_t
        return s * sigma**2 * (df * exp_t + f * dg)

    else:
        raise ValueError(f"Unsupported Matern nu={nu}")


def construct_intra_species_kernel(species_data, omega, delta, n, D, M, L, nu, partial_omega=False,
                                   device=None):
    """
    Build the intra-species GP kernel matrix.

    Each of the M*L snapshots contributes n*(n-1)/2 agent pairs. U_s encodes
    relative displacement vectors; R_s is the Matern kernel evaluated on pairwise
    distances between paired distances.

    Parameters:
    species_data : ndarray, shape (D*n, L, M)
    omega, delta : float, kernel length scale and signal variance
    n : int, number of agents in this species
    D : int, spatial dimension
    M, L : int, trajectories and time points
    nu : float, Matern smoothness
    total_n_for_normalization : int, N1+N2, used for the 1/n^2 scaling in the kernel
    partial_omega : bool, if True, return dK/domega instead of K
    device : torch.device or None, defaults to CUDA if available

    Returns:
    K : ndarray, shape (D*n*L*M, D*n*L*M)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_snapshots = M * L
    tilde_n = (n * (n - 1) * total_snapshots) // 2
    cap_n   = n * D * total_snapshots

    # Build list of (snapshot_idx, m, l, i, j) for all agent pairs across all snapshots
    combinations = []
    for snapshot_idx in range(total_snapshots):
        m = snapshot_idx // L
        l = snapshot_idx % L
        for i in range(n - 1):
            for j in range(i + 1, n):
                combinations.append((snapshot_idx, m, l, i, j))
    combinations = np.array(combinations)

    # Extract agent positions for each pair
    i_positions = np.zeros((tilde_n, D))
    j_positions = np.zeros((tilde_n, D))
    for idx in range(tilde_n):
        snapshot_idx, m, l, i, j = combinations[idx]
        i_positions[idx] = species_data[D*i:D*(i+1), l, m]
        j_positions[idx] = species_data[D*j:D*(j+1), l, m]

    u_ij = i_positions - j_positions
    u_ji = -u_ij

    # Fill U_s, each pair contributes to rows for agent i and agent j in its snapshot
    U_s = np.zeros((cap_n, tilde_n))
    for idx in range(tilde_n):
        snapshot_idx, m, l, i, j = combinations[idx]
        i_start = snapshot_idx * D * n + i * D
        j_start = snapshot_idx * D * n + j * D
        U_s[i_start:i_start+D, idx] = u_ij[idx]
        U_s[j_start:j_start+D, idx] = u_ji[idx]

    # Sort pairs by distance
    ds = np.linalg.norm(u_ij, axis=1)
    order = np.argsort(ds)

    # Move to device
    U_s_t = torch.from_numpy(U_s[:, order]).to(device)
    ds_t  = torch.from_numpy(ds[order]).to(device)
    ds_matrix = torch.abs(ds_t[:, None] - ds_t[None, :])

    kernel_fn = _matern_kernel_omega_partial_torch if partial_omega else _matern_kernel_torch
    
    R_s = kernel_fn(ds_matrix, omega, delta, nu)
    
    K = (U_s_t @ R_s @ U_s_t.T).cpu().numpy()

    return K