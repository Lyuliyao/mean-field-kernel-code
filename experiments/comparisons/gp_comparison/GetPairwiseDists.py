# Original code from Charles Kulick for the paper "Learning particle swarming models from data with Gaussian processes" by J. Feng, C. Kulick, Y. Ren, S. Tang (2023)

import numpy as np


def get_pairwise_dists(data_out_x, d, N, L, M):
    """
    Compute all pairwise distances from two-species trajectory data.

    Parameters:
    data_out_x : ndarray, shape (L, 2*N, M), species 1 trajectories
    data_out_z : ndarray, shape (L, 2*p, M), species 2 trajectories
    d : int, spatial dimension (2)
    N : int, number of species 1 agents
    L : int, number of time points
    p : int, number of species 2 agents
    M : int, number of trajectories

    Returns:
    (d11, d12, d21, d22) : ndarrays of pairwise distances aggregated over all (l, m)
    """
    d11 = []

    for m in range(M):
        for l in range(L):
            x = data_out_x[l, :, m]

            for i in range(N):
                xi = x[d*i : d*(i+1)]
                for j in range(i + 1, N):
                    d11.append(np.linalg.norm(xi - x[d*j : d*(j+1)]))

    d11 = np.array(d11)

    return d11