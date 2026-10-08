# Original code from Charles Kulick for the paper "Learning particle swarming models from data with Gaussian processes" by J. Feng, C. Kulick, Y. Ren, S. Tang (2023)

import numpy as np
from GPPredictionUtils import cov_matern


def k_r_explicit(r, source_data_3d, N, d, L, hyp, nu, kernel_type, influence_data_3d=None, P=None):
    """
    Compute the cross-covariance row vector K(r, X) for GP prediction at distance r.

    Parameters:
    r : float, query distance
    source_data_3d : ndarray, shape (d*N, L, M), source species positions
    N : int, number of source species agents
    d : int, spatial dimension
    L : int, number of time points
    hyp : array, [log(sigma), log(omega)]
    nu : float, Matern smoothness
    kernel_type : str, intra or inter
    influence_data_3d : ndarray, shape (d*P, L, M), influence species positions (inter only)
    P : int, number of influence species agents

    Returns:
    Kr : ndarray, shape (1, d*N*L*M)
    """
    M = source_data_3d.shape[2]
    sigma = np.exp(hyp[0])
    omega = np.exp(hyp[1])

    Kr = np.zeros((1, d * N * L * M))

    if kernel_type == 'intra':
        for m in range(M):
            for l in range(L):
                X = np.zeros((d, N))
                for i in range(N):
                    X[:, i] = source_data_3d[d*i:d*(i+1), l, m]

                for i in range(N):
                    diff = X[:, [i]] - X
                    norms = np.linalg.norm(diff, axis=0)
                    covs = np.array([cov_matern(norm, r, sigma, omega, nu) for norm in norms])
                    contribution = (diff @ covs) / N
                    start = ((m * L + l) * N + i) * d
                    Kr[0, start:start+d] = contribution

    elif kernel_type == 'inter':
        if influence_data_3d is None or P is None:
            raise ValueError("influence_data_3d and P required for inter kernel")
        assert M == influence_data_3d.shape[2]

        for m in range(M):
            for l in range(L):
                X = np.zeros((d, N))
                for i in range(N):
                    X[:, i] = source_data_3d[d*i:d*(i+1), l, m]

                Z = np.zeros((d, P))
                for j in range(P):
                    Z[:, j] = influence_data_3d[d*j:d*(j+1), l, m]

                for i in range(N):
                    diff = X[:, i].reshape(-1, 1) - Z
                    norms = np.linalg.norm(diff, axis=0)
                    covs = np.array([cov_matern(norm, r, sigma, omega, nu) for norm in norms])
                    contribution = (diff @ covs) / (N + P)
                    start = ((m * L + l) * N + i) * d
                    Kr[0, start:start+d] = contribution

    else:
        raise ValueError("Unknown kernel_type.")

    return Kr