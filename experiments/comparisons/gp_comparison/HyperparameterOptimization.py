# Original code from Charles Kulick for the paper "Learning particle swarming models from data with Gaussian processes" by J. Feng, C. Kulick, Y. Ren, S. Tang (2023)

import logging
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

from ConstructIntraSpeciesKernel import construct_intra_species_kernel

def compute_log_likelihood_and_solve(K_combined, Y):
    """
    Compute the negative log-likelihood and solve K*u = Y with Cholesky.

    Returns (fval, u, L_chol) where fval is the negative log-likelihood,
    u = K^{-1} Y, and L_chol is the lower Cholesky factor.
    """
    try:
        L_chol = cholesky(K_combined, lower=True)
    except np.linalg.LinAlgError as e:
        # This should never happen with the noisy setup but might for exact
        logging.warning("Cholesky failed, adding jitter 1e-6 and retrying")
        return compute_log_likelihood_and_solve(
            K_combined + 1e-6 * np.eye(K_combined.shape[0]), Y
        )

    v = solve_triangular(L_chol, Y, lower=True)
    u = solve_triangular(L_chol.T, v, lower=False)

    log_det_K = 2 * np.sum(np.log(np.diag(L_chol)))
    n = len(Y)
    fval = 0.5 * np.dot(Y, u) + 0.5 * log_det_K + 0.5 * n * np.log(2 * np.pi)

    return fval, u, L_chol


def build_combined_kernel(hyps, source_data_3d, 
                            N_source, d, M, L, nu, jitter,
                            device=None):
    """
    Build K = KE + KG + jitter*I for one species.

    hyps : [log(delta_intra), log(omega_intra), log(delta_inter), log(omega_inter)]

    Returns (K_combined, KE, KG).
    """
    delta_intra, omega_intra = np.exp(hyps[0]), np.exp(hyps[1])

    KE = construct_intra_species_kernel(
        source_data_3d, omega_intra, delta_intra,
        N_source, d, M, L, nu, partial_omega=False,
        device=device
    )
    
    K_combined = KE + jitter * np.eye(KE.shape[0])

    return K_combined, KE


def compute_gradients(hyps, source_data_3d, 
                               K_combined, KE, u,
                               N_source, d, M, L, nu,
                               device=None):
    """
    Gradient of the negative log-likelihood w.r.t. log-scale hyperparameters.

    Uses the standard GP gradient formula:
        d(-log p)/d(theta) = -0.5 * (u^T dK/dtheta u - tr(K^{-1} dK/dtheta))

    Returns dfval of length 4, one entry per element of hyps.
    """
    try:
        L_chol = cholesky(K_combined, lower=True)
        def solve_K_inv(rhs):
            return solve_triangular(L_chol.T,
                                    solve_triangular(L_chol, rhs, lower=True),
                                    lower=False)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.inv(K_combined)
        def solve_K_inv(rhs):
            return K_inv @ rhs

    def grad_term(dK):
        return -0.5 * (u.T @ dK @ u - np.trace(solve_K_inv(dK)))

    dfval = np.zeros(2)

    # dK/d(log delta) = 2*KE 
    dfval[0] = grad_term(2 * KE)

    # dK/d(log omega) = omega * dKE/domega
    omega_intra = np.exp(hyps[1])
    dKE_domega = construct_intra_species_kernel(
        source_data_3d, omega_intra, np.exp(hyps[0]),
        N_source, d, M, L, nu, partial_omega=True,
        device=device
    )
    dfval[1] = grad_term(omega_intra * dKE_domega)

    return dfval


def compute_glik(hyps, source_data_3d, Y,
                          N_source, d, M, L, nu, jitter,
                          device=None):
    """
    Compute the negative log-likelihood and its gradient for one species.

    Returns (fval, dfval) suitable for use as a scipy minimize objective with jac=True.
    """
    K_combined, KE = build_combined_kernel(
        hyps, source_data_3d, N_source, d, M, L, nu, jitter,
        device=device
    )
    fval, u, _ = compute_log_likelihood_and_solve(K_combined, Y)
    dfval = compute_gradients(
        hyps, source_data_3d, K_combined, KE, u,
        N_source, d, M, L, nu, device=device
    )

    logging.info(f"fval={fval:.6f}  grad_norm={np.linalg.norm(dfval):.6f}")

    return fval, dfval


def optimize_species_hyperparameters(source_data_3d, Y,
                                      initial_hyps, N_source, 
                                      d, M, L, nu, jitter, max_iterations=50,
                                      device=None):
    """
    Optimize the four log-scale hyperparameters for one species via L-BFGS-B.

    Hyperparameters: [log(delta_intra), log(omega_intra), log(delta_inter), log(omega_inter)]
    Bounds are [-5, 5] for each by default, can be expanded if needed.

    Returns the scipy OptimizeResult.
    """
    def objective(hyps):
        return compute_glik(
            hyps, source_data_3d, Y,
            N_source, d, M, L, nu, jitter, device=device
        )

    logging.info(f"Starting hyperparameter optimization (max_iter={max_iterations})")
    logging.info(f"Initial hyps: {initial_hyps}")

    result = minimize(
        objective,
        initial_hyps,
        method='L-BFGS-B',
        jac=True,
        # bounds=[(-5, 5)]*2,
        bounds=[(-5,2),(-2,8)],
        options={'maxiter': max_iterations}
    )

    logging.info(result.message)
    logging.info(result.nit)
    logging.info(result.nfev)

    logging.info(f"Optimization complete, final fval={result.fun:.6f}")
    logging.info(f"Optimized hyps: {result.x}")

    return result