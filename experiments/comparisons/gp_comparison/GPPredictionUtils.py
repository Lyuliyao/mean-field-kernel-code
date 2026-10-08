# Original code from Charles Kulick for the paper "Learning particle swarming models from data with Gaussian processes" by J. Feng, C. Kulick, Y. Ren, S. Tang (2023)

import numpy as np


def cov_matern(x1, x2, sigma, omega, v):
    """
    Matern covariance function evaluated at scalar distances x1, x2.

    k(x1, x2) depends only on r = |x1 - x2| * omega.
    Supported smoothness values: v = 0.5, 1.5, 2.5, 3.5.
    """
    r = np.abs(x1 - x2) * omega

    if v == 0.5:
        return sigma**2 * np.exp(-r)
    elif v == 1.5:
        return sigma**2 * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    elif v == 2.5:
        return sigma**2 * (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
    elif v == 3.5:
        return sigma**2 * (1 + np.sqrt(7) * r + 0.4 * r**2 + r**3 / 15) * np.exp(-np.sqrt(7) * r)
    else:
        raise ValueError(f"Unsupported Matern smoothness v={v}")


def cov_matern_omega_partial(x1, x2, sigma, omega, v):
    """
    Partial derivative of the Matern covariance with respect to omega: dk/domega.

    Used during gradient-based hyperparameter optimization.
    Supported smoothness values: v = 0.5, 1.5, 2.5, 3.5.
    """
    r = np.abs(x1 - x2)
    omega_r = omega * r

    if v == 0.5:
        return sigma**2 * (-r) * np.exp(-omega_r)

    elif v == 1.5:
        exp_term = np.exp(-np.sqrt(3) * omega_r)
        return -sigma**2 * 3 * omega * r**2 * exp_term

    elif v == 2.5:
        sqrt5_omega_r = np.sqrt(5) * omega_r
        exp_term = np.exp(-sqrt5_omega_r)
        f = 1 + sqrt5_omega_r + 5 * omega**2 * r**2 / 3
        df = np.sqrt(5) * r + 10 * omega * r**2 / 3
        dg = -np.sqrt(5) * r * exp_term
        return sigma**2 * (df * exp_term + f * dg)

    elif v == 3.5:
        sqrt7_omega_r = np.sqrt(7) * omega_r
        exp_term = np.exp(-sqrt7_omega_r)
        f = 1 + sqrt7_omega_r + 0.4 * omega**2 * r**2 + omega**3 * r**3 / 15
        df = np.sqrt(7) * r + 0.8 * omega * r**2 + 3 * omega**2 * r**3 / 15
        dg = -np.sqrt(7) * r * exp_term
        return sigma**2 * (df * exp_term + f * dg)

    else:
        raise ValueError(f"Unsupported Matern smoothness v={v}")