# Original code from Charles Kulick for the paper "Learning particle swarming models from data with Gaussian processes" by J. Feng, C. Kulick, Y. Ren, S. Tang (2023)

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

WARM_COLORS = ['#FF0000', '#FF4500', '#FF6A00', '#FF8C00', '#FFA500', '#FFB347']
COOL_COLORS = ['#0066FF', '#0080FF', '#00AAFF', '#00D4FF', '#00FFD4', '#00FFAA']


def create_interpolated_kernels(x, y_pred_11):
    """
    Wrap discrete kernel predictions into scalar callable functions via linear interpolation.

    Parameters:
    x : ndarray, distance grid
    y_pred_11/12/21/22 : ndarrays, predicted values for g11, g12, g21, g22

    Returns (g11_interp, g12_interp, g21_interp, g22_interp).
    """
    def make_interp(y):
        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        def scalar_safe(r):
            r_arr = np.atleast_1d(r)
            result = f(r_arr)
            return result.item() if result.size == 1 else result
        return scalar_safe

    return make_interp(y_pred_11)


def concatenate_temporal_data(data_training, data_temporal, L):
    """
    Concatenate training and temporal generalization arrays along the time axis.

    Inputs have shape (L, 2*N, M); output has shape (2*L, 2*N, M).
    """
    return np.concatenate([data_training, data_temporal], axis=0)


def compute_relative_trajectory_error(true_x, pred_x, 
                                      times, N1, t_training_end=None):
    """
    Compute the max relative L^2 trajectory error max_t ||x_true(t) - x_pred(t)|| / ||x_true(t)||

    Parameters:
    true_x, true_z : ndarrays, shape (L_total, N, 2)
    pred_x, pred_z : ndarrays, shape (L_total, N, 2)
    times : ndarray, shape (L_total,)
    N1, N2 : int
    t_training_end : float, optional, if given errors are split into training/generalization

    Returns a dict with keys 'relative_error' and, if t_training_end is given,
    also 'training_relative_error' and 'generalization_relative_error'.
    """
    L_total = len(times)

    # Flatten all agents and coordinates into (2*(N1+N2), L_total)
    true_traj = np.zeros((N1, L_total))
    pred_traj = np.zeros((N1, L_total))
    for i in range(N1):
        true_traj[i],   pred_traj[i]   = true_x[:, i, 0], pred_x[:, i, 0]
        true_traj[i+1], pred_traj[i+1] = true_x[:, i, 1], pred_x[:, i, 1]

    error_norms = np.sqrt(np.sum((true_traj - pred_traj)**2, axis=0))
    true_norms  = np.sqrt(np.sum(true_traj**2, axis=0))
    rel_errors  = error_norms / (true_norms + 1e-12)

    result = {'relative_error': rel_errors.max()}

    if t_training_end is not None:
        split = np.argmin(np.abs(times - t_training_end))
        result['training_relative_error']      = rel_errors[:split+1].max()
        result['generalization_relative_error'] = (
            rel_errors[split+1:].max() if split + 1 < L_total else 0.0
        )

    return result