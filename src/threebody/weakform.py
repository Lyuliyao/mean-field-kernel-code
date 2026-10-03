"""Weak-form test windows and design assembly for single-species densities.

Convention: for d_t rho + d_x(sum_k c_k F_k) - nu d_xx rho = 0 and compactly
supported psi(t, x), integration by parts gives

    sum_k c_k * INT F_k psi_x dt dx = -INT rho psi_t dt dx - nu INT rho psi_xx dt dx

so each design column is +INT F_k psi_x and the regression target is the
right-hand side. Integrals use plain dt*dx Riemann sums, matching the
validated r1c1 convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import uniform_spacing


@dataclass(frozen=True)
class WeakConfig:
    time_half_width: float
    space_half_width: float
    time_stride: int = 1
    space_stride: int = 1
    polynomial_order: int = 4
    max_windows_per_trajectory: int | None = None
    random_seed: int = 0


@dataclass(frozen=True)
class WeakWindow:
    time_indices: np.ndarray
    space_indices: np.ndarray
    psi_t: np.ndarray
    psi_x: np.ndarray
    psi_xx: np.ndarray
    time_center: float
    space_center: float


@dataclass
class WeakDesign:
    matrix: np.ndarray
    target: np.ndarray
    trajectory_id: np.ndarray
    coefficient_names: tuple[str, ...]


def compact_polynomial(coordinates: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(1-s^2)^p and its first two s-derivatives; p >= 3 keeps psi_xx continuous."""

    if order < 3:
        raise ValueError("polynomial_order must be at least three (psi_xx needed)")
    s = np.asarray(coordinates, dtype=np.float64)
    values = np.zeros_like(s)
    first = np.zeros_like(s)
    second = np.zeros_like(s)
    inside = np.abs(s) < 1.0
    u = s[inside]
    base = 1.0 - u**2
    values[inside] = base**order
    first[inside] = -2.0 * order * u * base ** (order - 1)
    second[inside] = base ** (order - 2) * (
        -2.0 * order * base + 4.0 * order * (order - 1) * u**2
    )
    return values, first, second


def weak_windows(
    times: np.ndarray,
    grid_centers: np.ndarray,
    config: WeakConfig,
    trajectory_index: int,
) -> list[WeakWindow]:
    time = np.asarray(times, dtype=np.float64)
    x = np.asarray(grid_centers, dtype=np.float64)
    dt = uniform_spacing(time)
    dx = uniform_spacing(x)
    if config.time_half_width <= dt or config.space_half_width <= dx:
        raise ValueError("weak half-widths must exceed one grid spacing")
    if config.time_stride < 1 or config.space_stride < 1:
        raise ValueError("weak strides must be positive integers")

    time_margin = int(np.ceil(config.time_half_width / dt))
    space_margin = int(np.ceil(config.space_half_width / dx))
    time_centers = np.arange(time_margin, time.size - time_margin, config.time_stride)
    space_centers = np.arange(space_margin, x.size - space_margin, config.space_stride)
    centers = np.asarray(
        [(int(t), int(s)) for t in time_centers for s in space_centers],
        dtype=np.int64,
    )
    if centers.size == 0:
        raise ValueError("weak supports leave no interior test-function centers")
    if (
        config.max_windows_per_trajectory is not None
        and centers.shape[0] > config.max_windows_per_trajectory
    ):
        rng = np.random.default_rng(config.random_seed + trajectory_index)
        chosen = np.sort(
            rng.choice(
                centers.shape[0],
                size=config.max_windows_per_trajectory,
                replace=False,
            )
        )
        centers = centers[chosen]

    windows: list[WeakWindow] = []
    for time_index, space_index in centers:
        time_indices = np.flatnonzero(
            np.abs(time - time[time_index]) <= config.time_half_width * (1.0 + 1e-12)
        )
        space_indices = np.flatnonzero(
            np.abs(x - x[space_index]) <= config.space_half_width * (1.0 + 1e-12)
        )
        local_time = (time[time_indices] - time[time_index]) / config.time_half_width
        local_space = (x[space_indices] - x[space_index]) / config.space_half_width
        wt, dwt, _ = compact_polynomial(local_time, config.polynomial_order)
        wx, dwx, ddwx = compact_polynomial(local_space, config.polynomial_order)
        psi_t = (dwt[:, None] / config.time_half_width) * wx[None, :]
        psi_x = wt[:, None] * (dwx[None, :] / config.space_half_width)
        psi_xx = wt[:, None] * (ddwx[None, :] / config.space_half_width**2)
        windows.append(
            WeakWindow(
                time_indices=time_indices,
                space_indices=space_indices,
                psi_t=psi_t,
                psi_x=psi_x,
                psi_xx=psi_xx,
                time_center=float(time[time_index]),
                space_center=float(x[space_index]),
            )
        )
    return windows


def window_integral(field: np.ndarray, window: WeakWindow, weight: np.ndarray, dt: float, dx: float) -> float:
    """INT field * weight dt dx over the window support (Riemann sum)."""

    sub = field[np.ix_(window.time_indices, window.space_indices)]
    return float(np.sum(sub * weight) * dt * dx)


def assemble_design(
    density: np.ndarray,
    times: np.ndarray,
    grid_centers: np.ndarray,
    flux_fields: dict[str, np.ndarray],
    windows: list[WeakWindow],
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Design rows for one trajectory.

    ``flux_fields`` maps coefficient names to (T, X) flux fields F_k.
    Returns (matrix (n_windows, n_features), target (n_windows,)).
    """

    dt = uniform_spacing(np.asarray(times, dtype=np.float64))
    dx = uniform_spacing(np.asarray(grid_centers, dtype=np.float64))
    rho = np.asarray(density, dtype=np.float64)
    if rho.shape != (np.asarray(times).size, np.asarray(grid_centers).size):
        raise ValueError("density must have shape (times, grid)")
    names = list(flux_fields.keys())
    matrix = np.empty((len(windows), len(names)), dtype=np.float64)
    target = np.empty(len(windows), dtype=np.float64)
    for row, window in enumerate(windows):
        target[row] = -window_integral(rho, window, window.psi_t, dt, dx)
        if nu != 0.0:
            target[row] -= nu * window_integral(rho, window, window.psi_xx, dt, dx)
        for col, name in enumerate(names):
            matrix[row, col] = window_integral(
                flux_fields[name], window, window.psi_x, dt, dx
            )
    return matrix, target
