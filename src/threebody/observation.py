"""Common particle-to-density observation operator shared by all methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    lower: float
    upper: float
    cells: int

    @property
    def dx(self) -> float:
        return (self.upper - self.lower) / self.cells

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.lower, self.upper, self.cells + 1)

    @property
    def centers(self) -> np.ndarray:
        edges = self.edges
        return 0.5 * (edges[:-1] + edges[1:])


def default_grid(protocol: dict) -> Grid:
    domain = protocol["ground_truth"]["domain"]
    cells = int(protocol["observation"]["grid_cells"])
    return Grid(float(domain[0]), float(domain[1]), cells)


def gaussian_smoothing_kernel(bandwidth: float, dx: float, truncate: float = 4.0) -> np.ndarray:
    if bandwidth < 0.0:
        raise ValueError("bandwidth must be nonnegative")
    if bandwidth == 0.0:
        return np.ones(1, dtype=np.float64)
    half_width = max(1, int(np.ceil(truncate * bandwidth / dx)))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64) * dx
    kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2)
    return kernel / kernel.sum()


def kde_density(samples: np.ndarray, grid: Grid, bandwidth: float) -> np.ndarray:
    """Histogram + discrete Gaussian smoothing, renormalized to unit mass.

    This single function is THE observation operator: reference densities,
    PDE initial conditions, and MVNN rollout projections all pass through it
    with the same grid and bandwidth, so every method carries the same t=0
    representation and smoothing error.
    """

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("samples must have a nonempty particle axis")
    kernel = gaussian_smoothing_kernel(bandwidth, grid.dx)
    edges = grid.edges
    flat = values.reshape((-1, values.shape[-1]))
    outside = (flat < grid.lower) | (flat > grid.upper)
    if np.any(outside):
        raise ValueError(
            f"{int(outside.sum())} particles lie outside the density grid"
        )
    densities = np.empty((flat.shape[0], grid.cells), dtype=np.float64)
    for index, frame in enumerate(flat):
        counts, _ = np.histogram(frame, bins=edges)
        density = counts.astype(np.float64) / (frame.size * grid.dx)
        if kernel.size > 1:
            density = np.convolve(density, kernel, mode="same")
        mass = float(density.sum() * grid.dx)
        if mass <= 0.0 or not np.isfinite(mass):
            raise FloatingPointError("KDE produced invalid mass")
        densities[index] = density / mass
    return densities.reshape(values.shape[:-1] + (grid.cells,))


def density_trajectory(particle_trajectory: np.ndarray, grid: Grid, bandwidth: float) -> np.ndarray:
    """(n_frames, N) particles -> (n_frames, cells) densities."""

    return kde_density(particle_trajectory, grid, bandwidth)
