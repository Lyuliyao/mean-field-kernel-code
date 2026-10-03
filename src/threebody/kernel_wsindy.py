"""Additive B-spline kernel WSINDy baseline.

Model: d_t rho = -d_x{ rho [ f_B(x) + INT K_B(y - x) rho(y) dy ] } + nu d_xx rho
with nu = sigma^2/2 known. Both f_B and K_B are generic cubic B-spline
expansions; no oracle information (true kernel, gamma, polynomial degree,
kernel radius, oddness, M, M^2) enters the fit.

Kernel argument convention: the induced drift at grid point x_i is
    (K conv rho)(x_i) = sum_j K(x_j - x_i) rho_j dx,
a CORRELATION with the stencil K evaluated at offsets (y - x).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .bsplines import (
    evaluate_bspline_basis,
    open_uniform_knots,
    second_difference_matrix,
)
from .observation import Grid
from .regression import FitResult, Group, fit_smooth_ridge
from .weakform import WeakConfig, assemble_design, weak_windows


def displacement_domain(trajectories: list[np.ndarray]) -> tuple[float, float]:
    """Symmetric observed displacement range from training particle data."""

    lo = min(float(np.min(t)) for t in trajectories)
    hi = max(float(np.max(t)) for t in trajectories)
    radius = hi - lo
    if radius <= 0.0:
        raise ValueError("degenerate particle range")
    return (-radius, radius)


def apply_stencil_correlation(values: np.ndarray, stencil: np.ndarray) -> np.ndarray:
    """result[..., i] = sum_o stencil[o + X - 1] * values[..., i + o] (zero-extended)."""

    rho = np.asarray(values, dtype=np.float64)
    weights = np.asarray(stencil, dtype=np.float64)
    x_size = rho.shape[-1]
    if weights.shape != (2 * x_size - 1,):
        raise ValueError("stencil must have length 2*X-1")
    kernel_shape = (1,) * (rho.ndim - 1) + (weights.size,)
    full = fftconvolve(rho, weights[::-1].reshape(kernel_shape), mode="full", axes=-1)
    return full[..., x_size - 1 : 2 * x_size - 1]


def kernel_stencils(grid: Grid, knots: np.ndarray) -> np.ndarray:
    """(basis_count, 2X-1) stencils B_m(offsets) * dx (quadrature weight baked in)."""

    offsets = np.arange(-(grid.cells - 1), grid.cells, dtype=np.float64) * grid.dx
    basis = evaluate_bspline_basis(offsets, knots)  # (2X-1, M)
    return basis.T * grid.dx


@dataclass
class KernelModel:
    grid: Grid
    f_knots: np.ndarray
    k_knots: np.ndarray
    f_coefficients: np.ndarray
    k_coefficients: np.ndarray
    nu: float

    def __post_init__(self) -> None:
        self._stencils = kernel_stencils(self.grid, self.k_knots)
        self._f_values = evaluate_bspline_basis(
            self.grid.centers, self.f_knots
        ) @ self.f_coefficients
        self._k_stencil = self._stencils.T @ self.k_coefficients

    def velocity(self, density: np.ndarray) -> np.ndarray:
        return self._f_values + apply_stencil_correlation(density, self._k_stencil)

    def drift_field(self, density: np.ndarray) -> np.ndarray:
        """Model drift b_hat(x, rho) on the grid (same as velocity)."""

        return self.velocity(density)

    def kernel_values(self, points: np.ndarray) -> np.ndarray:
        return evaluate_bspline_basis(points, self.k_knots) @ self.k_coefficients

    def f_values(self, points: np.ndarray) -> np.ndarray:
        return evaluate_bspline_basis(points, self.f_knots) @ self.f_coefficients


def gauge_fix(model: KernelModel) -> KernelModel:
    """Shift the constant mode: zero grid mean on K_B, absorbed into f_B.

    The transformation moves along the exact null direction of the additive
    model on unit-mass measures, so induced drifts on the data support are
    unchanged.
    """

    lower = float(model.k_knots[3])
    upper = float(model.k_knots[-4])
    offsets = np.arange(-(model.grid.cells - 1), model.grid.cells, dtype=np.float64)
    offsets = offsets * model.grid.dx
    inside = (offsets >= lower) & (offsets <= upper)
    k_values = evaluate_bspline_basis(offsets[inside], model.k_knots) @ model.k_coefficients
    constant = float(k_values.mean())
    return KernelModel(
        grid=model.grid,
        f_knots=model.f_knots,
        k_knots=model.k_knots,
        f_coefficients=model.f_coefficients + constant,
        k_coefficients=model.k_coefficients - constant,
        nu=model.nu,
    )


def save_kernel_model(path, model: KernelModel) -> None:
    np.savez_compressed(
        path,
        grid_lower=model.grid.lower,
        grid_upper=model.grid.upper,
        grid_cells=model.grid.cells,
        f_knots=model.f_knots,
        k_knots=model.k_knots,
        f_coefficients=model.f_coefficients,
        k_coefficients=model.k_coefficients,
        nu=model.nu,
    )


def load_kernel_model(path) -> KernelModel:
    payload = np.load(path)
    grid = Grid(
        float(payload["grid_lower"]),
        float(payload["grid_upper"]),
        int(payload["grid_cells"]),
    )
    return KernelModel(
        grid=grid,
        f_knots=payload["f_knots"],
        k_knots=payload["k_knots"],
        f_coefficients=payload["f_coefficients"],
        k_coefficients=payload["k_coefficients"],
        nu=float(payload["nu"]),
    )


def build_flux_fields(
    density: np.ndarray,
    grid: Grid,
    f_knots: np.ndarray,
    k_knots: np.ndarray,
) -> dict[str, np.ndarray]:
    """Flux fields rho*B_j(x) (f block) and rho*(B_m corr rho) (K block)."""

    rho = np.asarray(density, dtype=np.float64)
    f_basis = evaluate_bspline_basis(grid.centers, f_knots)  # (X, Mf)
    fields: dict[str, np.ndarray] = {}
    for j in range(f_basis.shape[1]):
        fields[f"f_{j:03d}"] = rho * f_basis[None, :, j]
    stencils = kernel_stencils(grid, k_knots)
    for m in range(stencils.shape[0]):
        convolved = apply_stencil_correlation(rho, stencils[m])
        fields[f"k_{m:03d}"] = rho * convolved
    return fields


def fit_kernel_wsindy(
    densities: list[np.ndarray],
    times: np.ndarray,
    grid: Grid,
    displacement: tuple[float, float],
    basis_count: int,
    nu: float,
    weak_config: WeakConfig,
    smoothness: float,
    ridge: float = 1.0e-8,
) -> tuple[KernelModel, FitResult]:
    """One shared (f_B, K_B) fit across all training density trajectories."""

    f_knots = open_uniform_knots(grid.lower, grid.upper, basis_count)
    k_knots = open_uniform_knots(displacement[0], displacement[1], basis_count)
    rows = []
    targets = []
    for index, density in enumerate(densities):
        windows = weak_windows(times, grid.centers, weak_config, index)
        fields = build_flux_fields(density, grid, f_knots, k_knots)
        matrix, target = assemble_design(
            density, times, grid.centers, fields, windows, nu
        )
        rows.append(matrix)
        targets.append(target)
    matrix = np.vstack(rows)
    target = np.concatenate(targets)
    roughness = second_difference_matrix(basis_count)
    groups = [
        Group("f", tuple(range(basis_count)), roughness),
        Group("K", tuple(range(basis_count, 2 * basis_count)), roughness),
    ]
    fit = fit_smooth_ridge(
        matrix,
        target,
        groups,
        smoothness=smoothness,
        ridge=ridge,
        threshold=0.0,  # both blocks always retained; no sparsity assumption
    )
    model = KernelModel(
        grid=grid,
        f_knots=f_knots,
        k_knots=k_knots,
        f_coefficients=fit.coefficients[:basis_count],
        k_coefficients=fit.coefficients[basis_count:],
        nu=nu,
    )
    return gauge_fix(model), fit
