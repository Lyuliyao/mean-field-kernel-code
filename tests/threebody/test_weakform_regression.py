"""Weak-form assembly, stencil orientation, and regression tests."""

import numpy as np

from threebody.bsplines import (
    evaluate_bspline_basis,
    open_uniform_knots,
    second_difference_matrix,
)
from threebody.kernel_wsindy import apply_stencil_correlation, kernel_stencils
from threebody.observation import Grid
from threebody.regression import Group, fit_smooth_ridge
from threebody.weakform import WeakConfig, assemble_design, weak_windows


def test_bspline_partition_of_unity():
    knots = open_uniform_knots(-2.0, 2.0, 10)
    points = np.linspace(-2.0, 2.0, 101)
    basis = evaluate_bspline_basis(points, knots)
    assert np.allclose(basis.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(basis >= 0.0)
    outside = evaluate_bspline_basis(np.asarray([-2.5, 2.5]), knots)
    assert np.array_equal(outside, np.zeros((2, 10)))


def test_stencil_correlation_orientation_matches_dense_sum():
    """drift(x_i) = sum_j K(x_j - x_i) rho_j dx, including asymmetric K."""

    grid = Grid(-2.0, 2.0, 64)
    knots = open_uniform_knots(-4.0, 4.0, 8)
    rng = np.random.default_rng(1)
    coefficients = rng.normal(size=8)
    stencil = kernel_stencils(grid, knots).T @ coefficients
    rho = rng.uniform(0.0, 1.0, grid.cells)
    fast = apply_stencil_correlation(rho, stencil)
    x = grid.centers
    dense = np.asarray(
        [
            float(
                np.sum(
                    (evaluate_bspline_basis(x - xi, knots) @ coefficients)
                    * rho
                    * grid.dx
                )
            )
            for xi in x
        ]
    )
    assert np.max(np.abs(fast - dense)) < 1e-10


def test_weak_design_recovers_manufactured_transport():
    """rho_t + d_x(rho v) = nu rho_xx with known v: weak design must solve for v."""

    grid = Grid(-3.0, 3.0, 240)
    x = grid.centers
    times = np.linspace(0.0, 0.5, 51)
    nu = 0.01

    # manufactured density: translating + spreading Gaussian, velocity constant
    velocity_true = 0.4
    from threebody.solver import SolverConfig, TransportSolver1D

    rho0 = np.exp(-0.5 * ((x + 0.5) / 0.35) ** 2)
    rho0 /= rho0.sum() * grid.dx
    solver = TransportSolver1D(
        x,
        lambda rho: velocity_true * np.ones_like(rho),
        SolverConfig(cfl=0.4, diffusion=nu, max_time_step=1e-3),
    )
    density, _ = solver.rollout(rho0, times)

    config = WeakConfig(
        time_half_width=0.06,
        space_half_width=0.3,
        time_stride=2,
        space_stride=3,
        max_windows_per_trajectory=400,
    )
    windows = weak_windows(times, x, config, 0)
    fields = {"rho": density}
    matrix, target = assemble_design(density, times, x, fields, windows, nu)
    coefficient, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    assert abs(coefficient[0] - velocity_true) < 0.02


def test_regression_recovers_coefficients_and_thresholds_null_groups():
    rng = np.random.default_rng(2)
    n_rows = 400
    a = rng.normal(size=(n_rows, 6))
    truth = np.asarray([1.5, -2.0, 0.0, 0.0, 0.7, 0.0])
    y = a @ truth + 1e-6 * rng.normal(size=n_rows)
    groups = [
        Group("g1", (0, 1)),
        Group("g2", (2, 3)),
        Group("g3", (4,)),
        Group("g4", (5,)),
    ]
    fit = fit_smooth_ridge(a, y, groups, ridge=1e-10, threshold=0.01)
    assert "g2" not in fit.active_groups and "g4" not in fit.active_groups
    assert np.allclose(fit.coefficients[[0, 1, 4]], [1.5, -2.0, 0.7], atol=1e-3)
    assert np.allclose(fit.coefficients[[2, 3, 5]], 0.0)


def test_smoothness_penalty_reduces_roughness():
    rng = np.random.default_rng(4)
    n_rows, n_cols = 300, 12
    a = rng.normal(size=(n_rows, n_cols))
    y = rng.normal(size=n_rows)
    groups = [Group("s", tuple(range(n_cols)), second_difference_matrix(n_cols))]
    rough = fit_smooth_ridge(a, y, groups, smoothness=0.0, ridge=1e-10)
    smooth = fit_smooth_ridge(a, y, groups, smoothness=10.0, ridge=1e-10)
    def roughness(c):
        return float(np.sum(np.diff(c, 2) ** 2))
    assert roughness(smooth.coefficients) < roughness(rough.coefficients)
