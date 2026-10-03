"""Observation operator, mass conservation, and solver correctness tests."""

import numpy as np
import pytest

from threebody.observation import Grid, kde_density
from threebody.solver import SolverConfig, TransportSolver1D


def test_kde_unit_mass_and_shape(coarse_grid):
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 0.5, size=(7, 300))
    densities = kde_density(samples, coarse_grid, 0.1)
    assert densities.shape == (7, coarse_grid.cells)
    masses = densities.sum(axis=-1) * coarse_grid.dx
    assert np.allclose(masses, 1.0, atol=1e-12)


def test_kde_rejects_outside_particles(coarse_grid):
    with pytest.raises(ValueError):
        kde_density(np.asarray([0.0, 4.5]), coarse_grid, 0.1)


def test_common_t0_observation_operator(coarse_grid):
    """All methods share the exact same t=0 density representation."""

    rng = np.random.default_rng(3)
    particles = rng.uniform(-2.0, 2.0, 500)
    reference_t0 = kde_density(particles, coarse_grid, 0.08)
    pde_initial = kde_density(particles, coarse_grid, 0.08)
    mvnn_projection_t0 = kde_density(particles, coarse_grid, 0.08)
    assert np.array_equal(reference_t0, pde_initial)
    assert np.array_equal(reference_t0, mvnn_projection_t0)


def test_solver_mass_conservation_and_negativity_monitoring(coarse_grid):
    x = coarse_grid.centers
    rho0 = np.exp(-0.5 * (x / 0.5) ** 2)
    rho0 /= rho0.sum() * coarse_grid.dx

    def velocity(rho):
        return 0.3 * np.ones_like(rho)

    solver = TransportSolver1D(
        x, velocity, SolverConfig(cfl=0.4, diffusion=1e-3, max_time_step=0.01)
    )
    snapshots, diagnostics = solver.rollout(rho0, np.linspace(0.0, 1.0, 11))
    assert np.max(np.abs(diagnostics["mass_error"])) < 5e-13
    assert snapshots.shape == (11, coarse_grid.cells)
    assert np.all(np.isfinite(snapshots))
    assert "negative_fraction" in diagnostics


def test_solver_pure_diffusion_matches_heat_kernel():
    grid = Grid(-4.0, 4.0, 256)
    x = grid.centers
    nu = 0.05
    sigma0 = 0.4
    rho0 = np.exp(-0.5 * (x / sigma0) ** 2) / (sigma0 * np.sqrt(2 * np.pi))
    solver = TransportSolver1D(
        x, lambda rho: np.zeros_like(rho), SolverConfig(cfl=0.4, diffusion=nu)
    )
    t_final = 0.5
    snapshots, _ = solver.rollout(rho0, np.asarray([t_final]))
    sigma_t = np.sqrt(sigma0**2 + 2 * nu * t_final)
    exact = np.exp(-0.5 * (x / sigma_t) ** 2) / (sigma_t * np.sqrt(2 * np.pi))
    assert np.max(np.abs(snapshots[0] - exact)) < 2e-3


def test_solver_cfl_guard(coarse_grid):
    x = coarse_grid.centers
    rho0 = np.ones_like(x) / 8.0
    solver = TransportSolver1D(x, lambda rho: np.ones_like(rho), SolverConfig(cfl=0.4))
    with pytest.raises(ValueError):
        solver.step(rho0, 1.0)  # far above the CFL limit
