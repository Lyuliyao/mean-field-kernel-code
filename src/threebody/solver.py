"""Single-species conservative finite-volume solver with explicit diffusion.

Numerics follow the validated r1c1 solvers: MUSCL minmod reconstruction with
velocity upwinding, exactly-zero outer faces (no-flux boundary), explicit
centered diffusion on interior faces, and adaptive forward-Euler stepping
under dt * (max|v|/dx + 2 nu/dx^2) <= cfl. Negative densities are monitored,
never clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def uniform_spacing(values: np.ndarray | Sequence[float]) -> float:
    grid = np.asarray(values, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("grid must be one-dimensional with at least three points")
    differences = np.diff(grid)
    spacing = float(differences[0])
    if spacing <= 0.0 or not np.allclose(differences, spacing, rtol=1e-10, atol=1e-12):
        raise ValueError("grid must be strictly increasing and uniform")
    return spacing


@dataclass(frozen=True)
class SolverConfig:
    cfl: float = 0.4
    diffusion: float = 0.0
    max_time_step: float | None = None
    boundary: str = "no_flux"
    spatial_order: int = 2
    max_steps: int = 2_000_000
    minimum_time_step: float = 1.0e-12


class TransportSolver1D:
    """Conservative solver for d_t rho = -d_x(rho v(x, rho)) + nu d_xx rho."""

    def __init__(
        self,
        grid: np.ndarray | Sequence[float],
        velocity_fn: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig | None = None,
    ) -> None:
        self.grid = np.asarray(grid, dtype=np.float64)
        self.dx = uniform_spacing(self.grid)
        self.velocity_fn = velocity_fn
        self.config = config or SolverConfig()
        if not 0.0 < self.config.cfl <= 1.0:
            raise ValueError("cfl must be in (0, 1]")
        if self.config.diffusion < 0.0:
            raise ValueError("diffusion must be nonnegative")
        if self.config.max_time_step is not None and self.config.max_time_step <= 0.0:
            raise ValueError("max_time_step must be positive")
        if self.config.boundary != "no_flux":
            raise ValueError("only the no_flux boundary is implemented")
        if self.config.spatial_order not in (1, 2):
            raise ValueError("spatial_order must be one or two")
        if self.config.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if (
            not np.isfinite(self.config.minimum_time_step)
            or self.config.minimum_time_step <= 0.0
        ):
            raise ValueError("minimum_time_step must be positive and finite")

    def _validate_density(self, density: np.ndarray) -> np.ndarray:
        rho = np.asarray(density, dtype=np.float64)
        if rho.shape != (self.grid.size,):
            raise ValueError(f"density must have shape ({self.grid.size},)")
        if not np.all(np.isfinite(rho)):
            raise ValueError("density contains non-finite values")
        return rho

    def velocity(self, density: np.ndarray) -> np.ndarray:
        speed = np.asarray(self.velocity_fn(density), dtype=np.float64)
        if speed.shape != density.shape or not np.all(np.isfinite(speed)):
            raise FloatingPointError("velocity must be finite and match density shape")
        return speed

    def _stable_time_step_from_velocity(self, velocity: np.ndarray) -> float:
        rate = float(np.max(np.abs(velocity))) / self.dx
        rate += 2.0 * self.config.diffusion / self.dx**2
        stable = np.inf if rate == 0.0 else self.config.cfl / rate
        if self.config.max_time_step is not None:
            stable = min(stable, self.config.max_time_step)
        return float(stable)

    def stable_time_step(self, density: np.ndarray) -> float:
        rho = self._validate_density(density)
        return self._stable_time_step_from_velocity(self.velocity(rho))

    def _flux_from_velocity(self, rho: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        face_velocity = 0.5 * (velocity[:-1] + velocity[1:])
        if self.config.spatial_order == 2:
            backward = rho[1:-1] - rho[:-2]
            forward = rho[2:] - rho[1:-1]
            limited = np.where(
                backward * forward > 0.0,
                np.sign(backward) * np.minimum(np.abs(backward), np.abs(forward)),
                0.0,
            )
            slope = np.zeros_like(rho)
            slope[1:-1] = limited
            left_state = rho[:-1] + 0.5 * slope[:-1]
            right_state = rho[1:] - 0.5 * slope[1:]
        else:
            left_state = rho[:-1]
            right_state = rho[1:]
        flux = np.zeros(self.grid.size + 1, dtype=np.float64)
        flux[1:-1] = (
            np.maximum(face_velocity, 0.0) * left_state
            + np.minimum(face_velocity, 0.0) * right_state
        )
        if self.config.diffusion > 0.0:
            flux[1:-1] -= self.config.diffusion * np.diff(rho) / self.dx
        return flux

    def step(self, density: np.ndarray, time_step: float) -> np.ndarray:
        rho = self._validate_density(density)
        if time_step <= 0.0:
            raise ValueError("time_step must be positive")
        velocity = self.velocity(rho)
        stable = self._stable_time_step_from_velocity(velocity)
        if time_step > stable * (1.0 + 1e-12):
            raise ValueError(f"time_step {time_step:g} exceeds CFL limit {stable:g}")
        flux = self._flux_from_velocity(rho, velocity)
        updated = rho - (time_step / self.dx) * np.diff(flux)
        if not np.all(np.isfinite(updated)):
            raise FloatingPointError("PDE step produced non-finite density")
        return updated

    def rollout(
        self,
        initial_density: np.ndarray,
        output_times: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        times = np.asarray(output_times, dtype=np.float64)
        if times.ndim != 1 or times.size == 0 or times[0] < 0.0:
            raise ValueError("output_times must be a nonempty nonnegative vector")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("output_times must be strictly increasing")

        rho = self._validate_density(initial_density).copy()
        snapshots = np.empty((times.size, self.grid.size), dtype=np.float64)
        masses = np.empty(times.size, dtype=np.float64)
        minima = np.empty(times.size, dtype=np.float64)
        negative_fraction = np.empty(times.size, dtype=np.float64)
        current_time = 0.0
        step_count = 0
        smallest_step = np.inf
        largest_step = 0.0

        for output_index, requested_time in enumerate(times):
            while current_time < requested_time - 32.0 * np.finfo(float).eps:
                if step_count >= self.config.max_steps:
                    raise FloatingPointError(
                        f"PDE rollout exceeded configured max_steps={self.config.max_steps}"
                    )
                remaining = float(requested_time - current_time)
                velocity = self.velocity(rho)
                stable = self._stable_time_step_from_velocity(velocity)
                if (
                    np.isfinite(stable)
                    and stable < self.config.minimum_time_step
                    and stable < remaining * (1.0 - 16.0 * np.finfo(float).eps)
                ):
                    raise FloatingPointError(
                        "PDE CFL time step fell below configured minimum_time_step: "
                        f"{stable:g} < {self.config.minimum_time_step:g}"
                    )
                time_step = min(stable, remaining)
                if not np.isfinite(time_step):
                    time_step = remaining
                if time_step <= 0.0 or current_time + time_step <= current_time:
                    raise FloatingPointError(
                        f"PDE rollout cannot advance time with time_step={time_step:g}"
                    )
                flux = self._flux_from_velocity(rho, velocity)
                rho = rho - (time_step / self.dx) * np.diff(flux)
                if not np.all(np.isfinite(rho)):
                    raise FloatingPointError("PDE step produced non-finite density")
                current_time += time_step
                step_count += 1
                smallest_step = min(smallest_step, time_step)
                largest_step = max(largest_step, time_step)
            snapshots[output_index] = rho
            masses[output_index] = float(rho.sum() * self.dx)
            minima[output_index] = float(rho.min())
            negative_fraction[output_index] = float(np.mean(rho < 0.0))

        initial_mass = float(np.sum(initial_density) * self.dx)
        diagnostics = {
            "mass": masses,
            "mass_error": masses - initial_mass,
            "minimum_density": minima,
            "negative_fraction": negative_fraction,
            "step_count": step_count,
            "minimum_time_step": float(smallest_step if step_count else 0.0),
            "maximum_time_step": float(largest_step),
            "cfl": float(self.config.cfl),
            "diffusion": float(self.config.diffusion),
            "boundary": self.config.boundary,
            "spatial_order": self.config.spatial_order,
        }
        return snapshots, diagnostics
