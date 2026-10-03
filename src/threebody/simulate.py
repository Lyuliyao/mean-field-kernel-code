"""Euler-Maruyama simulation of the three-body global-mean SDE.

dX_i = [gamma (M_N - X_i) + (a - alpha M_N^2) X_i - beta X_i^3] dt + sigma dW_i
with M_N the empirical mean. Particle states are never clipped.
"""

from __future__ import annotations

import numpy as np

from .protocol import GroundTruth


def true_drift(x: np.ndarray, mean: float, params: GroundTruth) -> np.ndarray:
    """Drift b(x, mu); the measure enters only through its mean."""

    x = np.asarray(x, dtype=np.float64)
    return (
        params.gamma * (mean - x)
        + (params.a - params.alpha * mean**2) * x
        - params.beta * x**3
    )


def affinity_defect_truth(x: np.ndarray, d: float, params: GroundTruth) -> np.ndarray:
    """b(x, mu0) - [b(x, mu+) + b(x, mu-)]/2 for means m0 +/- d = alpha x d^2."""

    return params.alpha * np.asarray(x, dtype=np.float64) * d**2


def simulate(
    initial_particles: np.ndarray,
    params: GroundTruth,
    brownian_seed: int,
) -> np.ndarray:
    """Simulate the particle system; returns (n_frames, N) float64."""

    x = np.asarray(initial_particles, dtype=np.float64).copy()
    if x.ndim != 1 or x.size < 2:
        raise ValueError("initial_particles must be a 1D vector with N >= 2")
    rng = np.random.default_rng(int(brownian_seed))
    frames = np.empty((params.n_frames, x.size), dtype=np.float64)
    frames[0] = x
    noise_scale = params.sigma * np.sqrt(params.dt)
    for frame in range(1, params.n_frames):
        mean = float(x.mean())
        x = x + params.dt * true_drift(x, mean, params)
        if noise_scale > 0.0:
            x = x + noise_scale * rng.standard_normal(x.size)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f"non-finite particle state at frame {frame}")
        frames[frame] = x
    return frames


def boundary_mass_fraction(
    trajectory: np.ndarray, domain: tuple[float, float], margin: float = 0.2
) -> float:
    """Largest per-frame fraction of particles within ``margin`` of a wall."""

    frames = np.asarray(trajectory, dtype=np.float64)
    lower, upper = domain
    near = (frames <= lower + margin) | (frames >= upper - margin)
    return float(near.mean(axis=-1).max())


def check_boundary_mass(
    trajectory: np.ndarray,
    domain: tuple[float, float],
    tolerance: float,
    margin: float = 0.2,
) -> float:
    fraction = boundary_mass_fraction(trajectory, domain, margin)
    if fraction > tolerance:
        raise ValueError(
            f"boundary mass fraction {fraction:g} exceeds tolerance {tolerance:g}"
        )
    return fraction
