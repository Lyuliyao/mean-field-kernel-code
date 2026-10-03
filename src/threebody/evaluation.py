"""Common evaluation of all three methods on a set of matched triplets.

Every method consumes the same particle data through the same observation
operator: PDE baselines start from the KDE of the empirical initial particles,
MVNN rollouts and reference trajectories are projected through the identical
KDE/grid, so all methods share the t=0 density representation exactly.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import triplets as tp
from .baselines import rollout_model_density
from .metrics import (
    bump_centroid,
    density_mean,
    direction_match,
    drift_rmse,
    short_time_velocity,
    w1_trajectory,
)
from .observation import Grid, kde_density
from .protocol import GroundTruth
from .simulate import true_drift

MVNN_EVAL_NOISE_STREAM = 404


def eval_noise_seed(stage_base: int, triplet_id: int, measure: str, model_seed: int) -> int:
    return int(
        np.random.SeedSequence(
            [
                MVNN_EVAL_NOISE_STREAM,
                int(stage_base),
                int(triplet_id),
                tp.MEASURES.index(measure),
                int(model_seed),
            ]
        ).generate_state(1)[0]
    )


def eval_frames_for(n_frames: int, stride: int = 5) -> np.ndarray:
    frames = np.arange(0, n_frames, stride)
    if frames[-1] != n_frames - 1:
        frames = np.concatenate([frames, [n_frames - 1]])
    return frames


def evaluate_pde_method(
    method: str,
    velocity_fn: Callable[[np.ndarray], np.ndarray],
    diffusion: float,
    drift_field_fn: Callable[[np.ndarray], np.ndarray],
    family_row: dict[str, Any],
    measure: str,
    trajectory: np.ndarray,
    grid: Grid,
    bandwidth: float,
    params: GroundTruth,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one PDE-type model on one trajectory; returns (row, density traj)."""

    frames = eval_frames_for(trajectory.shape[0])
    times = frames.astype(np.float64) * params.dt
    preprocessing_start = time.perf_counter()
    reference = kde_density(trajectory[frames], grid, bandwidth)
    preprocessing_seconds = time.perf_counter() - preprocessing_start
    rollout_start = time.perf_counter()
    predicted, diagnostics = rollout_model_density(
        velocity_fn, diffusion, reference[0], grid, times, params.dt
    )
    rollout_seconds = time.perf_counter() - rollout_start
    row = _metric_row(
        method,
        family_row,
        measure,
        predicted,
        reference,
        times,
        grid,
        trajectory,
        drift_field_fn(reference[0]),
        params,
    )
    row["rollout_seconds"] = rollout_seconds
    row["preprocessing_seconds"] = preprocessing_seconds
    row["max_abs_mass_error"] = float(np.max(np.abs(diagnostics["mass_error"])))
    row["max_negative_fraction"] = float(np.max(diagnostics["negative_fraction"]))
    return row, predicted


def evaluate_mvnn_method(
    model,
    params_tree,
    model_seed: int,
    stage_base: int,
    family_row: dict[str, Any],
    measure: str,
    trajectory: np.ndarray,
    grid: Grid,
    bandwidth: float,
    params: GroundTruth,
) -> tuple[dict[str, Any], np.ndarray]:
    import jax

    from .mvnn import rollout_sde

    frames = eval_frames_for(trajectory.shape[0])
    times = frames.astype(np.float64) * params.dt
    preprocessing_start = time.perf_counter()
    reference = kde_density(trajectory[frames], grid, bandwidth)
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    triplet_id = int(family_row["family"]["triplet_id"])
    noise_seed = eval_noise_seed(stage_base, triplet_id, measure, model_seed)
    rollout_start = time.perf_counter()
    rolled = np.asarray(
        rollout_sde(
            model,
            params_tree,
            trajectory[0],
            params.dt,
            int(frames[-1]),
            params.sigma,
            jax.random.PRNGKey(noise_seed),
        )
    )
    rollout_seconds = time.perf_counter() - rollout_start
    if not np.all(np.isfinite(rolled)):
        raise FloatingPointError("MVNN evaluation rollout produced non-finite states")
    if rolled.min() < grid.lower or rolled.max() > grid.upper:
        raise FloatingPointError(
            "MVNN evaluation rollout left the density domain (invalid state)"
        )
    projection_start = time.perf_counter()
    predicted = kde_density(rolled[frames], grid, bandwidth)
    preprocessing_seconds += time.perf_counter() - projection_start

    particles = trajectory[0][:, None]
    drift_particles = np.asarray(
        model.drift_at(params_tree, particles, particles)
    )[:, 0]
    # interpolate model drift onto the grid for the drift-field diagnostics
    order = np.argsort(trajectory[0])
    drift_grid = np.interp(
        grid.centers, trajectory[0][order], drift_particles[order]
    )
    row = _metric_row(
        f"mvnn_seed{model_seed}",
        family_row,
        measure,
        predicted,
        reference,
        times,
        grid,
        trajectory,
        drift_grid,
        params,
        drift_at_particles=drift_particles,
    )
    row["rollout_seconds"] = rollout_seconds
    row["preprocessing_seconds"] = preprocessing_seconds
    row["max_abs_mass_error"] = 0.0  # particle method conserves mass exactly
    row["max_negative_fraction"] = 0.0
    return row, predicted


def _metric_row(
    method: str,
    family_row: dict[str, Any],
    measure: str,
    predicted: np.ndarray,
    reference: np.ndarray,
    times: np.ndarray,
    grid: Grid,
    trajectory: np.ndarray,
    drift_on_grid: np.ndarray,
    params: GroundTruth,
    drift_at_particles: np.ndarray | None = None,
) -> dict[str, Any]:
    family = family_row["family"]
    w1_t = w1_trajectory(predicted, reference, grid)
    window_center = float(family["x_c"])
    window_half = float(family["window_half_width"])
    short_mask = times <= 0.3 + 1e-12
    predicted_centroids = bump_centroid(
        predicted[short_mask], grid, window_center, window_half
    )
    reference_centroids = bump_centroid(
        reference[short_mask], grid, window_center, window_half
    )
    predicted_velocity = short_time_velocity(
        predicted_centroids, times[short_mask], horizon=0.3
    )
    reference_velocity = short_time_velocity(
        reference_centroids, times[short_mask], horizon=0.3
    )

    # drift RMSE at t=0 against the true drift at the reference particles
    particles = trajectory[0]
    true_b = true_drift(particles, float(particles.mean()), params)
    if drift_at_particles is None:
        order = np.argsort(grid.centers)
        model_b = np.interp(particles, grid.centers[order], drift_on_grid[order])
    else:
        model_b = drift_at_particles

    return {
        "method": method,
        "triplet_id": int(family["triplet_id"]),
        "measure": measure,
        "time_averaged_w1": float(np.mean(w1_t)),
        "final_w1": float(w1_t[-1]),
        "w1_of_t": w1_t.tolist(),
        "times": times.tolist(),
        "predicted_short_time_velocity": float(predicted_velocity),
        "reference_short_time_velocity": float(reference_velocity),
        "direction_match": direction_match(predicted_velocity, reference_velocity),
        "predicted_M_of_t": density_mean(predicted, grid).tolist(),
        "reference_M_of_t": density_mean(reference, grid).tolist(),
        "drift_rmse_t0": drift_rmse(model_b, true_b),
        "window_center": window_center,
        "window_half_width": window_half,
    }


def affinity_defect_rows(
    method: str,
    drift_of_density: Callable[[np.ndarray], np.ndarray] | None,
    drift_of_particles: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    family_row: dict[str, Any],
    initial_particles: dict[str, np.ndarray],
    grid: Grid,
    bandwidth: float,
    params: GroundTruth,
) -> dict[str, Any]:
    """Affinity defect of a fitted model on one matched triplet.

    Every method sees exactly the same empirical objects: the three stored
    particle vectors, whose construction satisfies the convex-mixture
    identity mu_mixed = (mu_plus + mu_minus)/2 EXACTLY at the multiset level.
    Density models receive the shared KDE of each particle vector (KDE is
    linear in the empirical measure, so rho_mixed = (rho_plus + rho_minus)/2
    to floating point, which is recorded as a diagnostic); the MVNN receives
    the particle vectors directly. Any additive model must give zero defect
    to numerical tolerance; the defect is compared with the true
    alpha*x*d^2 on the central bump window.
    """

    family = family_row["family"]
    d = float(family["d"])
    window_center = float(family["x_c"])
    window_half = float(family["window_half_width"])
    window = (grid.centers >= window_center - window_half) & (
        grid.centers <= window_center + window_half
    )
    x = grid.centers[window]
    truth = params.alpha * x * d**2
    kde_mixture_gap = None

    if drift_of_density is not None:
        rho = {
            measure: kde_density(initial_particles[measure], grid, bandwidth)
            for measure in tp.MEASURES
        }
        kde_mixture_gap = float(
            np.max(np.abs(rho["mixed"] - 0.5 * (rho["plus"] + rho["minus"])))
        )
        defect_full = drift_of_density(rho["mixed"]) - 0.5 * (
            drift_of_density(rho["plus"]) + drift_of_density(rho["minus"])
        )
        defect = defect_full[window]
    else:
        assert drift_of_particles is not None
        query = x[:, None]
        defect = (
            drift_of_particles(query, initial_particles["mixed"][:, None])
            - 0.5
            * (
                drift_of_particles(query, initial_particles["plus"][:, None])
                + drift_of_particles(query, initial_particles["minus"][:, None])
            )
        )[:, 0]

    truth_rms = float(np.sqrt(np.mean(truth**2)))
    return {
        "method": method,
        "triplet_id": int(family["triplet_id"]),
        "d": d,
        "defect_rms": float(np.sqrt(np.mean(np.asarray(defect) ** 2))),
        "truth_defect_rms": truth_rms,
        "defect_vs_truth_relative_error": float(
            np.sqrt(np.mean((np.asarray(defect) - truth) ** 2)) / max(truth_rms, 1e-300)
        ),
        "kde_mixture_max_abs_gap": kde_mixture_gap,
    }


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    scalar_columns = [
        column
        for column in frame.columns
        if column not in ("w1_of_t", "times", "predicted_M_of_t", "reference_M_of_t")
    ]
    return frame[scalar_columns + [c for c in frame.columns if c not in scalar_columns]]
