"""Baseline fitting drivers with validation-rollout-W1 model selection.

Both WSINDy baselines fit ONE shared model across all training trajectories
and families. Hyperparameters (spline counts, smoothness, threshold) are
selected exclusively by mean autonomous validation rollout W1; candidates
whose rollouts fail (CFL collapse, non-finite states) receive infinite scores.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .kernel_wsindy import KernelModel, displacement_domain, fit_kernel_wsindy
from .local_wsindy import LocalModel, fit_local_wsindy
from .metrics import w1_trajectory
from .observation import Grid, kde_density
from .solver import SolverConfig, TransportSolver1D
from .weakform import WeakConfig


def rollout_model_density(
    velocity_fn: Callable[[np.ndarray], np.ndarray],
    diffusion: float,
    initial_density: np.ndarray,
    grid: Grid,
    output_times: np.ndarray,
    max_time_step: float,
    cfl: float = 0.4,
) -> tuple[np.ndarray, dict]:
    if diffusion < 0.0:
        raise FloatingPointError("negative diffusion coefficient is unstable")
    solver = TransportSolver1D(
        grid.centers,
        velocity_fn,
        SolverConfig(
            cfl=cfl,
            diffusion=diffusion,
            max_time_step=max_time_step,
            max_steps=500_000,
        ),
    )
    return solver.rollout(initial_density, output_times)


def validation_rollout_w1(
    velocity_fn: Callable[[np.ndarray], np.ndarray],
    diffusion: float,
    validation_trajectories: list[np.ndarray],
    grid: Grid,
    bandwidth: float,
    dt: float,
    eval_frame_stride: int = 5,
) -> float:
    """Mean time-averaged W1 of autonomous density rollouts on validation data."""

    scores = []
    for trajectory in validation_trajectories:
        n_frames = trajectory.shape[0]
        eval_frames = np.arange(0, n_frames, eval_frame_stride)
        if eval_frames[-1] != n_frames - 1:
            eval_frames = np.concatenate([eval_frames, [n_frames - 1]])
        reference = kde_density(trajectory[eval_frames], grid, bandwidth)
        initial = reference[0]
        output_times = eval_frames.astype(np.float64) * dt
        try:
            predicted, _ = rollout_model_density(
                velocity_fn, diffusion, initial, grid, output_times, dt
            )
        except FloatingPointError:
            return float("inf")
        scores.append(float(np.mean(w1_trajectory(predicted, reference, grid))))
    return float(np.mean(scores))


def fit_local_baseline(
    train_densities: list[np.ndarray],
    validation_trajectories: list[np.ndarray],
    times: np.ndarray,
    grid: Grid,
    bandwidth: float,
    nu: float,
    weak_config: WeakConfig,
    knot_candidates: list[int],
    smoothness_candidates: list[float],
    threshold_candidates: list[float],
    ridge: float = 1.0e-8,
) -> tuple[LocalModel, dict[str, Any]]:
    dt = float(times[1] - times[0])
    rows: list[dict[str, Any]] = []
    best: tuple[float, int, float, float] | None = None
    best_model: LocalModel | None = None
    best_fit_seconds = 0.0
    fit_seconds = 0.0
    for basis_count in knot_candidates:
        for smoothness in smoothness_candidates:
            for threshold in threshold_candidates:
                start = time.perf_counter()
                model, fit = fit_local_wsindy(
                    train_densities,
                    times,
                    grid,
                    basis_count,
                    nu,
                    weak_config,
                    smoothness,
                    threshold,
                    ridge,
                )
                candidate_fit_seconds = time.perf_counter() - start
                fit_seconds += candidate_fit_seconds
                score = validation_rollout_w1(
                    model.velocity,
                    model.diffusion,
                    validation_trajectories,
                    grid,
                    bandwidth,
                    dt,
                )
                rows.append(
                    {
                        "basis_count": basis_count,
                        "smoothness": smoothness,
                        "threshold": threshold,
                        "validation_rollout_w1": score,
                        "relative_residual": fit.relative_residual,
                        "active_groups": list(fit.active_groups),
                        "fit_seconds": candidate_fit_seconds,
                    }
                )
                key = (score, basis_count, smoothness, threshold)
                if best is None or key < best:
                    best = key
                    best_model = model
                    best_fit_seconds = candidate_fit_seconds
    if best_model is None or not np.isfinite(best[0]):
        raise RuntimeError("no stable local WSINDy candidate was found")
    selection = {
        "nu": nu,
        "candidates": rows,
        "selected": {
            "basis_count": best[1],
            "smoothness": best[2],
            "threshold": best[3],
            "validation_rollout_w1": best[0],
        },
        "selected_fit_seconds": best_fit_seconds,
        "fit_seconds_total": fit_seconds,
    }
    return best_model, selection


def fit_kernel_baseline(
    train_densities: list[np.ndarray],
    train_particle_trajectories: list[np.ndarray],
    validation_trajectories: list[np.ndarray],
    times: np.ndarray,
    grid: Grid,
    bandwidth: float,
    nu: float,
    weak_config: WeakConfig,
    spline_count_candidates: list[int],
    smoothness_candidates: list[float],
    ridge: float = 1.0e-8,
) -> tuple[KernelModel, dict[str, Any]]:
    dt = float(times[1] - times[0])
    displacement = displacement_domain(train_particle_trajectories)
    rows: list[dict[str, Any]] = []
    best: tuple[float, int, float] | None = None
    best_model: KernelModel | None = None
    best_fit_seconds = 0.0
    fit_seconds = 0.0
    for basis_count in spline_count_candidates:
        for smoothness in smoothness_candidates:
            start = time.perf_counter()
            model, fit = fit_kernel_wsindy(
                train_densities,
                times,
                grid,
                displacement,
                basis_count,
                nu,
                weak_config,
                smoothness,
                ridge,
            )
            candidate_fit_seconds = time.perf_counter() - start
            fit_seconds += candidate_fit_seconds
            score = validation_rollout_w1(
                model.velocity,
                model.nu,
                validation_trajectories,
                grid,
                bandwidth,
                dt,
            )
            rows.append(
                {
                    "basis_count": basis_count,
                    "smoothness": smoothness,
                    "validation_rollout_w1": score,
                    "relative_residual": fit.relative_residual,
                    "fit_seconds": candidate_fit_seconds,
                }
            )
            key = (score, basis_count, smoothness)
            if best is None or key < best:
                best = key
                best_model = model
                best_fit_seconds = candidate_fit_seconds
    if best_model is None or not np.isfinite(best[0]):
        raise RuntimeError("no stable kernel WSINDy candidate was found")
    selection = {
        "displacement_domain": list(displacement),
        "candidates": rows,
        "selected": {
            "basis_count": best[1],
            "smoothness": best[2],
            "validation_rollout_w1": best[0],
        },
        "selected_fit_seconds": best_fit_seconds,
        "fit_seconds_total": fit_seconds,
    }
    return best_model, selection
