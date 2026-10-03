"""Stage dataset generation, storage, and split handling.

Layout (under outputs/threebody/data/<stage>/):
  manifest.json                       families, roles, seeds, checksums
  triplet_<id>_<measure>.npy          (n_frames, N) float64 particle frames
  observation.json                    selected KDE bandwidth (train-only choice)

Splits are by entire triplet families; every family contributes its three
trajectories (plus/minus/mixed) to the same role. Confirmation families use a
separate seed base and are generated only after the freeze marker exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import triplets as tp
from .artifacts import read_json, sha256_file, write_json
from .observation import Grid, kde_density
from .protocol import GroundTruth
from .simulate import check_boundary_mass, simulate

CONFIRMATION_ID_OFFSET = 100


def trajectory_path(stage_dir: Path, triplet_id: int, measure: str) -> Path:
    return Path(stage_dir) / f"triplet_{triplet_id:03d}_{measure}.npy"


def generate_role(
    stage_dir: Path,
    families: list[tp.TripletFamily],
    role: str,
    params: GroundTruth,
    seed_base: int,
    boundary_tolerance: float,
) -> list[dict[str, Any]]:
    """Simulate and store all trajectories for one role; returns manifest rows."""

    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for family in families:
        initial = tp.build_triplet(family)
        row: dict[str, Any] = {
            "role": role,
            "family": family.to_dict(),
            "measures": {},
        }
        for measure in tp.MEASURES:
            seed = tp.brownian_seed(seed_base, family.triplet_id, measure)
            trajectory = simulate(initial[measure], params, seed)
            boundary_fraction = check_boundary_mass(
                trajectory, params.domain, boundary_tolerance
            )
            path = trajectory_path(stage_dir, family.triplet_id, measure)
            np.save(path, trajectory)
            row["measures"][measure] = {
                "brownian_seed": seed,
                "path": path.name,
                "sha256": sha256_file(path),
                "initial_mean": float(initial[measure].mean()),
                "max_boundary_fraction": boundary_fraction,
                "min_state": float(trajectory.min()),
                "max_state": float(trajectory.max()),
            }
        rows.append(row)
    return rows


def load_manifest(stage_dir: Path) -> dict[str, Any]:
    return read_json(Path(stage_dir) / "manifest.json")


def write_manifest(stage_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(Path(stage_dir) / "manifest.json", manifest, immutable=True)


def families_for_role(manifest: dict[str, Any], role: str) -> list[dict[str, Any]]:
    return [row for row in manifest["triplets"] if row["role"] == role]


def load_trajectory(stage_dir: Path, triplet_id: int, measure: str) -> np.ndarray:
    return np.load(trajectory_path(Path(stage_dir), triplet_id, measure))


def load_role_trajectories(
    stage_dir: Path, manifest: dict[str, Any], role: str
) -> list[tuple[dict[str, Any], str, np.ndarray]]:
    """[(family_row, measure, (n_frames, N) trajectory), ...] in stable order."""

    out = []
    for row in families_for_role(manifest, role):
        triplet_id = int(row["family"]["triplet_id"])
        for measure in tp.MEASURES:
            out.append((row, measure, load_trajectory(stage_dir, triplet_id, measure)))
    return out


def select_bandwidth(
    train_trajectories: list[np.ndarray],
    grid: Grid,
    candidates: list[float],
    seed: int = 20260711,
) -> tuple[float, list[dict[str, float]]]:
    """Held-out-particle W1 bandwidth selection on training data only.

    Particles of sampled frames are split 50/50; the KDE of one half is scored
    by quantile W1 against the held-out half. Ties break to the smaller
    bandwidth.
    """

    from .metrics import midpoint_probabilities, quantiles_from_density

    rng = np.random.default_rng(seed)
    probs = midpoint_probabilities(2048)
    frames = []
    for trajectory in train_trajectories:
        for frame_index in (0, trajectory.shape[0] // 2, trajectory.shape[0] - 1):
            frames.append(trajectory[frame_index])
    table: list[dict[str, float]] = []
    scores = []
    for bandwidth in candidates:
        errors = []
        for frame in frames:
            permutation = rng.permutation(frame.size)
            half = frame.size // 2
            fit_half = frame[permutation[:half]]
            held_half = frame[permutation[half:]]
            density = kde_density(fit_half, grid, bandwidth)
            quantiles_fit = quantiles_from_density(density, grid, probs)
            quantiles_held = np.quantile(held_half, probs, method="linear")
            errors.append(float(np.mean(np.abs(quantiles_fit - quantiles_held))))
        score = float(np.mean(errors))
        table.append({"bandwidth": float(bandwidth), "mean_w1": score})
        scores.append(score)
    best_index = int(np.argmin(scores))
    # ties (within 1e-12) break to the smaller bandwidth
    best_score = scores[best_index]
    for index, score in enumerate(scores):
        if score <= best_score + 1e-12:
            best_index = index
            break
    return float(candidates[best_index]), table
