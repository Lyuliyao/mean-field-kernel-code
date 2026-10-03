"""Fit both WSINDy baselines with validation-rollout-W1 model selection.

Usage:
  python scripts/threebody_fit_baselines.py --stage pilot
  python scripts/threebody_fit_baselines.py --stage final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from threebody import triplets as tp  # noqa: E402
from threebody.artifacts import atomic_write_text, read_json  # noqa: E402
from threebody.baselines import fit_kernel_baseline, fit_local_baseline  # noqa: E402
from threebody.data import load_manifest, load_trajectory  # noqa: E402
from threebody.kernel_wsindy import save_kernel_model  # noqa: E402
from threebody.local_wsindy import save_local_model  # noqa: E402
from threebody.observation import Grid, kde_density  # noqa: E402
from threebody.protocol import ground_truth_params, load_protocol  # noqa: E402
from threebody.weakform import WeakConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "final"), required=True)
    args = parser.parse_args()

    protocol = load_protocol(ROOT / "protocol.yaml")
    params = ground_truth_params(protocol)
    stage_dir = ROOT / "outputs" / "threebody" / "data" / args.stage
    manifest = load_manifest(stage_dir)
    observation = read_json(stage_dir / "observation.json")
    grid = Grid(
        float(observation["domain"][0]),
        float(observation["domain"][1]),
        int(observation["grid_cells"]),
    )
    bandwidth = float(observation["bandwidth"])
    output_dir = ROOT / "outputs" / "threebody" / args.stage / "baselines"
    marker = output_dir / "fit_summary.json"
    if marker.exists():
        raise FileExistsError(f"{marker} already exists; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    def role_trajectories(role: str) -> list[np.ndarray]:
        out = []
        for row in manifest["triplets"]:
            if row["role"] != role:
                continue
            triplet_id = int(row["family"]["triplet_id"])
            for measure in tp.MEASURES:
                out.append(load_trajectory(stage_dir, triplet_id, measure))
        return out

    train_trajectories = role_trajectories("train")
    val_trajectories = role_trajectories("validation")
    times = np.arange(params.n_frames) * params.dt

    preprocessing_start = time.perf_counter()
    train_densities = [
        kde_density(trajectory, grid, bandwidth) for trajectory in train_trajectories
    ]
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    weak_config = WeakConfig(
        time_half_width=0.05,
        space_half_width=0.3,
        time_stride=2,
        space_stride=3,
        max_windows_per_trajectory=400,
        random_seed=49157,
    )

    nu = params.sigma**2 / 2.0
    local_cfg = protocol["local_wsindy"]
    local_start = time.perf_counter()
    local_model, local_selection = fit_local_baseline(
        train_densities,
        val_trajectories,
        times,
        grid,
        bandwidth,
        nu,
        weak_config,
        [int(k) for k in local_cfg["knot_candidates"]],
        [float(s) for s in local_cfg["smoothness_candidates"]],
        [float(t) for t in local_cfg["group_threshold_candidates"]],
        float(local_cfg["ridge"]),
    )
    local_seconds = time.perf_counter() - local_start
    save_local_model(output_dir / "local_model.npz", local_model)

    kernel_cfg = protocol["kernel_wsindy"]
    kernel_start = time.perf_counter()
    kernel_model, kernel_selection = fit_kernel_baseline(
        train_densities,
        train_trajectories,
        val_trajectories,
        times,
        grid,
        bandwidth,
        nu,
        weak_config,
        [int(k) for k in kernel_cfg["spline_count_candidates"]],
        [float(s) for s in kernel_cfg["smoothness_candidates"]],
        float(kernel_cfg["ridge"]),
    )
    kernel_seconds = time.perf_counter() - kernel_start
    save_kernel_model(output_dir / "kernel_model.npz", kernel_model)

    summary = {
        "stage": args.stage,
        "bandwidth": bandwidth,
        "weak_config": {
            "time_half_width": weak_config.time_half_width,
            "space_half_width": weak_config.space_half_width,
            "time_stride": weak_config.time_stride,
            "space_stride": weak_config.space_stride,
            "polynomial_order": weak_config.polynomial_order,
            "max_windows_per_trajectory": weak_config.max_windows_per_trajectory,
            "random_seed": weak_config.random_seed,
        },
        "preprocessing_seconds": preprocessing_seconds,
        "local": {
            "selection": local_selection,
            "wall_seconds_including_selection": local_seconds,
            "model_file": "local_model.npz",
        },
        "kernel": {
            "selection": kernel_selection,
            "nu": nu,
            "wall_seconds_including_selection": kernel_seconds,
            "model_file": "kernel_model.npz",
        },
    }
    atomic_write_text(marker, json.dumps(summary, indent=2) + "\n")
    print(
        "local selected:",
        local_selection["selected"],
        "\nkernel selected:",
        kernel_selection["selected"],
    )


if __name__ == "__main__":
    main()
