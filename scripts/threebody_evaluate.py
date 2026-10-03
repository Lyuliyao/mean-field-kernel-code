"""Evaluate all methods on validation (development) or confirmation triplets.

Usage:
  python scripts/threebody_evaluate.py --stage pilot --role validation
  python scripts/threebody_evaluate.py --stage final --role validation
  python scripts/threebody_evaluate.py --stage final --role confirmation

The confirmation role requires the freeze marker, verifies the protocol
checksum, and refuses to run twice.
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
from threebody.data import load_manifest  # noqa: E402
from threebody.evaluation import (  # noqa: E402
    affinity_defect_rows,
    evaluate_mvnn_method,
    evaluate_pde_method,
)
from threebody.kernel_wsindy import load_kernel_model  # noqa: E402
from threebody.local_wsindy import load_local_model  # noqa: E402
from threebody.observation import Grid  # noqa: E402
from threebody.protocol import ground_truth_params, load_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "final"), required=True)
    parser.add_argument("--role", choices=("validation", "confirmation"), required=True)
    args = parser.parse_args()

    protocol = load_protocol(ROOT / "protocol.yaml")
    params = ground_truth_params(protocol)
    stage_cfg = protocol["stages"][args.stage]
    outputs_root = ROOT / "outputs" / "threebody"
    stage_dir = outputs_root / "data" / args.stage
    observation = read_json(stage_dir / "observation.json")
    grid = Grid(
        float(observation["domain"][0]),
        float(observation["domain"][1]),
        int(observation["grid_cells"]),
    )
    bandwidth = float(observation["bandwidth"])

    if args.role == "confirmation":
        from threebody.freeze import assert_artifacts_match, assert_frozen

        if args.stage != "final":
            raise RuntimeError("confirmation evaluation exists only for the final stage")
        marker = assert_frozen(ROOT, outputs_root)
        assert_artifacts_match(marker, outputs_root)
        data_dir = stage_dir / "confirmation"
        stage_base = int(stage_cfg["confirmation_seed_base"])
    else:
        data_dir = stage_dir
        stage_base = int(stage_cfg["data_seed_base"])
    manifest = load_manifest(data_dir)

    result_dir = outputs_root / args.stage / "evaluation" / args.role
    marker = result_dir / "raw_metrics.json"
    if marker.exists():
        raise FileExistsError(
            f"{marker} already exists; the {args.role} evaluation runs exactly once"
        )
    result_dir.mkdir(parents=True, exist_ok=True)

    # ---- load locked models -------------------------------------------------
    baseline_dir = outputs_root / args.stage / "baselines"
    local_model = load_local_model(baseline_dir / "local_model.npz")
    kernel_model = load_kernel_model(baseline_dir / "kernel_model.npz")

    from threebody.training import load_best_params

    mvnn_runs = []
    for seed in stage_cfg["mvnn_seeds"]:
        run_dir = outputs_root / args.stage / "mvnn" / f"seed_{seed}"
        model, params_tree = load_best_params(run_dir)
        mvnn_runs.append((int(seed), model, params_tree))

    role_rows = [row for row in manifest["triplets"] if row["role"] == args.role]
    if not role_rows:
        raise RuntimeError(f"no triplets with role {args.role} in manifest")

    metric_rows: list[dict] = []
    defect_rows: list[dict] = []
    timing = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    start = time.perf_counter()

    for family_row in role_rows:
        triplet_id = int(family_row["family"]["triplet_id"])
        family = tp.TripletFamily(**family_row["family"])
        initial = tp.build_triplet(family)
        trajectories = {
            measure: np.load(data_dir / family_row["measures"][measure]["path"])
            for measure in tp.MEASURES
        }
        for measure in tp.MEASURES:
            trajectory = trajectories[measure]
            row, _ = evaluate_pde_method(
                "local_wsindy",
                local_model.velocity,
                local_model.diffusion,
                local_model.drift_field,
                family_row,
                measure,
                trajectory,
                grid,
                bandwidth,
                params,
            )
            metric_rows.append(row)
            row, _ = evaluate_pde_method(
                "kernel_wsindy",
                kernel_model.velocity,
                kernel_model.nu,
                kernel_model.drift_field,
                family_row,
                measure,
                trajectory,
                grid,
                bandwidth,
                params,
            )
            metric_rows.append(row)
            for seed, model, params_tree in mvnn_runs:
                row, _ = evaluate_mvnn_method(
                    model,
                    params_tree,
                    seed,
                    stage_base,
                    family_row,
                    measure,
                    trajectory,
                    grid,
                    bandwidth,
                    params,
                )
                metric_rows.append(row)

        # affinity defect of every fitted model on this matched triplet
        defect_rows.append(
            affinity_defect_rows(
                "local_wsindy",
                local_model.drift_field,
                None,
                family_row,
                initial,
                grid,
                bandwidth,
                params,
            )
        )
        defect_rows.append(
            affinity_defect_rows(
                "kernel_wsindy",
                kernel_model.drift_field,
                None,
                family_row,
                initial,
                grid,
                bandwidth,
                params,
            )
        )
        for seed, model, params_tree in mvnn_runs:
            def particle_drift(query, measure_particles, _m=model, _p=params_tree):
                return np.asarray(_m.drift_at(_p, query, measure_particles))

            defect_rows.append(
                affinity_defect_rows(
                    f"mvnn_seed{seed}",
                    None,
                    particle_drift,
                    family_row,
                    initial,
                    grid,
                    bandwidth,
                    params,
                )
            )

    timing["wall_seconds"] = time.perf_counter() - start
    payload = {
        "stage": args.stage,
        "role": args.role,
        "bandwidth": bandwidth,
        "grid_cells": grid.cells,
        "n_triplets": len(role_rows),
        "timing": timing,
        "metrics": metric_rows,
        "affinity_defects": defect_rows,
    }
    atomic_write_text(marker, json.dumps(payload, indent=2) + "\n")

    # compact CSV for quick reading
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key
                not in ("w1_of_t", "times", "predicted_M_of_t", "reference_M_of_t")
            }
            for row in metric_rows
        ]
    )
    frame.to_csv(result_dir / "raw_metrics.csv", index=False)
    pd.DataFrame(defect_rows).to_csv(result_dir / "affinity_defects.csv", index=False)
    summary = (
        frame.groupby("method")
        .agg(
            time_averaged_w1=("time_averaged_w1", "mean"),
            final_w1=("final_w1", "mean"),
            direction_accuracy=("direction_match", "mean"),
            drift_rmse_t0=("drift_rmse_t0", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(result_dir / "method_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
