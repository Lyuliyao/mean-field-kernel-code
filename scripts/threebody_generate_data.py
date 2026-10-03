"""Generate stage datasets (triplet families, trajectories, bandwidth choice).

Usage:
  python scripts/threebody_generate_data.py --stage pilot
  python scripts/threebody_generate_data.py --stage final
  python scripts/threebody_generate_data.py --stage final --role confirmation

Confirmation data can only be generated after the freeze marker exists and is
verified; training/validation data for a stage are generated together.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threebody import triplets as tp  # noqa: E402
from threebody.artifacts import sha256_file, write_json  # noqa: E402
from threebody.data import (  # noqa: E402
    CONFIRMATION_ID_OFFSET,
    generate_role,
    load_manifest,
    select_bandwidth,
    write_manifest,
)


def verify_existing_dataset(data_dir: Path) -> bool:
    """If a manifest exists, verify every stored trajectory checksum and stop.

    Returns True when the dataset already exists and is intact; raises when it
    exists but does not match (a rerun must never silently rewrite data).
    """

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = load_manifest(data_dir)
    for row in manifest["triplets"]:
        for measure, record in row["measures"].items():
            path = data_dir / record["path"]
            if not path.exists() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(
                    f"existing dataset {data_dir} does not match its manifest "
                    f"({path.name}); refusing to overwrite"
                )
    return True
from threebody.observation import default_grid  # noqa: E402
from threebody.protocol import ground_truth_params, load_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "final"), required=True)
    parser.add_argument(
        "--role",
        choices=("development", "confirmation"),
        default="development",
        help="development = train+validation; confirmation requires the freeze marker",
    )
    args = parser.parse_args()

    protocol = load_protocol(ROOT / "protocol.yaml")
    params = ground_truth_params(protocol)
    stage_cfg = protocol["stages"][args.stage]
    grid = default_grid(protocol)
    boundary_tolerance = float(protocol["ground_truth"]["boundary_mass_tolerance"])
    stage_dir = ROOT / "outputs" / "threebody" / "data" / args.stage
    n_particles = int(stage_cfg["particles"])

    start = time.perf_counter()
    if args.role == "development":
        if verify_existing_dataset(stage_dir):
            print(f"{stage_dir} already exists and matches its manifest; nothing to do")
            return
        seed_base = int(stage_cfg["data_seed_base"])
        n_train = int(stage_cfg["n_train_triplets"])
        n_val = int(stage_cfg["n_validation_triplets"])
        train_families = tp.sample_families(
            n_train, n_particles, seed_base, id_offset=0
        )
        val_families = tp.sample_families(
            n_val, n_particles, seed_base, id_offset=n_train
        )
        rows = generate_role(
            stage_dir, train_families, "train", params, seed_base, boundary_tolerance
        )
        rows += generate_role(
            stage_dir, val_families, "validation", params, seed_base, boundary_tolerance
        )
        manifest = {
            "schema_version": 1,
            "stage": args.stage,
            "data_seed_base": seed_base,
            "particles": n_particles,
            "triplets": rows,
        }
        write_manifest(stage_dir, manifest)

        # KDE bandwidth: training triplets only (common observation operator)
        train_trajectories = []
        import numpy as np

        for row in rows:
            if row["role"] != "train":
                continue
            for measure in tp.MEASURES:
                train_trajectories.append(
                    np.load(stage_dir / row["measures"][measure]["path"])
                )
        candidates = [
            float(b) for b in protocol["observation"]["bandwidth_candidates"]
        ]
        bandwidth, table = select_bandwidth(train_trajectories, grid, candidates)
        write_json(
            stage_dir / "observation.json",
            {
                "grid_cells": grid.cells,
                "domain": [grid.lower, grid.upper],
                "bandwidth": bandwidth,
                "bandwidth_candidates": candidates,
                "selection_table": table,
                "selection_rule": "held_out_particle_w1_on_training_triplets_only",
            },
        )
        print(f"selected bandwidth: {bandwidth}")
    else:
        from threebody.freeze import assert_frozen

        assert_frozen(ROOT, ROOT / "outputs" / "threebody")
        seed_base = int(stage_cfg["confirmation_seed_base"])
        n_confirmation = int(stage_cfg["n_confirmation_triplets"])
        confirmation_dir = stage_dir / "confirmation"
        if verify_existing_dataset(confirmation_dir):
            print(f"{confirmation_dir} already exists and matches; nothing to do")
            return
        families = tp.sample_families(
            n_confirmation,
            n_particles,
            seed_base,
            id_offset=CONFIRMATION_ID_OFFSET,
        )
        rows = generate_role(
            confirmation_dir,
            families,
            "confirmation",
            params,
            seed_base,
            boundary_tolerance,
        )
        manifest = {
            "schema_version": 1,
            "stage": args.stage,
            "role": "confirmation",
            "confirmation_seed_base": seed_base,
            "particles": n_particles,
            "triplets": rows,
        }
        write_manifest(confirmation_dir, manifest)

    elapsed = time.perf_counter() - start
    print(f"generated {args.stage}/{args.role} data in {elapsed:.1f}s -> {stage_dir}")


if __name__ == "__main__":
    main()
