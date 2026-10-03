"""Generate the fixed validation/test sets and the trajectory manifest.

Usage:
  python scripts/mtcurve_generate_data.py [--verify-only]

Steps:
  1. Port verification: regenerate training-pool seed 1 with the ported
     simulator and match the stored trajectory on frames 0..200.
  2. Simulate 10 validation (seeds 1001-1010) and 30 test (seeds 2001-2030)
     trajectories from the SAME initial-condition family, 200 Euler steps.
  3. Write outputs/mtcurve/data/manifest.json recording path, IC parameters,
     seed, and sha256 of every training/validation/test trajectory.
Idempotent: existing generated files are verified against the manifest and
never overwritten.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from mtcurve import data as mtdata  # noqa: E402
from mtcurve.simulate import init_opinion, simulate  # noqa: E402
from threebody.artifacts import sha256_file, write_json  # noqa: E402

PORT_VERIFICATION_TOLERANCE = 1e-8


def _simulate_one(job: tuple[str, int]) -> str:
    """Worker: simulate one (role, seed) trajectory and save it."""

    role, seed = job
    path = mtdata.generated_path(role, seed)
    if path.exists():
        return f"exists {role} {seed}"
    path.parent.mkdir(parents=True, exist_ok=True)
    frames, _ = simulate(seed=seed, steps=mtdata.FRAMES_USED - 1)
    np.save(path, frames)
    return f"simulated {role} {seed}"


def verify_port() -> float:
    stored = np.load(mtdata.pool_path(1), mmap_mode="r")
    regenerated, _ = simulate(seed=1, steps=mtdata.FRAMES_USED - 1)
    difference = float(
        np.max(np.abs(regenerated - np.asarray(stored[: mtdata.FRAMES_USED])))
    )
    if difference > PORT_VERIFICATION_TOLERANCE:
        raise RuntimeError(
            f"ported simulator deviates from stored pool trajectory: {difference:g}"
        )
    return difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    port_difference = verify_port()
    print(f"port verification: max|diff| = {port_difference:.3e} on seed 1")
    if args.verify_only:
        return

    rows: list[dict] = []
    for seed in mtdata.POOL_SEEDS:
        path = mtdata.pool_path(seed)
        _, parameters = init_opinion(mtdata.load_frames("training", seed).shape[1], seed)
        rows.append(
            {
                "role": "training",
                "seed": seed,
                "path": str(path),
                "sha256": sha256_file(path),
                "initial_condition": parameters,
            }
        )
        if seed % 20 == 0:
            print(f"hashed training pool seed {seed}")

    jobs = [("validation", seed) for seed in mtdata.VALIDATION_SEEDS] + [
        ("test", seed) for seed in mtdata.TEST_SEEDS
    ]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for message in pool.map(_simulate_one, jobs):
            print(f"{message} ({time.perf_counter() - start:.0f}s elapsed)")

    for role, seed in jobs:
        path = mtdata.generated_path(role, seed)
        frames = np.load(path, mmap_mode="r")
        if frames.shape != (mtdata.FRAMES_USED, 16000):
            raise RuntimeError(f"{path} has unexpected shape {frames.shape}")
        _, parameters = init_opinion(16000, seed)
        rows.append(
            {
                "role": role,
                "seed": seed,
                "path": str(path),
                "sha256": sha256_file(path),
                "initial_condition": parameters,
            }
        )

    pool_hashes = {row["sha256"] for row in rows if row["role"] == "training"}
    generated_hashes = {row["sha256"] for row in rows if row["role"] != "training"}
    if pool_hashes & generated_hashes:
        raise RuntimeError("generated trajectories duplicate training-pool content")

    manifest = {
        "schema_version": 1,
        "frames_used": mtdata.FRAMES_USED,
        "n_agents": 16000,
        "dt": 0.01,
        "seed_ranges": {
            "training": [min(mtdata.POOL_SEEDS), max(mtdata.POOL_SEEDS)],
            "validation": [min(mtdata.VALIDATION_SEEDS), max(mtdata.VALIDATION_SEEDS)],
            "test": [min(mtdata.TEST_SEEDS), max(mtdata.TEST_SEEDS)],
        },
        "port_verification_max_abs_diff_seed1": port_difference,
        "trajectories": rows,
    }
    write_json(mtdata.DATA_ROOT / "manifest.json", manifest, immutable=True)
    print(f"manifest written with {len(rows)} trajectories "
          f"in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
