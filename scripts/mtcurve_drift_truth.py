"""Precompute the exact drift on reference frames for every trajectory.

Usage:
  python scripts/mtcurve_drift_truth.py

Caches b_true at frames t = 0.1, ..., 2.0 (stride 10) for all 100 training,
10 validation, and 30 test trajectories. Idempotent (skips existing caches).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from mtcurve import data as mtdata  # noqa: E402
from mtcurve.drift_truth import true_drift_frames  # noqa: E402


def main() -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    frame_indices = np.arange(
        10, mtdata.FRAMES_USED, 10
    )  # t = 0.1 .. 2.0, matching training.METRIC_FRAME_STRIDE
    jobs = (
        [("training", seed) for seed in mtdata.POOL_SEEDS]
        + [("validation", seed) for seed in mtdata.VALIDATION_SEEDS]
        + [("test", seed) for seed in mtdata.TEST_SEEDS]
    )
    start = time.perf_counter()
    done = 0
    for role, seed in jobs:
        path = mtdata.drift_truth_path(role, seed)
        if path.exists():
            done += 1
            continue
        frames = mtdata.load_frames(role, seed)
        drift = true_drift_frames(frames, frame_indices)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frame_indices=frame_indices, drift=drift)
        done += 1
        if done % 20 == 0:
            print(f"{done}/{len(jobs)} drift caches "
                  f"({time.perf_counter() - start:.0f}s)")
    print(f"all {len(jobs)} drift-truth caches ready "
          f"({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
