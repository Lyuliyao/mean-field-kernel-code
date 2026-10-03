"""Data sets, manifests, and nested training subsets for the learning curve."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = Path(os.environ.get('MVNN_MT_POOL', str(REPO_ROOT / 'data' / 'raw' / 'mt_training_pool')))
DATA_ROOT = REPO_ROOT / "outputs" / "mtcurve" / "data"

POOL_SEEDS = tuple(range(1, 101))
VALIDATION_SEEDS = tuple(range(1001, 1011))
TEST_SEEDS = tuple(range(2001, 2031))

FRAMES_USED = 201  # l = 0..200, T = 2
M_VALUES = (5, 10, 20, 40, 60, 80, 100)
REPLICATES = 3
PERMUTATION_STREAM = 777
MODEL_SEED_STREAM = 888
BATCH_SEED_STREAM = 999


def pool_path(seed: int) -> Path:
    return POOL_DIR / f"opinion_traj_{seed}.npy"


def generated_path(role: str, seed: int) -> Path:
    return DATA_ROOT / role / f"opinion_traj_{seed}.npy"


def trajectory_path(role: str, seed: int) -> Path:
    if role == "training":
        return pool_path(seed)
    return generated_path(role, seed)


def load_frames(role: str, seed: int) -> np.ndarray:
    """(FRAMES_USED, N) float64 slice used everywhere in this experiment."""

    frames = np.load(trajectory_path(role, seed), mmap_mode="r")
    return np.asarray(frames[:FRAMES_USED], dtype=np.float64)


def subset_permutation(replicate: int) -> np.ndarray:
    """Fixed recorded permutation of the pool seeds for one replicate."""

    rng = np.random.default_rng(
        np.random.SeedSequence([PERMUTATION_STREAM, int(replicate)])
    )
    return rng.permutation(np.asarray(POOL_SEEDS))


def nested_subset(replicate: int, m: int) -> list[int]:
    """First m seeds of the replicate permutation (nested in m by design)."""

    if m not in M_VALUES:
        raise ValueError(f"m must be one of {M_VALUES}")
    return [int(seed) for seed in subset_permutation(replicate)[:m]]


def model_seed(replicate: int) -> int:
    return int(
        np.random.SeedSequence([MODEL_SEED_STREAM, int(replicate)]).generate_state(1)[0]
    )


def batch_seed(replicate: int, m: int) -> int:
    return int(
        np.random.SeedSequence(
            [BATCH_SEED_STREAM, int(replicate), int(m)]
        ).generate_state(1)[0]
    )


def task_grid() -> list[tuple[int, int]]:
    """SLURM array task id -> (M, replicate); 21 tasks."""

    return [(m, r) for m in M_VALUES for r in range(REPLICATES)]


def model_dir(m: int, replicate: int) -> Path:
    return REPO_ROOT / "outputs" / "mtcurve" / "models" / f"M{m:03d}_r{replicate}"


def drift_truth_path(role: str, seed: int) -> Path:
    return REPO_ROOT / "outputs" / "mtcurve" / "drift_truth" / f"{role}_{seed}.npz"


def load_manifest() -> dict[str, Any]:
    import json

    with open(DATA_ROOT / "manifest.json", "r", encoding="utf-8") as handle:
        return json.load(handle)
