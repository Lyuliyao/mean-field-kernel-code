"""Full-scale alpha=0, sigma=0 sanity check (implementation validation only).

Usage:
  python scripts/threebody_sanity_alpha0.py

Runs at the protocol's sanity sizes (6 train + 2 validation triplets, N=512),
fits the B-spline kernel baseline, and verifies the acceptance gates. This
check never appears in the rebuttal figure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from threebody import triplets as tp  # noqa: E402
from threebody.artifacts import atomic_write_text  # noqa: E402
from threebody.baselines import validation_rollout_w1  # noqa: E402
from threebody.kernel_wsindy import displacement_domain, fit_kernel_wsindy  # noqa: E402
from threebody.observation import default_grid, kde_density  # noqa: E402
from threebody.protocol import load_protocol, sanity_alpha0_params  # noqa: E402
from threebody.simulate import simulate, true_drift  # noqa: E402
from threebody.weakform import WeakConfig  # noqa: E402

SANITY_SEED_BASE = 50260711


def main() -> None:
    protocol = load_protocol(ROOT / "protocol.yaml")
    params = sanity_alpha0_params(protocol)
    sanity_cfg = protocol["sanity_check_alpha0"]
    grid = default_grid(protocol)
    bandwidth = 0.05
    n_particles = int(sanity_cfg["particles"])
    times = np.arange(params.n_frames) * params.dt

    start = time.perf_counter()

    def generate(count, id_offset, base):
        trajectories = []
        for family in tp.sample_families(
            count, n_particles, base, id_offset=id_offset
        ):
            triplet = tp.build_triplet(family)
            for measure in tp.MEASURES:
                trajectories.append(
                    simulate(
                        triplet[measure],
                        params,
                        tp.brownian_seed(base, family.triplet_id, measure),
                    )
                )
        return trajectories

    train = generate(int(sanity_cfg["n_train_triplets"]), 0, SANITY_SEED_BASE)
    validation = generate(
        int(sanity_cfg["n_validation_triplets"]), 50, SANITY_SEED_BASE + 1
    )
    densities = [kde_density(t, grid, bandwidth) for t in train]
    weak_config = WeakConfig(
        time_half_width=0.05,
        space_half_width=0.3,
        time_stride=2,
        space_stride=3,
        max_windows_per_trajectory=400,
        random_seed=49157,
    )
    model, fit = fit_kernel_wsindy(
        densities,
        times,
        grid,
        displacement_domain(train),
        basis_count=12,
        nu=0.0,
        weak_config=weak_config,
        smoothness=1e-6,
    )
    w1 = validation_rollout_w1(
        model.velocity, model.nu, validation, grid, bandwidth, params.dt
    )
    relative_errors = []
    for index, trajectory in enumerate(train):
        rho = densities[index][0]
        fitted = model.velocity(rho)
        truth = true_drift(grid.centers, float(trajectory[0].mean()), params)
        support = rho > 0.05
        relative_errors.append(
            float(
                np.sqrt(np.mean((fitted[support] - truth[support]) ** 2))
                / np.sqrt(np.mean(truth[support] ** 2))
            )
        )
    drift_rel = float(np.mean(relative_errors))
    acceptance = sanity_cfg["acceptance"]
    passed = w1 <= float(acceptance["max_validation_time_avg_w1"]) and drift_rel <= float(
        acceptance["max_drift_rel_l2"]
    )
    report = {
        "alpha": 0.0,
        "sigma": 0.0,
        "bandwidth": bandwidth,
        "validation_rollout_w1": w1,
        "train_support_drift_rel_l2_mean": drift_rel,
        "weak_relative_residual": fit.relative_residual,
        "acceptance": acceptance,
        "passed": bool(passed),
        "wall_seconds": time.perf_counter() - start,
    }
    out = ROOT / "outputs" / "threebody" / "sanity_alpha0" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("alpha=0 sanity check FAILED; fix the baseline first")


if __name__ == "__main__":
    main()
