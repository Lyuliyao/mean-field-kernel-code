"""alpha = 0 sanity behavior: the additive kernel model must fit additive truth.

With alpha = 0 and sigma = 0 the true drift IS of the additive form
    b(x, mu) = (a x - beta x^3) + INT gamma (y - x) mu(dy),
so the B-spline kernel WSINDy baseline must produce accurate held-out
forecasts (protocol sanity_check_alpha0). The primary gate is autonomous
held-out rollout W1; the drift is additionally checked on the training
support (f_B is legitimately unconstrained where no training mass ever
lives, so arbitrary-support pointwise drift errors are not a valid gate).
"""

import numpy as np
import pytest

from threebody import triplets as tp
from threebody.baselines import validation_rollout_w1
from threebody.kernel_wsindy import displacement_domain, fit_kernel_wsindy
from threebody.observation import Grid, kde_density
from threebody.protocol import GroundTruth
from threebody.simulate import simulate, true_drift
from threebody.weakform import WeakConfig


@pytest.mark.slow
def test_alpha0_kernel_recovery():
    params = GroundTruth(
        a=1.0,
        alpha=0.0,
        beta=0.2,
        gamma=0.2,
        sigma=0.0,
        dt=0.01,
        n_frames=101,
        domain=(-4.0, 4.0),
    )
    grid = Grid(-4.0, 4.0, 512)
    bandwidth = 0.05
    times = np.arange(params.n_frames) * params.dt

    def generate(families, seed_base):
        trajectories = []
        for family in families:
            triplet = tp.build_triplet(family)
            for measure in tp.MEASURES:
                trajectories.append(
                    simulate(
                        triplet[measure],
                        params,
                        tp.brownian_seed(seed_base, family.triplet_id, measure),
                    )
                )
        return trajectories

    train_trajectories = generate(tp.sample_families(4, 384, stage_seed_base=777), 777)
    held_out = generate(tp.sample_families(2, 384, stage_seed_base=778, id_offset=20), 778)

    densities = [kde_density(t, grid, bandwidth) for t in train_trajectories]
    weak_config = WeakConfig(
        time_half_width=0.05,
        space_half_width=0.3,
        time_stride=2,
        space_stride=3,
        max_windows_per_trajectory=400,
    )
    model, fit = fit_kernel_wsindy(
        densities,
        times,
        grid,
        displacement_domain(train_trajectories),
        basis_count=12,
        nu=0.0,
        weak_config=weak_config,
        smoothness=1e-6,
    )

    # primary gate: accurate held-out autonomous forecasts
    w1 = validation_rollout_w1(
        model.velocity, model.nu, held_out, grid, bandwidth, params.dt
    )
    assert w1 < 0.02, f"alpha=0 held-out rollout W1 {w1:.4f} exceeds 0.02"

    # secondary: drift accuracy on the training measures' own support
    relative_errors = []
    for index, trajectory in enumerate(train_trajectories):
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
    mean_rel = float(np.mean(relative_errors))
    assert mean_rel < 0.12, f"alpha=0 training-support drift rel L2 {mean_rel:.3f}"
