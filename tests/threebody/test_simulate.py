"""Euler-Maruyama reproducibility and boundary-mass tests."""

import numpy as np

from threebody import triplets as tp
from threebody.simulate import boundary_mass_fraction, simulate, true_drift


def test_reproducibility(small_params):
    x0 = np.linspace(-1.0, 1.0, 64)
    a = simulate(x0, small_params, brownian_seed=42)
    b = simulate(x0, small_params, brownian_seed=42)
    c = simulate(x0, small_params, brownian_seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (small_params.n_frames, 64)


def test_matches_hand_rolled_euler_maruyama(small_params):
    x0 = np.asarray([0.3, -0.5, 1.1, 0.0])
    frames = simulate(x0, small_params, brownian_seed=5)
    rng = np.random.default_rng(5)
    x = x0.copy()
    for step in range(1, 4):
        mean = x.mean()
        x = (
            x
            + small_params.dt * true_drift(x, mean, small_params)
            + small_params.sigma * np.sqrt(small_params.dt) * rng.standard_normal(4)
        )
        assert np.allclose(frames[step], x, atol=0.0, rtol=0.0)


def test_no_artificial_clipping(small_params):
    # start particles outside the density domain: simulation must not clip them
    x0 = np.asarray([-5.0, 5.0, -6.0, 6.0])
    frames = simulate(x0, small_params, brownian_seed=1)
    assert np.max(np.abs(frames[0])) == 6.0
    assert np.all(np.isfinite(frames))


def test_boundary_mass_negligible_for_canonical_triplet(ground_truth, protocol):
    family = tp.canonical_family(512, sample_seed=17)
    triplet = tp.build_triplet(family)
    tolerance = float(protocol["ground_truth"]["boundary_mass_tolerance"])
    for measure in tp.MEASURES:
        trajectory = simulate(triplet[measure], ground_truth, brownian_seed=99)
        fraction = boundary_mass_fraction(trajectory, ground_truth.domain)
        assert fraction <= tolerance
