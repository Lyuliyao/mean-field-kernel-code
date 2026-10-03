"""Affinity-defect tests: the core discriminating identity of the experiment."""

import numpy as np

from threebody import triplets as tp
from threebody.bsplines import open_uniform_knots
from threebody.kernel_wsindy import KernelModel, gauge_fix
from threebody.observation import Grid, kde_density
from threebody.simulate import affinity_defect_truth, true_drift


def test_true_affinity_defect_is_alpha_x_d_squared(small_params):
    x = np.linspace(-4.0, 4.0, 41)
    for m0 in (0.0, 0.15, -0.2):
        for d in (0.9, 1.2, 1.3):
            defect = true_drift(x, m0, small_params) - 0.5 * (
                true_drift(x, m0 + d, small_params)
                + true_drift(x, m0 - d, small_params)
            )
            assert np.allclose(defect, affinity_defect_truth(x, d, small_params), atol=1e-12)


def test_defect_vanishes_at_x_zero(small_params):
    defect = affinity_defect_truth(np.asarray([0.0]), 1.2, small_params)
    assert defect[0] == 0.0  # why x_c must stay away from zero


def _arbitrary_kernel_model(grid: Grid) -> KernelModel:
    rng = np.random.default_rng(5)
    basis_count = 12
    return KernelModel(
        grid=grid,
        f_knots=open_uniform_knots(grid.lower, grid.upper, basis_count),
        k_knots=open_uniform_knots(-6.0, 6.0, basis_count),
        f_coefficients=rng.normal(size=basis_count),
        k_coefficients=rng.normal(size=basis_count),
        nu=0.0,
    )


def test_kde_is_linear_on_the_exact_mixture(coarse_grid):
    """rho_mixed == (rho_plus + rho_minus)/2 to roundoff for the stored triplets."""

    family = tp.canonical_family(1024, sample_seed=8)
    triplet = tp.build_triplet(family)
    rho = {m: kde_density(triplet[m], coarse_grid, 0.1) for m in tp.MEASURES}
    gap = np.max(np.abs(rho["mixed"] - 0.5 * (rho["plus"] + rho["minus"])))
    assert gap < 1e-12


def test_additive_kernel_model_has_zero_defect_on_constructed_triplet(coarse_grid):
    """Any f + K*rho model is affine in mu: zero defect on the SAME empirical
    objects every method sees (the stored particle triplets)."""

    model = _arbitrary_kernel_model(coarse_grid)
    family = tp.canonical_family(512, sample_seed=21)
    triplet = tp.build_triplet(family)
    rho = {m: kde_density(triplet[m], coarse_grid, 0.1) for m in tp.MEASURES}
    defect = model.velocity(rho["mixed"]) - 0.5 * (
        model.velocity(rho["plus"]) + model.velocity(rho["minus"])
    )
    assert np.max(np.abs(defect)) < 1e-11


def test_gauge_fix_preserves_induced_drift_and_zeroes_kernel_mean(coarse_grid):
    model = _arbitrary_kernel_model(coarse_grid)
    fixed = gauge_fix(model)
    family = tp.canonical_family(512, sample_seed=4)
    triplet = tp.build_triplet(family)
    rho = kde_density(triplet["plus"], coarse_grid, 0.1)
    # density-weighted drift must be unchanged (gauge acts on unit-mass measures)
    difference = (fixed.velocity(rho) - model.velocity(rho)) * rho
    assert np.max(np.abs(difference)) < 1e-9
    # kernel grid mean inside its knot domain is zero after the fix
    offsets = np.arange(-(coarse_grid.cells - 1), coarse_grid.cells) * coarse_grid.dx
    inside = (offsets >= fixed.k_knots[3]) & (offsets <= fixed.k_knots[-4])
    assert abs(float(fixed.kernel_values(offsets[inside]).mean())) < 1e-11
