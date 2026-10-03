"""Empirical mixture construction and exact-mean tests."""

import numpy as np
import pytest

from threebody import triplets as tp


def test_canonical_family_matches_protocol_values():
    family = tp.canonical_family(1024, sample_seed=7)
    assert family.x_c == 0.6
    assert family.d == 1.2
    assert family.m0 == 0.0
    # canonical remote centers approximately 1.6 and -2.4 (exact for c=0.4)
    assert abs(family.r_plus - 1.6) < 0.02
    assert abs(family.r_minus + 2.4) < 0.06


@pytest.mark.parametrize("n_particles", [512, 1024])
def test_exact_empirical_means(n_particles):
    family = tp.canonical_family(n_particles, sample_seed=3)
    triplet = tp.build_triplet(family)
    for measure, expected in (("plus", family.d), ("minus", -family.d), ("mixed", 0.0)):
        assert triplet[measure].size == n_particles
        assert abs(triplet[measure].mean() - expected) < 1e-12


def test_shared_central_samples_and_duplicated_remote_atoms():
    family = tp.canonical_family(512, sample_seed=11)
    triplet = tp.build_triplet(family)
    n_c = family.n_central
    n_r = family.n_particles - n_c
    central = triplet["plus"][:n_c]
    # identical central block in all three measures
    assert np.array_equal(central, triplet["minus"][:n_c])
    assert np.array_equal(central, triplet["mixed"][:n_c])
    # endpoints duplicate their remote base atoms (two identical copies)
    base_plus = triplet["plus"][n_c : n_c + n_r // 2]
    base_minus = triplet["minus"][n_c : n_c + n_r // 2]
    assert np.array_equal(base_plus, triplet["plus"][n_c + n_r // 2 :])
    assert np.array_equal(base_minus, triplet["minus"][n_c + n_r // 2 :])
    # the mixture takes exactly one copy of every positive and negative atom
    assert np.array_equal(base_plus, triplet["mixed"][n_c : n_c + n_r // 2])
    assert np.array_equal(base_minus, triplet["mixed"][n_c + n_r // 2 :])
    # exact bump-center means for the base blocks
    assert abs(base_plus.mean() - family.r_plus) < 1e-12
    assert abs(base_minus.mean() - family.r_minus) < 1e-12


def test_exact_convex_mixture_identity_at_multiset_level():
    """multiset(mu_plus) + multiset(mu_minus) == 2 x multiset(mu_mixed)."""

    for n_particles, seed in ((512, 11), (1024, 3)):
        family = tp.canonical_family(n_particles, sample_seed=seed)
        triplet = tp.build_triplet(family)
        endpoints = np.sort(np.concatenate([triplet["plus"], triplet["minus"]]))
        doubled_mixture = np.sort(np.concatenate([triplet["mixed"], triplet["mixed"]]))
        assert np.array_equal(endpoints, doubled_mixture)


def test_family_sampling_is_deterministic_and_safe():
    a = tp.sample_families(6, 512, stage_seed_base=123)
    b = tp.sample_families(6, 512, stage_seed_base=123)
    assert [fam.to_dict() for fam in a] == [fam.to_dict() for fam in b]
    for family in a:
        assert tp.family_is_safe(family)
        assert 0.4 <= family.x_c <= 0.8
        assert 0.9 <= family.d <= 1.3
        assert 0.08 <= family.width <= 0.15
        triplet = tp.build_triplet(family)
        for measure in tp.MEASURES:
            assert np.max(np.abs(triplet[measure])) <= 3.4


def test_initial_support_inside_domain():
    for family in tp.sample_families(4, 1024, stage_seed_base=99, id_offset=50):
        triplet = tp.build_triplet(family)
        for values in triplet.values():
            assert values.min() > -4.0 and values.max() < 4.0
