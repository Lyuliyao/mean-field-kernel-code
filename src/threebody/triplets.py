"""Matched mixture triplet construction with exact empirical means.

A triplet family consists of three empirical measures mu_plus, mu_minus and
mu_mixed = (mu_plus + mu_minus)/2 that share exactly the same central bump
samples. Remote bump centers are solved so that the empirical means are
exactly m0 + d, m0 - d and m0 (to floating point), using antithetic bump
samples and whole-antithetic-pair halves for the mixture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

MEASURES = ("plus", "minus", "mixed")

# streams for numpy SeedSequence spawning (recorded in protocol.yaml seeds:)
TRIPLET_PARAMETER_STREAM = 101
BUMP_SAMPLING_STREAM = 202
BROWNIAN_STREAM = 303


@dataclass(frozen=True)
class TripletFamily:
    triplet_id: int
    n_particles: int
    x_c: float
    c_nominal: float
    d: float
    width: float
    m0: float
    n_central: int
    c_eff: float
    r_plus: float
    r_minus: float
    window_half_width: float
    sample_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mollifier_ppf_table(resolution: int = 200_001) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(-1.0, 1.0, resolution)
    inner = np.zeros_like(u)
    interior = np.abs(u) < 1.0
    inner[interior] = np.exp(-1.0 / (1.0 - u[interior] ** 2))
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (inner[1:] + inner[:-1]))))
    cdf /= cdf[-1]
    # strictly increasing section only (endpoints have zero density)
    return cdf, u


_PPF_CDF, _PPF_U = _mollifier_ppf_table()


def mollifier_ppf(quantiles: np.ndarray) -> np.ndarray:
    """Inverse CDF of the standard mollifier bump on (-1, 1)."""

    q = np.asarray(quantiles, dtype=np.float64)
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("quantiles must lie in [0, 1]")
    return np.interp(q, _PPF_CDF, _PPF_U)


def antithetic_bump_samples(count: int, width: float, rng: np.random.Generator) -> np.ndarray:
    """Antithetic mollifier samples with exactly zero mean.

    Samples come in pairs (w*u, -w*u); ``count`` must be even. Pairs are kept
    adjacent-block ordered: the first count//2 entries are the positive-draw
    members and the last count//2 entries their antithetic partners, so any
    prefix of whole pairs {i, i + count//2} has exactly zero mean.
    """

    if count % 2 != 0:
        raise ValueError("antithetic sampling requires an even sample count")
    half = count // 2
    u = mollifier_ppf(rng.uniform(0.5, 1.0, size=half))
    return width * np.concatenate([u, -u])


def n_central_for(c: float, n_particles: int) -> int:
    """Even central count with the remote remainder divisible by four."""

    candidate = int(round(c * n_particles / 2.0)) * 2
    for offset in (0, 2, -2, 4, -4, 6, -6):
        n_central = candidate + offset
        n_remote = n_particles - n_central
        if n_central >= 4 and n_remote >= 8 and n_remote % 4 == 0:
            return n_central
    raise ValueError("could not find a valid central particle count")


def solve_remote_centers(x_c: float, c_eff: float, d: float, m0: float) -> tuple[float, float]:
    r_plus = (m0 + d - c_eff * x_c) / (1.0 - c_eff)
    r_minus = (m0 - d - c_eff * x_c) / (1.0 - c_eff)
    return r_plus, r_minus


def make_family(
    triplet_id: int,
    n_particles: int,
    x_c: float,
    c: float,
    d: float,
    width: float,
    m0: float,
    sample_seed: int,
) -> TripletFamily:
    n_central = n_central_for(c, n_particles)
    c_eff = n_central / n_particles
    r_plus, r_minus = solve_remote_centers(x_c, c_eff, d, m0)
    gap = min(abs(r_plus - x_c), abs(r_minus - x_c))
    window_half_width = min(0.45, 0.5 * gap)
    return TripletFamily(
        triplet_id=int(triplet_id),
        n_particles=int(n_particles),
        x_c=float(x_c),
        c_nominal=float(c),
        d=float(d),
        width=float(width),
        m0=float(m0),
        n_central=int(n_central),
        c_eff=float(c_eff),
        r_plus=float(r_plus),
        r_minus=float(r_minus),
        window_half_width=float(window_half_width),
        sample_seed=int(sample_seed),
    )


def family_is_safe(family: TripletFamily, support_bound: float = 3.4) -> bool:
    w = family.width
    min_gap = 2.0 * w + 0.25
    if abs(family.r_plus - family.x_c) < min_gap:
        return False
    if abs(family.r_minus - family.x_c) < min_gap:
        return False
    extremes = (
        abs(family.x_c) + w,
        abs(family.r_plus) + w,
        abs(family.r_minus) + w,
    )
    return max(extremes) <= support_bound


def canonical_family(n_particles: int, sample_seed: int, triplet_id: int = 0) -> TripletFamily:
    family = make_family(
        triplet_id=triplet_id,
        n_particles=n_particles,
        x_c=0.6,
        c=0.4,
        d=1.2,
        width=0.12,
        m0=0.0,
        sample_seed=sample_seed,
    )
    if not family_is_safe(family):
        raise AssertionError("canonical development triplet violates safety constraints")
    return family


def sample_families(
    count: int,
    n_particles: int,
    stage_seed_base: int,
    id_offset: int = 0,
    ranges: dict[str, tuple[float, float]] | None = None,
    support_bound: float = 3.4,
    include_canonical_first: bool = False,
) -> list[TripletFamily]:
    """Sample triplet families with rejection on the safety constraints.

    ``d`` is drawn from the intersection of its protocol range with the
    separation constraint d >= x_c + (2 w + 0.25)(1 - c); parameter draws whose
    intersection is empty are rejected and redrawn.
    """

    bounds = ranges or {
        "x_c": (0.4, 0.8),
        "c": (0.35, 0.45),
        "d": (0.9, 1.3),
        "width": (0.08, 0.15),
    }
    parameter_seed = np.random.SeedSequence(
        [TRIPLET_PARAMETER_STREAM, int(stage_seed_base), int(id_offset)]
    )
    rng = np.random.default_rng(parameter_seed)
    families: list[TripletFamily] = []
    attempts = 0
    while len(families) < count:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("triplet family rejection sampling did not converge")
        triplet_id = id_offset + len(families)
        sample_seed_value = int(
            np.random.SeedSequence(
                [BUMP_SAMPLING_STREAM, int(stage_seed_base), int(triplet_id)]
            ).generate_state(1)[0]
        )
        if include_canonical_first and len(families) == 0:
            family = canonical_family(n_particles, sample_seed_value, triplet_id)
            families.append(family)
            continue
        x_c = rng.uniform(*bounds["x_c"])
        c = rng.uniform(*bounds["c"])
        width = rng.uniform(*bounds["width"])
        d_low = max(bounds["d"][0], x_c + (2.0 * width + 0.25) * (1.0 - c))
        if d_low >= bounds["d"][1]:
            continue
        d = rng.uniform(d_low, bounds["d"][1])
        family = make_family(triplet_id, n_particles, x_c, c, d, width, 0.0, sample_seed_value)
        if family_is_safe(family, support_bound):
            families.append(family)
    return families


def build_triplet(family: TripletFamily) -> dict[str, np.ndarray]:
    """Construct the three initial particle vectors of a family.

    Central samples are shared identically across all three measures. Each
    endpoint duplicates its remote base atoms (two copies per atom) while the
    mixture takes exactly one copy of every positive and negative remote atom.
    All three empirical measures therefore have N particles and satisfy the
    convex-mixture identity EXACTLY at the multiset level:

        mu_mixed = (mu_plus + mu_minus) / 2

    (the multiset union of mu_plus and mu_minus equals two copies of
    mu_mixed). Antithetic base samples make all bump means exact.
    """

    n = family.n_particles
    n_central = family.n_central
    n_remote = n - n_central
    if n_remote % 4 != 0:
        raise ValueError("remote particle count must be divisible by four")
    rng = np.random.default_rng(
        np.random.SeedSequence([BUMP_SAMPLING_STREAM, family.sample_seed])
    )
    central = family.x_c + antithetic_bump_samples(n_central, family.width, rng)
    base_plus = family.r_plus + antithetic_bump_samples(n_remote // 2, family.width, rng)
    base_minus = family.r_minus + antithetic_bump_samples(
        n_remote // 2, family.width, rng
    )

    mu_plus = np.concatenate([central, base_plus, base_plus])
    mu_minus = np.concatenate([central, base_minus, base_minus])
    mu_mixed = np.concatenate([central, base_plus, base_minus])
    assert mu_plus.size == mu_minus.size == mu_mixed.size == n
    return {"plus": mu_plus, "minus": mu_minus, "mixed": mu_mixed}


def brownian_seed(stage_seed_base: int, triplet_id: int, measure: str) -> int:
    measure_index = MEASURES.index(measure)
    return int(
        np.random.SeedSequence(
            [BROWNIAN_STREAM, int(stage_seed_base), int(triplet_id), measure_index]
        ).generate_state(1)[0]
    )
