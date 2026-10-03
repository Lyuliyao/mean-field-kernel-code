"""Verbatim single-process port of CASE5/DATA_GENERATION3/simulation.py.

The blocked influence computation, self-term removal, denominator epsilon,
initial-condition sampler, and explicit Euler update replicate the original
MPI generator exactly (block sizes included), so regenerating a training-pool
seed reproduces the stored trajectory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

N_AGENTS = 16000
DT = 1e-2
ELL = 0.5
BI = 400
BJ = 400
DEN_EPS = 1e-12


def init_opinion(n: int, seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Original sampler plus a record of the drawn IC parameters."""

    rng = np.random.default_rng(seed)
    num_gaussian = rng.integers(low=2, high=9, size=1)[0]
    value = rng.uniform(low=0, high=3, size=num_gaussian)
    p = rng.dirichlet(np.ones(num_gaussian), size=1)[0]
    x = 0.5 * rng.standard_normal(n) + rng.choice(value, size=n, p=p)
    x = x - np.mean(x, axis=0)
    parameters = {
        "num_gaussian": int(num_gaussian),
        "means": [float(v) for v in value],
        "weights": [float(w) for w in p],
        "component_std": 0.5,
        "mean_centered": True,
    }
    return x.astype(np.float64), parameters


def phi(r: np.ndarray, ell: float = ELL) -> np.ndarray:
    return np.exp(-((r / ell) ** 2))


def blocked_influence(x: np.ndarray, bi: int = BI, bj: int = BJ) -> np.ndarray:
    """g_i = [sum_j w_ij (x_j - x_i)] / [sum_j w_ij + eps], self-terms removed."""

    n = x.shape[0]
    num = np.zeros(n, dtype=x.dtype)
    den = np.zeros(n, dtype=x.dtype)
    ib = 0
    while ib < n:
        i1 = min(ib + bi, n)
        xi = x[ib:i1][:, None]
        num_b = np.zeros((i1 - ib,), dtype=x.dtype)
        den_b = np.zeros((i1 - ib,), dtype=x.dtype)
        jb = 0
        while jb < n:
            j1 = min(jb + bj, n)
            xj = x[jb:j1][None, :]
            diff = xj - xi
            w = phi(diff)
            if (ib < j1) and (jb < i1):
                mself = min(i1 - max(ib, jb), j1 - max(ib, jb))
                a0 = max(ib, jb) - ib
                b0 = max(ib, jb) - jb
                w[np.arange(a0, a0 + mself), np.arange(b0, b0 + mself)] = 0.0
            sum_w = w.sum(axis=1)
            num_b += (w @ x[jb:j1]) - (sum_w * x[ib:i1])
            den_b += sum_w
            jb = j1
        num[ib:i1] = num_b
        den[ib:i1] = den_b + DEN_EPS
        ib = i1
    return num / den


def simulate(seed: int, steps: int, n: int = N_AGENTS) -> tuple[np.ndarray, dict[str, Any]]:
    """Explicit Euler trajectory (steps + 1, n) from the seeded initial condition."""

    x, parameters = init_opinion(n, seed=seed)
    frames = np.empty((steps + 1, n), dtype=np.float64)
    frames[0] = x
    for t in range(1, steps + 1):
        x = x + DT * blocked_influence(x)
        frames[t] = x
    return frames, parameters
