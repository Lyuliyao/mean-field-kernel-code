"""Exact Motsch-Tadmor drift on reference states (JAX, chunked, float64)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .simulate import DEN_EPS, ELL

jax.config.update("jax_enable_x64", True)

CHUNK = 2000


def _chunk_influence(x_chunk: jnp.ndarray, x_all: jnp.ndarray) -> jnp.ndarray:
    diff = x_all[None, :] - x_chunk[:, None]
    w = jnp.exp(-((diff / ELL) ** 2))
    num = (w * diff).sum(axis=1)  # self term contributes w_ii * 0 = 0
    den = w.sum(axis=1) - 1.0  # remove self weight w_ii = exp(0) = 1
    return num / (den + DEN_EPS)


_chunk_influence_jit = jax.jit(_chunk_influence)


def true_drift(states: np.ndarray) -> np.ndarray:
    """b_true for one frame (N,) with self-terms removed, matching the generator."""

    x = jnp.asarray(states, dtype=jnp.float64)
    n = x.shape[0]
    pieces = []
    for start in range(0, n, CHUNK):
        pieces.append(np.asarray(_chunk_influence_jit(x[start : start + CHUNK], x)))
    return np.concatenate(pieces)


def true_drift_frames(frames: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    """(len(frame_indices), N) exact drift at the requested reference frames."""

    return np.stack([true_drift(frames[index]) for index in frame_indices])
