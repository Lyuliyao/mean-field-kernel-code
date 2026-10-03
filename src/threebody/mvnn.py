"""Single-species MVNN drift model (manuscript convention, small architecture).

Architecture: per-particle tanh feature MLP whose outputs are mean-pooled over
the particle axis to give the measure embedding, concatenated with the
particle state, and fed to a tanh describe MLP that outputs the scalar drift.
No handcrafted mean or moment feature is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class MVNNConfig:
    dimension: int = 1
    embedding_hidden: tuple[int, ...] = (32, 32)
    embedding_out: int = 8
    interaction_hidden: tuple[int, ...] = (64, 64)
    dtype: str = "float64"

    @property
    def feature_layers(self) -> tuple[int, ...]:
        return self.embedding_hidden + (self.embedding_out,)

    @property
    def describe_layers(self) -> tuple[int, ...]:
        return self.interaction_hidden + (self.dimension,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "embedding_hidden": list(self.embedding_hidden),
            "embedding_out": self.embedding_out,
            "interaction_hidden": list(self.interaction_hidden),
            "dtype": self.dtype,
        }


class MLP(nn.Module):
    features: Sequence[int]
    dtype: Any

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = inputs
        for index, width in enumerate(self.features):
            x = nn.Dense(width, dtype=self.dtype, param_dtype=self.dtype)(x)
            if index + 1 < len(self.features):
                x = nn.tanh(x)
        return x


class MVNN:
    """b_theta(x_i, mu) = describe([x_i, mean_j feature(x_j)])."""

    def __init__(self, config: MVNNConfig) -> None:
        self.config = config
        dtype = jnp.float64 if config.dtype == "float64" else jnp.float32
        self._dtype = dtype
        self.feature_net = MLP(config.feature_layers, dtype)
        self.describe_net = MLP(config.describe_layers, dtype)

    def init(self, key: jax.Array) -> dict[str, Mapping]:
        feature_key, describe_key = jax.random.split(key)
        sample = jnp.zeros((2, self.config.dimension), dtype=self._dtype)
        feature_params = self.feature_net.init(feature_key, sample)
        embedded = jnp.zeros(
            (2, self.config.dimension + self.config.embedding_out), dtype=self._dtype
        )
        describe_params = self.describe_net.init(describe_key, embedded)
        return {"feature": feature_params, "describe": describe_params}

    def apply(self, params: Mapping, x: jax.Array) -> jax.Array:
        """x: (..., N, dimension) -> drift (..., N, dimension)."""

        features = self.feature_net.apply(params["feature"], x)
        pooled = jnp.mean(features, axis=-2, keepdims=True)
        tiled = jnp.broadcast_to(pooled, x.shape[:-1] + (self.config.embedding_out,))
        return self.describe_net.apply(
            params["describe"], jnp.concatenate([x, tiled], axis=-1)
        )

    def drift_at(self, params: Mapping, query: jax.Array, measure: jax.Array) -> jax.Array:
        """Drift at query points (Q, dim) given measure particles (N, dim)."""

        features = self.feature_net.apply(params["feature"], measure)
        pooled = jnp.mean(features, axis=-2, keepdims=True)
        tiled = jnp.broadcast_to(pooled, query.shape[:-1] + (self.config.embedding_out,))
        return self.describe_net.apply(
            params["describe"], jnp.concatenate([query, tiled], axis=-1)
        )


def relative_l2_loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Manuscript loss: sqrt(sum((v - b)^2)) / sqrt(sum(v^2))."""

    numerator = jnp.sqrt(jnp.sum((targets - predictions) ** 2))
    denominator = jnp.sqrt(jnp.sum(targets**2))
    return numerator / jnp.maximum(denominator, 1e-30)


def rollout_sde(
    model: MVNN,
    params: Mapping,
    initial_particles: jax.Array,
    dt: float,
    n_steps: int,
    sigma: float,
    noise_key: jax.Array,
) -> jax.Array:
    """Euler-Maruyama rollout with the learned drift and the known sigma.

    initial_particles: (N,) -> returns (n_steps + 1, N).
    """

    x0 = jnp.asarray(initial_particles, dtype=self_dtype(model))[:, None]
    noise_scale = sigma * jnp.sqrt(dt)

    def step(carry, key):
        x = carry
        drift = model.apply(params, x)
        noise = jax.random.normal(key, x.shape, dtype=x.dtype)
        x_next = x + dt * drift + noise_scale * noise
        return x_next, x_next

    keys = jax.random.split(noise_key, n_steps)
    _, frames = jax.lax.scan(step, x0, keys)
    trajectory = jnp.concatenate([x0[None], frames], axis=0)
    return trajectory[:, :, 0]


def self_dtype(model: MVNN) -> jnp.dtype:
    return jnp.float64 if model.config.dtype == "float64" else jnp.float32


def parameter_count(params: Mapping) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(leaf.size for leaf in leaves))
