from typing import Callable, MutableMapping, Optional,Tuple
import jax
import jax.numpy as jnp
from jax import random
import chex
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Sequence, Optional, Callable
import logging
import numpy as np
import math
class MLP(nn.Module):
    """
    Simple MLP (Multilayer Perceptron) function copied from Flax
    """
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = jnp.tanh(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

def create_model(feature_net_layer, describe_net_layer,dim):
    """
    Function to create the model
    """

    feature_net_mlp = MLP(feature_net_layer)
    describe_net_mlp = MLP(describe_net_layer)

    def init(key):
        params = {}
        key1, key2 = jax.random.split(key)
        batch = jnp.ones((32, dim))
        params["feature_net_param"] = feature_net_mlp.init(key1, batch)
        batch = jnp.ones((32, feature_net_layer[-1]+2*dim))
        params["describe_net_param"] = describe_net_mlp.init(key2, batch)
        return params

    displacement_vmap = None

    @jax.jit
    def fcn_f(params,x, v):
        feature = feature_net_mlp.apply(params["feature_net_param"], x)
        mu =  jnp.mean(feature,axis=-2)[...,None,:].repeat(feature.shape[-2], axis=-2)
        force = describe_net_mlp.apply(params["describe_net_param"], jnp.concatenate([x, v, mu], axis=-1))
        return force
    return init, fcn_f
