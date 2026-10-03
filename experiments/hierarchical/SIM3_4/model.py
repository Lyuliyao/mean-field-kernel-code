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
        params["feature_net_param1"] = feature_net_mlp.init(key1, batch)
        params["feature_net_param2"] = feature_net_mlp.init(key1, batch)
        params["feature_net_param3"] = feature_net_mlp.init(key1, batch)
        batch = jnp.ones((32, 3*feature_net_layer[-1]+dim))
        params["describe_net_param1"] = describe_net_mlp.init(key2, batch)
        params["describe_net_param2"] = describe_net_mlp.init(key2, batch)
        params["describe_net_param3"] = describe_net_mlp.init(key2, batch)
        return params

    displacement_vmap = None

    @jax.jit
    def fcn_f(params,x1,x2,x3):
        feature1 = feature_net_mlp.apply(params["feature_net_param1"], x1)
        feature2 = feature_net_mlp.apply(params["feature_net_param2"], x2)
        feature3 = feature_net_mlp.apply(params["feature_net_param3"], x3)
        mu1 =  jnp.mean(feature1,axis=-2)[...,None,:]#.repeat(feature1.shape[-2], axis=-2)
        mu2 =  jnp.mean(feature2,axis=-2)[...,None,:]#.repeat(feature2.shape[-2], axis=-2)
        mu3 =  jnp.mean(feature3,axis=-2)[...,None,:]#.repeat(feature3.shape[-2], axis=-2)
        force1 = describe_net_mlp.apply(params["describe_net_param1"], jnp.concatenate([x1,
                                                                                        mu1.repeat(x1.shape[-2], axis=-2),
                                                                                        mu2.repeat(x1.shape[-2], axis=-2),
                                                                                        mu3.repeat(x1.shape[-2], axis=-2)], axis=-1))
        force2 = describe_net_mlp.apply(params["describe_net_param2"], jnp.concatenate([x2,
                                                                                        mu1.repeat(x2.shape[-2], axis=-2),
                                                                                        mu2.repeat(x2.shape[-2], axis=-2),
                                                                                        mu3.repeat(x2.shape[-2], axis=-2)], axis=-1))
        force3 = describe_net_mlp.apply(params["describe_net_param3"], jnp.concatenate([x3, 
                                                                                        mu1.repeat(x3.shape[-2], axis=-2),
                                                                                        mu2.repeat(x3.shape[-2], axis=-2),
                                                                                        mu3.repeat(x3.shape[-2], axis=-2)], axis=-1))
        return force1, force2, force3
    return init, fcn_f
