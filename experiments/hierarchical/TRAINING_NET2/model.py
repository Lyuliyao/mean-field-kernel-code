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
import pdb
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

    feature_net_mlp_list = [];
    describe_net_mlp_list = [];
    for i in range(len(feature_net_layer)):
        feature_net_mlp_list.append(MLP(feature_net_layer[i]))
        describe_net_mlp_list.append(MLP(describe_net_layer[i]))

    def init(key):
        params = {}
        key1, key2 = jax.random.split(key)
        batch = jnp.ones((32, dim))
        ll = dim
        for i in range(len(feature_net_layer)):
            params[f"feature_net_param{i+1}_1"] = feature_net_mlp_list[i].init(key1, batch)
            params[f"feature_net_param{i+1}_2"] = feature_net_mlp_list[i].init(key1, batch)
            params[f"feature_net_param{i+1}_3"] = feature_net_mlp_list[i].init(key1, batch)
            batch = jnp.ones((32, 3*feature_net_layer[i][-1]+ll))
            params[f"describe_net_param{i+1}_1"] = describe_net_mlp_list[i].init(key2, batch)
            params[f"describe_net_param{i+1}_2"] = describe_net_mlp_list[i].init(key2, batch)
            params[f"describe_net_param{i+1}_3"] = describe_net_mlp_list[i].init(key2, batch)
            batch = jnp.ones((32, describe_net_layer[i][-1]))
            ll = describe_net_layer[i][-1]
        return params

    displacement_vmap = None

    @jax.jit
    def fcn_f(params,x1,x2,x3):
        for i in range(len(feature_net_layer)):
            feature1 = feature_net_mlp_list[i].apply(params[f"feature_net_param{i+1}_1"], x1)
            feature2 = feature_net_mlp_list[i].apply(params[f"feature_net_param{i+1}_2"], x2)
            feature3 = feature_net_mlp_list[i].apply(params[f"feature_net_param{i+1}_3"], x3)
            mu1 =  jnp.mean(feature1,axis=-2)[...,None,:].repeat(feature1.shape[-2], axis=-2)
            mu2 =  jnp.mean(feature2,axis=-2)[...,None,:].repeat(feature2.shape[-2], axis=-2)
            mu3 =  jnp.mean(feature3,axis=-2)[...,None,:].repeat(feature3.shape[-2], axis=-2)
            x1  =  describe_net_mlp_list[i].apply(params[f"describe_net_param{i+1}_1"], jnp.concatenate([x1, mu1,mu2,mu3], axis=-1))
            x2  =  describe_net_mlp_list[i].apply(params[f"describe_net_param{i+1}_2"], jnp.concatenate([x2, mu1,mu2,mu3], axis=-1))
            x3  =  describe_net_mlp_list[i].apply(params[f"describe_net_param{i+1}_3"], jnp.concatenate([x3, mu1,mu2,mu3], axis=-1))
        return x1, x2, x3
    return init, fcn_f
