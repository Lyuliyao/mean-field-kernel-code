import jax
import jax.numpy as jnp
import sys  
import numpy as np
import matplotlib.pyplot as plt 
from jax import config as jax_config
import os
from model import create_model
jax_config.update("jax_enable_x64", True)
outdir = "data"


seed = int(sys.argv[1])
DIM = 1
TRAIN_ITERATION = 10000
BATCH_SIZE = 100
DT = 1e-2
FEATURE_NET_LAYER = [12,24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, DIM]

params = jnp.load(f"../TRAINING/model_save_path/jax_ckpt_000400.npz", allow_pickle=True)['params'].item()
init, apply_fn = create_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)




def init_opinion(N, seed=0):
    rng = np.random.default_rng(seed)
    value = rng.integers(low=0, high=3, size=1)[0]
    x = 0.5 * rng.standard_normal(N) + rng.choice([-value, 0 , +value], size=N, p=[0.3,0.4,0.3])
    return x.astype(np.float64), x.copy().astype(np.float64), rng

x, _, _ = init_opinion(16000,seed=seed)
x = x[...,None]

def step(xn):
    f = apply_fn(params, xn)
    return xn + DT *f



x_save = x[None,...]
for i in range(400):
    x = step(x)
    x_save = jnp.concatenate([x_save, x[None,...]], axis=0)




os.makedirs(outdir, exist_ok=True)
np.save(f"{outdir}/x_save_{seed}.npy", x_save)
