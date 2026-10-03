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
N1, N2, N3 = 4000, 4000, 4000          # workers, managers, CEOs
TRAIN_ITERATION = 10000
BATCH_SIZE = 100
DT = 1e-3
FEATURE_NET_LAYER = [12,24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, DIM]

params = jnp.load(f"../TRAINING3/model_save_path/jax_ckpt_000100.npz", allow_pickle=True)['params'].item()
init, apply_fn = create_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)



def init_opinion(N, seed=0):
    rng = np.random.default_rng(seed)
    num_gaussian = rng.integers(low=2, high=9, size=1)[0]
    value = rng.uniform(low=-5, high=5, size=num_gaussian)
    p = rng.dirichlet(np.ones(num_gaussian), size=1)[0]
    x = 2*rng.standard_normal(N) + rng.choice(value, size=N, p=p)
    x = x - np.mean(x, axis=0)
    return x.astype(np.float64)

x1 = init_opinion(N1, seed=seed)
x2 = init_opinion(N2, seed=seed+1)
x3 = init_opinion(N3, seed=seed+2)
x1 = x1[...,None]
x2 = x2[...,None]
x3 = x3[...,None]
def step(x1,x2,x3):
    f1,f2,f3 = apply_fn(params, x1,x2,x3)
    return x1 + DT *f1,x2 + DT *f2,x3 + DT *f3



x1_save = x1[None,...]
x2_save = x2[None,...]
x3_save = x3[None,...]
for i in range(500):
    x1,x2,x3 = step(x1,x2,x3)
    x1_save = jnp.concatenate([x1_save, x1[None,...]], axis=0)
    x2_save = jnp.concatenate([x2_save, x2[None,...]], axis=0)
    x3_save = jnp.concatenate([x3_save, x3[None,...]], axis=0)




os.makedirs(outdir, exist_ok=True)
for x_save, name in zip([x1_save, x2_save, x3_save], ["opinion_workers_mpi", "opinion_managers_mpi", "opinion_ceos_mpi"]):
    np.save(f"{outdir}/{name}_{seed}.npy", x_save)
