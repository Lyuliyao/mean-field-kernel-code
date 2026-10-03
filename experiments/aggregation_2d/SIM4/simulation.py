import jax
import jax.numpy as jnp
import sys  
import numpy as np
import matplotlib.pyplot as plt 
from jax import config as jax_config
import os
path = "../TRAINING"
from model import create_model
jax_config.update("jax_enable_x64", True)
outdir = "data"

DIM = 2
TRAIN_ITERATION = 10000
BATCH_SIZE = 100
DT = 1e-2
FEATURE_NET_LAYER = [12,24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, DIM]

params = jnp.load(f"../TRAINING/model_save_path/jax_ckpt_001000.npz", allow_pickle=True)['params'].item()
_, apply_fn = create_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)
seed = int(sys.argv[1]) 


def ic_density_step(N, d=2, box=2.0, bias=0.7, seed=0):
    # bias 越大，左半区越密
    rng=np.random.default_rng(seed)
    nL=int(N*bias); nR=N-nL
    xL=np.c_[rng.uniform(-box,0,(nL,1)), rng.uniform(-box,box,(nL,1))]
    xR=np.c_[rng.uniform(0, box,(nR,1)), rng.uniform(-box,box,(nR,1))]
    x = np.vstack([xL,xR]).astype(np.float64)
    x = x - np.mean(x, axis=0)
    return x

def step(xn):
    f = apply_fn(params, xn)
    return xn + DT *f



x = jnp.array(ic_density_step(16000,seed=seed))

x_save = x[None,...]
for i in range(200):
    x = step(x)
    x_save = jnp.concatenate([x_save, x[None,...]], axis=0)


os.makedirs(outdir, exist_ok=True)
np.save(f"{outdir}/x_save_{seed}.npy", x_save)