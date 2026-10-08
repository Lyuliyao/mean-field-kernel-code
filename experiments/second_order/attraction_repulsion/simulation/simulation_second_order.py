import jax
import jax.numpy as jnp
import sys  
import numpy as np
import matplotlib.pyplot as plt 
import os
path = "../training"
from model_second_order import create_model
jax.config.update("jax_enable_x64", True)
outdir = "data"

DIM = 2
TRAIN_ITERATION = 10000
BATCH_SIZE = 100
DT = 1e-2
FEATURE_NET_LAYER = [12,24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, DIM]

params = jnp.load(f"../training/model_save_path/jax_ckpt_001000.npz", allow_pickle=True)['params'].item()
_, apply_fn = create_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)
seed = int(sys.argv[1]) 


def ic_ring(N, R=1.0, thickness=0.05, seed=0):
    rng=np.random.default_rng(seed)
    angles=rng.uniform(0,2*np.pi,size=N)
    radii=R + rng.normal(0, thickness, size=N)
    x = np.stack([radii*np.cos(angles), radii*np.sin(angles)],1)
    return x.astype(np.float64)

def step(xn,vn):
    f = apply_fn(params, xn, vn)
    v = vn + DT*f
    x = xn + DT*vn
    return x, v




x = jnp.array(ic_ring(16000,seed=seed))
v = jnp.zeros_like(x)

x_save = x[None,...]
v_save = v[None,...]
for i in range(200):
    x, v = step(x,v)
    x_save = jnp.concatenate([x_save, x[None,...]], axis=0)
    v_save = jnp.concatenate([v_save, v[None,...]], axis=0)


os.makedirs(outdir, exist_ok=True)
np.save(f"{outdir}/x_save_second_order_{seed}.npy", x_save)
np.save(f"{outdir}/v_save_second_order_{seed}.npy", v_save)