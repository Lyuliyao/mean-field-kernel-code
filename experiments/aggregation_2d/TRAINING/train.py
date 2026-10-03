"""
Optimized DeepMD code script.
"""

# Import common packages
from pathlib import Path
from typing import Sequence, Callable
import numpy as np
# Import JAX related packages
import jax
import jax.numpy as jnp
from jax import random, vmap
# Debugging packages; uncomment if needed
from jax import config
config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)
# config.update("jax_debug_nans", True)
import pdb
# Flax imports
import flax
from flax import linen as nn
from flax.training import train_state
from flax import jax_utils

# Import Optax for optimization
import optax
# import pdb
# Import numpy and other utility libraries
import numpy as onp
import functools
import glob
import sys
from model import create_model


# Network settings
DIM = 2
TRAIN_ITERATION = 10000
BATCH_SIZE = 100
DT = 1e-2
FEATURE_NET_LAYER = [12,24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, DIM]
SAVE_PATH = Path(f"model_save_path")
SAVE_PATH.mkdir(parents=True, exist_ok=True)




x_save = None
v_save = None



data_file_list = glob.glob("../DATA_GENERATION/data/simulation_*.npy")
for data_file in data_file_list:
    data = np.load(data_file)
    x = data[:-1]
    v = (data[1:]-data[:-1])/DT
    if x_save is None:
        x_save = x
        v_save = v
    else:
        x_save = np.concatenate([x_save, x], axis=0)
        v_save = np.concatenate([v_save, v], axis=0)
        
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)        
perms = jax.random.permutation(init_rng, x_save.shape[0])
 
        
TRAIN_DATA = {
    "x_save":x_save[perms[:-BATCH_SIZE]],
    "v_save":v_save[perms[:-BATCH_SIZE]]}

TEST_DATA = {
    "x_save":x_save[perms[-BATCH_SIZE:]],
    "v_save":v_save[perms[-BATCH_SIZE:]]}




# Initialize the training state
def create_train_state(rng):
    learning_rate_fn = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=100000,
        decay_rate=0.9,
        end_value =1e-7
    )

    cf_init,fcn_f = create_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)

    params = cf_init(rng)

    tx = optax.adam(learning_rate=learning_rate_fn)
    state = train_state.TrainState.create(
        apply_fn = fcn_f,
        params = params,
        tx = tx
    )

    return state



def update_model(state, grads):
    return state.apply_gradients(grads=grads)

# Define the function to apply the model
def apply_model(state,x_save_,v_save_):
    def loss_fn(params):
        # input = jnp.concatenate([fft_data_,grad_fft_data_],axis=-1)
        f_save = state.apply_fn(params,x_save_)
        loss = jnp.sqrt(jnp.sum((v_save_ - f_save)**2))/jnp.sqrt(jnp.sum(v_save_**2))
        return loss

    # Compute loss and gradients
    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = loss_grad_fn(state.params)
    
    return loss_val, grads



def train_epoch(state, train_ds, batch_size, rng):
    train_ds_size = train_ds["x_save"].shape[0]
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
    epoch_loss = []
    for perm in perms:
        batch_data = {
            f"{key}_": train_ds[key][perm]
            for key in train_ds
        }
        loss_val, grads = apply_model(state, **batch_data)
        state = update_model(state, grads)
        epoch_loss.append(loss_val)

    return state, onp.mean(epoch_loss)

def test_epoch(state, train_ds, batch_size, rng):
    train_ds_size = train_ds["x_save"].shape[0]
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
    epoch_loss = []
    for perm in perms:
        batch_data = {
            f"{key}_": train_ds[key][perm]
            for key in train_ds
        }
        loss_val, grads = apply_model(state, **batch_data)
        epoch_loss.append(loss_val)

    return onp.mean(epoch_loss)


# Initialize the random number generator


state = create_train_state(init_rng)

error_save = onp.zeros(TRAIN_ITERATION)

# Start training
for t in range(TRAIN_ITERATION):
    rng, input_rng = jax.random.split(rng)
    state, loss_val = train_epoch(state, TRAIN_DATA,BATCH_SIZE,input_rng)
    error_save[t] = loss_val
    if t % 10 == 0:
        onp.save(SAVE_PATH/ "error.npy",error_save)
        ckpt_filename = SAVE_PATH / f'jax_ckpt_{t:06d}.npz'
        params = jax.device_get(state.params)
        with open(ckpt_filename, 'wb') as f:
            jnp.savez(f,t=t,params=params)
        error_val = test_epoch(state, TEST_DATA,BATCH_SIZE,input_rng)   
        print('step: {}, loss: {}, error: {}'.format(state.step,loss_val,error_val))
