"""
Optimized DeepMD code script with multi-GPU support.
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
jax.config.update("jax_enable_x64", True)
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
from model_second_order import create_model


# Network settings
DIM = 2
TRAIN_ITERATION = 10000
BATCH_SIZE = 100
DT = 1e-2
FEATURE_NET_LAYER = [12, 24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, DIM]
SAVE_PATH = Path(f"model_save_path")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Multi-GPU setup
num_devices = jax.device_count()
print(f"Number of devices available: {num_devices}")

# Ensure batch size is divisible by number of devices
assert BATCH_SIZE % num_devices == 0, f"BATCH_SIZE ({BATCH_SIZE}) must be divisible by num_devices ({num_devices})"
PER_DEVICE_BATCH_SIZE = BATCH_SIZE // num_devices


x_save = None
v_save = None
a_save = None


data_file_list = glob.glob("../data_generation/data/simulation_*.npz")
for data_file in data_file_list:
    data = np.load(data_file)
    x = data["x"][:-1]   # positions
    v = data["v"][:-1]   # velocities
    a = (data["v"][1:] - data["v"][:-1]) / DT
    if x_save is None:
        x_save = x
        v_save = v
        a_save = a
    else:
        x_save = np.concatenate([x_save, x], axis=0)
        v_save = np.concatenate([v_save, v], axis=0)
        a_save = np.concatenate([a_save, a], axis=0)

        
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)        
perms = jax.random.permutation(init_rng, x_save.shape[0])

        
TRAIN_DATA = {
    "x_save": x_save[perms[:-BATCH_SIZE]],
    "v_save": v_save[perms[:-BATCH_SIZE]],
    "a_save": a_save[perms[:-BATCH_SIZE]]}

TEST_DATA = {
    "x_save": x_save[perms[-BATCH_SIZE:]],
    "v_save": v_save[perms[-BATCH_SIZE:]],
    "a_save": a_save[perms[-BATCH_SIZE:]]}


train_ds_size = TRAIN_DATA["x_save"].shape[0]
steps_per_epoch = train_ds_size // BATCH_SIZE
total_state_steps = TRAIN_ITERATION * steps_per_epoch
print(f"Total expected state.step at the end of training: {total_state_steps}")


# Initialize the training state
def create_train_state(rng):
    learning_rate_fn = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=100000,
        decay_rate=0.9,
        end_value=1e-7
    )

    cf_init, fcn_f = create_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)

    params = cf_init(rng)

    tx = optax.adam(learning_rate=learning_rate_fn)
    state = train_state.TrainState.create(
        apply_fn=fcn_f,
        params=params,
        tx=tx
    )

    return state


def update_model(state, grads):
    return state.apply_gradients(grads=grads)


# Define the function to apply the model
def apply_model(state, x_save_, v_save_, a_save_):
    def loss_fn(params):
        f_save = state.apply_fn(params, x_save_, v_save_)
        loss = jnp.sqrt(jnp.sum((a_save_ - f_save)**2)) / jnp.sqrt(jnp.sum(a_save_**2))
        return loss

    # Compute loss and gradients
    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = loss_grad_fn(state.params)
    
    return loss_val, grads


# Multi-GPU training step (parallelized across devices)
@functools.partial(jax.pmap, axis_name='devices')
def train_step(state, x_batch, v_batch, a_batch):
    """Single training step parallelized across devices."""
    loss_val, grads = apply_model(state, x_batch, v_batch, a_batch)
    
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='devices')
    # Update model
    state = update_model(state, grads)
    # Average loss across devices for reporting
    loss_val = jax.lax.pmean(loss_val, axis_name='devices')
    
    return state, loss_val


# Multi-GPU evaluation step
@functools.partial(jax.pmap, axis_name='devices')
def eval_step(state, x_batch, v_batch, a_batch):
    """Single evaluation step parallelized across devices."""
    loss_val, _ = apply_model(state, x_batch, v_batch, a_batch)

    # Average loss across devices
    loss_val = jax.lax.pmean(loss_val, axis_name='devices')
    
    return loss_val


def train_epoch(state, train_ds, batch_size, rng):
    """Train for one epoch with multi-GPU support."""
    train_ds_size = train_ds["x_save"].shape[0]
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
    epoch_loss = []
    
    for perm in perms:
        # Get batch data
        x_batch = train_ds["x_save"][perm]
        v_batch = train_ds["v_save"][perm]
        a_batch = train_ds["a_save"][perm]
        
        # Reshape for multi-GPU: (num_devices, per_device_batch_size, ...)
        x_batch = x_batch.reshape((num_devices, PER_DEVICE_BATCH_SIZE) + x_batch.shape[1:])
        v_batch = v_batch.reshape((num_devices, PER_DEVICE_BATCH_SIZE) + v_batch.shape[1:])
        a_batch = a_batch.reshape((num_devices, PER_DEVICE_BATCH_SIZE) + a_batch.shape[1:])
        
        # Run parallel training step
        state, loss_val = train_step(state, x_batch, v_batch, a_batch)
        
        # loss_val is replicated across devices, take first value
        epoch_loss.append(loss_val[0])

    return state, onp.mean(epoch_loss)


def test_epoch(state, test_ds, batch_size, rng):
    """Evaluate for one epoch with multi-GPU support."""
    test_ds_size = test_ds["x_save"].shape[0]
    steps_per_epoch = test_ds_size // batch_size
    perms = jax.random.permutation(rng, test_ds_size)
    perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
    epoch_loss = []
    
    for perm in perms:
        # Get batch data
        x_batch = test_ds["x_save"][perm]
        v_batch = test_ds["v_save"][perm]
        a_batch = test_ds["a_save"][perm]
        
        # Reshape for multi-GPU: (num_devices, per_device_batch_size, ...)
        x_batch = x_batch.reshape((num_devices, PER_DEVICE_BATCH_SIZE) + x_batch.shape[1:])
        v_batch = v_batch.reshape((num_devices, PER_DEVICE_BATCH_SIZE) + v_batch.shape[1:])
        a_batch = a_batch.reshape((num_devices, PER_DEVICE_BATCH_SIZE) + a_batch.shape[1:])
        
        # Run parallel evaluation step
        loss_val = eval_step(state, x_batch, v_batch, a_batch)
        
        # loss_val is replicated across devices, take first value
        epoch_loss.append(loss_val[0])

    return onp.mean(epoch_loss)


# Initialize the random number generator and create initial state
state = create_train_state(init_rng)

# Replicate state across devices
state = jax_utils.replicate(state)

error_save = onp.zeros(TRAIN_ITERATION)

# Start training
for t in range(TRAIN_ITERATION):
    rng, input_rng = jax.random.split(rng)
    state, loss_val = train_epoch(state, TRAIN_DATA, BATCH_SIZE, input_rng)
    error_save[t] = loss_val
    
    if t % 10 == 0:
        onp.save(SAVE_PATH / "error.npy", error_save)
        error_val = test_epoch(state, TEST_DATA, BATCH_SIZE, input_rng)   
        # Unreplicate state for saving (take from first device)
        state_cpu = jax_utils.unreplicate(state)
        params = jax.device_get(state_cpu.params)
        ckpt_filename = SAVE_PATH / f'jax_ckpt_{t:06d}.npz'
        with open(ckpt_filename, 'wb') as f:
            jnp.savez(f, t=t, params=params)
        print('step: {}, loss: {}, error: {}'.format(state_cpu.step,loss_val,error_val))




# Save final model
# state_cpu = jax_utils.unreplicate(state)
# params = jax.device_get(state_cpu.params)
# final_ckpt = SAVE_PATH / 'jax_ckpt_final.npz'
# with open(final_ckpt, 'wb') as f:
#     jnp.savez(f, t=TRAIN_ITERATION-1, params=params)
# print('step: {}'.format(state_cpu.step))