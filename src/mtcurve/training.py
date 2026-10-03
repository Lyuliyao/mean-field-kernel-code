"""Training and evaluation of one learning-curve model (one M, one replicate).

Manuscript conventions throughout: MVNN with feature layers [12, 24, 48] and
describe layers [128, 128, 128, 1], batch relative-L2 increment loss, Adam
with exponential decay (1e-3, 100000 steps, rate 0.9, end 1e-7), batches of
100 full-population snapshots. Early stopping uses ONLY the 10 fixed
validation trajectories; the 30 test trajectories are opened exclusively by
the final autonomous-rollout evaluation, which applies the identical protocol
to training, validation, and test trajectories.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from threebody.artifacts import atomic_write_text, sha256_file
from threebody.metrics import w1_samples
from threebody.mvnn import MVNN, MVNNConfig, relative_l2_loss
from threebody.observation import Grid, kde_density
from threebody.training import save_params

from . import data as mtdata

jax.config.update("jax_enable_x64", True)

DT = 1e-2
BATCH_SNAPSHOTS = 100
MAX_STEPS = 30000
VAL_EVERY = 250
PATIENCE_EVALS = 8
MIN_RELATIVE_IMPROVEMENT = 0.001
METRIC_FRAME_STRIDE = 10
GRID = Grid(-5.0, 5.0, 640)
KDE_BANDWIDTH = 0.1
DENSITY_EPS = 1e-8
ROLLOUT_BATCH = 32

MODEL_CONFIG = MVNNConfig(
    dimension=1,
    embedding_hidden=(12, 24),
    embedding_out=48,
    interaction_hidden=(128, 128, 128),
    dtype="float64",
)

PROTOCOL_FINGERPRINT = {
    "dt": DT,
    "rollout_steps": mtdata.FRAMES_USED - 1,
    "metric_frame_stride": METRIC_FRAME_STRIDE,
    "grid": [GRID.lower, GRID.upper, GRID.cells],
    "kde_bandwidth": KDE_BANDWIDTH,
    "density_eps": DENSITY_EPS,
    "quantile_count": 4096,
    "drift_frames": "stride 10, t in [0.1, 2.0]",
}


def _increments(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = frames[:-1]
    v = (frames[1:] - frames[:-1]) / DT
    return x, v


def _stack_role(role: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    xs, vs = [], []
    for seed in seeds:
        x, v = _increments(mtdata.load_frames(role, seed))
        xs.append(x)
        vs.append(v)
    return np.concatenate(xs, axis=0), np.concatenate(vs, axis=0)


def train_and_evaluate(
    m: int,
    replicate: int,
    *,
    max_steps: int = MAX_STEPS,
    val_every: int = VAL_EVERY,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or mtdata.model_dir(m, replicate)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "training_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"{summary_path} exists; refusing to retrain")

    manifest = mtdata.load_manifest()
    test_rows = [row for row in manifest["trajectories"] if row["role"] == "test"]
    test_manifest_sha = {row["seed"]: row["sha256"] for row in test_rows}

    subset = mtdata.nested_subset(replicate, m)
    train_x, train_v = _stack_role("training", subset)
    val_x, val_v = _stack_role("validation", list(mtdata.VALIDATION_SEEDS))

    device = jax.devices()[0]
    train_x_d = jax.device_put(jnp.asarray(train_x)[..., None], device)
    train_v_d = jax.device_put(jnp.asarray(train_v)[..., None], device)
    val_x_d = jax.device_put(jnp.asarray(val_x)[..., None], device)
    val_v_d = jax.device_put(jnp.asarray(val_v)[..., None], device)
    del train_x, train_v, val_x, val_v

    model = MVNN(MODEL_CONFIG)
    params = model.init(jax.random.PRNGKey(mtdata.model_seed(replicate)))
    schedule = optax.exponential_decay(
        init_value=1e-3, transition_steps=100000, decay_rate=0.9, end_value=1e-7
    )
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, indices):
        xb = jnp.take(train_x_d, indices, axis=0)
        vb = jnp.take(train_v_d, indices, axis=0)

        def loss_fn(p):
            return relative_l2_loss(model.apply(p, xb), vb)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    @jax.jit
    def val_sums(params, xb, vb):
        f = model.apply(params, xb)
        return jnp.sum((vb - f) ** 2), jnp.sum(vb**2)

    def validation_loss(params) -> float:
        num = 0.0
        den = 0.0
        chunk = 200
        for start in range(0, val_x_d.shape[0], chunk):
            n_part, d_part = val_sums(
                params, val_x_d[start : start + chunk], val_v_d[start : start + chunk]
            )
            num += float(n_part)
            den += float(d_part)
        return float(np.sqrt(num) / np.sqrt(den))

    rng = np.random.default_rng(mtdata.batch_seed(replicate, m))
    n_snapshots = int(train_x_d.shape[0])

    best_val = np.inf
    best_step = 0
    best_params_host = jax.device_get(params)
    evals_since_improvement = 0
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()
    stopping_step = max_steps

    for step in range(1, max_steps + 1):
        indices = jnp.asarray(rng.integers(0, n_snapshots, size=BATCH_SNAPSHOTS))
        params, opt_state, loss = update(params, opt_state, indices)
        if step % val_every == 0:
            loss_value = float(loss)
            if not np.isfinite(loss_value):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            val_loss = validation_loss(params)
            history.append(
                {"step": step, "train_loss": loss_value, "val_loss": val_loss}
            )
            if val_loss < best_val * (1.0 - MIN_RELATIVE_IMPROVEMENT):
                best_val = val_loss
                best_step = step
                best_params_host = jax.device_get(params)
                evals_since_improvement = 0
            else:
                if val_loss < best_val:
                    best_val = val_loss
                    best_step = step
                    best_params_host = jax.device_get(params)
                evals_since_improvement += 1
            if evals_since_improvement >= PATIENCE_EVALS:
                stopping_step = step
                break

    training_walltime = time.perf_counter() - start_time
    params = jax.device_put(best_params_host)  # restore best checkpoint
    save_params(output_dir / "best_params.msgpack", best_params_host)
    del train_x_d, train_v_d

    # ---- evaluation: identical autonomous-rollout protocol for all roles ----
    n_steps = mtdata.FRAMES_USED - 1
    n_segments = n_steps // METRIC_FRAME_STRIDE

    @jax.jit
    def rollout(params, x0):
        def outer(state, _):
            def inner(s, __):
                return s + DT * model.apply(params, s), None

            state, _ = jax.lax.scan(inner, state, None, length=METRIC_FRAME_STRIDE)
            return state, state

        _, frames = jax.lax.scan(outer, x0, None, length=n_segments)
        return frames  # (n_segments, B, N, 1)

    @jax.jit
    def drift_at(params, states):
        return model.apply(params, states)

    metric_frames = np.arange(0, mtdata.FRAMES_USED, METRIC_FRAME_STRIDE)  # 0..200
    times = metric_frames * DT
    quantiles = 4096
    eval_start = time.perf_counter()

    rows: list[dict[str, Any]] = []
    role_sets = (
        ("training", subset),
        ("validation", list(mtdata.VALIDATION_SEEDS)),
        ("test", list(mtdata.TEST_SEEDS)),
    )
    for role, seeds in role_sets:
        for batch_start in range(0, len(seeds), ROLLOUT_BATCH):
            batch_seeds = seeds[batch_start : batch_start + ROLLOUT_BATCH]
            reference = np.stack(
                [mtdata.load_frames(role, seed) for seed in batch_seeds]
            )  # (B, 201, N)
            x0 = jnp.asarray(reference[:, 0, :, None])
            predicted = np.asarray(rollout(params, x0))[:, :, :, 0]  # (seg, B, N)
            predicted = np.concatenate(
                [reference[:, 0:1, :], predicted.transpose(1, 0, 2)], axis=1
            )  # (B, seg+1, N) with frame 0 = shared initial condition
            for b, seed in enumerate(batch_seeds):
                truth_cache = np.load(mtdata.drift_truth_path(role, seed))
                true_drift_frames = truth_cache["drift"]  # (n_segments, N), t=0.1..2.0
                ref_states = reference[b, metric_frames[1:], :]  # (n_segments, N)
                model_drift = np.asarray(
                    drift_at(params, jnp.asarray(ref_states[..., None]))
                )[..., 0]
                drift_num = np.sum((model_drift - true_drift_frames) ** 2, axis=1)
                drift_den = np.sum(true_drift_frames**2, axis=1)

                for f_index, frame in enumerate(metric_frames):
                    ref_particles = reference[b, frame]
                    pred_particles = predicted[b, f_index]
                    rho_ref = kde_density(ref_particles, GRID, KDE_BANDWIDTH)
                    rho_pred = kde_density(pred_particles, GRID, KDE_BANDWIDTH)
                    l2_ref = float(np.sqrt(np.sum(rho_ref**2) * GRID.dx))
                    l2_diff = float(
                        np.sqrt(np.sum((rho_pred - rho_ref) ** 2) * GRID.dx)
                    )
                    row: dict[str, Any] = {
                        "training_M": m,
                        "training_replicate": replicate,
                        "trajectory_type": role,
                        "trajectory_seed": seed,
                        "time": round(float(times[f_index]), 3),
                        "relative_density_L2": l2_diff / (l2_ref + DENSITY_EPS),
                        "W1": w1_samples(pred_particles, ref_particles, quantiles),
                        "drift_NRMSE": (
                            float(np.sqrt(drift_num[f_index - 1] / drift_den[f_index - 1]))
                            if f_index > 0
                            else ""
                        ),
                        "drift_num": drift_num[f_index - 1] if f_index > 0 else "",
                        "drift_den": drift_den[f_index - 1] if f_index > 0 else "",
                        "best_training_step": best_step,
                        "stopping_step": stopping_step,
                        "training_walltime": round(training_walltime, 2),
                    }
                    rows.append(row)

    evaluation_walltime = time.perf_counter() - eval_start
    fieldnames = list(rows[0].keys())
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "training_M": m,
        "training_replicate": replicate,
        "training_subset_seeds": subset,
        "model_seed": mtdata.model_seed(replicate),
        "batch_seed": mtdata.batch_seed(replicate, m),
        "best_training_step": best_step,
        "stopping_step": stopping_step,
        "early_stopped": stopping_step < max_steps,
        "best_validation_loss": best_val,
        "training_walltime_seconds": training_walltime,
        "evaluation_walltime_seconds": evaluation_walltime,
        "max_steps": max_steps,
        "history": history,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "test_manifest_sha256": test_manifest_sha,
        "model_config": MODEL_CONFIG.to_dict(),
        "jax_backend": jax.default_backend(),
    }
    atomic_write_text(summary_path, json.dumps(summary, indent=2) + "\n")
    summary["model_sha256"] = sha256_file(output_dir / "best_params.msgpack")
    return summary
