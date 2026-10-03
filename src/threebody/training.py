"""Step-based MVNN trainer with rollout-W1 checkpoint selection.

Budget discipline (protocol section mvnn.budgets):
  - one-step validation every 250 steps (monitoring only)
  - autonomous validation rollout W1 every 1000 steps (checkpoint selection)
  - early stop after 5 rollout evaluations with < 0.5% relative improvement
  - best and last checkpoints only; best restored at the end
  - JAX compilation time recorded separately from training time
  - abort immediately on non-finite losses, gradients, or rollout states
"""

from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from .artifacts import atomic_write_bytes, atomic_write_text
from .metrics import w1_densities
from .mvnn import MVNN, MVNNConfig, relative_l2_loss, rollout_sde
from .observation import Grid, kde_density


class EarlyStopper:
    """Stop after ``patience`` evaluations without ``min_relative`` improvement."""

    def __init__(self, patience: int, min_relative: float) -> None:
        if patience < 1 or min_relative < 0.0:
            raise ValueError("invalid early stopping configuration")
        self.patience = patience
        self.min_relative = min_relative
        self.best_value = np.inf
        self.best_step: int | None = None
        self.evals_since_improvement = 0

    def update(self, step: int, value: float) -> bool:
        """Record an evaluation; returns True when training should stop."""

        if not np.isfinite(value):
            raise FloatingPointError("early stopping received a non-finite value")
        improved = value < self.best_value * (1.0 - self.min_relative)
        if improved:
            self.best_value = float(value)
            self.best_step = int(step)
            self.evals_since_improvement = 0
        else:
            # first evaluation initializes the incumbent without counting
            if self.best_step is None:
                self.best_value = float(value)
                self.best_step = int(step)
            else:
                if value < self.best_value:
                    self.best_value = float(value)
                    self.best_step = int(step)
                self.evals_since_improvement += 1
        return self.evals_since_improvement >= self.patience

    @property
    def improved_at_last_update(self) -> bool:
        return self.evals_since_improvement == 0


class GPUSampler:
    """Background nvidia-smi sampler (utilization %, memory MiB)."""

    def __init__(self, interval_seconds: float = 5.0) -> None:
        self.interval = interval_seconds
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                line = output.stdout.strip().splitlines()
                if line:
                    util, mem = line[0].split(",")
                    self.samples.append((float(util), float(mem)))
            except (OSError, ValueError, subprocess.SubprocessError):
                return

    def start(self) -> None:
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 10)
        if not self.samples:
            return {
                "gpu_utilization_mean": None,
                "gpu_utilization_max": None,
                "gpu_memory_max_mib": None,
            }
        utils = [sample[0] for sample in self.samples]
        mems = [sample[1] for sample in self.samples]
        return {
            "gpu_utilization_mean": float(np.mean(utils)),
            "gpu_utilization_max": float(np.max(utils)),
            "gpu_memory_max_mib": float(np.max(mems)),
        }


@dataclass
class TrainingData:
    positions: np.ndarray  # (F, N) training frames
    velocities: np.ndarray  # (F, N) forward-difference velocities
    val_positions: np.ndarray  # (Fv, N)
    val_velocities: np.ndarray  # (Fv, N)
    val_initial: np.ndarray  # (V, N) validation t=0 particles
    val_reference_densities: np.ndarray  # (V, T_eval, cells)
    val_noise_seeds: np.ndarray  # (V,) rollout noise seeds
    eval_frame_indices: np.ndarray  # (T_eval,)


def build_training_data(
    train_trajectories: list[np.ndarray],
    val_trajectories: list[np.ndarray],
    val_noise_seeds: list[int],
    dt: float,
    grid: Grid,
    bandwidth: float,
    eval_frame_stride: int = 5,
) -> TrainingData:
    def frames_and_velocities(trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        positions = []
        velocities = []
        for trajectory in trajectories:
            positions.append(trajectory[:-1])
            velocities.append((trajectory[1:] - trajectory[:-1]) / dt)
        return np.concatenate(positions, axis=0), np.concatenate(velocities, axis=0)

    positions, velocities = frames_and_velocities(train_trajectories)
    val_positions, val_velocities = frames_and_velocities(val_trajectories)
    n_frames = val_trajectories[0].shape[0]
    eval_frames = np.arange(0, n_frames, eval_frame_stride)
    if eval_frames[-1] != n_frames - 1:
        eval_frames = np.concatenate([eval_frames, [n_frames - 1]])
    reference = np.stack(
        [
            kde_density(trajectory[eval_frames], grid, bandwidth)
            for trajectory in val_trajectories
        ]
    )
    return TrainingData(
        positions=positions,
        velocities=velocities,
        val_positions=val_positions,
        val_velocities=val_velocities,
        val_initial=np.stack([trajectory[0] for trajectory in val_trajectories]),
        val_reference_densities=reference,
        val_noise_seeds=np.asarray(val_noise_seeds, dtype=np.int64),
        eval_frame_indices=eval_frames,
    )


def save_params(path: Path, params: Any) -> None:
    atomic_write_bytes(path, serialization.to_bytes(params))


def load_params(path: Path, template: Any) -> Any:
    return serialization.from_bytes(template, Path(path).read_bytes())


def train_mvnn(
    data: TrainingData,
    output_dir: Path,
    *,
    model_seed: int,
    dt: float,
    sigma: float,
    grid: Grid,
    bandwidth: float,
    max_steps: int,
    learning_rate: float = 1.0e-3,
    gradient_clip: float = 1.0,
    batch_frames: int = 32,
    onestep_every: int = 250,
    rollout_every: int = 1000,
    early_stop_patience: int = 5,
    early_stop_min_relative: float = 0.005,
    config: MVNNConfig | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = config or MVNNConfig()
    model = MVNN(model_config)
    params = model.init(jax.random.PRNGKey(model_seed))
    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, xb, vb):
        def loss_fn(p):
            return relative_l2_loss(model.apply(p, xb), vb)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, grad_norm

    @jax.jit
    def onestep_loss(params, xb, vb):
        return relative_l2_loss(model.apply(params, xb), vb)

    n_rollout_steps = int(data.eval_frame_indices[-1])

    @jax.jit
    def rollout_fn(params, x0, key):
        return rollout_sde(model, params, x0, dt, n_rollout_steps, sigma, key)

    def rollout_metric(params) -> float:
        w1_values = []
        for v_index in range(data.val_initial.shape[0]):
            trajectory = rollout_fn(
                params,
                jnp.asarray(data.val_initial[v_index]),
                jax.random.PRNGKey(int(data.val_noise_seeds[v_index])),
            )
            frames = np.asarray(trajectory)[data.eval_frame_indices]
            if not np.all(np.isfinite(frames)):
                raise FloatingPointError("MVNN rollout produced non-finite states")
            if frames.min() < grid.lower or frames.max() > grid.upper:
                raise FloatingPointError(
                    "MVNN rollout left the evaluation domain (invalid state)"
                )
            densities = kde_density(frames, grid, bandwidth)
            per_frame = [
                w1_densities(
                    densities[t], data.val_reference_densities[v_index, t], grid
                )
                for t in range(densities.shape[0])
            ]
            w1_values.append(float(np.mean(per_frame)))
        return float(np.mean(w1_values))

    # fixed one-step validation batch (monitoring only)
    val_x = jnp.asarray(data.val_positions[:, :, None])
    val_v = jnp.asarray(data.val_velocities[:, :, None])

    rng = np.random.default_rng(1_000_003 * (model_seed + 1))
    n_train_frames = data.positions.shape[0]

    sampler = GPUSampler()
    sampler.start()

    log_path = output_dir / "training_log.csv"
    log_fields = [
        "step",
        "train_loss",
        "grad_norm",
        "onestep_validation",
        "rollout_w1",
        "elapsed_seconds",
    ]
    log_handle = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_handle, fieldnames=log_fields)
    log_writer.writeheader()

    stopper = EarlyStopper(early_stop_patience, early_stop_min_relative)
    best_params_host = jax.device_get(params)
    timings: dict[str, float] = {}
    start_time = time.perf_counter()
    compile_time = 0.0
    rollout_compile_time = 0.0
    early_stopped = False
    aborted_reason: str | None = None
    last_step = 0

    for step in range(1, max_steps + 1):
        batch_index = rng.integers(0, n_train_frames, size=batch_frames)
        xb = jnp.asarray(data.positions[batch_index][:, :, None])
        vb = jnp.asarray(data.velocities[batch_index][:, :, None])
        step_start = time.perf_counter()
        params, opt_state, loss, grad_norm = update(params, opt_state, xb, vb)
        if step == 1:
            loss.block_until_ready()
            compile_time = time.perf_counter() - step_start
        loss_value = float(loss)
        grad_norm_value = float(grad_norm)
        if not np.isfinite(loss_value) or not np.isfinite(grad_norm_value):
            aborted_reason = f"non-finite loss/gradient at step {step}"
            last_step = step
            break
        row: dict[str, Any] = {
            "step": step,
            "train_loss": loss_value,
            "grad_norm": grad_norm_value,
            "onestep_validation": "",
            "rollout_w1": "",
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        if step % onestep_every == 0:
            row["onestep_validation"] = float(onestep_loss(params, val_x, val_v))
        if step % rollout_every == 0:
            rollout_start = time.perf_counter()
            metric = rollout_metric(params)
            if step == rollout_every:
                rollout_compile_time = time.perf_counter() - rollout_start
            row["rollout_w1"] = metric
            should_stop = stopper.update(step, metric)
            if stopper.best_step == step:
                best_params_host = jax.device_get(params)
                save_params(output_dir / "best_params.msgpack", best_params_host)
            if should_stop:
                early_stopped = True
                log_writer.writerow(row)
                last_step = step
                break
        log_writer.writerow(row)
        if step % 50 == 0:
            log_handle.flush()
        last_step = step

    log_handle.flush()
    log_handle.close()
    wall_time = time.perf_counter() - start_time

    if aborted_reason is not None:
        raise FloatingPointError(aborted_reason)

    # if no rollout evaluation ever ran (e.g., smoke test), best = last
    if stopper.best_step is None:
        best_params_host = jax.device_get(params)
        stopper.best_step = last_step
        stopper.best_value = float("nan")
        save_params(output_dir / "best_params.msgpack", best_params_host)

    save_params(output_dir / "last_params.msgpack", jax.device_get(params))
    if not (output_dir / "best_params.msgpack").exists():
        save_params(output_dir / "best_params.msgpack", best_params_host)

    gpu_stats = sampler.stop()
    device = jax.devices()[0]
    memory_stats = getattr(device, "memory_stats", lambda: None)() or {}
    peak_device_bytes = memory_stats.get("peak_bytes_in_use")

    timings = {
        "wall_time_seconds": wall_time,
        "first_step_time_seconds_includes_compile": compile_time,
        "first_rollout_eval_seconds_includes_compile": rollout_compile_time,
        "training_time_seconds_excluding_first_step": wall_time - compile_time,
    }
    summary: dict[str, Any] = {
        "model_seed": model_seed,
        "model_config": model_config.to_dict(),
        "learning_rate": learning_rate,
        "gradient_clip_global_norm": gradient_clip,
        "batch_frames": batch_frames,
        "max_steps": max_steps,
        "last_step": last_step,
        "best_step": stopper.best_step,
        "best_rollout_w1": (
            stopper.best_value if np.isfinite(stopper.best_value) else None
        ),
        "early_stopped": early_stopped,
        "onestep_every": onestep_every,
        "rollout_every": rollout_every,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_relative": early_stop_min_relative,
        "timings": timings,
        "gpu": gpu_stats,
        "peak_device_bytes_in_use": peak_device_bytes,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "checkpoints": ["best_params.msgpack", "last_params.msgpack"],
    }
    if extra_summary:
        summary.update(extra_summary)
    atomic_write_text(
        output_dir / "training_summary.json", json.dumps(summary, indent=2) + "\n"
    )
    return summary


def load_best_params(output_dir: Path, config: MVNNConfig | None = None):
    model = MVNN(config or MVNNConfig())
    template = model.init(jax.random.PRNGKey(0))
    return model, load_params(Path(output_dir) / "best_params.msgpack", template)
