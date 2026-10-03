"""Train the MVNN drift model for one seed under the locked budget discipline.

Usage:
  python scripts/threebody_train_mvnn.py --stage pilot --model-seed 0 --max-steps 5000
  python scripts/threebody_train_mvnn.py --stage pilot --model-seed 0 --max-steps 200 --tag smoke
  python scripts/threebody_train_mvnn.py --stage final --model-seed 1 --max-steps <budget>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from threebody import triplets as tp  # noqa: E402
from threebody.artifacts import read_json  # noqa: E402
from threebody.data import load_manifest, load_trajectory  # noqa: E402
from threebody.observation import Grid  # noqa: E402
from threebody.protocol import ground_truth_params, load_protocol  # noqa: E402
from threebody.training import build_training_data, train_mvnn  # noqa: E402

ROLLOUT_NOISE_STREAM = 505  # training-time validation rollout noise (never reused)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "final"), required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--tag", default=None, help="optional run tag, e.g. smoke")
    args = parser.parse_args()

    protocol = load_protocol(ROOT / "protocol.yaml")
    params = ground_truth_params(protocol)
    mvnn_cfg = protocol["mvnn"]
    budgets = mvnn_cfg["budgets"]
    stage_dir = ROOT / "outputs" / "threebody" / "data" / args.stage
    manifest = load_manifest(stage_dir)
    observation = read_json(stage_dir / "observation.json")
    grid = Grid(
        float(observation["domain"][0]),
        float(observation["domain"][1]),
        int(observation["grid_cells"]),
    )
    bandwidth = float(observation["bandwidth"])

    def role_trajectories(role: str) -> tuple[list[np.ndarray], list[int]]:
        trajectories = []
        seeds = []
        for row in manifest["triplets"]:
            if row["role"] != role:
                continue
            triplet_id = int(row["family"]["triplet_id"])
            for measure in tp.MEASURES:
                trajectories.append(load_trajectory(stage_dir, triplet_id, measure))
                seeds.append(
                    int(
                        np.random.SeedSequence(
                            [
                                ROLLOUT_NOISE_STREAM,
                                int(manifest["data_seed_base"]),
                                triplet_id,
                                tp.MEASURES.index(measure),
                            ]
                        ).generate_state(1)[0]
                    )
                )
        return trajectories, seeds

    train_trajectories, _ = role_trajectories("train")
    val_trajectories, val_seeds = role_trajectories("validation")

    data = build_training_data(
        train_trajectories,
        val_trajectories,
        val_seeds,
        params.dt,
        grid,
        bandwidth,
    )

    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(mvnn_cfg["learning_rate"])
    )
    run_name = f"seed_{args.model_seed}" + (f"_{args.tag}" if args.tag else "")
    output_dir = ROOT / "outputs" / "threebody" / args.stage / "mvnn" / run_name
    if (output_dir / "training_summary.json").exists():
        raise FileExistsError(f"{output_dir} already contains a completed run")

    onestep_every = int(budgets["onestep_validation_every"])
    rollout_every = int(budgets["rollout_validation_every"])
    if args.tag == "smoke":
        # exercise every evaluation path within the 200-step smoke budget
        onestep_every, rollout_every = 50, 100

    summary = train_mvnn(
        data,
        output_dir,
        model_seed=args.model_seed,
        dt=params.dt,
        sigma=params.sigma,
        grid=grid,
        bandwidth=bandwidth,
        max_steps=args.max_steps,
        learning_rate=learning_rate,
        gradient_clip=float(mvnn_cfg["gradient_clip_global_norm"]),
        batch_frames=int(mvnn_cfg["batch_frames"]),
        onestep_every=onestep_every,
        rollout_every=rollout_every,
        early_stop_patience=int(budgets["early_stop_patience_rollout_evals"]),
        early_stop_min_relative=float(budgets["early_stop_min_relative_improvement"]),
        extra_summary={"stage": args.stage, "tag": args.tag, "bandwidth": bandwidth},
    )
    best_w1 = summary["best_rollout_w1"]
    best_w1_text = f"{best_w1:.5f}" if best_w1 is not None else "n/a"
    print(
        f"seed {args.model_seed}: best_step={summary['best_step']} "
        f"best_rollout_w1={best_w1_text} "
        f"last_step={summary['last_step']} early_stopped={summary['early_stopped']} "
        f"wall={summary['timings']['wall_time_seconds']:.1f}s "
        f"compile={summary['timings']['first_step_time_seconds_includes_compile']:.1f}s"
    )


if __name__ == "__main__":
    main()
