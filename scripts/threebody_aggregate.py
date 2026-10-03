"""Aggregate results into machine-readable summaries, figures, and the report.

Usage:
  python scripts/threebody_aggregate.py --stage final --role confirmation
  python scripts/threebody_aggregate.py --stage pilot --role validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from threebody import triplets as tp  # noqa: E402
from threebody.artifacts import atomic_write_text, read_json  # noqa: E402
from threebody.data import load_manifest  # noqa: E402
from threebody.metrics import bootstrap_mean, paired_bootstrap  # noqa: E402
from threebody.observation import Grid, kde_density  # noqa: E402
from threebody.protocol import load_protocol  # noqa: E402

plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


def save_figure(figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(directory / f"{stem}.png", bbox_inches="tight")
    plt.close(figure)


def method_display(method: str) -> str:
    if method.startswith("mvnn"):
        return method.replace("mvnn_seed", "MVNN seed ")
    return {"local_wsindy": "Local PDE-WSINDy", "kernel_wsindy": "B-spline kernel WSINDy"}[
        method
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "final"), required=True)
    parser.add_argument("--role", choices=("validation", "confirmation"), required=True)
    args = parser.parse_args()

    protocol = load_protocol(ROOT / "protocol.yaml")
    outputs_root = ROOT / "outputs" / "threebody"
    eval_dir = outputs_root / args.stage / "evaluation" / args.role
    payload = read_json(eval_dir / "raw_metrics.json")
    metrics = pd.DataFrame(
        [
            {k: v for k, v in row.items() if not isinstance(v, list)}
            for row in payload["metrics"]
        ]
    )
    defects = pd.DataFrame(payload["affinity_defects"])
    curves = {
        (row["method"], row["triplet_id"], row["measure"]): (
            np.asarray(row["times"]),
            np.asarray(row["w1_of_t"]),
            np.asarray(row["predicted_M_of_t"]),
            np.asarray(row["reference_M_of_t"]),
        )
        for row in payload["metrics"]
    }
    summary_dir = outputs_root / args.stage / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = ROOT / "figures" / "threebody"

    methods = sorted(metrics["method"].unique())
    mvnn_methods = [m for m in methods if m.startswith("mvnn")]

    # ---- per-triplet aggregation (paired across methods) --------------------
    per_triplet = (
        metrics.groupby(["method", "triplet_id"])
        .agg(
            time_averaged_w1=("time_averaged_w1", "mean"),
            final_w1=("final_w1", "mean"),
            direction_match=("direction_match", "mean"),
            drift_rmse_t0=("drift_rmse_t0", "mean"),
        )
        .reset_index()
    )
    summary_rows = []
    for method in methods:
        sub = per_triplet[per_triplet["method"] == method].sort_values("triplet_id")
        stats = bootstrap_mean(sub["time_averaged_w1"].to_numpy())
        summary_rows.append(
            {
                "method": method,
                "time_averaged_w1_mean": stats["mean"],
                "time_averaged_w1_ci95_low": stats["ci95_low"],
                "time_averaged_w1_ci95_high": stats["ci95_high"],
                "final_w1_mean": float(sub["final_w1"].mean()),
                "direction_accuracy": float(sub["direction_match"].mean()),
                "drift_rmse_t0_mean": float(sub["drift_rmse_t0"].mean()),
                "defect_vs_truth_relative_error_mean": float(
                    defects[defects["method"] == method][
                        "defect_vs_truth_relative_error"
                    ].mean()
                ),
                "defect_rms_mean": float(
                    defects[defects["method"] == method]["defect_rms"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_dir / f"method_summary_{args.role}.csv", index=False)

    # paired bootstrap: each baseline vs the mean MVNN (across seeds)
    mvnn_by_triplet = (
        per_triplet[per_triplet["method"].isin(mvnn_methods)]
        .groupby("triplet_id")["time_averaged_w1"]
        .mean()
        .sort_index()
    )
    paired = {}
    for method in ("local_wsindy", "kernel_wsindy"):
        sub = (
            per_triplet[per_triplet["method"] == method]
            .set_index("triplet_id")["time_averaged_w1"]
            .sort_index()
        )
        paired[method] = paired_bootstrap(
            sub.to_numpy(), mvnn_by_triplet.to_numpy()
        )
    with open(summary_dir / f"paired_bootstrap_{args.role}.json", "w") as handle:
        json.dump(paired, handle, indent=2)

    # ---- timing/cost table ---------------------------------------------------
    costs: dict[str, dict] = {}
    fit_summary = read_json(outputs_root / args.stage / "baselines" / "fit_summary.json")
    costs["preprocessing_seconds_train_densities"] = fit_summary[
        "preprocessing_seconds"
    ]
    costs["hardware"] = {
        "wsindy_fits_and_rollouts": "CPU (SLURM, 16 cores fits / 8 cores rollouts)",
        "mvnn_training": "single NVIDIA A100 GPU",
    }
    costs["local_selected_fit_seconds"] = fit_summary["local"]["selection"].get(
        "selected_fit_seconds"
    )
    costs["local_selection_total_seconds"] = fit_summary["local"][
        "wall_seconds_including_selection"
    ]
    costs["kernel_selected_fit_seconds"] = fit_summary["kernel"]["selection"].get(
        "selected_fit_seconds"
    )
    costs["kernel_selection_total_seconds"] = fit_summary["kernel"][
        "wall_seconds_including_selection"
    ]
    for seed in protocol["stages"][args.stage]["mvnn_seeds"]:
        training = read_json(
            outputs_root / args.stage / "mvnn" / f"seed_{seed}" / "training_summary.json"
        )
        costs[f"mvnn_seed{seed}"] = {
            "best_step": training["best_step"],
            "last_step": training["last_step"],
            "early_stopped": training["early_stopped"],
            "wall_time_seconds": training["timings"]["wall_time_seconds"],
            "compile_seconds": training["timings"][
                "first_step_time_seconds_includes_compile"
            ],
            "gpu": training["gpu"],
            "peak_device_bytes_in_use": training.get("peak_device_bytes_in_use"),
        }
    rollout_costs = (
        metrics.groupby("method")[["rollout_seconds", "preprocessing_seconds"]]
        .mean()
        .to_dict("index")
    )
    costs["mean_rollout_seconds_per_trajectory"] = rollout_costs
    with open(summary_dir / f"costs_{args.role}.json", "w") as handle:
        json.dump(costs, handle, indent=2)

    # ---- main comparison figure ----------------------------------------------
    domain = protocol["ground_truth"]["domain"]
    protocol_grid = Grid(
        float(domain[0]), float(domain[1]), int(payload["grid_cells"])
    )
    bandwidth = float(payload["bandwidth"])
    if args.role == "confirmation":
        data_dir = outputs_root / "data" / args.stage / "confirmation"
    else:
        data_dir = outputs_root / "data" / args.stage
    manifest = load_manifest(data_dir)
    display_row = [row for row in manifest["triplets"] if row["role"] == args.role][0]
    display_family = tp.TripletFamily(**display_row["family"])
    display_initial = tp.build_triplet(display_family)
    display_id = display_family.triplet_id

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    for measure, style in (("plus", "C0-"), ("minus", "C1-"), ("mixed", "C2--")):
        rho = kde_density(display_initial[measure], protocol_grid, bandwidth)
        ax.plot(protocol_grid.centers, rho, style, label=rf"$\mu_{{{measure}}}$")
    ax.set_title(f"Matched initial densities (triplet {display_id})")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho(x, 0)$")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    # presentation-only re-rollout of every method on the display triplet to
    # show the central-bump centroid evolution (raw metrics are untouched)
    from threebody.baselines import rollout_model_density
    from threebody.data import load_trajectory as _load_traj
    from threebody.evaluation import eval_frames_for, eval_noise_seed
    from threebody.kernel_wsindy import load_kernel_model
    from threebody.local_wsindy import load_local_model
    from threebody.metrics import bump_centroid
    from threebody.protocol import ground_truth_params

    params = ground_truth_params(protocol)
    display_traj = np.load(
        data_dir / f"triplet_{display_id:03d}_mixed.npy"
    )
    frames = eval_frames_for(display_traj.shape[0])
    times_disp = frames.astype(np.float64) * params.dt
    bump_mask = times_disp <= 0.3 + 1e-12
    reference_density = kde_density(display_traj[frames], protocol_grid, bandwidth)
    window_c = display_family.x_c
    window_h = display_family.window_half_width

    def centroids(density_traj):
        return bump_centroid(density_traj[bump_mask], protocol_grid, window_c, window_h)

    ax.plot(times_disp[bump_mask], centroids(reference_density), "k-", lw=2, label="reference")
    baseline_dir = outputs_root / args.stage / "baselines"
    local_model = load_local_model(baseline_dir / "local_model.npz")
    kernel_model = load_kernel_model(baseline_dir / "kernel_model.npz")
    for name, model, diffusion in (
        ("kernel_wsindy", kernel_model, kernel_model.nu),
        ("local_wsindy", local_model, local_model.diffusion),
    ):
        predicted, _ = rollout_model_density(
            model.velocity, diffusion, reference_density[0], protocol_grid,
            times_disp, params.dt,
        )
        ax.plot(times_disp[bump_mask], centroids(predicted), "--", label=method_display(name))
    import jax as _jax

    from threebody.mvnn import rollout_sde
    from threebody.training import load_best_params

    stage_cfg = protocol["stages"][args.stage]
    stage_base = int(
        stage_cfg["confirmation_seed_base"]
        if args.role == "confirmation"
        else stage_cfg["data_seed_base"]
    )
    for seed in stage_cfg["mvnn_seeds"]:
        model, tree = load_best_params(outputs_root / args.stage / "mvnn" / f"seed_{seed}")
        rolled = np.asarray(
            rollout_sde(
                model, tree, display_traj[0], params.dt, int(frames[-1]),
                params.sigma,
                _jax.random.PRNGKey(eval_noise_seed(stage_base, display_id, "mixed", seed)),
            )
        )
        density = kde_density(rolled[frames], protocol_grid, bandwidth)
        ax.plot(times_disp[bump_mask], centroids(density), "--",
                label=method_display(f"mvnn_seed{seed}"))
    ax.set_title(f"Central-bump centroid, mixed measure (triplet {display_id})")
    ax.set_xlabel("t")
    ax.set_ylabel("bump centroid")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for method in methods:
        all_curves = [
            curves[key][1] for key in curves if key[0] == method
        ]
        times = curves[next(key for key in curves if key[0] == method)][0]
        ax.plot(times, np.mean(all_curves, axis=0), label=method_display(method))
    ax.set_title(rf"Mean $W_1(t)$ across {args.role} triplets")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$W_1$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    mvnn_walls = [
        costs[f"mvnn_seed{seed}"]["wall_time_seconds"]
        for seed in protocol["stages"][args.stage]["mvnn_seeds"]
    ]
    mvnn_rollouts = [
        value["rollout_seconds"]
        for method, value in rollout_costs.items()
        if method.startswith("mvnn")
    ]
    # raw seconds, no rescaling; hardware stated per bar group
    categories = (
        "single fit / train\n(local, kernel: CPU;\nMVNN: A100 GPU)",
        "hyperparameter\nselection total (CPU;\nMVNN: none)",
        "autonomous rollout\nper trajectory (CPU)",
    )
    series = {
        "local WSINDy": (
            costs["local_selected_fit_seconds"] or float("nan"),
            costs["local_selection_total_seconds"],
            rollout_costs.get("local_wsindy", {}).get("rollout_seconds", float("nan")),
        ),
        "kernel WSINDy": (
            costs["kernel_selected_fit_seconds"] or float("nan"),
            costs["kernel_selection_total_seconds"],
            rollout_costs.get("kernel_wsindy", {}).get("rollout_seconds", float("nan")),
        ),
        "MVNN (mean seed)": (
            float(np.mean(mvnn_walls)),
            float("nan"),
            float(np.mean(mvnn_rollouts)) if mvnn_rollouts else float("nan"),
        ),
    }
    x_pos = np.arange(len(categories))
    width = 0.26
    for offset, (label, values) in zip((-width, 0.0, width), series.items()):
        ax.bar(x_pos + offset, values, width=width, label=label)
    ax.set_xticks(x_pos, categories, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("seconds")
    ax.set_title("Cost (raw wall seconds, hardware as labeled)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    figure.suptitle(
        f"Three-body global-mean experiment - {args.stage}/{args.role}", y=1.0
    )
    figure.tight_layout()
    save_figure(figure, figures_dir, f"main_comparison_{args.stage}_{args.role}")

    # ---- learning curves -------------------------------------------------------
    figure, ax = plt.subplots(figsize=(7, 4.2))
    for seed in protocol["stages"][args.stage]["mvnn_seeds"]:
        run_dir = outputs_root / args.stage / "mvnn" / f"seed_{seed}"
        log = pd.read_csv(run_dir / "training_log.csv")
        rollout = log[log["rollout_w1"].notna() & (log["rollout_w1"] != "")]
        training = read_json(run_dir / "training_summary.json")
        ax.plot(
            rollout["step"],
            rollout["rollout_w1"].astype(float),
            marker="o",
            label=f"seed {seed}",
        )
        ax.axvline(
            training["best_step"], color=ax.lines[-1].get_color(), ls=":", alpha=0.7
        )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation rollout $W_1$")
    ax.set_title(f"MVNN learning curves ({args.stage}); dotted = best step")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(figure, figures_dir, f"learning_curves_{args.stage}")

    # ---- machine-readable master summary --------------------------------------
    master = {
        "stage": args.stage,
        "role": args.role,
        "n_triplets": payload["n_triplets"],
        "bandwidth": bandwidth,
        "methods": summary.to_dict("records"),
        "paired_bootstrap_vs_mvnn_mean": paired,
        "costs": costs,
    }
    atomic_write_text(
        summary_dir / f"master_summary_{args.role}.json",
        json.dumps(master, indent=2, default=float) + "\n",
    )
    print(summary.to_string(index=False))
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
