"""Rebuttal figures: L2 density error vs time and density evolution snapshots.

Presentation-only replay of the frozen models on the confirmation set (the
model artifacts are verified against the freeze marker; MVNN rollouts reuse
the exact evaluation noise seeds, so the trajectories equal the ones behind
the published confirmation metrics).

Style matches the paper figures (SIMPLE/result/figure_3way_local.py, which
mirrors result.ipynb): serif text with Computer Modern mathtext, font size
24, inward ticks, no legend frame, fonttype 42, savefig dpi 300; square
panels with subplots_adjust(wspace=0) and hidden interior y axes.

Usage:
  python scripts/threebody_rebuttal_figures.py [--recompute]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.pyplot import MultipleLocator  # noqa: E402

SNAPSHOT_TIMES = (0.0, 0.25, 0.5, 1.0)
DISPLAY_MEASURE = "mixed"
F_SIZE = 24
CACHE = ROOT / "outputs" / "threebody" / "final" / "summary" / "rebuttal_figure_data.npz"

METHOD_STYLE = {
    "kernel_wsindy": ("C0", "Kernel WSINDy"),
    "local_wsindy": ("C1", "Local WSINDy"),
    "mvnn": ("C2", "MVNN"),
}


def configure_fonts() -> None:
    """Paper style block, copied verbatim from SIMPLE/result/figure_3way_local.py."""

    matplotlib.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "font.size": 24,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "axes.linewidth": 1.2,
            "axes.unicode_minus": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.4,
            "legend.borderpad": 0.3,
            "legend.labelspacing": 0.3,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax, show_y: bool = True, f_size: int | None = None) -> None:
    """Hide interior y axes in gapless rows; everything else comes from rcParams."""

    del f_size
    if not show_y:
        ax.yaxis.set_visible(False)


def l2_density_error(predicted: np.ndarray, reference: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Manuscript convention: sqrt(trapz((p_pred - p_ref)^2, x)) per frame."""

    difference = (predicted - reference) ** 2
    return np.sqrt(np.trapz(difference, x, axis=-1))


def compute_data() -> dict[str, np.ndarray]:
    import jax

    from threebody import triplets as tp
    from threebody.artifacts import read_json
    from threebody.baselines import rollout_model_density
    from threebody.data import load_manifest
    from threebody.evaluation import eval_frames_for, eval_noise_seed
    from threebody.freeze import assert_artifacts_match
    from threebody.kernel_wsindy import load_kernel_model
    from threebody.local_wsindy import load_local_model
    from threebody.mvnn import rollout_sde
    from threebody.observation import Grid, kde_density
    from threebody.protocol import ground_truth_params, load_protocol
    from threebody.training import load_best_params

    protocol = load_protocol(ROOT / "protocol.yaml")
    params = ground_truth_params(protocol)
    outputs_root = ROOT / "outputs" / "threebody"
    observation = read_json(outputs_root / "data" / "final" / "observation.json")
    grid = Grid(
        float(observation["domain"][0]),
        float(observation["domain"][1]),
        int(observation["grid_cells"]),
    )
    bandwidth = float(observation["bandwidth"])
    stage_cfg = protocol["stages"]["final"]
    stage_base = int(stage_cfg["confirmation_seed_base"])
    seeds = [int(s) for s in stage_cfg["mvnn_seeds"]]

    marker = read_json(outputs_root / "FREEZE.json")
    assert_artifacts_match(marker, outputs_root)

    baseline_dir = outputs_root / "final" / "baselines"
    local_model = load_local_model(baseline_dir / "local_model.npz")
    kernel_model = load_kernel_model(baseline_dir / "kernel_model.npz")
    mvnn_runs = {
        seed: load_best_params(outputs_root / "final" / "mvnn" / f"seed_{seed}")
        for seed in seeds
    }

    data_dir = outputs_root / "data" / "final" / "confirmation"
    manifest = load_manifest(data_dir)
    rows = [row for row in manifest["triplets"] if row["role"] == "confirmation"]

    frames = eval_frames_for(params.n_frames)
    times = frames.astype(np.float64) * params.dt

    l2_curves: dict[str, list[np.ndarray]] = {
        "kernel_wsindy": [],
        "local_wsindy": [],
        **{f"mvnn_seed{seed}": [] for seed in seeds},
    }
    snapshots: dict[str, np.ndarray] = {}
    display_id = int(rows[0]["family"]["triplet_id"])

    for row in rows:
        triplet_id = int(row["family"]["triplet_id"])
        for measure in tp.MEASURES:
            trajectory = np.load(data_dir / row["measures"][measure]["path"])
            reference = kde_density(trajectory[frames], grid, bandwidth)
            keep_snapshot = triplet_id == display_id and measure == DISPLAY_MEASURE
            if keep_snapshot:
                snapshots["reference"] = reference

            for name, model, diffusion in (
                ("kernel_wsindy", kernel_model, kernel_model.nu),
                ("local_wsindy", local_model, local_model.diffusion),
            ):
                predicted, _ = rollout_model_density(
                    model.velocity, diffusion, reference[0], grid, times, params.dt
                )
                l2_curves[name].append(l2_density_error(predicted, reference, grid.centers))
                if keep_snapshot:
                    snapshots[name] = predicted

            for seed in seeds:
                model, tree = mvnn_runs[seed]
                rolled = np.asarray(
                    rollout_sde(
                        model,
                        tree,
                        trajectory[0],
                        params.dt,
                        int(frames[-1]),
                        params.sigma,
                        jax.random.PRNGKey(
                            eval_noise_seed(stage_base, triplet_id, measure, seed)
                        ),
                    )
                )
                predicted = kde_density(rolled[frames], grid, bandwidth)
                l2_curves[f"mvnn_seed{seed}"].append(
                    l2_density_error(predicted, reference, grid.centers)
                )
                if keep_snapshot and seed == seeds[0]:
                    snapshots[f"mvnn_seed{seed}"] = predicted

    payload = {
        "times": times,
        "x": grid.centers,
        "seeds": np.asarray(seeds),
        "display_id": np.asarray(display_id),
        "l2_kernel": np.mean(l2_curves["kernel_wsindy"], axis=0),
        "l2_local": np.mean(l2_curves["local_wsindy"], axis=0),
        "l2_mvnn": np.stack(
            [np.mean(l2_curves[f"mvnn_seed{seed}"], axis=0) for seed in seeds]
        ),
        "snap_reference": snapshots["reference"],
        "snap_kernel": snapshots["kernel_wsindy"],
        "snap_local": snapshots["local_wsindy"],
        "snap_mvnn": snapshots[f"mvnn_seed{seeds[0]}"],
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    if CACHE.exists() and not args.recompute:
        payload = dict(np.load(CACHE))
    else:
        payload = compute_data()

    configure_fonts()
    figures_dir = ROOT / "figures" / "threebody"
    figures_dir.mkdir(parents=True, exist_ok=True)

    times = payload["times"]
    x = payload["x"]
    l2_mvnn = payload["l2_mvnn"]

    # ---- Figure 1: L2 density error versus time (single square panel) ------
    fig = plt.figure(figsize=(8.2, 6.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_box_aspect(1)
    style_axes(ax, show_y=True)
    ax.plot(times, payload["l2_local"], color="C3", linewidth=2.0,
            label="Local WSINDy")
    ax.plot(times, payload["l2_kernel"], color="C2", linewidth=2.0,
            label="Kernel WSINDy")
    ax.plot(times, l2_mvnn.mean(axis=0), color="C0", linewidth=2.0,
            label="MVNN")
    ax.fill_between(
        times,
        l2_mvnn.min(axis=0),
        l2_mvnn.max(axis=0),
        color="C0",
        alpha=0.3,
        linewidth=0,
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|\rho_{\mathrm{model}}-\rho_{\mathrm{ref}}\|_{L^2}$")
    ax.set_xticks([0, 0.5, 1])
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.60))
    fig.savefig(figures_dir / "l2_error_vs_time_confirmation.pdf", pad_inches=0.15)
    fig.savefig(figures_dir / "l2_error_vs_time_confirmation.png", pad_inches=0.15)
    plt.close(fig)

    # ---- Figure 2: density evolution at selected times (1x4, no gaps) ------
    fig = plt.figure(figsize=(24, 6))
    fig.subplots_adjust(wspace=0, hspace=0)
    snap_indices = [int(np.argmin(np.abs(times - t_snap))) for t_snap in SNAPSHOT_TIMES]
    first_ax = None
    for column, (t_snap, index) in enumerate(zip(SNAPSHOT_TIMES, snap_indices)):
        ax = fig.add_subplot(1, 4, column + 1, sharey=first_ax)
        if first_ax is None:
            first_ax = ax
        ax.set_title(rf"$t={t_snap:g}$")
        style_axes(ax, show_y=(column == 0))
        ax.plot(x, payload["snap_local"][index], color="C3", linewidth=2.0,
                label="Local PDE-WSINDy")
        ax.plot(x, payload["snap_kernel"][index], color="C2", linewidth=2.0,
                label="Kernel WSINDy")
        ax.plot(x, payload["snap_reference"][index], color="C1", linewidth=2.0,
                label="Simulation (Reference)")
        ax.plot(x, payload["snap_mvnn"][index], color="C0", linewidth=2.0,
                label="Learned MVNN dynamics")
        ax.set_xlim(-3.5, 3.0)
        ax.set_xticks([-2, 0, 2])
        ax.set_xlabel(r"$x$")
        if column == 0:
            ax.set_ylabel(r"$\rho(x)$")
            ax.yaxis.set_major_locator(MultipleLocator(2))
        if column == 1:
            handles, labels = ax.get_legend_handles_labels()
            order = [3, 1, 0, 2]  # MVNN, Kernel, Local, Reference
            ax.legend([handles[i] for i in order], [labels[i] for i in order],
                      loc="upper left", fontsize=20)
    fig.savefig(figures_dir / "density_evolution_confirmation.pdf", pad_inches=0.15)
    fig.savefig(figures_dir / "density_evolution_confirmation.png", pad_inches=0.15)
    plt.close(fig)

    # ---- Figure 3: combined paper strip (3 density panels + L2 panel) ------
    strip_times = (0.0, 0.5, 1.0)
    strip_indices = [int(np.argmin(np.abs(times - t_snap))) for t_snap in strip_times]
    fig = plt.figure(figsize=(24, 6))
    fig.subplots_adjust(wspace=0, hspace=0)
    first_ax = None
    for column, (t_snap, index) in enumerate(zip(strip_times, strip_indices)):
        ax = fig.add_subplot(1, 4, column + 1, sharey=first_ax)
        if first_ax is None:
            first_ax = ax
        ax.set_title(rf"$t={t_snap:g}$")
        style_axes(ax, show_y=(column == 0))
        ax.plot(x, payload["snap_local"][index], color="C3", linewidth=2.0,
                label="Local PDE-WSINDy")
        ax.plot(x, payload["snap_kernel"][index], color="C2", linewidth=2.0,
                label="Kernel WSINDy")
        ax.plot(x, payload["snap_reference"][index], color="C1", linewidth=2.0,
                label="Simulation (Reference)")
        ax.plot(x, payload["snap_mvnn"][index], color="C0", linewidth=2.0,
                label="Learned MVNN dynamics")
        ax.set_xlim(-3.5, 3.0)
        ax.set_xticks([-2, 0, 2])
        ax.set_xlabel(r"$x$")
        if column == 0:
            ax.set_ylabel(r"$\rho(x)$")
            ax.yaxis.set_major_locator(MultipleLocator(2))
        if column == 1:
            handles, labels = ax.get_legend_handles_labels()
            order = [3, 1, 0, 2]
            ax.legend([handles[i] for i in order], [labels[i] for i in order],
                      loc="upper left", fontsize=20)
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_title(r"$L^2$ error")
    style_axes(ax4, show_y=True)
    ax4.plot(times, payload["l2_local"], color="C3", linewidth=2.0)
    ax4.plot(times, payload["l2_kernel"], color="C2", linewidth=2.0)
    ax4.plot(times, l2_mvnn.mean(axis=0), color="C0", linewidth=2.0)
    ax4.fill_between(times, l2_mvnn.min(axis=0), l2_mvnn.max(axis=0),
                     color="C0", alpha=0.3, linewidth=0)
    ax4.set_xlabel(r"$t$")
    ax4.set_ylabel(r"$\|\rho_{\mathrm{model}}-\rho_{\mathrm{ref}}\|_{L^2}$")
    ax4.set_xticks([0, 0.5, 1])
    ax4.set_ylim(bottom=0)
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.tick_params(axis="y", labelleft=False, left=False, right=True)
    fig.savefig(figures_dir / "paper_strip_confirmation.pdf", pad_inches=0.15)
    fig.savefig(figures_dir / "paper_strip_confirmation.png", pad_inches=0.15)
    plt.close(fig)

    print(f"display triplet {int(payload['display_id'])} ({DISPLAY_MEASURE} measure)")
    print(f"time-mean L2: local {payload['l2_local'].mean():.4f}, "
          f"kernel {payload['l2_kernel'].mean():.4f}, "
          f"mvnn {l2_mvnn.mean():.4f}")
    print(f"wrote 2 figures to {figures_dir}")


if __name__ == "__main__":
    main()
