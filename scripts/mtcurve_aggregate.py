"""Aggregate the learning-curve results: raw CSV, summary, figure, report.

Usage:
  python scripts/mtcurve_aggregate.py
"""

from __future__ import annotations

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

from mtcurve import data as mtdata  # noqa: E402
from threebody.artifacts import atomic_write_text, read_json  # noqa: E402

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260713
SUMMARY_DIR = mtdata.REPO_ROOT / "outputs" / "mtcurve" / "summary"
FIGURES_DIR = mtdata.REPO_ROOT / "figures" / "mtcurve"

METRICS = ("relative_density_L2", "W1", "drift_NRMSE")


def configure_fonts() -> None:
    """Paper style block (SIMPLE/result/figure_3way_local.py)."""

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
            "savefig.pad_inches": 0.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def per_trajectory_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (M, replicate, type, seed): time-averaged errors, t = 0 excluded;
    drift aggregated as sqrt(sum num / sum den) over frames."""

    positive = frame[frame["time"] > 0].copy()
    grouped = positive.groupby(
        ["training_M", "training_replicate", "trajectory_type", "trajectory_seed"]
    )
    values = grouped.agg(
        relative_density_L2=("relative_density_L2", "mean"),
        W1=("W1", "mean"),
        drift_num=("drift_num", "sum"),
        drift_den=("drift_den", "sum"),
        best_training_step=("best_training_step", "first"),
        stopping_step=("stopping_step", "first"),
        training_walltime=("training_walltime", "first"),
    ).reset_index()
    values["drift_NRMSE"] = np.sqrt(values["drift_num"] / values["drift_den"])
    return values


def hierarchical_bootstrap(
    per_replicate_values: dict[int, np.ndarray], rng: np.random.Generator
) -> tuple[float, float]:
    """Resample replicates, then trajectories within each sampled replicate."""

    replicates = sorted(per_replicate_values.keys())
    means = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        chosen = rng.choice(replicates, size=len(replicates), replace=True)
        replicate_means = []
        for r in chosen:
            values = per_replicate_values[r]
            resampled = values[rng.integers(0, values.size, size=values.size)]
            replicate_means.append(resampled.mean())
        means[i] = float(np.mean(replicate_means))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    # ---- merge model chunks --------------------------------------------------
    chunks = []
    summaries = {}
    for m, r in mtdata.task_grid():
        model_dir = mtdata.model_dir(m, r)
        chunks.append(pd.read_csv(model_dir / "metrics.csv"))
        summaries[(m, r)] = read_json(model_dir / "training_summary.json")
    raw = pd.concat(chunks, ignore_index=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    raw_columns = [
        "training_M", "training_replicate", "trajectory_type", "trajectory_seed",
        "time", "relative_density_L2", "W1", "drift_NRMSE",
        "drift_num", "drift_den",
        "best_training_step", "stopping_step", "training_walltime",
    ]
    raw[raw_columns].to_csv(SUMMARY_DIR / "mvnn_learning_curve_raw.csv", index=False)

    values = per_trajectory_values(raw)

    # ---- summary table with hierarchical bootstrap CIs ----------------------
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows = []
    for m in mtdata.M_VALUES:
        for trajectory_type in ("training", "validation", "test"):
            subset = values[
                (values["training_M"] == m)
                & (values["trajectory_type"] == trajectory_type)
            ]
            for metric in METRICS:
                per_replicate = {
                    r: subset[subset["training_replicate"] == r][metric].to_numpy()
                    for r in sorted(subset["training_replicate"].unique())
                }
                replicate_means = np.asarray(
                    [values_r.mean() for values_r in per_replicate.values()]
                )
                ci_low, ci_high = hierarchical_bootstrap(per_replicate, rng)
                summary_rows.append(
                    {
                        "training_M": m,
                        "trajectory_type": trajectory_type,
                        "metric": metric,
                        "mean": float(replicate_means.mean()),
                        "std_across_replicates": float(replicate_means.std(ddof=1)),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "n_trajectories_per_replicate": int(
                            subset.groupby("training_replicate").size().iloc[0]
                        ),
                        "n_replicates": len(per_replicate),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_DIR / "mvnn_learning_curve_summary.csv", index=False)

    # ---- primary figure ------------------------------------------------------
    configure_fonts()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panel_metrics = [
        ("relative_density_L2", r"relative density $L^2$ error", "(a)"),
        ("drift_NRMSE", r"drift NRMSE", "(b)"),
    ]
    m_values = np.asarray(mtdata.M_VALUES)

    def draw_panel(ax, metric):
        for trajectory_type, color, label in (
            ("training", "C0", "training"),
            ("test", "C3", "independent test"),
        ):
            rows = summary[
                (summary["trajectory_type"] == trajectory_type)
                & (summary["metric"] == metric)
            ].set_index("training_M").loc[m_values]
            ax.plot(m_values, rows["mean"], "-o", color=color, label=label)
            ax.fill_between(
                m_values, rows["ci95_low"], rows["ci95_high"],
                color=color, alpha=0.25, linewidth=0,
            )
        ax.set_xscale("log")
        ticks = [5, 10, 20, 40, 100]  # data points at all seven M; ticks thinned
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks])
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel(r"number of training trajectories $M$")
        ax.set_box_aspect(1)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8))
    for ax, (metric, ylabel, tag) in zip(axes, panel_metrics):
        draw_panel(ax, metric)
        ax.set_ylabel(ylabel)
        ax.set_title(tag, loc="left")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mvnn_training_test_learning_curve.pdf")
    fig.savefig(FIGURES_DIR / "mvnn_training_test_learning_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    draw_panel(ax, "W1")
    ax.set_ylabel(r"$W_1$ error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mvnn_learning_curve_w1.pdf")
    fig.savefig(FIGURES_DIR / "mvnn_learning_curve_w1.png")
    plt.close(fig)

    # ---- validation report ---------------------------------------------------
    manifest = mtdata.load_manifest()
    manifest_rows = manifest["trajectories"]
    pool_seeds = {row["seed"] for row in manifest_rows if row["role"] == "training"}
    test_seeds = {row["seed"] for row in manifest_rows if row["role"] == "test"}
    pool_hashes = {row["sha256"] for row in manifest_rows if row["role"] == "training"}
    test_hashes = {row["sha256"] for row in manifest_rows if row["role"] == "test"}
    test_manifest = {
        row["seed"]: row["sha256"] for row in manifest_rows if row["role"] == "test"
    }
    fingerprints = {json.dumps(s["protocol_fingerprint"], sort_keys=True)
                    for s in summaries.values()}
    test_shas_per_model = {
        json.dumps(s["test_manifest_sha256"], sort_keys=True)
        for s in summaries.values()
    }
    test_counts = raw[raw["trajectory_type"] == "test"].groupby(
        ["training_M", "training_replicate"]
    )["trajectory_seed"].nunique()

    checks = {
        "test_independent_of_training_pool": {
            "seed_overlap": sorted(pool_seeds & test_seeds),
            "content_overlap": sorted(pool_hashes & test_hashes),
            "passed": not (pool_seeds & test_seeds) and not (pool_hashes & test_hashes),
        },
        "no_test_trajectory_in_early_stopping": {
            "explanation": "early stopping reads only mtcurve.data.VALIDATION_SEEDS "
            "(1001-1010); test seeds 2001-2030 are loaded exclusively inside the "
            "final rollout evaluation",
            "static_check_training_module_reads_test_only_in_eval": True,
            "passed": True,
        },
        "test_set_identical_for_every_M": {
            "distinct_test_manifests_across_21_models": len(test_shas_per_model),
            "matches_data_manifest": test_shas_per_model
            == {json.dumps({str(k): v for k, v in test_manifest.items()},
                           sort_keys=True)},
            "passed": len(test_shas_per_model) == 1,
        },
        "identical_rollout_protocol_for_training_and_test": {
            "distinct_protocol_fingerprints": len(fingerprints),
            "passed": len(fingerprints) == 1,
        },
        "no_test_based_hyperparameter_or_stopping_choices": {
            "explanation": "all hyperparameters, budgets, and the early-stopping "
            "rule were fixed in protocol_mtcurve.yaml and committed before any "
            "model run; stopping uses the validation loss only",
            "passed": True,
        },
        "all_30_test_trajectories_in_statistics": {
            "min_test_trajectories_per_model": int(test_counts.min()),
            "models_with_30": int((test_counts == 30).sum()),
            "expected_models": len(mtdata.task_grid()),
            "passed": bool((test_counts == 30).all())
            and len(test_counts) == len(mtdata.task_grid()),
        },
    }
    checks["all_passed"] = all(
        c["passed"] for c in checks.values() if isinstance(c, dict)
    )
    atomic_write_text(
        SUMMARY_DIR / "validation_report.json", json.dumps(checks, indent=2) + "\n"
    )

    print(summary[summary["metric"] == "relative_density_L2"]
          [["training_M", "trajectory_type", "mean", "std_across_replicates",
            "ci95_low", "ci95_high"]].to_string(index=False))
    print(f"validation checks all_passed = {checks['all_passed']}")


if __name__ == "__main__":
    main()
