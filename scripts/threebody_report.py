"""Generate results_report.md from measured artifacts only.

Usage:
  python scripts/threebody_report.py --stage final --role confirmation

The report contains only measured numbers read from committed artifact files;
it never contains manuscript or rebuttal language.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threebody.artifacts import read_json  # noqa: E402
from threebody.protocol import load_protocol  # noqa: E402


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "final"), default="final")
    parser.add_argument("--role", choices=("validation", "confirmation"), default="confirmation")
    args = parser.parse_args()

    protocol = load_protocol(ROOT / "protocol.yaml")
    outputs_root = ROOT / "outputs" / "threebody"
    summary_dir = outputs_root / args.stage / "summary"
    master = read_json(summary_dir / f"master_summary_{args.role}.json")
    paired = master["paired_bootstrap_vs_mvnn_mean"]
    costs = master["costs"]
    sanity = read_json(outputs_root / "sanity_alpha0" / "report.json")
    fit_summary = read_json(outputs_root / args.stage / "baselines" / "fit_summary.json")
    freeze_path = outputs_root / "FREEZE.json"
    freeze = read_json(freeze_path) if freeze_path.exists() else None
    defect_csv = (
        outputs_root / args.stage / "evaluation" / args.role / "affinity_defects.csv"
    )

    lines: list[str] = []
    add = lines.append
    add("# Results Report: Three-Body Global-Mean Experiment")
    add("")
    add(
        "Measured results only. Ground truth: "
        "dX = [gamma (M - X) + (a - alpha M^2) X - beta X^3] dt + sigma dW with "
        f"a={protocol['ground_truth']['a']}, alpha={protocol['ground_truth']['alpha']}, "
        f"beta={protocol['ground_truth']['beta']}, gamma={protocol['ground_truth']['gamma']}, "
        f"sigma={protocol['ground_truth']['sigma']}. "
        "The three-body term -alpha X M^2 cannot be represented by any additive "
        "pairwise model f(x) + int K(y-x) mu(dy): on matched mixture triplets "
        "(mu+, mu-, mu0=(mu+ + mu-)/2 with means +/-d and 0) the true drift "
        "violates mixture affinity by exactly alpha*x*d^2 while every additive "
        "model has zero affinity defect."
    )
    add("")
    add(f"Stage: `{args.stage}`; evaluation set: `{args.role}` "
        f"({master['n_triplets']} untouched matched triplets). "
        f"KDE bandwidth {master['bandwidth']} on a {protocol['observation']['grid_cells']}-cell "
        f"grid over {protocol['ground_truth']['domain']}; identical observation "
        "operator for all methods.")
    add("")
    add("## Primary comparison (per-triplet means, bootstrap 95% CIs)")
    add("")
    add("| method | time-avg W1 | 95% CI | final W1 | direction acc. | drift RMSE (t=0) | affinity defect vs truth (rel. err.) |")
    add("|---|---|---|---|---|---|---|")
    for row in master["methods"]:
        add(
            f"| {row['method']} | {fmt(row['time_averaged_w1_mean'])} | "
            f"[{fmt(row['time_averaged_w1_ci95_low'])}, {fmt(row['time_averaged_w1_ci95_high'])}] | "
            f"{fmt(row['final_w1_mean'])} | {fmt(100 * row['direction_accuracy'], 3)}% | "
            f"{fmt(row['drift_rmse_t0_mean'])} | "
            f"{fmt(row['defect_vs_truth_relative_error_mean'], 3)} |"
        )
    add("")
    add("Direction accuracy = fraction of triplet/measure pairs whose central-bump "
        "short-time motion direction matches the reference. Affinity defect relative "
        "error = ||defect_model - alpha*x*d^2|| / ||alpha*x*d^2|| on the central "
        "window; 1.0 means the model shows no three-body defect at all (additive), "
        "0 means the defect is captured exactly.")
    add("")
    import pandas as _pd

    defects = _pd.read_csv(defect_csv)
    gaps = defects["kde_mixture_max_abs_gap"].dropna()
    add(
        "Same-condition defect evaluation: the stored particle triplets satisfy "
        "mu_mixed = (mu_plus + mu_minus)/2 exactly at the multiset level, so every "
        "method is probed with the same empirical objects (density models via the "
        "shared KDE of each particle vector, MVNN via the particle vectors). "
        f"Measured max |rho_mixed - (rho_plus + rho_minus)/2| over triplets: "
        f"{fmt(float(gaps.max()), 3) if len(gaps) else 'n/a'} "
        "(floating-point roundoff of the linear KDE)."
    )
    add("")
    add("## Paired differences vs MVNN (paired bootstrap across triplets)")
    add("")
    add("| baseline | mean(baseline - MVNN) time-avg W1 | 95% CI |")
    add("|---|---|---|")
    for method, stats in paired.items():
        add(
            f"| {method} | {fmt(stats['mean_difference'])} | "
            f"[{fmt(stats['ci95_low'])}, {fmt(stats['ci95_high'])}] |"
        )
    add("")
    add("## Model selection (validation only)")
    add("")
    add(f"- Local WSINDy selected: `{json.dumps(fit_summary['local']['selection']['selected'])}`")
    add(f"- Kernel WSINDy selected: `{json.dumps(fit_summary['kernel']['selection']['selected'])}`")
    add(f"- Kernel model displacement domain: {fit_summary['kernel']['selection']['displacement_domain']}")
    add("- MVNN checkpoints: best mean autonomous validation rollout W1 (never training loss).")
    add("")
    add("## MVNN training cost and early stopping")
    add("")
    add("| seed | best step | last step | early stopped | wall (s) | compile (s) | GPU util mean/max (%) | peak GPU mem (MiB) |")
    add("|---|---|---|---|---|---|---|---|")
    for seed in protocol["stages"][args.stage]["mvnn_seeds"]:
        c = costs[f"mvnn_seed{seed}"]
        gpu = c["gpu"]
        util = (
            f"{fmt(gpu['gpu_utilization_mean'], 3)}/{fmt(gpu['gpu_utilization_max'], 3)}"
            if gpu.get("gpu_utilization_mean") is not None
            else "n/a"
        )
        mem = fmt(gpu["gpu_memory_max_mib"], 5) if gpu.get("gpu_memory_max_mib") else "n/a"
        add(
            f"| {seed} | {c['best_step']} | {c['last_step']} | {c['early_stopped']} | "
            f"{fmt(c['wall_time_seconds'], 4)} | {fmt(c['compile_seconds'], 3)} | {util} | {mem} |"
        )
    add("")
    add("## Computational cost")
    add("")
    add("Raw wall seconds; hardware reported per row. WSINDy fitting/selection "
        "and every autonomous rollout ran on CPU SLURM nodes; MVNN training ran "
        "on a single NVIDIA A100 GPU. Single-model cost and hyperparameter-"
        "selection cost are reported separately (MVNN used no hyperparameter "
        "search).")
    add("")
    add(f"- Particle-to-density preprocessing (all training trajectories, CPU): "
        f"{fmt(costs['preprocessing_seconds_train_densities'], 4)} s")
    add(f"- Local WSINDy single selected fit (CPU): "
        f"{fmt(costs['local_selected_fit_seconds'], 4)} s; "
        f"full selection over all candidates: "
        f"{fmt(costs['local_selection_total_seconds'], 4)} s")
    add(f"- Kernel WSINDy single selected fit (CPU): "
        f"{fmt(costs['kernel_selected_fit_seconds'], 4)} s; "
        f"full selection over all candidates: "
        f"{fmt(costs['kernel_selection_total_seconds'], 4)} s")
    add("- MVNN training (single A100 GPU, per seed): see the table above; no "
        "hyperparameter search was run.")
    add("- Mean autonomous rollout seconds per trajectory (CPU):")
    for method, value in costs["mean_rollout_seconds_per_trajectory"].items():
        add(f"  - {method}: rollout {fmt(value['rollout_seconds'], 3)} s, "
            f"preprocessing {fmt(value['preprocessing_seconds'], 3)} s")
    add("")
    add("## Sanity check (alpha = 0, sigma = 0; implementation validation only)")
    add("")
    add(
        f"Additive kernel WSINDy on additive ground truth: held-out rollout W1 = "
        f"{fmt(sanity['validation_rollout_w1'])} (gate {sanity['acceptance']['max_validation_time_avg_w1']}), "
        f"training-support drift rel. L2 = {fmt(sanity['train_support_drift_rel_l2_mean'])} "
        f"(gate {sanity['acceptance']['max_drift_rel_l2']}); passed = {sanity['passed']}."
    )
    add("")
    if freeze is not None:
        add("## Freeze discipline")
        add("")
        add(f"- Freeze commit: `{freeze['git_commit']}`")
        add(f"- Protocol sha256 at freeze: `{freeze['protocol_sha256']}`")
        add("- Confirmation data were generated and evaluated exactly once after this freeze.")
        add("")
    add("## Artifacts")
    add("")
    add(f"- Raw metrics: `outputs/threebody/{args.stage}/evaluation/{args.role}/raw_metrics.{{json,csv}}`")
    add(f"- Affinity defects: `{defect_csv.relative_to(ROOT)}`")
    add(f"- Summaries: `outputs/threebody/{args.stage}/summary/`")
    add(f"- Figures: `figures/threebody/main_comparison_{args.stage}_{args.role}.{{pdf,png}}`, "
        f"`figures/threebody/learning_curves_{args.stage}.{{pdf,png}}`")
    add("- Protocol of record: `protocol.yaml`; splits and seeds: "
        f"`outputs/threebody/data/{args.stage}/manifest.json`")
    add("")

    report_path = ROOT / "results_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
