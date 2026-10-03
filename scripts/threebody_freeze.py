"""Write the freeze marker after protocol, code, and selections are committed.

Usage:
  python scripts/threebody_freeze.py

Requires a clean committed worktree. Records the git commit, the protocol
checksum, and the locked model-selection summaries. After this step the
confirmation set may be generated and evaluated exactly once.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threebody.artifacts import read_json, sha256_file  # noqa: E402
from threebody.freeze import write_freeze  # noqa: E402
from threebody.protocol import load_protocol  # noqa: E402


def main() -> None:
    protocol = load_protocol(ROOT / "protocol.yaml")
    outputs_root = ROOT / "outputs" / "threebody"
    fit_summary_path = outputs_root / "final" / "baselines" / "fit_summary.json"
    fit_summary = read_json(fit_summary_path)
    selection = {
        "local_selected": fit_summary["local"]["selection"]["selected"],
        "kernel_selected": fit_summary["kernel"]["selection"]["selected"],
        "bandwidth": fit_summary["bandwidth"],
        "fit_summary_sha256": sha256_file(fit_summary_path),
        "mvnn_checkpoint_rule": "best mean autonomous validation rollout W1",
        "mvnn_summaries": {},
        # pinned model artifacts: the exact files the confirmation evaluation
        # is allowed to load (verified by freeze.assert_artifacts_match)
        "model_artifacts": {},
    }
    artifact_paths = [
        "final/baselines/local_model.npz",
        "final/baselines/kernel_model.npz",
    ]
    for seed in protocol["stages"]["final"]["mvnn_seeds"]:
        summary_path = (
            outputs_root / "final" / "mvnn" / f"seed_{seed}" / "training_summary.json"
        )
        summary = read_json(summary_path)
        selection["mvnn_summaries"][str(seed)] = {
            "best_step": summary["best_step"],
            "best_rollout_w1": summary["best_rollout_w1"],
            "sha256": sha256_file(summary_path),
        }
        artifact_paths.append(f"final/mvnn/seed_{seed}/best_params.msgpack")
    for relative_path in artifact_paths:
        selection["model_artifacts"][relative_path] = sha256_file(
            outputs_root / relative_path
        )
    marker = write_freeze(ROOT, outputs_root, selection)
    print(f"freeze marker written: {marker}")


if __name__ == "__main__":
    main()
