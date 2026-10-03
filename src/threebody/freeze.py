"""Freeze marker and confirmation-set access guard.

The confirmation set may be generated and evaluated only after the protocol,
model selection rules, and code are frozen and committed. The marker records
the git commit and the protocol checksum; the confirmation entry point
verifies both and refuses to run twice.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .artifacts import read_json, sha256_file, write_json

FREEZE_MARKER = "FREEZE.json"


def git_head(repo_root: Path) -> str:
    return (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def git_worktree_clean(repo_root: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return status == ""


def write_freeze(
    repo_root: Path,
    outputs_root: Path,
    selection_summary: dict[str, Any],
) -> Path:
    repo_root = Path(repo_root)
    marker_path = Path(outputs_root) / FREEZE_MARKER
    if marker_path.exists():
        raise RuntimeError("freeze marker already exists; refusing to re-freeze")
    if not git_worktree_clean(repo_root):
        raise RuntimeError("worktree must be clean (all code committed) to freeze")
    payload = {
        "git_commit": git_head(repo_root),
        "protocol_sha256": sha256_file(repo_root / "protocol.yaml"),
        "selection_summary": selection_summary,
    }
    write_json(marker_path, payload, immutable=True)
    return marker_path


def assert_frozen(repo_root: Path, outputs_root: Path) -> dict[str, Any]:
    marker_path = Path(outputs_root) / FREEZE_MARKER
    if not marker_path.exists():
        raise RuntimeError(
            "confirmation access denied: freeze marker does not exist; "
            "commit the protocol and run the freeze step first"
        )
    marker = read_json(marker_path)
    protocol_sha = sha256_file(Path(repo_root) / "protocol.yaml")
    if marker["protocol_sha256"] != protocol_sha:
        raise RuntimeError(
            "confirmation access denied: protocol.yaml changed after the freeze"
        )
    head = git_head(Path(repo_root))
    if marker["git_commit"] != head:
        raise RuntimeError(
            f"confirmation access denied: HEAD {head} differs from the freeze "
            f"commit {marker['git_commit']}"
        )
    return marker


def assert_artifacts_match(marker: dict[str, Any], outputs_root: Path) -> None:
    """Verify every model artifact pinned at freeze time is byte-identical.

    The freeze marker records SHA-256 digests of the locked MVNN checkpoints
    and both WSINDy model files; confirmation evaluation refuses to run
    against any modified or missing artifact.
    """

    artifacts = marker.get("selection_summary", {}).get("model_artifacts")
    if not artifacts:
        raise RuntimeError(
            "confirmation access denied: freeze marker does not pin model artifacts"
        )
    outputs_root = Path(outputs_root)
    for relative_path, expected_sha in artifacts.items():
        path = outputs_root / relative_path
        if not path.exists():
            raise RuntimeError(
                f"confirmation access denied: pinned artifact {relative_path} is missing"
            )
        actual = sha256_file(path)
        if actual != expected_sha:
            raise RuntimeError(
                f"confirmation access denied: {relative_path} changed after the "
                f"freeze (sha256 {actual} != {expected_sha})"
            )
