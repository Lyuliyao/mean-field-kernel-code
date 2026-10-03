"""Atomic artifact writes, checksums, and immutable manifests (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: str | Path, value: Any, *, immutable: bool = True) -> None:
    """JSON writer; refuses to change an existing immutable manifest."""

    path = Path(path)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == rendered:
            return
        if immutable:
            raise RuntimeError(f"Refusing to overwrite immutable manifest {path}")
    atomic_write_text(path, rendered)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def completed_summary(final_dir: str | Path, marker: str) -> dict[str, Any] | None:
    final_dir = Path(final_dir)
    marker_path = final_dir / marker
    if final_dir.exists():
        if marker_path.exists():
            return read_json(marker_path)
        raise RuntimeError(
            f"{final_dir} exists without its completion marker {marker}; "
            "remove the partial directory deliberately before rerunning"
        )
    return None


def attempt_directory(final_dir: str | Path) -> Path:
    final_dir = Path(final_dir)
    parts = [
        os.environ.get("SLURM_JOB_ID", ""),
        os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        str(os.getpid()),
    ]
    suffix = "_".join(part for part in parts if part)
    attempt = final_dir.parent / f".{final_dir.name}.attempt_{suffix}"
    if attempt.exists():
        raise FileExistsError(f"attempt directory {attempt} already exists")
    attempt.mkdir(parents=True)
    return attempt


def publish_attempt(attempt_dir: str | Path, final_dir: str | Path, marker: str) -> None:
    attempt_dir = Path(attempt_dir)
    final_dir = Path(final_dir)
    if not (attempt_dir / marker).exists():
        raise RuntimeError(f"attempt {attempt_dir} is missing its marker {marker}")
    if final_dir.exists():
        raise FileExistsError(f"refusing to replace existing final dir {final_dir}")
    os.replace(attempt_dir, final_dir)
