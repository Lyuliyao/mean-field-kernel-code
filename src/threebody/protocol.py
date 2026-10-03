"""Protocol loading and ground-truth parameter access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "protocol.yaml"


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    protocol_path = Path(path) if path is not None else PROTOCOL_PATH
    with open(protocol_path, "r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported protocol schema_version")
    protocol["_protocol_path"] = str(protocol_path)
    return protocol


@dataclass(frozen=True)
class GroundTruth:
    a: float
    alpha: float
    beta: float
    gamma: float
    sigma: float
    dt: float
    n_frames: int
    domain: tuple[float, float]

    @property
    def final_time(self) -> float:
        return self.dt * (self.n_frames - 1)


def ground_truth_params(protocol: dict[str, Any]) -> GroundTruth:
    section = protocol["ground_truth"]
    return GroundTruth(
        a=float(section["a"]),
        alpha=float(section["alpha"]),
        beta=float(section["beta"]),
        gamma=float(section["gamma"]),
        sigma=float(section["sigma"]),
        dt=float(section["dt"]),
        n_frames=int(section["n_frames"]),
        domain=(float(section["domain"][0]), float(section["domain"][1])),
    )


def sanity_alpha0_params(protocol: dict[str, Any]) -> GroundTruth:
    base = protocol["ground_truth"]
    section = protocol["sanity_check_alpha0"]
    return GroundTruth(
        a=float(section["a"]),
        alpha=float(section["alpha"]),
        beta=float(section["beta"]),
        gamma=float(section["gamma"]),
        sigma=float(section["sigma"]),
        dt=float(base["dt"]),
        n_frames=int(base["n_frames"]),
        domain=(float(base["domain"][0]), float(base["domain"][1])),
    )
