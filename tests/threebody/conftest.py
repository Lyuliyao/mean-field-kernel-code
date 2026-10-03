import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from threebody.observation import Grid  # noqa: E402
from threebody.protocol import GroundTruth, load_protocol  # noqa: E402


@pytest.fixture(scope="session")
def protocol():
    return load_protocol(REPO_ROOT / "protocol.yaml")


@pytest.fixture(scope="session")
def ground_truth(protocol):
    from threebody.protocol import ground_truth_params

    return ground_truth_params(protocol)


@pytest.fixture(scope="session")
def small_params():
    """Reduced ground truth for fast tests."""

    return GroundTruth(
        a=1.0,
        alpha=1.0,
        beta=0.2,
        gamma=0.2,
        sigma=0.03,
        dt=0.01,
        n_frames=21,
        domain=(-4.0, 4.0),
    )


@pytest.fixture(scope="session")
def coarse_grid():
    return Grid(-4.0, 4.0, 128)
