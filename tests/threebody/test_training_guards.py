"""Early stopping, best-checkpoint restoration, and freeze-guard tests."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from threebody.training import EarlyStopper

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_early_stopper_stops_after_patience_without_improvement():
    stopper = EarlyStopper(patience=5, min_relative=0.005)
    assert not stopper.update(1000, 1.0)
    # five evaluations with < 0.5% relative improvement trigger the stop
    values = [0.999, 0.998, 0.997, 0.9965, 0.996]
    stops = [stopper.update(2000 + 1000 * i, v) for i, v in enumerate(values)]
    assert stops == [False, False, False, False, True]
    # the incumbent best still tracked the small improvements
    assert stopper.best_value == 0.996
    assert stopper.best_step == 6000


def test_early_stopper_resets_on_real_improvement():
    stopper = EarlyStopper(patience=3, min_relative=0.005)
    stopper.update(1000, 1.0)
    stopper.update(2000, 0.999)
    stopper.update(3000, 0.9985)
    assert not stopper.update(4000, 0.90)  # > 0.5% improvement resets patience
    assert stopper.evals_since_improvement == 0
    assert stopper.best_step == 4000


def test_early_stopper_rejects_nonfinite():
    stopper = EarlyStopper(patience=2, min_relative=0.005)
    with pytest.raises(FloatingPointError):
        stopper.update(1000, float("nan"))


@pytest.mark.slow
def test_best_checkpoint_restoration(tmp_path, small_params, coarse_grid):
    """Training saves best+last; the best checkpoint reproduces its metric."""

    jax = pytest.importorskip("jax")
    from threebody import triplets as tp
    from threebody.mvnn import MVNNConfig
    from threebody.simulate import simulate
    from threebody.training import build_training_data, load_best_params, train_mvnn

    families = tp.sample_families(2, 128, stage_seed_base=555)
    trajectories = []
    for family in families:
        triplet = tp.build_triplet(family)
        trajectories.append(simulate(triplet["plus"], small_params, brownian_seed=1))
    data = build_training_data(
        train_trajectories=[trajectories[0]],
        val_trajectories=[trajectories[1]],
        val_noise_seeds=[7],
        dt=small_params.dt,
        grid=coarse_grid,
        bandwidth=0.1,
    )
    config = MVNNConfig(embedding_hidden=(8,), embedding_out=4, interaction_hidden=(8,))
    summary = train_mvnn(
        data,
        tmp_path,
        model_seed=0,
        dt=small_params.dt,
        sigma=small_params.sigma,
        grid=coarse_grid,
        bandwidth=0.1,
        max_steps=60,
        rollout_every=20,
        onestep_every=20,
        batch_frames=4,
        config=config,
    )
    assert (tmp_path / "best_params.msgpack").exists()
    assert (tmp_path / "last_params.msgpack").exists()
    assert summary["best_step"] is not None
    assert summary["checkpoints"] == ["best_params.msgpack", "last_params.msgpack"]
    model, params = load_best_params(tmp_path, config)
    drift = model.drift_at(
        params,
        np.zeros((3, 1)),
        np.asarray(trajectories[0][0])[:, None],
    )
    assert np.all(np.isfinite(np.asarray(drift)))
    summary_on_disk = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary_on_disk["best_step"] == summary["best_step"]


def test_no_confirmation_access_in_selection_code():
    """Static guard: fitting/training/selection modules never touch confirmation data."""

    forbidden = "confirmation"
    for module in ("training", "baselines", "kernel_wsindy", "local_wsindy", "regression"):
        source = (REPO_ROOT / "src" / "threebody" / f"{module}.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert forbidden not in node.value.lower(), (
                    f"{module}.py references confirmation data in a string constant"
                )
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                assert forbidden not in name.lower(), (
                    f"{module}.py references confirmation data in identifier {name}"
                )


def test_confirmation_guard_requires_freeze(tmp_path):
    from threebody.freeze import assert_frozen

    with pytest.raises(RuntimeError, match="freeze marker does not exist"):
        assert_frozen(REPO_ROOT, tmp_path)


def test_freeze_artifact_pinning(tmp_path):
    """assert_artifacts_match accepts identical files and rejects any change."""

    from threebody.artifacts import sha256_file
    from threebody.freeze import assert_artifacts_match

    artifact = tmp_path / "final" / "baselines" / "kernel_model.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"locked-model-bytes")
    marker = {
        "selection_summary": {
            "model_artifacts": {
                "final/baselines/kernel_model.npz": sha256_file(artifact)
            }
        }
    }
    assert_artifacts_match(marker, tmp_path)  # identical -> passes

    artifact.write_bytes(b"tampered-model-bytes")
    with pytest.raises(RuntimeError, match="changed after the"):
        assert_artifacts_match(marker, tmp_path)

    artifact.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        assert_artifacts_match(marker, tmp_path)

    with pytest.raises(RuntimeError, match="does not pin"):
        assert_artifacts_match({"selection_summary": {}}, tmp_path)
