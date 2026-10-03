# Reproduction recipes and limits

## Saved-result plotting

```bash
python -m pip install -e .
python scripts/verify_repository.py
python scripts/replot.py --figures 1 5 9 --output-dir runs/figures
python scripts/plot_snapshots.py --case aggregation_2d --scenario ring
python scripts/plot_snapshots.py --case second_order_ar --scenario binary --quantity velocity
```

The quick plotting commands use saved statistics, caches or explicitly sampled arrays. They require no JAX, GPU, MPI, TeX, cluster account, or remote paths. Plotting Fig. 9 uses the cached confirmation results and does not reopen the test set for model selection.

## Model environment and checks

For the reference CPU stack on Linux or Apple Silicon:

```bash
python -m pip install -r requirements-reference-cpu.txt
python -m pip install -e '.[test]'
python -m pytest -q tests/threebody
python scripts/smoke_models.py
```

On macOS Intel use `requirements-macos-intel.txt`: the reference JAX 0.5.3 does not have a matching Intel macOS jaxlib wheel; JAX 0.4.38 is the available Intel wheel. On Apple Silicon use a native ARM Python environment; an Intel Python under Rosetta may fail the jaxlib AVX check. `environment.yml` records the original Python 3.10 / JAX 0.5.3 / CUDA 12 environment. For the legacy MPI generators install `.[mpi]` and provide a working MPI runtime. The original generator/training settings remain in their source files.

Formal generation, fitting and large rollouts belong on compute nodes, not login nodes. Original cluster launchers were omitted because their accounts and environment paths were specific to old runs. First-order scripts can be invoked from their own subdirectory, for example:

```bash
cd experiments/aggregation_2d/DATA_GENERATION
mpiexec -n 4 python simulation.py 1
# Repeat the recorded seeds to populate the training directory.
cd ../TRAINING
python train.py
cd ../SIM
python simulation.py 1
```

These are research scripts, not a general-purpose training CLI: inspect directory relationships and recorded settings before starting a full run. Archived selected checkpoints are already present. Create a source-only run directory using `python scripts/prepare_training_run.py runs/retrain` and work inside it when retraining; this preserves the measured outputs. Initialize and commit a separate Git repository in the run directory before using the final freeze/confirmation workflow.

## Training-size study

`protocol_mtcurve.yaml` fixes N=16000, dt=0.01, 201 frames, M in 5/10/20/40/60/80/100, three model replicates, 10 validation trajectories and 30 held-out test trajectories. The original sampler and blocked deterministic simulator are in `src/mtcurve/simulate.py`.

```bash
# Run in a source-only working directory / compute allocation.
python scripts/prepare_training_run.py runs/retrain
cd runs/retrain
python scripts/generate_mt_pool.py --seeds 1:100
export MVNN_MT_POOL="$PWD/data/raw/mt_training_pool"
python scripts/mtcurve_generate_data.py
python scripts/mtcurve_drift_truth.py --help
python scripts/mtcurve_train_eval.py --help
```

Regenerating a pool with 201 frames reproduces the time slice used by the learning curve but does not reproduce the hash of an original file containing more frames. The recorded port verification matched original seed 1 to approximately 7.8e-16 on the used slice. Archived models and metrics are sufficient to redraw Fig. 1 without regenerating the large pool. Original training summaries record validation-based checkpoint selection; training loss alone was not the selection rule.

## Nonlinear global-mean experiment

`protocol.yaml` specifies the ground truth, staged sample sizes, observation operator, baselines, budgets, validation selection and confirmation discipline. `src/threebody/` contains the recovered full implementation. Entry points expose their options with `--help`:

```bash
python scripts/threebody_generate_data.py --help
python scripts/threebody_train_mvnn.py --help
python scripts/threebody_fit_baselines.py --help
python scripts/threebody_freeze.py --help
python scripts/threebody_evaluate.py --help
```

The archived `FREEZE.json` refers to the original experiment commit and frozen artifact hashes. It is provenance for those results, not the new archive's Git commit. Run a new freeze/confirmation workflow for a genuinely new experiment rather than modifying the old marker. The original training/validation data can be regenerated from the locked protocol; the actual 36 confirmation trajectories are included and verified against the source manifest.

## Gaps and historical inconsistencies

1. Second-order attraction–repulsion and Cucker–Smale training/generation code was not found in the supplied roots. Only plotting sources and stored trajectories are available; the archive does not claim end-to-end retraining for these models.
2. GP and online parameter-estimator fitting code was not found. Saved predictions/curves and aggregate timing values are included. Per-trial timing samples for the ten-trial statement were not located, so the timing utility redraws recorded averages only.
3. Several second-order plotting paths are stale (`sim_second_order_1` versus the unsuffixed directory, and `test_data_sim_second_order` versus `test_data_second_order`). The snapshot loader uses the existing directory and records its actual source.
4. The manuscript copy of `figure.py` reads the prediction file for both reference and prediction in its binary second-order panels. The original source is preserved under `paper/plot_sources/`; the snapshot utility reads distinct reference/prediction files. It is not claimed to be a numerically identical re-rendering of the published binary panel.
5. Some old first-order plots map the full stored time range to `[0,2]` even though generators save longer or slightly shorter trajectories. Frame indices are retained in the extract manifests instead of silently rewriting the scientific time axis.
6. The old hierarchical plotting script refers to `SIM3/TEST_CASE3`; the requested population sizes are represented by `SIM3_4/TEST_CASE4`. Both source variants remain inspectable; the compact data package uses the latter and makes no claim that the historical script and all published panels have been numerically reconciled.
7. The legacy `motsch_tadmor/TEST_DATA/simulation.py` contained three accidentally embedded shell commands. They are commented in the archived Python copy, and that transformation is recorded in the source manifest. The canonical training-pool source is `DATA_GENERATION3`.
8. First-order ring initialization in the recovered generator uses Gaussian radial noise. This differs from the uniform annulus notation in the paper. The source sampler is retained; no new sampler was substituted.

Validation performed for this archive is recorded in `docs/VALIDATION.md`. It does not substitute for retraining all large experiments.
