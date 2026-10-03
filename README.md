# Mean-field kernel / MVNN

Code and curated data accompanying **MVNN: A Measure-Valued Neural Network for Learning McKean-Vlasov Dynamics from Particle Data**, by Liyao Lyu, Xinyue Yu, and Hayden Schaeffer.

[Figure-to-code map](docs/PAPER_MAP.md) · [Data](docs/DATA.md) · [Reproduction and limitations](docs/REPRODUCTION.md)

This private repository collects the implementations and results located in the authors' supplied amd20 and Anvil directories. It includes the first-order MVNN experiments, the training-size study, the nonlinear global-mean benchmark, recovered second-order plotting scripts, trained parameters, measured results, and selected reference/prediction data. See the coverage table below before treating it as a complete end-to-end reproduction of every experiment.

## Quick start: redraw saved results

Python 3.10–3.12 is recommended. Redrawing saved results does not require a GPU, MPI, a TeX installation, or access to the original clusters.

```bash
git clone git@github.com:Lyuliyao/mean-field-kernel-code.git
cd mean-field-kernel-code
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/verify_repository.py
python scripts/replot.py --figures 1 5 9 --output-dir runs/figures
python scripts/plot_snapshots.py --case aggregation_2d --scenario ring
```

Figure 1 uses the saved seven-size, three-replicate learning-curve statistics. Figure 5 uses the recorded aggregate timing values. Figure 9 uses the frozen confirmation cache and the archived online-estimator vector curves. These commands redraw measured results; they do not retrain the models or remeasure runtimes. Fonts and layout may differ from the manuscript's TeX-rendered plots.

## What is included

| Experiment | Implementation | Data/results included | Coverage |
|---|---|---|---|
| Deterministic Motsch–Tadmor | Generation, MVNN training, rollout | Checkpoints, representative snapshots | Source implementation recovered |
| Training-size study, Fig. 1 | Generation, training, evaluation, aggregation | 21 models, per-trajectory metrics, fixed split/seed manifest | Saved numerical results can be redrawn |
| Stochastic Motsch–Tadmor | Generation, MVNN training, rollout | Checkpoints, representative snapshots | Source implementation recovered |
| Nonlinear McKean–Vlasov benchmark, Fig. 9 | MVNN, local PDE-WSINDy, B-spline kernel WSINDy | Frozen models, 36 confirmation trajectories, cache, raw metrics | MVNN/WSINDy implementation recovered; online-estimator code missing |
| First-order 2D aggregation | Generation, training, rollout | Checkpoint, ring/double-ring/disk/binary snapshots | Source implementation recovered |
| Hierarchical three-group system | Generation, training, rollout | Referenced checkpoint and snapshots at population sizes 16000/4000/200 | Source implementation recovered; historical plotting paths differ |
| Second-order attraction–repulsion and Cucker–Smale | Plotting scripts | Position/velocity snapshots and source data catalog | Training and generation code not located |
| GP and online parameter-estimation baselines | Recovered plotting scripts | Stored forecasts, aggregate timing values, archived curves | Original fitting implementations not located |

## Layout

```text
src/mtcurve/                 training-size experiment
src/threebody/               nonlinear global-mean experiment and WSINDy baselines
scripts/                    redraw, validate, generate, train, fit, evaluate
experiments/                 recovered first-order and second-order source trees
outputs/mtcurve/              archived models, metrics and seed/split manifest
outputs/threebody/            frozen models, metrics and confirmation trajectories
data/representative/          small, explicitly indexed extracts of original arrays
data/baseline_predictions/    recovered GP / MVNN / online-estimator trajectories
paper/                       supplied PDF, saved manuscript plots and historical plot sources
provenance/                  source paths, SHA-256 records and full raw-data catalog
tests/threebody/              recovered scientific implementation tests
docs/                        recipes, source mapping and unresolved gaps
```

Only necessary saved results and compact examples are tracked. Full training pools, pooled multi-gigabyte trajectories, intermediate checkpoints, animation frames, environments, and unrelated data-assimilation projects remain outside this repository. Their relevant source locations are recorded in the raw-data catalog. Synthetic training data can be regenerated using the recovered generators; data hashes distinguish regenerated arrays from the original artifacts.

For training and model evaluation, install the appropriate environment described in [docs/REPRODUCTION.md](docs/REPRODUCTION.md). The original Linux GPU environment is recorded separately in `environment.yml`; it is not required for plotting.

No open-source license has been selected for this private research archive. Citation metadata is provided in `CITATION.cff`.
