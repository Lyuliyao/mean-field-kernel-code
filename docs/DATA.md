# Data included and data retained on the clusters

## Included artifacts

- The supplied final PDF and the recovered manuscript plot PDFs.
- Fig. 1: all 21 best MVNN parameter files, their training/evaluation summaries and per-trajectory metrics, aggregate CSVs, and the original seed/split manifest.
- Fig. 9: three frozen MVNN parameter sets, the two WSINDy fits, confirmation metrics, 36 reference trajectories, and the cache used to build the manuscript plot. The online estimator's saved trajectories are recovered separately from Tracy's directory. Its plotted vector paths are archived in `paper/plot_sources/compare_wsindy_src/purple_curves.npz`; those paths were recovered from a prior plot and are not a substitute for raw fitting output.
- Checkpoints explicitly referenced by the recovered first-order rollout scripts, plus a small number of latest archived checkpoints for context.
- GP/MVNN prediction arrays and online-estimator forecasts recovered from Anvil.
- Representative trajectory extracts from the four first-order families and the two second-order families.

## Representative extracts

Every extract has an entry in `data/representative/<host>/manifest.json` containing the original path, shape, dtype, size, selected frame indices, selected shape, and SHA-256. Values are copied without rounding or dtype conversion. Where the original file pools multiple simulations, only the first contiguous 16000 particles are retained, so these extracts are examples rather than full pooled data.

- Deterministic/stochastic Motsch–Tadmor: source frames `[0, 100, 200]`, 16000 particles, seeds 1/2/3, reference and MVNN.
- First-order 2D aggregation: source frames `[0, 100, last]`, up to 16000 particles, four initial geometries, reference and MVNN.
- Three-group system: frames `[0, 250, last]`, seeds 1/2, groups of 16000/4000/200 particles, reference and MVNN.
- Second-order systems: frames `[0, 100, last]`, first 16000 particles, both position and velocity arrays, reference and MVNN.

The frame indices are authoritative. Some old scripts label the final stored frame as `t=2` even when array lengths and generator settings differ. The snapshot utility labels source frames rather than asserting unverified physical times. Compact extracts cannot reproduce the original full-time error curves or the exact KDE bandwidth of an entire pooled array.

## Large raw data

`provenance/raw_data_catalog.json` records relevant files across the four supplied roots. Duplicate copies, intermediate checkpoints, raw training pools, and pooled trajectories are cataloged but not tracked in Git. Examples of pooled second-order files exceed 5 GB each. Data-assimilation projects, exploratory symmetrized models, videos and rendered frame collections are outside the paper's selected scope.

Use `python scripts/fetch_raw_data.py --list --contains CASE5/DATA_GENERATION3` to inspect matching files. With cluster access, `--index INDEX --fetch` downloads one original file to `data/raw/`, checks the source size and remote/local SHA-256, and writes a receipt. This is an optional retrieval tool, not a public data-hosting service. A collaborator without cluster access can use the tracked results and compact examples; full retraining may require regenerating synthetic data or separately sharing the large pools.

## Integrity and provenance

`provenance/files.json` maps copied files to their source and SHA-256. Notebook outputs and execution counts were cleared; compatibility fixes replace deprecated `jax.config` import paths, and the Fig. 1 training-pool location is configurable using `MVNN_MT_POOL`. These transformations are recorded separately from original hashes. The scientific formulas and trained parameters are unchanged.

`provenance/repository_checksums.json` covers all tracked deliverables. `python scripts/verify_repository.py` validates those hashes and the per-extract manifest hashes. The frozen Fig. 9 model hashes are also checked against `outputs/threebody/FREEZE.json`.
