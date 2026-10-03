# Archive validation

Validated on 2026-10-03. This records checks of the curated archive, not a new full training run.

| Check | Result |
|---|---|
| Independent source SHA-256 comparison | 429 copied amd20 files and 57 copied Anvil files matched fresh server hashes; local PDF is included unchanged |
| Frozen experiment artifacts | Five pinned model artifacts and all 36 confirmation trajectories matched their original manifests |
| Representative trajectory extracts | All 56 arrays matched the remote extraction hashes; original shapes, frame indices and particle counts are retained |
| Python syntax | Every archived Python file parsed successfully |
| Scientific tests without JAX-dependent training guards | 28 passed in 2.64 seconds (`pytest -q tests/threebody --ignore=tests/threebody/test_training_guards.py`) |
| Package installation | Editable installation succeeded with Python 3.12 |
| Saved-result plotting | Figures 1, 5 and 9 rendered; first-order ring, hierarchical and second-order position/velocity example plots rendered |
| Small-scale generation | Deterministic seeds 1–2, N=32, 3 steps produced indexed arrays and SHA-256 receipts |
| Independent run preparation | Source-only copy succeeded and omitted archived model directories |
| Raw-data catalog utility | Listing relevant remote files succeeded; bulk raw-data retrieval was not performed |

The plotted Figure 1, Figure 9, first-order ring and second-order binary-velocity outputs were visually inspected. Replotting saved values does not establish a fresh reproduction of the training or runtime measurements.

JAX checkpoint-loading / equivariance checks and the JAX-dependent training-guard tests are provided in `scripts/smoke_models.py` and `tests/threebody/test_training_guards.py`, but were not completed in this archive preparation. The local Intel Python ran under Rosetta on Apple Silicon and failed jaxlib's AVX check; a native-environment dependency download timed out. A lightweight CPU check in a temporary copy on amd20 was interrupted by an SSH disconnect. These checks must therefore not be counted as passed. Use a native ARM Python or the reference Linux CPU stack for them.

Full-scale training, generation, pooled trajectory reproduction and fresh ten-trial timing measurements were not run. Missing baseline and second-order sources and historical plot discrepancies are documented in `REPRODUCTION.md` and `PAPER_MAP.md`.

`provenance/files.json` records source paths, original hashes and the few explicit transformations (notebook output removal, updated JAX imports, a configurable training-pool location, and commenting accidentally embedded shell commands). `provenance/transfer_validation.json` records independent source comparisons. `provenance/repository_checksums.json` covers the distributed files; run `python scripts/verify_repository.py` to check them.
