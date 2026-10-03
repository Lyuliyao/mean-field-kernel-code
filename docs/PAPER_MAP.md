# Paper-to-code map

The numbering refers to the supplied final PDF, excluding its unnumbered highlights page. Coverage describes recovered evidence; original plotting files are historical source material, not automatically verified entry points.

| Paper figure(s) | Experiment / saved figure | Source in this repository | Included data |
|---|---|---|---|
| 1 | Training/test learning curve | `src/mtcurve/`, `scripts/mtcurve_*.py`, `scripts/replot.py` | `outputs/mtcurve/models/`, `outputs/mtcurve/summary/`, split manifest |
| 2 | Deterministic Motsch–Tadmor | `experiments/motsch_tadmor/`, `paper/plot_sources/figure.py` | `data/representative/amd20/CASE5/`, referenced checkpoints |
| 3–4 | Matched-data MVNN vs GP, and large-data MVNN | `experiments/second_order_and_baselines/case5/result/` | `data/baseline_predictions/case5/`, saved paper plots; GP fitting code absent |
| 5 | Agent count vs rollout time | `scripts/replot.py`, `experiments/second_order_and_baselines/case5/result/mvnn_gp_time.py` | `data/timing/rollout_times.csv` (aggregate values, no per-trial records found) |
| 6 | Deterministic WSINDy / parameter-estimator comparison | Saved paper plots and source catalog | Original fitting source for these Motsch–Tadmor baselines not recovered |
| 7 | Stochastic Motsch–Tadmor | `experiments/stochastic_motsch_tadmor/` | `data/representative/amd20/CASE5_stochastic/`, referenced checkpoints |
| 8 | Stochastic WSINDy / parameter-estimator comparison | Saved paper plots and source catalog | Original fitting source for these Motsch–Tadmor baselines not recovered |
| 9 | Nonlinear global-mean McKean–Vlasov benchmark | `src/threebody/`, `scripts/threebody_*.py`, `scripts/replot.py` | Frozen parameters, baseline fits, confirmation trajectories, raw metrics, figure cache; online-estimator forecasts / traced vector curves |
| 10–13 | First-order ring / double-ring / disk / binary aggregation | `experiments/aggregation_2d/`, `paper/plot_sources/figure.py` | `data/representative/amd20/CASE4/` and checkpoint |
| 14–15 | Hierarchical three-group system | `experiments/hierarchical/`, `paper/plot_sources/figure.py` | `data/representative/amd20/CASE7/` and checkpoint |
| 16–19 | Second-order attraction–repulsion | `experiments/second_order_and_baselines/case4/result/` | Position/velocity extracts in `data/representative/anvil/case4/`; generation/training code absent |
| 20–21 | Second-order Cucker–Smale | `experiments/second_order_and_baselines/case4_cs/result/` | Position/velocity extracts in `data/representative/anvil/case4_cs/`; generation/training code absent |

## First-order source variants

- `motsch_tadmor/DATA_GENERATION3` is the training-pool source recorded by the Fig. 1 protocol. `TRAINING2`, `TRAINING3`, and other historical variants are retained because rollout scripts refer to different checkpoints. Do not infer the paper's selected checkpoint from the greatest filename.
- `stochastic_motsch_tadmor/TRAINING3` is referenced by `SIM3`; `TRAINING4` is referenced by `SIM4`. Both referenced checkpoints are included.
- `aggregation_2d/TRAINING/model_save_path/jax_ckpt_001000.npz` is the checkpoint referenced by the recovered rollout scripts.
- `hierarchical/SIM3_4` and `TEST_CASE4` use the paper's population sizes `(16000, 4000, 200)`. Historical `SIM3` uses different equal-size input sets; its plotting path is recorded as a discrepancy, not silently substituted.

No new GP, online-estimation, or second-order training implementation was invented to fill the missing sources.
