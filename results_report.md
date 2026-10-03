# Results Report: Three-Body Global-Mean Experiment

Measured results only. Ground truth: dX = [gamma (M - X) + (a - alpha M^2) X - beta X^3] dt + sigma dW with a=1.0, alpha=1.0, beta=0.2, gamma=0.2, sigma=0.03. The three-body term -alpha X M^2 cannot be represented by any additive pairwise model f(x) + int K(y-x) mu(dy): on matched mixture triplets (mu+, mu-, mu0=(mu+ + mu-)/2 with means +/-d and 0) the true drift violates mixture affinity by exactly alpha*x*d^2 while every additive model has zero affinity defect.

Stage: `final`; evaluation set: `confirmation` (12 untouched matched triplets). KDE bandwidth 0.05 on a 512-cell grid over [-4.0, 4.0]; identical observation operator for all methods.

## Primary comparison (per-triplet means, bootstrap 95% CIs)

| method | time-avg W1 | 95% CI | final W1 | direction acc. | drift RMSE (t=0) | affinity defect vs truth (rel. err.) |
|---|---|---|---|---|---|---|
| kernel_wsindy | 0.09129 | [0.07619, 0.1078] | 0.1305 | 50% | 0.6688 | 1 |
| local_wsindy | 0.2661 | [0.2427, 0.2852] | 0.4673 | 44.4% | 0.7345 | 1 |
| mvnn_seed0 | 0.004811 | [0.004286, 0.005338] | 0.007903 | 100% | 0.04316 | 0.0376 |
| mvnn_seed1 | 0.004677 | [0.003706, 0.005728] | 0.006196 | 100% | 0.05529 | 0.0437 |
| mvnn_seed2 | 0.006271 | [0.005389, 0.007112] | 0.009834 | 100% | 0.07237 | 0.0971 |

Direction accuracy = fraction of triplet/measure pairs whose central-bump short-time motion direction matches the reference. Affinity defect relative error = ||defect_model - alpha*x*d^2|| / ||alpha*x*d^2|| on the central window; 1.0 means the model shows no three-body defect at all (additive), 0 means the defect is captured exactly.

Same-condition defect evaluation: the stored particle triplets satisfy mu_mixed = (mu_plus + mu_minus)/2 exactly at the multiset level, so every method is probed with the same empirical objects (density models via the shared KDE of each particle vector, MVNN via the particle vectors). Measured max |rho_mixed - (rho_plus + rho_minus)/2| over triplets: 4.44e-16 (floating-point roundoff of the linear KDE).

## Paired differences vs MVNN (paired bootstrap across triplets)

| baseline | mean(baseline - MVNN) time-avg W1 | 95% CI |
|---|---|---|
| local_wsindy | 0.2608 | [0.2377, 0.2797] |
| kernel_wsindy | 0.08604 | [0.07124, 0.1022] |

## Model selection (validation only)

- Local WSINDy selected: `{"basis_count": 16, "smoothness": 1e-08, "threshold": 0.0, "validation_rollout_w1": 0.2671977815955525}`
- Kernel WSINDy selected: `{"basis_count": 24, "smoothness": 1e-08, "validation_rollout_w1": 0.08519048592753128}`
- Kernel model displacement domain: [-4.9071153450854315, 4.9071153450854315]
- MVNN checkpoints: best mean autonomous validation rollout W1 (never training loss).

## MVNN training cost and early stopping

| seed | best step | last step | early stopped | wall (s) | compile (s) | GPU util mean/max (%) | peak GPU mem (MiB) |
|---|---|---|---|---|---|---|---|
| 0 | 11000 | 12000 | False | 21.25 | 1.42 | 44.2/47 | 2655 |
| 1 | 4000 | 9000 | True | 16.73 | 1.38 | 44.7/46 | 2655 |
| 2 | 3000 | 8000 | True | 14.59 | 1.41 | 46/47 | 2655 |

## Computational cost

Raw wall seconds; hardware reported per row. WSINDy fitting/selection and every autonomous rollout ran on CPU SLURM nodes; MVNN training ran on a single NVIDIA A100 GPU. Single-model cost and hyperparameter-selection cost are reported separately (MVNN used no hyperparameter search).

- Particle-to-density preprocessing (all training trajectories, CPU): 0.7086 s
- Local WSINDy single selected fit (CPU): 23.58 s; full selection over all candidates: 611.2 s
- Kernel WSINDy single selected fit (CPU): 23.47 s; full selection over all candidates: 245.8 s
- MVNN training (single A100 GPU, per seed): see the table above; no hyperparameter search was run.
- Mean autonomous rollout seconds per trajectory (CPU):
  - kernel_wsindy: rollout 0.327 s, preprocessing 0.00167 s
  - local_wsindy: rollout 0.365 s, preprocessing 0.00164 s
  - mvnn_seed0: rollout 0.432 s, preprocessing 0.00332 s
  - mvnn_seed1: rollout 0.431 s, preprocessing 0.00339 s
  - mvnn_seed2: rollout 0.418 s, preprocessing 0.00336 s

## Sanity check (alpha = 0, sigma = 0; implementation validation only)

Additive kernel WSINDy on additive ground truth: held-out rollout W1 = 0.01132 (gate 0.02), training-support drift rel. L2 = 0.07069 (gate 0.12); passed = True.

## Freeze discipline

- Freeze commit: `dfc366947f05a7a0f0049d8cf1e3b90aa5b662d9`
- Protocol sha256 at freeze: `e0946cdb9bf08a933f24e329a8a009dfa5969c4f0854da9dd585b521c0f5b26c`
- Confirmation data were generated and evaluated exactly once after this freeze.

## Artifacts

- Raw metrics: `outputs/threebody/final/evaluation/confirmation/raw_metrics.{json,csv}`
- Affinity defects: `outputs/threebody/final/evaluation/confirmation/affinity_defects.csv`
- Summaries: `outputs/threebody/final/summary/`
- Figures: `figures/threebody/main_comparison_final_confirmation.{pdf,png}`, `figures/threebody/learning_curves_final.{pdf,png}`
- Protocol of record: `protocol.yaml`; splits and seeds: `outputs/threebody/data/final/manifest.json`
