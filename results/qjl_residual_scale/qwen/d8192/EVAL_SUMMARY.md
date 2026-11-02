# Qwen residual-scale sweep: ARC evaluation

Model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`; QJL dimension 8192; Triton backend. Each checkpoint was evaluated zero-shot with batch size 16 and no dataset limit. **This run evaluated only `arc_challenge` and `arc_easy`**, because the sweep default task list was changed to those two tasks before evaluation. The originally planned eight-task accuracy is not available from this run.

| Alpha | ARC-C acc | ARC-C acc_norm | ARC-E acc | ARC-E acc_norm | Mean acc | Held-out MoE relative L2 error | Checkpoint size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 39.76% | 40.02% | 70.58% | 68.43% | 55.17% | 0.0000 | 29.07 GB |
| 0.25 | 38.31% | 39.85% | 70.16% | 68.10% | 54.24% | 0.0758 | 29.07 GB |
| 0.5 | 37.03% | 39.08% | 69.19% | 65.45% | 53.11% | 0.1555 | 29.07 GB |
| 1 | 33.36% | 35.15% | 61.62% | 58.46% | 47.49% | 0.3430 | 16.13 GB |

The mean uses `acc`, with equal weight for the two tasks. Relative to the alpha-0 identity control, the mean drops by 0.94, 2.06, and 7.68 percentage points at alpha 0.25, 0.5, and 1. The residual Frobenius ratio rises from 0 through 0.248, 0.496, and 0.992 across the same settings. Alpha 1 has a checkpoint 44.5% smaller than alpha 0; alpha 0.25 and 0.5 have approximately the same checkpoint size as alpha 0.

Sources: `sweep_summary.csv`, `sweep_summary.json`, and each `alpha_*/eval/metrics.json` and `lm_eval.txt`. All four evaluation records have status `evaluated`. The large `qjl_state.pt` checkpoints and shared scale cache remain on the server and are not included in Git.
