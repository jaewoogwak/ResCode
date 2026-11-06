# Row-wise residual element outliers before QJL

Scope: **full 24-layer analysis**; layers 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23.
Group/saliency cache: `/home/jaewoo/ResCode/results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt`. Original BF16 model: `/root/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be`.
R = FP32(W) − FP32(BF16 ResCode saliency base), delta=0.001.
Random seed: 2026. No QJL projection or quantization was applied.
Each matrix row is an output row, not a token. Norm and magnitude statistics cover every row; only Jaccard row pairs and plotted rows are sampled.

## Element concentration

| Projection | Residual row max/P99 median | W row max/P99 median | Residual top-1 squared norm removed | Random top-1 | Residual top-16 | Random top-16 |
|---|---:|---:|---:|---:|---:|---:|
| gate | 1.40 | 1.40 | 0.65% | 0.05% | 6.95% | 0.78% |
| up | 1.40 | 1.40 | 0.66% | 0.05% | 6.97% | 0.78% |
| down | 1.36 | 1.36 | 0.91% | 0.07% | 9.33% | 1.14% |

## Position patterns

- gate: within-expert sampled row Jaccard at k=16: 0.004 versus random 0.004; top 1% of input dimensions account for 2.1% of top-16 positions; within-group cross-expert Jaccard 0.018 versus random 0.004.
- up: within-expert sampled row Jaccard at k=16: 0.004 versus random 0.004; top 1% of input dimensions account for 2.0% of top-16 positions; within-group cross-expert Jaccard 0.011 versus random 0.004.
- down: within-expert sampled row Jaccard at k=16: 0.006 versus random 0.006; top 1% of input dimensions account for 2.3% of top-16 positions; within-group cross-expert Jaccard 0.006 versus random 0.006.

## Validation and interpretation

- Reused existing ResCode groups, saliency, and shared-base initializer. Singleton experts were excluded.
- Maximum relative reconstruction error for W = S + R: 3.23e-08.
- Checked BF16 tensor shapes, coefficient sums, independent per-row selection, and row/matrix squared energy identities.
- Large reductions from a few elements would support testing QJL outlier separation; this experiment does not measure quantization quality.
