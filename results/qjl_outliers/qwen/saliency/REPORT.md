# Saliency residual outliers before QJL

Scope: **full 24-layer analysis**; layers 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23.
Groups and saliency: `/home/jaewoo/ResCode/results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt`. Original BF16 model: `/root/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be`.
R = FP32(W) − FP32(BF16 ResCode saliency-weighted base), delta=0.001. Seed: 2026.
No QJL projection, quantization, packing, or compressed checkpoint was used.
Heatmap rows are output rows, replacing tokens in QJL Figure 2.

## Outlier concentration

- **gate:** Top 1% of input channels remove median 1.18% of residual squared norm versus 1.03% for random channels and 1.19% for original W. Top/median channel L2 is 1.11×; by the 2× random-control criterion, the residual is not strongly concentrated relative to random.
- **up:** Top 1% of input channels remove median 1.16% of residual squared norm versus 1.03% for random channels and 1.16% for original W. Top/median channel L2 is 1.08×; by the 2× random-control criterion, the residual is not strongly concentrated relative to random.
- **down:** Top 1% of input channels remove median 1.33% of residual squared norm versus 1.06% for random channels and 1.33% for original W. Top/median channel L2 is 1.19×; by the 2× random-control criterion, the residual is not strongly concentrated relative to random.

## Layer and channel patterns

Median top-1% squared energy removed from residuals (%):

| Layer | Gate | Up | Down |
|---:|---:|---:|---:|
| 0 | 1.20 | 1.17 | 1.15 |
| 1 | 1.17 | 1.16 | 1.17 |
| 2 | 1.18 | 1.16 | 1.18 |
| 3 | 1.18 | 1.16 | 1.17 |
| 4 | 1.17 | 1.16 | 1.19 |
| 5 | 1.17 | 1.16 | 1.23 |
| 6 | 1.18 | 1.16 | 1.29 |
| 7 | 1.17 | 1.16 | 1.30 |
| 8 | 1.18 | 1.16 | 1.33 |
| 9 | 1.18 | 1.16 | 1.34 |
| 10 | 1.18 | 1.16 | 1.33 |
| 11 | 1.18 | 1.16 | 1.33 |
| 12 | 1.19 | 1.16 | 1.37 |
| 13 | 1.19 | 1.17 | 1.41 |
| 14 | 1.19 | 1.17 | 1.45 |
| 15 | 1.18 | 1.16 | 1.47 |
| 16 | 1.17 | 1.16 | 1.45 |
| 17 | 1.18 | 1.16 | 1.46 |
| 18 | 1.18 | 1.15 | 1.42 |
| 19 | 1.18 | 1.15 | 1.39 |
| 20 | 1.18 | 1.15 | 1.35 |
| 21 | 1.19 | 1.15 | 1.45 |
| 22 | 1.17 | 1.16 | 1.37 |
| 23 | 1.18 | 1.17 | 1.37 |

Within-group mean top-32 Jaccard: 0.021.
Cross-layer top-32 Jaccard from pooled median-normalized expert channel L2 (successive analyzed layers): gate 0.226 (random-set reference 0.008), up 0.084 (random-set reference 0.008), down 0.013 (random-set reference 0.011).

## Validation and limitations

- Reused the ResCode `form_clusters` cache and its per-layer saliency; checked configuration, partitions, labels, and tensor shapes.
- The shared base uses the repository's saliency initializer; every coefficient vector sums to one before BF16 rounding.
- Maximum relative reconstruction error for W = S + R: 3.23e-08.
- Verified squared norm decomposition, zero-all and zero-none masking for every matrix; fixed seeds govern random controls and plotted row samples.
- Full-matrix statistics are unsampled. The experiment measures expert weight matrices, whereas QJL Figure 2 measures KV-cache key embeddings.
