# ResCode: Accurate Expert Compression of Sparse MoE Models via Residual-code Restoration

This repository contains the implementation of **RESCODE**, an accurate expert compression method for sparse mixture-of-experts (SMoE) models.
RESCODE forms compact expert clusters, coalesces residuals into cluster-shared low-rank bases, and learns compact codes to accurately recover the original routed outputs.

## Overview

<p align="center">
  <img src="./figure/Figure3.png" width="800"/>
</p>

Given a pretrained SMoE model, RESCODE compresses experts through three main ideas:
**(I1)** saliency-aware cluster-wise restoration,
**(I2)** diagonal residual-code coalescing, and
**(I3)** output-guided fitting.

## Prerequisites

We tested the code with the following environment.

- Python 3.10
- PyTorch 2.8
- HuggingFace Transformers 4.40
- NVIDIA GPU (we used a single B200 180GB GPU)

Install required packages:
```bash
pip install -r requirements.txt
```

## Setup

Download the pretrained SMoE models from HuggingFace (loaded automatically on first run):

- Mixtral-8x7B-v0.1 (`mistralai/Mixtral-8x7B-v0.1`)
- Qwen1.5-MoE-A2.7B-Chat (`Qwen/Qwen1.5-MoE-A2.7B-Chat`)
- DeepSeek-MoE-16B (`deepseek-ai/deepseek-moe-16b-base`)

Prepare a calibration corpus once via the HuggingFace `datasets` library
(DCLM is the default; SlimPajama, English Wikipedia, and StarCoder are also
supported):

```bash
python scripts/RESCODE/prepare_calib_datasets.py --dataset dclm
```

The `c4` option instead expects a pre-downloaded shard at
`rescode/data/c4-train.00000-of-01024.json`.

## Running

### Compress an SMoE model

Per-model launch scripts live under `scripts/RESCODE/` (see
[scripts/README.md](scripts/README.md) for the full table); each sets its
paper defaults and calls the shared launcher `scripts/RESCODE/run.sh`:

```bash
bash scripts/RESCODE/run_mixtral_4x.sh    # Mixtral-8x7B,  8 -> 4 experts  (50%)
bash scripts/RESCODE/run_qwen_30e.sh      # Qwen1.5-MoE,  60 -> 30 experts (50%)
bash scripts/RESCODE/run_dsmoe16b_32e.sh  # DeepSeek-MoE, 64 -> 32 experts (50%)
```

Main knobs (override via env vars; all documented inline in `run.sh`):
- `MODEL`: HuggingFace model name
- `NUM_EXPERTS`: compressed expert (cluster) count per layer
- `RANK`: residual rank `r` (default `128`)
- `ALPHA`: residual scale applied to the low-rank residual (`2.0` for Qwen/DeepSeek, `1.0` for Mixtral)
- `SALIENCY_TOP_TAU`: singleton ratio ρ in % (`15.0` at moderate compression, `0` at aggressive compression)
- `TARGET`: projection branches to restore (default `gate_up_down`)
- `JOINT_EPOCHS`: output-guided fitting epochs `n_ep` (default `3`)
- `CALIB_DATASET`: calibration corpus (`dclm` by default; `c4`, `slimpajama`, `wikipedia`, `starcoder`)
- `DESCRIPTOR_TOKENS`: token set for output-similarity clustering (`all` by default; `routed` for the conditional-mean variant)
- `DOWN_INIT`: down-projection residual init (`covariance` by default; `zero_L`, `zero_R`)
- `LAMBDA_MSE` / `LAMBDA_LM`: weights of the output-reconstruction and LM terms in Idea 3 (defaults `0.01` / `1.0`)
- `SEED` / `CALIB_SEED`: optimization and calibration-sampling seeds (defaults `0` / `42`); multi-seed runs vary both jointly

Example:

```bash
GPU=0 NUM_EXPERTS=4 bash scripts/RESCODE/run_mixtral_4x.sh
```

### Evaluate

Every compression run ends with zero-shot evaluation on the eight benchmarks
(ARC-c/e, BoolQ, HellaSwag, MMLU, OpenBookQA, RTE, WinoGrande) via
`lm-evaluation-harness`; per-task accuracies and the average are appended to
`results/<family>/<run>/result.txt`. To re-evaluate a finished run without
re-fitting, reuse its saved residual state:

```bash
LOAD=auto bash scripts/RESCODE/run_mixtral_4x.sh
```

## Reference

Parts of this code are adapted from the following repositories:

- **HC-SMoE** (https://github.com/wazenmai/HC-SMoE): expert similarity-based hierarchical clustering procedure
- **ResMoE** (https://github.com/iDEA-iSAIL-Lab-UIUC/ResMoE): baseline residual restoration implementation for comparison
- **REAP** (https://github.com/CerebrasResearch/reap): routing-aware expert pruning baseline

We thank the authors for releasing their code.

## License

This repository is released under the MIT License (see [LICENSE](LICENSE)).
`rescode/models/mixtral/modeling_mixtral.py` is derived from HuggingFace
Transformers and retains its original Apache-2.0 license header.