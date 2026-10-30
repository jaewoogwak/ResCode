# Direct residual QJL experiment

This path is independent of `scripts/RESCODE/run.sh`. It keeps the original
router, original top-k expert IDs and weights, Qwen's shared expert, and the
SwiGLU order. ResCode's existing calibration statistics and `form_clusters`
select the groups and pinned singletons. No low-rank residual factors or OGF
training are used.

For a group, each projection stores one BF16 shared base `S` and each member
stores either an exact FP32 residual `W-S` or QJL row norms (FP32) plus
little-endian packed sign bits. A seeded Gaussian matrix is shared across all
layers with the same input width and lives only in the runtime projection bank.
It is regenerated on checkpoint load. Both the PyTorch reference and Triton
backend consume packed bytes; neither reconstructs a dense residual in
inference. Unselected targets remain exact residuals. Singleton experts stay
BF16. The checkpoint contains only the expert-side tensors and metadata; it
loads alongside the unmodified original Hugging Face backbone and router.

`qjl_aware` minimizes the weighted variance objective using matrix-free CG:

```text
q_i(x) = 1[i in top-k(x)] * p_i(x)^2
V(R,x) = ((pi/2)*||x||^2*||R||_F^2 - ||R x||^2)/D
```

Gate/up use original hidden states; down uses the original expert's SwiGLU
middle activation. `mean` takes a uniform weight average; `saliency` uses
ResCode's `(saliency+delta)` weight average. The `1/D` factor is common to
the fixed-D optimum. Minimizing this objective need not minimize the residual
Frobenius norm. Training and disjoint held-out token objectives are reported
separately. The capped routed sample count may affect the optimum; all three
base methods use identical grouping and deterministic samples when invoked
with identical calibration flags.

The two-sequence smoke checks are too small to establish held-out gains: the
QJL-aware training objective fell on both models, while the held-out objective
rose on those tiny samples. Full runs should be judged by held-out error and
the eight-task evaluation, not the training objective alone. Qwen's one-layer
exact-residual probe had MoE-output relative error about `1.5e-4`.

## Environment and calibration data

On a fresh Linux x86_64 GPU instance, run `setup.sh` from the repo root. It
recreates the current server's Python and package versions, including
`peft==0.10.0` with `accelerate==0.28.0`. The current server uses
`/venv/rescode`; new clones use `.venv` by default. Then prepare calibration
data separately:

```bash
bash setup.sh
source .venv/bin/activate
export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false
python scripts/RESCODE/prepare_calib_datasets.py --dataset wikipedia --n_docs 40000 --overwrite
python -m unittest discover -s tests -p test_direct_qjl.py -v
```

The model files have been downloaded to
`/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat` and
`/home/jaewoo/models/Mixtral-8x7B-v0.1`. The CLI can also use Hugging Face
repository names via `--model-name`. The commands below use Wikipedia because
its preparation and loader were verified on this server. DCLM is also
supported through the unchanged ResCode data preparation script; use the same
corpus and sampling flags in every comparison.

## Smoke checks

These commands compress one MoE layer and run two WinoGrande questions. They
test plumbing, not benchmark accuracy. Qwen layer 1 and Mixtral layer 0 were
both tested with `D=127`, including lm-eval; both checkpoints were reloaded
and evaluated. The Mixtral whole-model device map was also checked: the
default `--gpu-memory-fraction 0.94` places all weights on this server's
96 GiB GPU. Compression checks selected layers are on GPU before starting.
An undersized GPU can require multiple GPUs or a different memory fraction.

```bash
python -u scripts/QJL/run.py --mode compress --model qwen \
  --model-name /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --layers 1 --base-method qjl_aware --qjl-d 1024 --targets gate_up_down \
  --train-data-path rescode/data/wikipedia-calib.jsonl --n-sentences 2 \
  --calib-seqlen 128 --max-train-per-expert 12 --max-heldout-per-expert 4 \
  --task winogrande --eval-limit 2 --eval-batch-size 1 \
  --output-dir results/qjl_direct/smoke/qwen_d1024

python -u scripts/QJL/run.py --mode compress --model mixtral \
  --model-name /home/jaewoo/models/Mixtral-8x7B-v0.1 \
  --layers 0 --base-method qjl_aware --qjl-d 1024 --targets gate_up_down \
  --train-data-path rescode/data/wikipedia-calib.jsonl --n-sentences 2 \
  --calib-seqlen 128 --max-train-per-expert 12 --max-heldout-per-expert 4 \
  --task winogrande --eval-limit 2 --eval-batch-size 1 \
  --output-dir results/qjl_direct/smoke/mixtral_d1024
```

## Full compression and eight-task evaluation

`--eval-limit 0` evaluates the full task sets. The model, calibration corpus,
grouping flags, seed, sample caps, tasks and evaluation settings should be
identical in every base ablation. The default task list is
`winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte`.

```bash
python -u scripts/QJL/run.py --mode compress --model qwen \
  --model-name /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --num-groups 30 --base-method qjl_aware --qjl-d 8192 \
  --targets gate_up_down --projection-seed 0 --backend triton \
  --train-data-path rescode/data/wikipedia-calib.jsonl --n-sentences 2400 \
  --calib-seqlen 2048 --max-train-per-expert 512 \
  --max-heldout-per-expert 128 --eval-limit 0 \
  --output-dir results/qjl_direct/qwen/qjl_aware_d8192_full_wiki

python -u scripts/QJL/run.py --mode compress --model mixtral \
  --model-name /home/jaewoo/models/Mixtral-8x7B-v0.1 \
  --num-groups 4 --base-method qjl_aware --qjl-d 8192 \
  --targets gate_up_down --projection-seed 0 --backend triton \
  --train-data-path rescode/data/wikipedia-calib.jsonl --n-sentences 2400 \
  --calib-seqlen 2048 --max-train-per-expert 512 \
  --max-heldout-per-expert 128 --eval-limit 0 \
  --output-dir results/qjl_direct/mixtral/qjl_aware_d8192_full_wiki
```

`--mode eval` loads a compressed expert checkpoint without calibration or
recompression. A new output directory avoids replacing prior metrics:

```bash
python -u scripts/QJL/run.py --mode eval --model qwen \
  --model-name /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --checkpoint results/qjl_direct/qwen/qjl_aware_d8192_full_wiki/qjl_state.pt \
  --backend triton --eval-limit 0 \
  --output-dir results/qjl_direct/qwen/qjl_aware_d8192_reval_wiki
```

For the three shared-base ablations at fixed settings:

```bash
for base in mean saliency qjl_aware; do
  python -u scripts/QJL/run.py --mode compress --model qwen \
    --model-name /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
    --num-groups 30 --base-method "$base" --qjl-d 8192 \
    --targets gate_up_down --projection-seed 0 --backend triton \
    --train-data-path rescode/data/wikipedia-calib.jsonl --n-sentences 2400 \
    --calib-seqlen 2048 --max-train-per-expert 512 \
    --max-heldout-per-expert 128 --eval-limit 0 \
    --output-dir "results/qjl_direct/qwen/ablation_${base}_d8192_full_wiki"
done
```

For controls, `--mode original` runs the same lm-eval pipeline on original
BF16; `--residual-mode exact` uses shared base plus exact residuals and is an
identity check; `--mode rescode` invokes the untouched ResCode launcher with a
distinct `--rescode-tag`. These modes use the same `--model`, `--task`, and
`--eval-limit` flags. `--skip-eval` saves compression without lm-eval.

```bash
python -u scripts/QJL/run.py --mode original --model qwen \
  --model-name /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --eval-limit 0 --output-dir results/qjl_direct/qwen/original_bf16_full

python -u scripts/QJL/run.py --mode compress --model qwen \
  --model-name /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --base-method mean --residual-mode exact --qjl-d 8192 \
  --train-data-path rescode/data/wikipedia-calib.jsonl --n-sentences 2400 \
  --calib-seqlen 2048 --max-train-per-expert 512 \
  --max-heldout-per-expert 128 --eval-limit 0 \
  --output-dir results/qjl_direct/qwen/exact_identity_full

python -u scripts/QJL/run.py --mode rescode --model qwen \
  --model-name /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --num-groups 30 --train-data-path rescode/data/wikipedia-calib.jsonl \
  --n-sentences 2400 --eval-limit 0 --rescode-tag qjl-control-wiki \
  --output-dir results/qjl_direct/qwen/rescode_control_config
```

The ResCode launcher writes its own model outputs under
`results/qwen/rescode-qwen-30e-qjl-control-wiki/`.

Each output directory contains `config.json`, `metrics.json`, `lm_eval.txt`
after evaluation, and `qjl_state.pt` after compression. `metrics.json`
records per-layer/cluster/projection residual ratios, variance objective
before/after and on held-out tokens, theoretical and empirical weighted
linear MSE, expert and MoE output relative error, task accuracy, expert-side
checkpoint bytes, original expert BF16 bytes, runtime projection bytes, GPU
memory, and fixed short-prompt forward latency. The checkpoint compression
ratio is **expert-side** and includes actual zip/metadata bytes. It excludes
the unmodified backbone, which is loaded from the original model. The group
file and calibration statistics live under `results/qjl_direct/shared/` and
are reused for matching configurations. Existing ResCode results are not
replaced unless explicitly requested with `--overwrite`.
