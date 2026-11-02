# Exponent-aware Compact Base + QJL Residual

This is an independent Qwen experiment. It does not use ResCode grouping or
BF16 singleton experts. Every routed expert's Gate, Up and Down projection is
converted; the original router, top-k and shared expert remain in the HF model.
The unmodified HF model provides non-expert parameters at checkpoint load.

## Representation and arithmetic

`k=1..4` retains `k` BF16 mantissa bits after ties-to-even rounding. A carry can
change the exponent, including subnormal-to-normal and finite-to-infinity;
the **rounded base's** sign and exponent are then stored exactly. NaN/Inf words
are kept in a special-value table. Sign and retained mantissa bits are packed
at `k+1` bits/weight. Exponent bytes are losslessly DEFLATE-coded with zlib in
the checkpoint. The checkpoint reports actual compressed Base bytes and bits
per weight, including the special-value table. It also reports the Shannon
entropy of the exponent symbols; entropy is an analysis statistic, not the
claimed storage cost.

At load, zlib is decoded into one uint8 exponent per weight on the GPU. Packed
sign/mantissa bits also reside on the GPU. `compact_base_kernel.py` directly
unpacks them in Triton and multiplies by BF16 inputs, without a resident or
transient dense BF16 weight. `CompactBase.decode()` is an exact reference path
used for tests and compression-time residual construction. The metrics report
both actual checkpoint bytes and GPU-resident compact buffers; GPU exponent
residency is larger than its compressed checkpoint representation.

The residual is computed as `original_weight.float() - decoded_base.float()`.
QJL uses the existing FP16 Gaussian projection and FP32 row norms, packing,
and Triton backend. `--d-factor 1` sets `D=n` and `--d-factor 2` sets `D=2n`
for each projection independently (`n=2048` Gate/Up, `n=1408` Down).
`--alpha calibrated` fits an unconstrained least-squares scalar per expert and
projection from the routed **training** activations only. Held-out activations
are used only for diagnostics. Evaluation tasks are not used for fitting.

The checkpoint is `rescode-exponent-base-qjl-v1`; it stores compact Base data,
packed QJL signs, FP32 row norms and alpha values for routed experts. It does
not store the original BF16 expert weights. The GPU may hold the original
weights transiently during model conversion, which is included in the
compression/load peak metric. The post-conversion inference resident and peak
metrics are measured separately. Short fixed-prompt forward latency is measured
after warmup with both CUDA Events and synchronized wall time; tokenizer work
is excluded.

## Run

From `/home/jaewoo/ResCode` with `.venv/bin/python` (the `rescode` environment):

```bash
# Prepare the dedicated calibration source once.
.venv/bin/python scripts/RESCODE/prepare_calib_datasets.py --dataset wikipedia --n_docs 38400

# Synthetic Gate/Up/Down kernel parity at D=n and D=2n.
.venv/bin/python scripts/QJL/check_exponent_kernels.py

# Small smoke test: one MoE layer, two ARC tasks and four examples per task.
.venv/bin/python scripts/QJL/run_exponent_base_sweep.py \
  --output-root results/qjl_exponent_base/smoke \
  --only base_k2,qjl_k2_d1n_a1 --layers 1 --n-sentences 2 \
  --calib-seqlen 128 --max-train-per-expert 4 \
  --max-heldout-per-expert 2 --diagnostic-samples 2 \
  --task arc_easy,arc_challenge --eval-limit 4 --eval-batch-size 2 --resume

# Full comparison: original, k=1..4 Base Only, k=1..3 QJL at D=n and 2n.
.venv/bin/python scripts/QJL/run_exponent_base_sweep.py --resume

# Add calibrated-alpha variants, reusing completed checkpoints and samples.
.venv/bin/python scripts/QJL/run_exponent_base_sweep.py \
  --include-calibrated --resume

# Re-evaluate existing checkpoints without recompression; preserve prior evals.
.venv/bin/python scripts/QJL/run_exponent_base_sweep.py \
  --eval-only --rerun-eval --resume
```

The full default evaluation is zero-shot, batch size 16, full dataset (`limit=0`)
on `winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte`.
Each compression and evaluation runs in its own process. An existing incomplete
or mismatched output raises an error; no prior results are overwritten. The
sweep writes per-setting checkpoints, diagnostics, metrics, `lm_eval.txt`, and
`sweep_summary.csv/json` under `results/qjl_exponent_base/qwen/`. Result files
are ignored by Git by default; large checkpoints should remain outside Git.

The two budget comparisons are `base_k3` versus `qjl_k2_d1n_a1`, and `base_k4`
versus `qjl_k2_d2n_a1`. Their **actual checkpoint-byte difference**, including
QJL norms and metadata, is recorded. Equality of nominal bit widths is not
assumed to imply equal physical storage.
