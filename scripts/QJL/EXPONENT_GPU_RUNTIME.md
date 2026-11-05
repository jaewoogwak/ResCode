# GPU-resident exponent base runtime

The v2 checkpoint keeps each BF16 base's sign and retained mantissa in the
existing packed code stream. Exponents use independently addressable blocks of
4096 weights. Each block chooses the smallest exact encoding among literal
bytes and local palettes with packed 3, 4, or 5 bit indices. The GPU stores the
packed codes, block offsets, modes, palettes, and special-value table. A Triton
kernel decodes only the weight tile needed for a linear operation and accumulates
in FP32. There is no permanent dense exponent array or BF16 base weight on GPU.

The old v1 checkpoint remains readable. Conversion works in either direction
without changing QJL signs, norms, alpha, projection seed, or any rounded BF16
base bit:

```bash
.venv/bin/python scripts/QJL/convert_exponent_checkpoint.py \
  --source results/qjl_exponent_base/qwen/base_k4/compact_qjl_state.pt \
  --output results/qjl_exponent_base/optimized/checkpoints/base_k4/compact_qjl_state.pt
```

The source is never overwritten. `conversion_report.json` records the exact
bitwise check and actual GPU exponent bytes. A v2 source produces a v1 target,
which allows same-checkpoint legacy comparisons when old v1 files are gone.
New compression through
`run_exponent_base.py` writes v2 directly. Use `--mode eval --checkpoint ...`
to evaluate either format with the same evaluation path.

Compare candidate exponent encodings on a real Qwen expert:

```bash
.venv/bin/python scripts/QJL/compare_exponent_encodings.py \
  --output results/qjl_exponent_base/optimized/encoding_candidates.json
```

For fair latency comparisons, finish other GPU work first, then run
`benchmark_exponent_runtime.py` in separate processes for the v1 and v2
checkpoints with the same model, GPU, and options. Each run saves CUDA Event
median and p95 latency, allocated resident and peak VRAM, and deterministic
output traces. `summarize_exponent_optimized.py` compares the traces and joins
benchmark and ARC-E/C evaluation results in `summary.csv/json` and `parity.json`.

Run the complete same-GPU validation and full ARC-E/C evaluation with:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/QJL/run_optimized_validation.sh \
  results/qjl_exponent_base/optimized
```

The runner compresses four v2 configurations with the existing sample cache,
losslessly recodes each one to v1, then benchmarks and evaluates both formats.
It skips verified compression and finished benchmark/evaluation directories. It
never retrains a base or QJL residual between the v1 and v2 runtime comparisons.
