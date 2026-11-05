# GPU-resident exponent base runtime

The v2 checkpoint keeps each BF16 base's sign and retained mantissa in the
existing packed code stream. Exponents use independently addressable blocks of
4096 weights. Each block chooses the smallest exact encoding among literal
bytes and local palettes with packed 3, 4, or 5 bit indices. The GPU stores the
packed codes, block offsets, modes, palettes, and special-value table. A Triton
kernel decodes only the weight tile needed for a linear operation and accumulates
in FP32. There is no permanent dense exponent array or BF16 base weight on GPU.

The old v1 checkpoint remains readable. Convert it without changing QJL signs,
norms, alpha, projection seed, or any rounded BF16 base bit:

```bash
.venv/bin/python scripts/QJL/convert_exponent_checkpoint.py \
  --source results/qjl_exponent_base/qwen/base_k4/compact_qjl_state.pt \
  --output results/qjl_exponent_base/optimized/checkpoints/base_k4/compact_qjl_state.pt
```

The source is never overwritten. `conversion_report.json` records the exact
bitwise check and actual GPU exponent bytes. New compression through
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

After preparing v1 and v2 checkpoints under the `optimized` result root, run
the complete same-GPU validation and full ARC-E/C evaluation with:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/QJL/run_optimized_validation.sh \
  results/qjl_exponent_base/optimized
```

The runner waits for all four compression and conversion reports and skips
finished benchmark/evaluation directories. It uses the exact same checkpoints
for both runtime versions and never retrains a base or QJL residual.
