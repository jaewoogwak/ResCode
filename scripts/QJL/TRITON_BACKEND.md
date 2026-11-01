# ResCode QJL Triton inference backend

`--backend triton` uses the original ResCode packed FP32 reduction for at most
20 expert tokens or QJL dimension at most 256. Larger groups use an on-chip
packed-sign decode and TF32x3 tiled FP32 dot. Gate and Up share one tiled launch
when both are QJL; Down uses the same dispatch. `triton_legacy` retains the old
kernel, and `reference` retains the PyTorch unpack-and-dot implementation.

Only inference changes. The shared base, individual residuals, FP16 Gaussian
projection and its seed, FP32 row norms and QJL scale, bit order, routing,
singleton experts, SwiGLU order, and `rescode-direct-qjl-v1` checkpoint format
are unchanged. RAM's FP32 projection and FP16 row scales were not imported.

## Verification

Run from `/home/jaewoo/ResCode` with its installed `.venv`:

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_direct_qjl -v
.venv/bin/python scripts/QJL/benchmark_triton_backend.py \
  --output results/qjl_direct/triton_backend_benchmark.json
```

The unit tests reload one saved checkpoint into each backend for both Qwen and
Mixtral block structures. They compare packed signs, FP32 norms, Gate/Up/Down,
shared-plus-residual outputs, SwiGLU, expert and MoE outputs, router logits and
top-k, and BF16 singleton outputs. The direct packed-dot test covers 1, 4, 32,
and 128 tokens with QJL dimensions 127, 8192, 8193, and 16384. All eight tests passed
on the server GPU.

The benchmark creates one **synthetic** checkpoint per family: a BF16 block
with hidden width 2048, intermediate width 1408, two QJL residual experts,
two BF16 singleton experts, and original top-2 routing. Qwen also has its
shared expert. Each backend reloads the same checkpoint. CUDA Events measure
five warmups and 20 timed iterations; table values are medians in milliseconds
on an NVIDIA RTX PRO 6000 Blackwell Server Edition. The JSON above contains
the full per-stage latency, FP32/BF16 error, routing checks, and peak allocated
bytes.

| Family | Tokens | Gate legacy/new | Up legacy/new | Down legacy/new | Expert legacy/new | MoE legacy/new |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 1 | 0.058/0.057 | 0.059/0.057 | 0.064/0.062 | 0.333/0.282 | 1.181/0.867 |
| Qwen | 4 | 0.072/0.073 | 0.071/0.071 | 0.091/0.091 | 0.310/0.314 | 1.197/1.171 |
| Qwen | 32 | 0.300/0.225 | 0.299/0.225 | 0.412/0.225 | 1.033/0.713 | 2.026/2.029 |
| Qwen | 128 | 1.067/0.363 | 1.052/0.362 | 1.506/0.668 | 3.704/1.462 | 4.514/2.416 |
| Mixtral | 128 | 1.052/0.364 | 1.052/0.363 | 1.511/0.668 | 3.707/1.459 | 4.434/2.346 |

At 128 tokens, the new Qwen Expert path is 2.53 times faster and MoE is
1.87 times faster. Mixtral is 2.54 and 1.89 times faster respectively.
At 32 total tokens, each routed expert gets fewer tokens and usually takes
the old reduction path; the full MoE time is therefore effectively unchanged.
Small-case timing varies due to routing and host launch overhead.

The maximum direct FP32 QJL relative L2 error across Gate/Up/Down and token
counts was `1.71e-7` against `reference` on the same projected input.
The largest BF16 expert-output relative L2 error was `2.89e-4`; the largest
MoE-output relative L2 error was `1.04e-4`. Router logits, top-k selection,
and singleton outputs were identical in all measured cases. The direct FP32
differences come from tiled dot accumulation order. BF16 rounding can amplify
their relative error in near-zero outputs; the synthetic MoE test stayed within
the unit test's `atol=0.02, rtol=0.02` bound.

At 128 tokens, the measured extra allocated memory for Qwen MoE was 5,087,744
bytes with either Triton backend, versus 16,327,168 bytes for `reference`.
For one Gate call it was 1,081,344 bytes with either Triton backend, versus
15,335,424 bytes for `reference`. This is allocated memory above the resident
checkpoint and projection, not total GPU memory.

## Evaluation with an existing checkpoint

The existing lm-eval path accepts all three backends. Use a distinct output
directory per backend:

```bash
.venv/bin/python scripts/QJL/run.py --mode eval --model qwen \
  --checkpoint /path/to/qjl_state.pt --backend reference \
  --task arc_easy --eval-limit 32 --output-dir /tmp/qjl_ref_eval
.venv/bin/python scripts/QJL/run.py --mode eval --model qwen \
  --checkpoint /path/to/qjl_state.pt --backend triton_legacy \
  --task arc_easy --eval-limit 32 --output-dir /tmp/qjl_legacy_eval
.venv/bin/python scripts/QJL/run.py --mode eval --model qwen \
  --checkpoint /path/to/qjl_state.pt --backend triton \
  --task arc_easy --eval-limit 32 --output-dir /tmp/qjl_triton_eval
```

Set `--model mixtral` and the matching checkpoint for Mixtral. The server has
no full ResCode checkpoint, model weights, or calibration dataset, so full
HF-model inference and lm-eval accuracy comparison have **not** been run.
The benchmark above measures representative saved synthetic checkpoints, not
end-to-end language-model latency.
