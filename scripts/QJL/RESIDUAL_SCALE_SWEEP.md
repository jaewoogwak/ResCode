# QJL residual-scale sweep

`run_residual_scale_sweep.py` runs four sequential Qwen experiments with the
same expert groups, routed calibration samples, and fitted BF16 shared bases.
Only the grouped experts receive a separate BF16 base per expert. Singleton
experts, the router, shared expert, projection generation, packed signs,
scaling, and Triton kernels retain their existing implementation.

For expert weight `W` and fitted shared base `Sg`, the stored individual base is
`Si = BF16(W - alpha * (W - Sg))`. The residual is then calculated as
`R = FP32(W) - FP32(Si)`. At alpha 1, the existing shared base is used directly
and no individual base tensors are added. At alpha 0, `Si` is the original BF16
weight and the QJL residual is zero.

## Environment and calibration data

```bash
cd /home/jaewoo/ResCode
source .venv/bin/activate
python scripts/RESCODE/prepare_calib_datasets.py --dataset wikipedia --n_docs 40000
```

The calibration JSONL is local data and is not committed. The checked-in group
file `results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt` pins the
partition from the original 8k experiment. Explicit use of that file checks
the model and grouping parameters while allowing dataset file metadata to
change after copying to a new server. The sweep cache records the SHA256 of
both the group file and dataset and rejects a different input on reuse.

## Full sweep

```bash
python scripts/QJL/run_residual_scale_sweep.py
```

The default evaluates all eight tasks with no limit. To prepare checkpoints
and diagnostics without evaluation, append `--skip-eval`. To resume completed
compressions and retry an incomplete evaluation, append `--resume`. Partial
compression and evaluation directories are renamed with a `.partial_...`
suffix before retrying, so incomplete files are retained for inspection.

All outputs are under `results/qjl_residual_scale/qwen/d8192/`, with
`alpha_0`, `alpha_0_25`, `alpha_0_5`, and `alpha_1` subdirectories. Each has a
`qjl_state.pt`, `config.json`, `diagnostics.json`, and compression `metrics.json`.
Evaluation writes `eval/metrics.json` and `eval/lm_eval.txt`. The root has
`shared_scale_cache.pt` and `sweep_summary.json/csv`. The cache holds routed
samples and fitted shared bases and can be large. Checkpoints are retained so
each alpha can be evaluated again.
The summary's Gate/Up/Down QJL and full-projection errors are means over
held-out routed samples. The detailed JSON contains train and held-out
records separately.

## Re-evaluation

An alpha checkpoint can be evaluated independently with the existing path:

```bash
python scripts/QJL/run.py \
  --mode eval --model qwen --backend triton \
  --checkpoint results/qjl_residual_scale/qwen/d8192/alpha_0_5/qjl_state.pt \
  --task winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte \
  --eval-batch-size 16 --eval-limit 0 \
  --output-dir results/qjl_residual_scale/qwen/d8192/alpha_0_5/reeval
```

The original direct-QJL checkpoint, if available locally, can be passed to
the same `--mode eval` command with its own new output directory. The old
checkpoint is not included in Git because it exceeds 16 GB.

## Diagnostics and smoke test

Diagnostics use at most 32 routed inputs per expert and split by default.
They run only during compression and are stored as per-layer, per-expert,
per-projection records plus mean, median, p90, max, valid count, and null
reasons. A zero reference norm produces `null` with
`zero_reference_norm`; this is expected for alpha 0 residual-QJL error.
No diagnostic hooks or original-output comparisons run during evaluation.

```bash
python scripts/QJL/run_residual_scale_sweep.py \
  --alphas 0,1 --layers 0 --n-sentences 1 --calib-seqlen 32 \
  --max-train-per-expert 2 --max-heldout-per-expert 1 \
  --diagnostic-samples 2 --cg-iterations 1 --skip-eval \
  --output-root results/qjl_residual_scale/smoke_qwen
```

This smoke test uses the pinned grouping but fits a small calibration sample,
so its accuracy and shared bases are not comparable to the full experiment.
`--dry-run` prints the subprocess commands without running them.

On the Qwen one-layer smoke test (`n_sentences=1`, two diagnostic samples),
all four compressions and reloadable checkpoints completed. Mean residual
Frobenius ratios for alpha 0, 0.25, 0.5, and 1 were 0, 0.269, 0.538, and
1.075. Mean held-out MoE relative errors were 0, 0.087, 0.179, and 0.403.
The nonzero-alpha residual-QJL *relative* errors stayed near 0.55 for Gate,
0.60 for Up, and 0.50 for Down; the absolute error scales with the residual
weight. On a short real-model prompt, alpha 0 logits matched original BF16
bit for bit, including router logits and top-k. ARC-Easy with `limit=4`
returned `acc=0.50`, `acc_norm=0.75` for both original and alpha 0.

The old 8k QJL checkpoint is absent on this server. Alpha 1 parity with the
existing conversion path was tested on a toy model by comparing its fitted
BF16 bases, packed QJL bits, and outputs. Real-checkpoint parity remains
unmeasured.
No full eight-task evaluation was run while developing this sweep.
