#!/usr/bin/env bash
# Same-GPU v1/v2 benchmarks and full ARC-E/C evaluation after compression.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${1:-$REPO_ROOT/results/qjl_exponent_base/optimized}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

if [[ ! -s rescode/data/wikipedia-calib.jsonl ]]; then
  "$PYTHON_BIN" scripts/RESCODE/prepare_calib_datasets.py \
    --dataset wikipedia --n_docs 40000
fi

# Generate the four v2 checkpoints once. The sweep checks configuration and
# checkpoint completeness before reusing any existing compression result.
"$PYTHON_BIN" scripts/QJL/run_exponent_base_sweep.py \
  --output-root "$OUTPUT_ROOT/checkpoints" \
  --only base_k2,base_k4,qjl_k2_d1n_a1,qjl_k2_d2n_a1 \
  --task arc_easy,arc_challenge --skip-eval --resume

# Legacy checkpoints were lost with the previous instance. Recreate the exact
# v1 exponent stream from v2, retaining QJL packed signs, norms, and alpha.
for method in base_k2 base_k4 qjl_k2_d1n_a1 qjl_k2_d2n_a1; do
  if [[ ! -f "$OUTPUT_ROOT/source_checkpoints/$method/conversion_report.json" ]]; then
    "$PYTHON_BIN" scripts/QJL/convert_exponent_checkpoint.py \
      --source "$OUTPUT_ROOT/checkpoints/$method/compact_qjl_state.pt" \
      --output "$OUTPUT_ROOT/source_checkpoints/$method/compact_qjl_state.pt"
  fi
done

"$PYTHON_BIN" scripts/QJL/compare_exponent_encodings.py --gpu-timing \
  --output "$OUTPUT_ROOT/encoding_candidates_timed.json"

if [[ ! -f "$OUTPUT_ROOT/benchmark/original/metrics.json" ]]; then
  "$PYTHON_BIN" scripts/QJL/benchmark_exponent_runtime.py \
    --output "$OUTPUT_ROOT/benchmark/original"
fi

for method in base_k2 base_k4 qjl_k2_d1n_a1 qjl_k2_d2n_a1; do
  if [[ ! -f "$OUTPUT_ROOT/benchmark/$method/legacy/metrics.json" ]]; then
    "$PYTHON_BIN" scripts/QJL/benchmark_exponent_runtime.py \
      --checkpoint "$OUTPUT_ROOT/source_checkpoints/$method/compact_qjl_state.pt" \
      --output "$OUTPUT_ROOT/benchmark/$method/legacy"
  fi
  if [[ ! -f "$OUTPUT_ROOT/benchmark/$method/optimized/metrics.json" ]]; then
    "$PYTHON_BIN" scripts/QJL/benchmark_exponent_runtime.py \
      --checkpoint "$OUTPUT_ROOT/checkpoints/$method/compact_qjl_state.pt" \
      --output "$OUTPUT_ROOT/benchmark/$method/optimized"
  fi
done

if [[ ! -f "$OUTPUT_ROOT/evaluation/original/metrics.json" ]]; then
  "$PYTHON_BIN" scripts/QJL/run_exponent_base.py --mode original \
    --output-dir "$OUTPUT_ROOT/evaluation/original" \
    --task arc_easy,arc_challenge --eval-limit 0 --eval-batch-size 16
fi
for method in base_k4 qjl_k2_d2n_a1; do
  for variant in legacy optimized; do
    if [[ ! -f "$OUTPUT_ROOT/evaluation/$method/$variant/metrics.json" ]]; then
      source_dir=source_checkpoints
      if [[ "$variant" == optimized ]]; then source_dir=checkpoints; fi
      "$PYTHON_BIN" scripts/QJL/run_exponent_base.py --mode eval \
        --checkpoint "$OUTPUT_ROOT/$source_dir/$method/compact_qjl_state.pt" \
        --output-dir "$OUTPUT_ROOT/evaluation/$method/$variant" \
        --backend triton --task arc_easy,arc_challenge \
        --eval-limit 0 --eval-batch-size 16
    fi
  done
done

"$PYTHON_BIN" scripts/QJL/summarize_exponent_optimized.py --root "$OUTPUT_ROOT"
