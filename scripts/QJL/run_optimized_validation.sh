#!/usr/bin/env bash
# Same-GPU v1/v2 benchmarks and full ARC-E/C evaluation after compression.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${1:-$REPO_ROOT/results/qjl_exponent_base/optimized}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
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

if [[ ! -f "$OUTPUT_ROOT/encoding_candidates_timed.json" ]]; then
  "$PYTHON_BIN" scripts/QJL/compare_exponent_encodings.py --gpu-timing \
    --output "$OUTPUT_ROOT/encoding_candidates_timed.json"
fi

if [[ ! -f "$OUTPUT_ROOT/benchmark/original/metrics.json" ]]; then
  "$PYTHON_BIN" scripts/QJL/benchmark_exponent_runtime.py \
    --output "$OUTPUT_ROOT/benchmark/original"
fi

for method in base_k2 base_k4 qjl_k2_d1n_a1 qjl_k2_d2n_a1; do
  legacy_checkpoint="$OUTPUT_ROOT/source_checkpoints/$method/compact_qjl_state.pt"
  legacy_created=0
  needs_legacy=0
  if [[ ! -f "$OUTPUT_ROOT/benchmark/$method/legacy/metrics.json" ]]; then
    needs_legacy=1
  fi
  if [[ "$method" == base_k4 || "$method" == qjl_k2_d2n_a1 ]]; then
    if [[ ! -f "$OUTPUT_ROOT/evaluation/$method/legacy/metrics.json" ]]; then
      needs_legacy=1
    fi
  fi
  if [[ "$needs_legacy" == 1 && ! -f "$legacy_checkpoint" ]]; then
    # The legacy file is derived solely from this method's v2 checkpoint.
    # Only files created in this invocation may be removed after validation.
    report="${legacy_checkpoint%/*}/conversion_report.json"
    if [[ -f "$report" ]]; then
      mv "$report" "${report%.json}.prior_$(date +%s).json"
    fi
    "$PYTHON_BIN" scripts/QJL/convert_exponent_checkpoint.py \
      --source "$OUTPUT_ROOT/checkpoints/$method/compact_qjl_state.pt" \
      --output "$legacy_checkpoint"
    legacy_created=1
  fi
  if [[ ! -f "$OUTPUT_ROOT/benchmark/$method/legacy/metrics.json" ]]; then
    "$PYTHON_BIN" scripts/QJL/benchmark_exponent_runtime.py \
      --checkpoint "$legacy_checkpoint" \
      --output "$OUTPUT_ROOT/benchmark/$method/legacy"
  fi
  if [[ ! -f "$OUTPUT_ROOT/benchmark/$method/optimized/metrics.json" ]]; then
    "$PYTHON_BIN" scripts/QJL/benchmark_exponent_runtime.py \
      --checkpoint "$OUTPUT_ROOT/checkpoints/$method/compact_qjl_state.pt" \
      --output "$OUTPUT_ROOT/benchmark/$method/optimized"
  fi
  if [[ "$method" == base_k4 || "$method" == qjl_k2_d2n_a1 ]]; then
    if [[ ! -f "$OUTPUT_ROOT/evaluation/$method/legacy/metrics.json" ]]; then
      "$PYTHON_BIN" scripts/QJL/run_exponent_base.py --mode eval \
        --checkpoint "$legacy_checkpoint" \
        --output-dir "$OUTPUT_ROOT/evaluation/$method/legacy" \
        --backend triton --task arc_easy,arc_challenge \
        --eval-limit 0 --eval-batch-size 16
    fi
    if [[ ! -f "$OUTPUT_ROOT/evaluation/$method/optimized/metrics.json" ]]; then
      "$PYTHON_BIN" scripts/QJL/run_exponent_base.py --mode eval \
        --checkpoint "$OUTPUT_ROOT/checkpoints/$method/compact_qjl_state.pt" \
        --output-dir "$OUTPUT_ROOT/evaluation/$method/optimized" \
        --backend triton --task arc_easy,arc_challenge \
        --eval-limit 0 --eval-batch-size 16
    fi
  fi
  if [[ "$legacy_created" == 1 && "${KEEP_GENERATED_LEGACY:-0}" != 1 ]]; then
    rm -- "$legacy_checkpoint"
  fi
done

if [[ ! -f "$OUTPUT_ROOT/evaluation/original/metrics.json" ]]; then
  "$PYTHON_BIN" scripts/QJL/run_exponent_base.py --mode original \
    --output-dir "$OUTPUT_ROOT/evaluation/original" \
    --task arc_easy,arc_challenge --eval-limit 0 --eval-batch-size 16
fi
"$PYTHON_BIN" scripts/QJL/summarize_exponent_optimized.py --root "$OUTPUT_ROOT"
