#!/usr/bin/env python3
"""Compare random-access exponent block formats on an actual Qwen expert weight."""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from rescode.qjl.compact_base import (CompactBase, convert_base_payload,
                                     encode_base, rounded_bf16_bits)
from rescode.qjl.packed_exponent import decode_exponents, encode_exponents, resident_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    p.add_argument("--weight", default="model.layers.0.mlp.experts.0.gate_proj.weight")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--gpu-timing", action="store_true",
                   help="Time selected encodings with CUDA Events after other GPU jobs finish")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    folder = Path(snapshot_download(args.model_name, local_files_only=True))
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    shard = folder / index["weight_map"][args.weight]
    with safe_open(shard, framework="pt", device="cpu") as file:
        weight = file.get_tensor(args.weight)
    bits = rounded_bf16_bits(weight, args.k).numpy().astype(np.uint16, copy=False)
    exponents = ((bits.reshape(-1) >> 7) & 255).astype(np.uint8)
    zlib_bytes = len(zlib.compress(exponents.tobytes(), level=1))
    torch.manual_seed(2026)
    legacy = encode_base(weight, args.k, legacy=True) if args.gpu_timing else None
    inputs = ({m: torch.randn(m, weight.shape[1], device="cuda", dtype=torch.bfloat16)
               for m in (1, 32, 128)} if args.gpu_timing else {})

    def gpu_ms(base, x):
        for _ in range(3):
            base(x)
        torch.cuda.synchronize()
        samples = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            base(x)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return float(np.median(samples))

    old_timing = None
    if args.gpu_timing:
        old_base = CompactBase(legacy, "cuda")
        old_timing = {str(m): gpu_ms(old_base, x) for m, x in inputs.items()}
    rows = []
    for block_size in (128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        for widths in ((3,), (4,), (3, 4, 5)):
            started = time.perf_counter()
            payload = encode_exponents(exponents, block_size, widths)
            elapsed = time.perf_counter() - started
            if not np.array_equal(decode_exponents(payload, len(exponents)), exponents):
                raise AssertionError("lossless exponent roundtrip failed")
            nbytes = resident_bytes(payload)
            modes = np.frombuffer(payload["exp_modes"], dtype=np.uint8)
            row = {"block_size": block_size, "widths": list(widths),
                         "gpu_exponent_bytes": nbytes,
                         "gpu_bits_per_weight": 8 * nbytes / len(exponents),
                         "encode_seconds": elapsed,
                         "mode_counts": {str(v): int((modes == v).sum())
                                         for v in (0, 3, 4, 5)}}
            if args.gpu_timing and block_size in (2048, 4096) and widths in ((4,), (3, 4, 5)):
                base = CompactBase(convert_base_payload(legacy, block_size, widths), "cuda")
                row["cuda_median_ms"] = {str(m): gpu_ms(base, x) for m, x in inputs.items()}
            rows.append(row)
    result = {"model": args.model_name, "weight": args.weight,
              "shape": list(weight.shape), "k": args.k,
              "dense_gpu_bits_per_weight": 8,
              "zlib_disk_bits_per_weight": 8 * zlib_bytes / len(exponents),
              "legacy_cuda_median_ms": old_timing,
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
