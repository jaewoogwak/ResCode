"""Losslessly recode v1/v2 compact-base checkpoints without recompressing QJL.

QJL packed signs, FP32 row norms, fitted/fixed alpha, router, and all other
checkpoint settings are reused verbatim. The source file is never modified.
"""

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rescode.qjl.compact_base import (byte_payload, convert_base_payload,
                                      convert_base_payload_legacy, encoded_bytes,
                                      storage_payload)
from rescode.qjl.exponent_model import CHECKPOINT_V1, CHECKPOINT_V2
from rescode.qjl.packed_exponent import decode_exponents, resident_bytes


def convert(source: Path, target: Path, block_size: int, widths: tuple[int, ...]):
    if target.exists():
        raise FileExistsError(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    started = time.perf_counter()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    source_format = payload.get("format")
    if source_format not in (CHECKPOINT_V1, CHECKPOINT_V2):
        raise ValueError("source must be a v1 or v2 compact-base checkpoint")
    target_format = CHECKPOINT_V2 if source_format == CHECKPOINT_V1 else CHECKPOINT_V1
    old_base_bytes = new_base_bytes = exponent_gpu_bytes = count = 0
    mode_counts = {str(mode): 0 for mode in (0, 3, 4, 5)}
    for layer in payload["layers"].values():
        for expert in layer.values():
            for part in expert.values():
                original = byte_payload(part["base"])
                n = math.prod(original["shape"])
                old_base_bytes += encoded_bytes(original)
                if source_format == CHECKPOINT_V1:
                    converted = convert_base_payload(original, block_size, widths)
                    before = np.frombuffer(zlib.decompress(original["exponents"]), dtype=np.uint8)
                    after = decode_exponents(converted, n)
                    exponent_gpu_bytes += resident_bytes(converted)
                    for mode in np.frombuffer(converted["exp_modes"], dtype=np.uint8):
                        mode_counts[str(int(mode))] += 1
                else:
                    converted = convert_base_payload_legacy(original)
                    before = decode_exponents(original, n)
                    after = np.frombuffer(zlib.decompress(converted["exponents"]), dtype=np.uint8)
                    exponent_gpu_bytes += n
                    for mode in np.frombuffer(original["exp_modes"], dtype=np.uint8):
                        mode_counts[str(int(mode))] += 1
                if not np.array_equal(before, after):
                    raise AssertionError("exponent conversion changed BF16 bits")
                if (original["codes"] != converted["codes"]
                        or original["special_indices"] != converted["special_indices"]
                        or original["special_bits"] != converted["special_bits"]):
                    raise AssertionError("conversion changed sign, mantissa or special bits")
                new_base_bytes += encoded_bytes(converted)
                part["base"] = storage_payload(converted)
                count += n
    payload["format"] = target_format
    payload["config"]["format"] = target_format
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    report = {"source": str(source), "target": str(target),
              "source_format": source_format, "target_format": target_format,
              "source_checkpoint_bytes": source.stat().st_size,
              "target_checkpoint_bytes": target.stat().st_size,
              "weights": count, "source_base_bytes": old_base_bytes,
              "target_base_bytes": new_base_bytes,
              "gpu_exponent_bytes": exponent_gpu_bytes,
              "gpu_exponent_bits_per_weight": 8 * exponent_gpu_bytes / count,
              "block_modes": mode_counts, "block_size": block_size,
              "widths": widths, "seconds": time.perf_counter() - started,
              "qjl_tensors_reused": True, "bitwise_base_verified": True}
    report_path = target.with_name("conversion_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--block-size", type=int, default=4096)
    p.add_argument("--widths", default="3,4,5")
    args = p.parse_args()
    widths = tuple(int(v) for v in args.widths.split(","))
    print(json.dumps(convert(args.source, args.output, args.block_size, widths), indent=2))


if __name__ == "__main__":
    main()
