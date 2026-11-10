#!/usr/bin/env python3
"""Calibration-free Frobenius diagnostics for Qwen 2-bit codebook bases."""

import argparse
import csv
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open

from rescode.qjl.codebook_base import PackedCodebookBase, encode_base

MODEL_PATH = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
              / "snapshots/ec052fda178e241c7c443468d2fa1db6618996be")
PROJECTIONS = ("gate", "up", "down")


def select(value, size):
    values = list(range(size)) if value == "all" else [int(x) for x in value.split(",")]
    if not values or len(values) != len(set(values)) or any(x < 0 or x >= size for x in values):
        raise ValueError(f"invalid selection {value!r}; valid range 0..{size-1}")
    return values


def aggregate(records):
    def summarize(subset):
        w2 = sum(r["weight_squared_norm"] for r in subset)
        r2 = sum(r["residual_squared_norm"] for r in subset)
        return {"matrix_count": len(subset), "weight_squared_norm": w2,
                "residual_squared_norm": r2,
                "aggregated_frobenius_ratio": math.sqrt(r2 / w2) if w2 else None,
                "base_only_relative_frobenius_error": math.sqrt(r2 / w2) if w2 else None,
                "base_tensor_bytes": sum(r["base_bytes"] for r in subset),
                "qjl_tensor_bytes": sum(r["qjl_bytes"] for r in subset),
                "weight_count": sum(r["rows"] * r["cols"] for r in subset),
                "bits_per_weight": (8 * sum(r["base_bytes"] + r["qjl_bytes"] for r in subset)
                                    / sum(r["rows"] * r["cols"] for r in subset))}
    return {"overall": summarize(records),
            "by_projection": {p: summarize([r for r in records if r["projection"] == p])
                              for p in PROJECTIONS if any(r["projection"] == p for r in records)}}


def atomic_json(path, value):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--experts", default="all")
    parser.add_argument("--projections", default="gate,up,down")
    parser.add_argument("--qjl-d-factor", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--base-row-chunk", type=int, default=512)
    parser.add_argument("--row-chunk", type=int, default=None,
                        help="Legacy alias for --base-row-chunk")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="Use one weight matrix at a time on GPU when available")
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional existing checkpoint for actual file bytes; diagnostics do not create it")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/qjl_codebook_base")
    args = parser.parse_args()
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"--checkpoint must name an existing file: {args.checkpoint}; "
                     "omit it to report projected QJL storage without a checkpoint")
    if args.qjl_d_factor <= 0:
        parser.error("--qjl-d-factor must be positive")
    if args.row_chunk is not None:
        args.base_row_chunk = args.row_chunk
    if args.base_row_chunk < 1:
        parser.error("--base-row-chunk must be positive")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    model_path = args.model_path.expanduser().resolve()
    config = json.loads((model_path / "config.json").read_text())
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    layers = select(args.layers, config["num_hidden_layers"])
    experts = select(args.experts, config["num_experts"])
    projections = args.projections.split(",")
    if not projections or len(projections) != len(set(projections)) or any(p not in PROJECTIONS for p in projections):
        parser.error("--projections must be a comma-separated subset of gate,up,down")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    records = []
    weight_map = index["weight_map"]
    with ExitStack() as stack:
        handles = {}
        for layer in layers:
            for expert in experts:
                for projection in projections:
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
                    shard = weight_map[name]
                    if shard not in handles:
                        handles[shard] = stack.enter_context(safe_open(str(model_path / shard),
                                                                          framework="pt", device="cpu"))
                    weight = handles[shard].get_tensor(name).to(device)
                    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
                        raise ValueError(f"expected BF16 matrix: {name}")
                    state = encode_base(weight, args.iterations, args.base_row_chunk,
                                        return_cpu=False)
                    decoded = PackedCodebookBase(state, device).decode()
                    residual = weight.float() - decoded
                    if not torch.allclose(decoded + residual, weight.float(), rtol=1e-6, atol=1e-6):
                        raise AssertionError(f"residual reconstruction failed: {name}")
                    w2 = weight.float().square().sum().item()
                    r2 = residual.square().sum().item()
                    row_w = torch.linalg.vector_norm(weight.float(), dim=1)
                    row_r = torch.linalg.vector_norm(residual, dim=1)
                    ratios = row_r / row_w.clamp_min(1e-30)
                    base_bytes = (state["packed"].numel() + state["scales"].numel() * 2
                                  + state["zero_indices"].numel() * 4)
                    d = round(weight.shape[1] * args.qjl_d_factor)
                    qjl_bytes = weight.shape[0] * ((d + 7) // 8 + 4)
                    ratio = math.sqrt(r2 / w2) if w2 else None
                    records.append({"layer": layer, "expert": expert, "projection": projection,
                                    "rows": weight.shape[0], "cols": weight.shape[1],
                                    "weight_squared_norm": w2, "residual_squared_norm": r2,
                                    "residual_frobenius_ratio": ratio,
                                    "base_only_relative_frobenius_error": ratio,
                                    "row_ratio_mean": ratios.mean().item(),
                                    "row_ratio_median": ratios.median().item(),
                                    "row_ratio_p90": torch.quantile(ratios, 0.9).item(),
                                    "row_ratio_max": ratios.max().item(),
                                    "base_bytes": base_bytes, "qjl_bytes": qjl_bytes,
                                    "bits_per_weight": 8 * (base_bytes + qjl_bytes) / weight.numel()})
                    print(f"[diagnostics] layer={layer} expert={expert} projection={projection} "
                          f"rho_F={ratio:.6f}", flush=True)
                    del weight, decoded, residual, state
            # Handles are kept only for current shard set; weight tensors are transient.
    summary = aggregate(records)
    summary["model_path"] = str(model_path)
    summary["layers"] = layers
    summary["experts"] = experts
    summary["projections"] = projections
    summary["qjl_d_factor"] = args.qjl_d_factor
    summary["device"] = device
    summary["checkpoint_bytes"] = args.checkpoint.stat().st_size if args.checkpoint else None
    summary["checkpoint_bytes_note"] = ("actual on-disk file size" if args.checkpoint
                                        else "no checkpoint supplied; qjl_tensor_bytes is projected storage")
    atomic_json(out / "diagnostics.json", {"records": records})
    with (out / "diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    atomic_json(out / "summary.json", summary)
    print(f"[diagnostics] saved {len(records)} matrices to {out}", flush=True)


if __name__ == "__main__":
    main()
