#!/usr/bin/env python3
"""Measure compact Base and QJL Reference/Triton errors on fixed synthetic inputs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from rescode.qjl.compact_base import CompactBase, encode_base, rounded_bf16_bits
from rescode.qjl.core import PackedResidual, ProjectionBank, pack_residual


def errors(actual, reference):
    delta = actual.float() - reference.float()
    norm = torch.linalg.vector_norm(reference.float()).item()
    return {"max_abs": delta.abs().max().item(),
            "relative_l2": torch.linalg.vector_norm(delta).item() / norm if norm else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/qjl_exponent_base/kernel_check.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(2026)
    rows = []
    for projection, n, m in (("gate", 2048, 1408), ("up", 2048, 1408),
                              ("down", 1408, 2048)):
        weight = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.01
        x = torch.randn(4, n, device="cuda", dtype=torch.bfloat16)
        for k in (1, 2, 3, 4):
            base = CompactBase(encode_base(weight, k), "cuda")
            decoded = base.decode()
            assert torch.equal(decoded.view(torch.uint16), rounded_bf16_bits(weight, k))
            base_error = errors(base(x), F.linear(x, decoded))
            if k != 2:
                rows.append({"projection": projection, "k": k, "d_factor": 0,
                             "base_triton_vs_decoded_bf16": base_error})
                continue
            residual = weight.float() - decoded.float()
            for factor in (1, 2):
                bank = ProjectionBank(n * factor, seed=0)
                signs, norms = pack_residual(residual, bank)
                reference = PackedResidual(signs, norms, n, bank.d, bank, "reference")(x)
                triton = PackedResidual(signs, norms, n, bank.d, bank, "triton")(x)
                rows.append({"projection": projection, "k": k, "d_factor": factor,
                             "base_triton_vs_decoded_bf16": base_error,
                             "qjl_triton_vs_reference": errors(triton, reference),
                             "residual_norm_ratio": (
                                 torch.linalg.vector_norm(residual).item()
                                 / torch.linalg.vector_norm(weight.float()).item())})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": 2026, "rows": rows}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
