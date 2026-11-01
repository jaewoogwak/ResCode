#!/usr/bin/env python3
"""Compare ResCode QJL backends on one saved synthetic checkpoint per family.

The representative block has original routing, two QJL residual experts, two
BF16 singletons, and the Qwen shared expert. It does not measure full HF models.
"""

import argparse
import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.qjl.model import convert_model, load_qjl_checkpoint, save_qjl_checkpoint
from rescode.qjl.core import qjl_triton_fused_gate_up


BACKENDS = ("reference", "triton_legacy", "triton")


class Expert(nn.Module):
    def __init__(self, hidden, intermediate, family):
        super().__init__()
        self.family = family
        if family == "qwen":
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        else:
            self.w1 = nn.Linear(hidden, intermediate, bias=False)
            self.w3 = nn.Linear(hidden, intermediate, bias=False)
            self.w2 = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        if self.family == "qwen":
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, hidden, intermediate, family):
        super().__init__()
        self.gate = nn.Linear(hidden, 4, bias=False)
        self.experts = nn.ModuleList([Expert(hidden, intermediate, family) for _ in range(4)])
        self.top_k = 2
        self.norm_topk_prob = family == "mixtral"
        if family == "qwen":
            self.shared_expert = Expert(hidden, intermediate, family)
            self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)


class Model(nn.Module):
    def __init__(self, hidden, intermediate, family):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        if family == "qwen":
            layer.mlp = Block(hidden, intermediate, family)
        else:
            layer.block_sparse_moe = Block(hidden, intermediate, family)
        self.model.layers = nn.ModuleList([layer])


def error(reference, candidate):
    left, right = reference.float(), candidate.float()
    delta = left - right
    return {"rmse": float(delta.square().mean().sqrt()),
            "relative_l2": float(torch.linalg.vector_norm(delta) /
                                 torch.linalg.vector_norm(left).clamp_min(1e-12)),
            "max_abs": float(delta.abs().max()),
            "bf16_exact_fraction": float((reference == candidate).float().mean())}


def timing(fn, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        del result
        times = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
            del result
    return {"median_ms": statistics.median(times), "peak_extra_allocated_bytes": peak}


def stages(block, x, bank, expert_id=0):
    cluster = block._qjl_clusters["0"]
    residual = cluster.residuals[str(expert_id)]
    p = bank.get(x.shape[-1], x.device)
    z = x.to(p.dtype) @ p
    gate_r = residual["gate"].forward_projected(z, x.dtype)
    up_r = residual["up"].forward_projected(z, x.dtype)
    gate = cluster.bases["gate"](x) + gate_r
    up = cluster.bases["up"](x) + up_r
    middle = F.silu(gate) * up
    p_down = bank.get(middle.shape[-1], x.device)
    down_z = middle.to(p_down.dtype) @ p_down
    down_r = residual["down"].forward_projected(down_z, x.dtype)
    output = cluster.bases["down"](middle) + down_r
    return {"gate_residual": gate_r, "up_residual": up_r,
            "down_residual": down_r, "gate_shared_plus_residual": gate,
            "up_shared_plus_residual": up, "swiglu": middle,
            "expert": output, "expert_forward": cluster.forward_expert(expert_id, x),
            "gate_residual_fp32": residual["gate"].forward_projected(z, torch.float32),
            "up_residual_fp32": residual["up"].forward_projected(z, torch.float32),
            "down_residual_fp32": residual["down"].forward_projected(down_z, torch.float32)}


def run_family(family, args):
    torch.manual_seed(2026)
    original = Model(args.hidden, args.intermediate, family).to("cuda", dtype=torch.bfloat16)
    untouched = copy.deepcopy(original)
    groups = {0: [[0, 1], [2], [3]]}
    config = {"qjl_d": args.qjl_d, "projection_seed": 0, "base_method": "mean",
              "residual_mode": "qjl", "targets": "gate_up_down", "backend": "triton",
              "delta": 1e-3, "cg_iterations": 30}
    convert_model(original, family, groups, {0: torch.ones(4)}, {}, {}, config)
    with tempfile.TemporaryDirectory() as temp:
        checkpoint = str(Path(temp) / "qjl_state.pt")
        saved = save_qjl_checkpoint(original, family, groups, config, checkpoint)
        models = {}
        for backend in BACKENDS:
            model = copy.deepcopy(untouched)
            load_qjl_checkpoint(model, checkpoint, backend=backend)
            models[backend] = model
        del original, untouched
        result = {"checkpoint_format": "rescode-direct-qjl-v1",
                  "checkpoint_bytes": saved["checkpoint_bytes"],
                  "shape": {"hidden": args.hidden, "intermediate": args.intermediate,
                            "qjl_d": args.qjl_d, "experts": 4,
                            "qjl_experts": 2, "bf16_singletons": 2},
                  "tokens": {}}
        blocks = {backend: getattr(models[backend].model.layers[0],
                                   "mlp" if family == "qwen" else "block_sparse_moe")
                  for backend in BACKENDS}
        for count in args.token_counts:
            torch.manual_seed(1000 + count)
            x = torch.randn(count, args.hidden, device="cuda", dtype=torch.bfloat16)
            reference_stage = stages(blocks["reference"], x,
                                     models["reference"]._qjl_projection_bank)
            reference_output, reference_logits = blocks["reference"](x)
            selected = torch.topk(reference_logits, 2, dim=-1).indices
            row = {"numerical": {}, "timing": {}, "routing_identical": {},
                   "singleton_identical": {}}
            for backend in BACKENDS:
                block = blocks[backend]
                stage = stages(block, x, models[backend]._qjl_projection_bank)
                output, logits = block(x)
                row["numerical"][backend] = {name: error(reference_stage[name], stage[name])
                                              for name in reference_stage}
                candidate_down_common = block._qjl_clusters["0"].residuals["0"]["down"](
                    reference_stage["swiglu"])
                row["numerical"][backend]["down_residual_common_input"] = error(
                    reference_stage["down_residual"], candidate_down_common)
                p_down_common = models[backend]._qjl_projection_bank.get(
                    args.intermediate, x.device)
                down_z_common = reference_stage["swiglu"].to(p_down_common.dtype) @ p_down_common
                candidate_down_fp32_common = block._qjl_clusters["0"].residuals["0"]["down"].forward_projected(
                    down_z_common, torch.float32)
                row["numerical"][backend]["down_residual_fp32_common_input"] = error(
                    reference_stage["down_residual_fp32"], candidate_down_fp32_common)
                row["numerical"][backend]["moe"] = error(reference_output, output)
                row["routing_identical"][backend] = (torch.equal(logits, reference_logits)
                    and torch.equal(torch.topk(logits, 2, dim=-1).indices, selected))
                row["singleton_identical"][backend] = all(torch.equal(
                    block.experts[i](x), blocks["reference"].experts[i](x)) for i in (2, 3))
                cluster = block._qjl_clusters["0"]
                residual = cluster.residuals["0"]
                p = models[backend]._qjl_projection_bank.get(args.hidden, x.device)
                z = x.to(p.dtype) @ p
                middle = reference_stage["swiglu"]
                p_down = models[backend]._qjl_projection_bank.get(args.intermediate, x.device)
                down_z = middle.to(p_down.dtype) @ p_down
                functions = {
                    "gate": lambda: residual["gate"].forward_projected(z, x.dtype),
                    "up": lambda: residual["up"].forward_projected(z, x.dtype),
                    "down": lambda: residual["down"].forward_projected(down_z, x.dtype),
                    "gate_up_pair": (lambda: qjl_triton_fused_gate_up(
                        z, residual["gate"], residual["up"], x.dtype))
                        if backend == "triton" else
                        (lambda: (residual["gate"].forward_projected(z, x.dtype),
                                  residual["up"].forward_projected(z, x.dtype))),
                    "expert": lambda: cluster.forward_expert(0, x),
                    "moe": lambda: block(x),
                }
                row["timing"][backend] = {name: timing(fn, args.warmup, args.repeats)
                                           for name, fn in functions.items()}
            result["tokens"][str(count)] = row
            print(f"[{family}] {count} tokens measured", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/qjl_direct/triton_backend_benchmark.json")
    parser.add_argument("--families", default="qwen,mixtral")
    parser.add_argument("--token-counts", default="1,4,32,128")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1408)
    parser.add_argument("--qjl-d", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    args.token_counts = tuple(int(v) for v in args.token_counts.split(","))
    families = tuple(args.families.split(","))
    if not torch.cuda.is_available() or any(f not in ("qwen", "mixtral") for f in families):
        parser.error("CUDA and qwen/mixtral families are required")
    if min(args.hidden, args.intermediate, args.qjl_d, args.warmup, args.repeats,
           *args.token_counts) < 1:
        parser.error("all dimensions and iteration counts must be positive")
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "triton": __import__("triton").__version__,
              "protocol": {"warmup": args.warmup, "repeats": args.repeats,
                           "same_checkpoint_per_family": True,
                           "cuda_events": True, "synthetic_block": True},
              "families": {family: run_family(family, args) for family in families}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
