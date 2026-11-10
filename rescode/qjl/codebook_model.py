"""Qwen packed Base (codebook or exponent/mantissa) plus QJL residual model."""

import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook_base import PackedCodebookBase, encode_base
from .core import PackedResidual, ProjectionBank, pack_residual, qjl_triton_fused_gate_up
from .exponent_mantissa_base import (PackedExponentMantissaBase,
    encode_exponent_mantissa_base, exponent_mantissa_base_bytes,
    reconstruct_base_from_weight)
from .exponent_model import PROJECTIONS, _install


FORMAT = "rescode-codebook-base-qjl-v1"


def base_state_bytes(state):
    if state.get("format"):
        return exponent_mantissa_base_bytes(state)
    return (state["packed"].numel() + state["scales"].numel() * 2
            + state["zero_indices"].numel() * 4)


def restore_base(state, device, method):
    if method == "codebook":
        return PackedCodebookBase(state, device)
    if method == "exponent-mantissa":
        return PackedExponentMantissaBase(state, device)
    raise ValueError(f"unknown base method: {method}")


def gpu_resident_breakdown(model):
    """Count compressed expert buffers actually placed on CUDA devices."""
    base_bytes = qjl_bytes = 0
    for block in model.model.layers:
        experts = getattr(block.mlp, "_compact_experts", ())
        for expert in experts:
            for base in expert.bases.values():
                first = next(base.buffers())
                if first.is_cuda:
                    base_bytes += base.resident_bytes()
            for residual in expert.residuals.values():
                if residual.packed.is_cuda:
                    qjl_bytes += (residual.packed.numel() * residual.packed.element_size()
                                  + residual.norms.numel() * residual.norms.element_size())
    projection_bytes = sum(bank.resident_bytes("cuda") for bank in
                           getattr(model, "_codebook_projection_banks", {}).values())
    return {"gpu_resident_base_bytes": base_bytes,
            "gpu_resident_qjl_bytes": qjl_bytes,
            "gpu_resident_projection_bytes": projection_bytes,
            "gpu_resident_compact_total_bytes": base_bytes + qjl_bytes + projection_bytes}


def bank_for(banks, input_dim, factor, seed):
    if input_dim not in banks:
        d = int(round(input_dim * factor))
        if d < 1:
            raise ValueError("QJL dimension must be positive")
        banks[input_dim] = ProjectionBank(d, seed)
    return banks[input_dim]


class CodebookExpert(nn.Module):
    def __init__(self, bases: dict, residuals: dict):
        super().__init__()
        self.bases = nn.ModuleDict(bases)
        self.residuals = nn.ModuleDict(residuals)

    def _projection(self, key, x, z=None):
        residual = self.residuals[key]
        return self.bases[key](x) + (residual.forward_projected(z, x.dtype)
                                      if z is not None else residual(x))

    def forward(self, x):
        gate, up = self.residuals["gate"], self.residuals["up"]
        p = gate.bank.get(x.shape[-1], x.device)
        z = x.reshape(-1, x.shape[-1]).to(p.dtype) @ p
        if (gate.backend == up.backend == "triton" and gate.d == up.d
                and gate.packed.shape == up.packed.shape):
            gr, ur = qjl_triton_fused_gate_up(z, gate, up, x.dtype)
            g = self.bases["gate"](x) + gr
            u = self.bases["up"](x) + ur
        else:
            g = self._projection("gate", x, z)
            u = self._projection("up", x, z)
        return self._projection("down", F.silu(g) * u)


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def encode_projection(weight, bank, backend, iterations=8, row_chunk=None,
                      *, base_row_chunk=512, qjl_row_chunk=128,
                      base_method="codebook"):
    """Encode one matrix and report synchronized stage timings.

    ``row_chunk`` is retained for callers of the original independent API.
    The normal compression path uses separate Base and QJL chunks.
    """
    if row_chunk is not None:
        base_row_chunk = qjl_row_chunk = row_chunk
    _synchronize(weight.device)
    if weight.is_cuda:
        torch.cuda.reset_peak_memory_stats(weight.device)
        starting_bytes = torch.cuda.memory_allocated(weight.device)
    else:
        starting_bytes = None
    start = time.perf_counter()
    if base_method == "codebook":
        state = encode_base(weight, iterations=iterations, row_chunk=base_row_chunk,
                            return_cpu=False)
    elif base_method == "exponent-mantissa":
        state = encode_exponent_mantissa_base(weight)
    else:
        raise ValueError(f"unknown base method: {base_method}")
    _synchronize(weight.device)
    after_base = time.perf_counter()
    base = restore_base(state, weight.device, base_method)
    decoded = (reconstruct_base_from_weight(weight, base.beta)
               if base_method == "exponent-mantissa" else base.decode())
    weight_fp32 = weight.float()
    residual = torch.where(torch.isfinite(weight_fp32), weight_fp32 - decoded,
                           torch.zeros_like(decoded))
    if not torch.allclose(decoded + residual, weight_fp32, rtol=1e-6, atol=1e-6,
                          equal_nan=True):
        raise AssertionError("base + residual reconstruction failed")
    _synchronize(weight.device)
    after_residual = time.perf_counter()
    packed, norms = pack_residual(residual, bank, row_chunk=qjl_row_chunk)
    qjl = PackedResidual(packed, norms, weight.shape[1], bank.d, bank, backend)
    _synchronize(weight.device)
    after_qjl = time.perf_counter()
    row_weight = torch.linalg.vector_norm(weight_fp32, dim=1)
    row_residual = torch.linalg.vector_norm(residual, dim=1)
    row_ratio = row_residual / row_weight.clamp_min(1e-30)
    values = torch.stack((weight_fp32.square().sum(), residual.square().sum(),
                          row_ratio.mean(), row_ratio.median(),
                          torch.quantile(row_ratio, 0.9), row_ratio.max()))
    weight_sq, residual_sq, row_mean, row_median, row_p90, row_max = values.cpu().tolist()
    _synchronize(weight.device)
    after_diagnostics = time.perf_counter()
    stored_base = {key: (value.cpu() if isinstance(value, torch.Tensor) else value)
                   for key, value in state.items()}
    stored = {"base": stored_base, "packed": packed.cpu(), "norms": norms.cpu()}
    _synchronize(weight.device)
    after_transfer = time.perf_counter()
    base_bytes = base_state_bytes(state)
    qjl_bytes = packed.numel() + norms.numel() * norms.element_size()
    record = {"rows": weight.shape[0], "cols": weight.shape[1],
              "weight_squared_norm": weight_sq,
              "residual_squared_norm": residual_sq,
              "residual_frobenius_ratio": math.sqrt(residual_sq / weight_sq) if weight_sq else None,
              "base_only_relative_frobenius_error": math.sqrt(residual_sq / weight_sq) if weight_sq else None,
              "row_ratio_mean": row_mean,
              "row_ratio_median": row_median,
              "row_ratio_p90": row_p90,
              "row_ratio_max": row_max,
              "base_method": base_method,
              "base_bytes": base_bytes,
              "base_gpu_resident_bytes": base.resident_bytes(),
              "qjl_bytes": qjl_bytes,
              "qjl_gpu_resident_bytes": qjl_bytes,
              "base_bits_per_weight": 8 * base_bytes / weight.numel(),
              "qjl_bits_per_weight": 8 * qjl_bytes / weight.numel(),
              "bits_per_weight": 8 * (base_bytes + qjl_bytes) / weight.numel(),
              "time_encode_base_s": after_base - start,
              "time_residual_s": after_residual - after_base,
              "time_pack_residual_s": after_qjl - after_residual,
              "time_diagnostics_s": after_diagnostics - after_qjl,
              "time_cpu_transfer_s": after_transfer - after_diagnostics,
              "time_total_s": after_transfer - start,
              "peak_extra_cuda_bytes": (torch.cuda.max_memory_allocated(weight.device)
                                         - starting_bytes if starting_bytes is not None else None)}
    return base, qjl, stored, record


@torch.no_grad()
def compress_model(model, config):
    """Replace routed experts only; router, shared expert and other modules survive."""
    banks, data, records, timing_records = {}, {}, [], []
    model._codebook_projection_banks = banks
    for layer in config["layers"]:
        layer_start = time.perf_counter()
        block = model.model.layers[layer].mlp
        if len(block.experts) != 60:
            raise ValueError(f"layer {layer}: expected 60 routed experts")
        router, shared, shared_gate, top_k = (block.gate,
            getattr(block, "shared_expert", None),
            getattr(block, "shared_expert_gate", None), block.top_k)
        experts, states = [], {}
        original = block.experts
        for expert_id, expert in enumerate(original):
            _synchronize(next(expert.parameters()).device)
            expert_start = time.perf_counter()
            bases, residuals, parts = {}, {}, {}
            projection_timings = {}
            for key, name in PROJECTIONS.items():
                weight = getattr(expert, name).weight.detach()
                bank = bank_for(banks, weight.shape[1], config["qjl_d_factor"],
                                config["projection_seed"])
                base, qjl, state, record = encode_projection(
                    weight, bank, config["backend"], config["iterations"],
                    base_row_chunk=config.get("base_row_chunk", config.get("row_chunk", 128)),
                    qjl_row_chunk=config.get("qjl_row_chunk", config.get("row_chunk", 128)),
                    base_method=config.get("base_method", "codebook"))
                bases[key], residuals[key], parts[key] = base, qjl, state
                projection_timings[key] = record["time_total_s"]
                records.append({"layer": layer, "expert": expert_id,
                                "projection": key, **record})
            experts.append(CodebookExpert(bases, residuals))
            states[expert_id] = parts
            _synchronize(next(expert.parameters()).device)
            timing_records.append({"layer": layer, "expert": expert_id,
                                   "seconds": time.perf_counter() - expert_start,
                                   "projection_seconds": projection_timings})
        _install(block, experts)
        if (block.gate is not router or getattr(block, "shared_expert", None) is not shared
                or getattr(block, "shared_expert_gate", None) is not shared_gate
                or block.top_k != top_k):
            raise AssertionError("router, shared expert or routing topology changed")
        data[layer] = states
        del original
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[codebook] layer {layer}: 60 routed experts converted", flush=True)
        timing_records.append({"layer": layer, "expert": None,
                               "seconds": time.perf_counter() - layer_start,
                               "projection_seconds": None})
    return data, records, timing_records


def save_checkpoint(data, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save({"format": FORMAT, "config": config, "layers": data}, tmp)
    os.replace(tmp, path)
    base_bytes = sum(base_state_bytes(p["base"])
                     for experts in data.values() for expert in experts.values()
                     for p in expert.values())
    qjl_bytes = sum(p["packed"].numel() + p["norms"].numel() * 4
                    for experts in data.values() for expert in experts.values()
                    for p in expert.values())
    total = path.stat().st_size
    return {"checkpoint_bytes": total, "base_tensor_bytes": base_bytes,
            "qjl_tensor_bytes": qjl_bytes, "metadata_bytes": total - base_bytes - qjl_bytes}


def load_checkpoint(model, path, backend=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("wrong Codebook Base checkpoint format")
    config = payload["config"]
    banks = {}
    model._codebook_projection_banks = banks
    for layer, expert_states in payload["layers"].items():
        block = model.model.layers[int(layer)].mlp
        device = next(block.parameters()).device
        experts = []
        for expert_id in range(60):
            states = expert_states[expert_id]
            bases, residuals = {}, {}
            for key, part in states.items():
                base = restore_base(part["base"], device,
                                    config.get("base_method", "codebook"))
                bank = bank_for(banks, base.shape[1], config["qjl_d_factor"],
                                config["projection_seed"])
                residual = PackedResidual(part["packed"].to(device),
                    part["norms"].to(device), base.shape[1], bank.d,
                    bank, backend or config["backend"])
                bases[key], residuals[key] = base, residual
            experts.append(CodebookExpert(bases, residuals))
        _install(block, experts)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload
