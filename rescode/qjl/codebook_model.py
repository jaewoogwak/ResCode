"""Independent Qwen 2-bit codebook base plus packed QJL residual model."""

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook_base import PackedCodebookBase, encode_base
from .core import PackedResidual, ProjectionBank, pack_residual, qjl_triton_fused_gate_up
from .exponent_model import PROJECTIONS, _install


FORMAT = "rescode-codebook-base-qjl-v1"


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


def encode_projection(weight, bank, backend, iterations=8, row_chunk=128):
    state = encode_base(weight, iterations=iterations, row_chunk=row_chunk)
    base = PackedCodebookBase(state, weight.device)
    decoded = base.decode()
    residual = weight.float() - decoded
    if not torch.allclose(decoded + residual, weight.float(), rtol=1e-6, atol=1e-6):
        raise AssertionError("base + residual reconstruction failed")
    packed, norms = pack_residual(residual, bank, row_chunk=row_chunk)
    qjl = PackedResidual(packed, norms, weight.shape[1], bank.d, bank, backend)
    weight_sq = weight.float().square().sum().item()
    residual_sq = residual.square().sum().item()
    row_weight = torch.linalg.vector_norm(weight.float(), dim=1)
    row_residual = torch.linalg.vector_norm(residual, dim=1)
    row_ratio = row_residual / row_weight.clamp_min(1e-30)
    record = {"rows": weight.shape[0], "cols": weight.shape[1],
              "weight_squared_norm": weight_sq,
              "residual_squared_norm": residual_sq,
              "residual_frobenius_ratio": math.sqrt(residual_sq / weight_sq) if weight_sq else None,
              "base_only_relative_frobenius_error": math.sqrt(residual_sq / weight_sq) if weight_sq else None,
              "row_ratio_mean": row_ratio.mean().item(),
              "row_ratio_median": row_ratio.median().item(),
              "row_ratio_p90": torch.quantile(row_ratio, 0.9).item(),
              "row_ratio_max": row_ratio.max().item(),
              "base_bytes": base.resident_bytes(),
              "qjl_bytes": packed.numel() + norms.numel() * norms.element_size(),
              "bits_per_weight": 8 * (base.resident_bytes() + packed.numel()
                                      + norms.numel() * norms.element_size()) / weight.numel()}
    stored = {"base": state, "packed": packed.cpu(), "norms": norms.cpu()}
    return base, qjl, stored, record


@torch.no_grad()
def compress_model(model, config):
    """Replace routed experts only; router, shared expert and other modules survive."""
    banks, data, records = {}, {}, []
    model._codebook_projection_banks = banks
    for layer in config["layers"]:
        block = model.model.layers[layer].mlp
        if len(block.experts) != 60:
            raise ValueError(f"layer {layer}: expected 60 routed experts")
        router, shared, shared_gate, top_k = (block.gate,
            getattr(block, "shared_expert", None),
            getattr(block, "shared_expert_gate", None), block.top_k)
        experts, states = [], {}
        original = block.experts
        for expert_id, expert in enumerate(original):
            bases, residuals, parts = {}, {}, {}
            for key, name in PROJECTIONS.items():
                weight = getattr(expert, name).weight.detach()
                bank = bank_for(banks, weight.shape[1], config["qjl_d_factor"],
                                config["projection_seed"])
                base, qjl, state, record = encode_projection(
                    weight, bank, config["backend"], config["iterations"],
                    config["row_chunk"])
                bases[key], residuals[key], parts[key] = base, qjl, state
                records.append({"layer": layer, "expert": expert_id,
                                "projection": key, **record})
            experts.append(CodebookExpert(bases, residuals))
            states[expert_id] = parts
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
    return data, records


def save_checkpoint(data, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save({"format": FORMAT, "config": config, "layers": data}, tmp)
    os.replace(tmp, path)
    base_bytes = sum(p["base"]["packed"].numel() + p["base"]["scales"].numel() * 2
                     + p["base"]["zero_indices"].numel() * 4
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
                base = PackedCodebookBase(part["base"], device)
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
