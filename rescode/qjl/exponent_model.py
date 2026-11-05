"""Independent Qwen compact-base plus QJL-residual inference and checkpoint."""

import os
import time
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compact_base import (CompactBase, encode_base, encoded_bytes,
                           rounded_bf16_bits, storage_payload)
from .packed_exponent import BLOCK_FORMAT


CHECKPOINT_V1 = "rescode-exponent-base-qjl-v1"
CHECKPOINT_V2 = "rescode-exponent-base-qjl-v2"
from .core import PackedResidual, ProjectionBank, pack_residual, qjl_triton_fused_gate_up


PROJECTIONS = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def _error(output, target):
    difference = output.float() - target.float()
    denominator = torch.linalg.vector_norm(target.float()).item()
    return {"relative_l2": (torch.linalg.vector_norm(difference).item() / denominator
                            if denominator else None),
            "mse": difference.square().mean().item(),
            "null_reason": None if denominator else "zero_reference_norm"}


def fit_alpha(exact: torch.Tensor, approximate: torch.Tensor) -> float:
    """Unconstrained calibration-only least-squares scalar."""
    denominator = approximate.float().square().sum().item()
    if denominator == 0:
        return 0.0
    value = (exact.float() * approximate.float()).sum().item() / denominator
    return float(value)


class CompactExpert(nn.Module):
    def __init__(self, bases: dict, residuals: dict, alphas: dict):
        super().__init__()
        self.bases = nn.ModuleDict(bases)
        self.residuals = nn.ModuleDict(residuals)
        self.alphas = dict(alphas)

    def _projection(self, key, x, projected=None):
        y = self.bases[key](x)
        if key in self.residuals and self.alphas[key] != 0:
            r = (self.residuals[key].forward_projected(projected, x.dtype)
                 if projected is not None else self.residuals[key](x))
            y = y + self.alphas[key] * r
        return y

    def forward(self, x):
        gate = self.residuals["gate"] if "gate" in self.residuals else None
        up = self.residuals["up"] if "up" in self.residuals else None
        z = None
        if gate is not None and up is not None and self.alphas["gate"] and self.alphas["up"]:
            p = gate.bank.get(x.shape[-1], x.device)
            z = x.to(p.dtype) @ p
        if (z is not None and gate.backend == up.backend == "triton"
                and gate.d == up.d and gate.packed.shape == up.packed.shape):
            gr, ur = qjl_triton_fused_gate_up(z, gate, up, x.dtype)
            g = self.bases["gate"](x) + self.alphas["gate"] * gr
            u = self.bases["up"](x) + self.alphas["up"] * ur
        else:
            g = self._projection("gate", x, z)
            u = self._projection("up", x, z)
        middle = F.silu(g) * u
        return self._projection("down", middle)


def _compact_moe_forward(self, hidden_states):
    shape = hidden_states.shape
    flat = hidden_states.reshape(-1, shape[-1])
    router_logits = self.gate(flat)
    weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    weights, selected = torch.topk(weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(flat.dtype)
    output = torch.zeros_like(flat)
    for expert_id in selected.unique().tolist():
        token, slot = torch.where(selected == expert_id)
        x = flat.index_select(0, token)
        y = self._compact_experts[expert_id](x)
        output.index_add_(0, token, (y * weights[token, slot, None]).to(output.dtype))
    if hasattr(self, "shared_expert"):
        shared = self.shared_expert(flat)
        if getattr(self, "shared_expert_gate", None) is not None:
            shared = torch.sigmoid(self.shared_expert_gate(flat)) * shared
        output += shared
    return output.reshape(shape), router_logits


def _install(block, experts):
    if len(experts) != len(block.experts):
        raise ValueError("all original routed experts must be converted")
    block.experts = nn.ModuleList([nn.Identity() for _ in experts])
    block._compact_experts = nn.ModuleList(experts)
    block.forward = MethodType(_compact_moe_forward, block)


def _samples(samples, layer, proj, expert, device, count):
    entry = samples.get(layer, {}).get(proj, {}).get(expert)
    return None if entry is None else entry[0][:count].to(device)


@torch.no_grad()
def compress_model(model, samples, heldout, probes, config, preencoded=None):
    """Convert every routed expert in selected Qwen MoE layers in place."""
    factor = config["d_factor"]
    seed = config["projection_seed"]
    banks = {2048: ProjectionBank(2048 * factor, seed),
             1408: ProjectionBank(1408 * factor, seed)} if factor else {}
    model._compact_projection_banks = banks
    records = []
    data = {}
    diagnostics_seconds = 0.0
    for layer in config["layers"]:
        block = model.model.layers[layer].mlp
        if len(block.experts) != 60:
            raise ValueError(f"layer {layer}: expected 60 routed experts")
        router = block.gate
        shared = getattr(block, "shared_expert", None)
        shared_gate = getattr(block, "shared_expert_gate", None)
        top_k = block.top_k
        original = block.experts
        probe = probes.get(layer)
        original_moe = None
        original_logits = None
        if probe:
            probe_x = probe[0].to(next(block.parameters()).device)
            original_moe, original_logits = block(probe_x)
        experts = []
        layer_data = {}
        for expert_id, expert in enumerate(original):
            bases, residuals, alphas = {}, {}, {}
            expert_data = {}
            weights = {key: getattr(expert, name).weight for key, name in PROJECTIONS.items()}
            for key, weight in weights.items():
                encoded = (preencoded[layer][expert_id][key]["base"]
                           if preencoded is not None
                           else encode_base(weight.detach(), config["k"]))
                base = CompactBase(encoded, weight.device)
                decoded = base.decode()
                diag_start = time.perf_counter()
                # Bitwise roundtrip is checked against the independent BF16 rounder.
                if not torch.equal(decoded.view(torch.uint16),
                                   rounded_bf16_bits(weight.detach(), config["k"])):
                    raise AssertionError("compact base roundtrip changed BF16 bits")
                residual = weight.float() - decoded.float()
                weight_norm = torch.linalg.vector_norm(weight.float()).item()
                ratio = (torch.linalg.vector_norm(residual).item() / weight_norm
                         if weight_norm else None)
                reconstruction = decoded.float() + residual
                decomposition = _error(reconstruction, weight)
                bases[key] = base
                state = {"base": encoded, "alpha": 0.0}
                x = _samples(heldout, layer, key, expert_id, weight.device,
                             config["diagnostic_samples"])
                if factor:
                    bank = banks[weight.shape[1]]
                    bits, norms = pack_residual(residual, bank)
                    packed = PackedResidual(bits, norms, weight.shape[1], bank.d,
                                            bank, config["backend"])
                    residuals[key] = packed
                    if config["alpha"] == "calibrated":
                        train_x = _samples(samples, layer, key, expert_id, weight.device,
                                           config["diagnostic_samples"])
                        if train_x is None or not train_x.numel():
                            raise ValueError(f"missing calibration input: {layer}/{expert_id}/{key}")
                        exact = F.linear(train_x.float(), residual)
                        approx = packed(train_x).float()
                        alpha = fit_alpha(exact, approx)
                        state["alpha_training_samples"] = train_x.shape[0]
                    else:
                        alpha = float(config["alpha"])
                    alphas[key] = alpha
                    state.update({"packed": bits.cpu(), "norms": norms.cpu(),
                                  "alpha": alpha})
                else:
                    alphas[key] = 0.0
                if x is not None and x.numel():
                    target = F.linear(x, weight)
                    approx_y = base(x)
                    if factor:
                        approx_y = approx_y + alphas[key] * residuals[key](x)
                    output_error = _error(approx_y, target)
                else:
                    output_error = {"relative_l2": None, "mse": None,
                                    "null_reason": "no_heldout_activation"}
                records.append({"layer": layer, "expert": expert_id,
                                "projection": key, "k": config["k"], "d": factor * weight.shape[1],
                                "alpha": alphas[key], "residual_norm_ratio": ratio,
                                "residual_norm_null_reason": None if weight_norm else "zero_weight_norm",
                                "decomposition": decomposition,
                                "output": output_error,
                                "exponent_entropy_bits": encoded["exponent_entropy_bits"],
                                "base_stored_bytes": encoded_bytes(encoded),
                                "base_stored_bits_per_weight": 8 * encoded_bytes(encoded) / weight.numel(),
                                "base_gpu_resident_bytes": base.gpu_resident_bytes()})
                diagnostics_seconds += time.perf_counter() - diag_start
                expert_data[key] = state
            # Expert output uses the same held-out hidden activation for all methods.
            hidden = _samples(heldout, layer, "gate", expert_id,
                              weights["gate"].device, config["diagnostic_samples"])
            converted = CompactExpert(bases, residuals, alphas)
            if hidden is not None and hidden.numel():
                diag_start = time.perf_counter()
                target = expert(hidden)
                actual = converted(hidden)
                records.append({"layer": layer, "expert": expert_id,
                                "projection": "expert_output", "output": _error(actual, target)})
                diagnostics_seconds += time.perf_counter() - diag_start
            experts.append(converted)
            layer_data[expert_id] = expert_data
        _install(block, experts)
        if (block.gate is not router or getattr(block, "shared_expert", None) is not shared
                or getattr(block, "shared_expert_gate", None) is not shared_gate
                or block.top_k != top_k):
            raise AssertionError(f"router, top-k or shared expert changed in layer {layer}")
        if original_moe is not None:
            diag_start = time.perf_counter()
            actual, logits = block(probe_x)
            if not torch.equal(logits, original_logits):
                raise AssertionError(f"router changed in layer {layer}")
            records.append({"layer": layer, "projection": "moe_output",
                            "output": _error(actual, original_moe)})
            diagnostics_seconds += time.perf_counter() - diag_start
        data[layer] = layer_data
        del original
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[compact] layer {layer}: 60 experts converted", flush=True)
    return data, records, diagnostics_seconds


def save_checkpoint(model, data, config, path):
    stored_data = {
        layer: {
            expert_id: {
                proj: {**part, "base": storage_payload(part["base"])}
                for proj, part in expert.items()
            } for expert_id, expert in experts.items()
        } for layer, experts in data.items()
    }
    first_base = next(iter(next(iter(next(iter(data.values())).values())).values()))["base"]
    checkpoint_format = (CHECKPOINT_V2 if first_base["format"] == BLOCK_FORMAT
                         else CHECKPOINT_V1)
    payload = {"format": checkpoint_format, "config": config,
               "layers": stored_data}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    base_bytes = sum(encoded_bytes(p["base"]) for layer in data.values()
                     for expert in layer.values() for p in expert.values())
    qjl_bytes = sum(p["packed"].numel() * p["packed"].element_size()
                    + p["norms"].numel() * p["norms"].element_size()
                    for layer in data.values() for expert in layer.values()
                    for p in expert.values() if "packed" in p)
    return {"checkpoint_bytes": os.path.getsize(path),
            "base_encoded_bytes": base_bytes, "qjl_tensor_bytes": qjl_bytes,
            "metadata_bytes": os.path.getsize(path) - base_bytes - qjl_bytes}


def load_checkpoint(model, path, backend=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") not in (CHECKPOINT_V1, CHECKPOINT_V2):
        raise ValueError("wrong checkpoint format")
    config = payload["config"]
    factor = config["d_factor"]
    banks = {2048: ProjectionBank(2048 * factor, config["projection_seed"]),
             1408: ProjectionBank(1408 * factor, config["projection_seed"])} if factor else {}
    model._compact_projection_banks = banks
    for layer, expert_states in payload["layers"].items():
        block = model.model.layers[int(layer)].mlp
        device = next(block.parameters()).device
        experts = []
        for expert_id in range(60):
            states = expert_states[expert_id]
            bases, residuals, alphas = {}, {}, {}
            for key, state in states.items():
                base = CompactBase(state["base"], device)
                bases[key] = base
                alphas[key] = float(state["alpha"])
                if "packed" in state:
                    bank = banks[base.shape[1]]
                    residuals[key] = PackedResidual(
                        state["packed"].to(device), state["norms"].to(device),
                        base.shape[1], bank.d, bank, backend or config["backend"])
            experts.append(CompactExpert(bases, residuals, alphas))
        _install(block, experts)
    if torch.cuda.is_available():
        # The original routed BF16 experts have been replaced. Release their
        # allocator cache so process-level VRAM reflects the packed runtime.
        torch.cuda.empty_cache()
    return payload
