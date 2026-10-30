"""Qwen/Mixtral MoE conversion with unchanged routing and expert identities."""

import os
from types import MethodType
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import fit_shared_base, variance_objective
from .core import ExactResidual, PackedResidual, ProjectionBank, pack_residual, parse_targets


def projections(expert: nn.Module, family: str):
    if family == "mixtral":
        return {"gate": expert.w1.weight, "up": expert.w3.weight,
                "down": expert.w2.weight}
    return {"gate": expert.gate_proj.weight, "up": expert.up_proj.weight,
            "down": expert.down_proj.weight}


class SharedBaseCluster(nn.Module):
    def __init__(self, bases: dict, residuals: dict):
        super().__init__()
        self.bases = nn.ModuleDict()
        self.residuals = nn.ModuleDict()
        for key, weight in bases.items():
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                               device=weight.device, dtype=weight.dtype)
            linear.weight = nn.Parameter(weight.detach().contiguous(), requires_grad=False)
            self.bases[key] = linear
        for expert_id, parts in residuals.items():
            self.residuals[str(expert_id)] = nn.ModuleDict(parts)

    def forward_expert(self, expert_id: int, x: torch.Tensor,
                       hidden_z: torch.Tensor = None) -> torch.Tensor:
        residual = self.residuals[str(expert_id)]
        gate_r = (residual["gate"].forward_projected(hidden_z, x.dtype)
                  if hidden_z is not None and isinstance(residual["gate"], PackedResidual)
                  else residual["gate"](x))
        up_r = (residual["up"].forward_projected(hidden_z, x.dtype)
                if hidden_z is not None and isinstance(residual["up"], PackedResidual)
                else residual["up"](x))
        gate = self.bases["gate"](x) + gate_r
        up = self.bases["up"](x) + up_r
        middle = F.silu(gate) * up
        return self.bases["down"](middle) + residual["down"](middle)


def _qjl_moe_forward(self, hidden_states: torch.Tensor):
    shape = hidden_states.shape
    flat = hidden_states.reshape(-1, shape[-1])
    router_logits = self.gate(flat)
    weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    weights, selected = torch.topk(weights, self.top_k, dim=-1)
    if self._qjl_family == "mixtral" or self.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(flat.dtype)
    output = torch.zeros_like(flat)
    # Exact expert IDs and original top-k weights, including singleton paths.
    for expert_id in selected.unique().tolist():
        token, slot = torch.where(selected == expert_id)
        if token.numel() == 0:
            continue
        x = flat.index_select(0, token)
        cluster_id = self._qjl_expert_to_cluster[expert_id]
        if cluster_id is None:
            y = self.experts[expert_id](x)
        else:
            hidden_z = None
            if self._qjl_hidden_bank is not None:
                p = self._qjl_hidden_bank.get(x.shape[-1], x.device)
                hidden_z = x.to(p.dtype) @ p
            y = self._qjl_clusters[str(cluster_id)].forward_expert(
                expert_id, x, hidden_z,
            )
        output.index_add_(0, token, (y * weights[token, slot, None]).to(output.dtype))
    if self._qjl_family == "qwen" and hasattr(self, "shared_expert"):
        shared = self.shared_expert(flat)
        if getattr(self, "shared_expert_gate", None) is not None:
            shared = torch.sigmoid(self.shared_expert_gate(flat)) * shared
        output += shared
    return output.reshape(shape), router_logits


def _block(model, family: str, layer: int):
    layer_module = model.model.layers[layer]
    return layer_module.block_sparse_moe if family == "mixtral" else layer_module.mlp


def _make_cluster(block, family, members, saliency, samples, heldout, config, bank):
    bases, residuals, diagnostics = {}, {i: {} for i in members}, {}
    target_set = set(parse_targets(config["targets"]))
    for proj in ("gate", "up", "down"):
        weights = {i: projections(block.experts[i], family)[proj].detach()
                   for i in members}
        sample = samples.get(proj, {})
        held = heldout.get(proj, {}) if heldout else None
        base, report = fit_shared_base(weights, sample, config["base_method"],
                                       saliency, config["qjl_d"], held,
                                       delta=config["delta"],
                                       max_iter=config["cg_iterations"])
        base = base.to(weights[members[0]].dtype)
        bases[proj] = base
        diagnostics[proj] = report
        diagnostics[proj]["experts"] = {}
        for i, weight in weights.items():
            residual = weight.float() - base.float()
            if config["residual_mode"] == "qjl" and proj in target_set:
                bits, norms = pack_residual(residual, bank)
                module = PackedResidual(bits, norms, weight.shape[1],
                                        config["qjl_d"], bank, config["backend"])
            else:
                module = ExactResidual(residual)
            residuals[i][proj] = module
            norm_w = torch.linalg.vector_norm(weight.float())
            ratio = float(torch.linalg.vector_norm(residual) / norm_w) if norm_w > 0 else 0.0
            theory = variance_objective(base.float(), {i: weight},
                                        {i: sample[i]} if i in sample else {},
                                        config["qjl_d"])
            empirical = None
            if i in sample:
                x, q = sample[i]
                x, q = x.to(weight.device), q.to(weight.device)
                predicted = module(x).float()
                exact = F.linear(x.float(), residual.float())
                empirical = float(((predicted - exact).square() * q[:, None]).sum())
            q_sum = float(sample[i][1].sum()) if i in sample else 0.0
            diagnostics[proj]["experts"][str(i)] = {
                "residual_frobenius_ratio": ratio,
                "theoretical_mse_sum_weighted": theory,
                "empirical_mse_sum_weighted": empirical,
                "theoretical_mse_per_weight": theory / q_sum if q_sum else None,
                "empirical_mse_per_weight": empirical / q_sum if q_sum and empirical is not None else None,
                "routed_weight_square_sum": q_sum,
            }
        print(f"[QJL] projection {proj}: base={config['base_method']} "
              f"objective {report['objective_before']:.4g} -> "
              f"{report['objective_after']:.4g}", flush=True)
    cluster = SharedBaseCluster(bases, residuals)
    expert_error = {}
    for i in members:
        if i in samples.get("gate", {}):
            x = samples["gate"][i][0].to(next(block.parameters()).device)
            original = block.experts[i](x).float()
            approx = cluster.forward_expert(i, x).float()
            expert_error[str(i)] = float(
                torch.linalg.vector_norm(approx - original)
                / torch.linalg.vector_norm(original).clamp_min(1e-12)
            )
    diagnostics["expert_output_relative_error"] = expert_error
    return cluster, diagnostics


def _install_block(block, family: str, groups: list, clusters: nn.ModuleDict):
    flattened = [i for group in groups for i in group]
    if sorted(flattened) != list(range(len(block.experts))):
        raise ValueError("groups must partition original expert IDs")
    ids = [None] * len(block.experts)
    singleton_ids = set()
    for cid, members in enumerate(groups):
        for i in members:
            ids[i] = cid if len(members) > 1 else None
        if len(members) == 1:
            singleton_ids.add(members[0])
    # ModuleList replacement releases all original non-singleton weights.
    block.experts = nn.ModuleList([
        block.experts[i] if i in singleton_ids else nn.Identity()
        for i in range(len(ids))
    ])
    block.add_module("_qjl_clusters", clusters)
    block._qjl_expert_to_cluster = ids
    block._qjl_family = family
    block._qjl_hidden_bank = None
    for cluster in clusters.values():
        for parts in cluster.residuals.values():
            for key in ("gate", "up"):
                if isinstance(parts[key], PackedResidual):
                    block._qjl_hidden_bank = parts[key].bank
                    break
            if block._qjl_hidden_bank is not None:
                break
        if block._qjl_hidden_bank is not None:
            break
    block.forward = MethodType(_qjl_moe_forward, block)


@torch.no_grad()
def convert_model(model, family: str, groups: Dict[int, list], saliency: dict,
                  samples: dict, heldout: dict, config: dict, probes: dict = None):
    """Compress selected layers in place. `groups` uses original expert IDs."""
    bank = ProjectionBank(config["qjl_d"], config["projection_seed"])
    model._qjl_projection_bank = bank
    reports = {}
    for layer_idx, layer_groups in sorted(groups.items()):
        block = _block(model, family, layer_idx)
        probe = None if probes is None else probes.get(layer_idx)
        original_probe = None
        if probe is not None:
            probe_x = probe[0].to(next(block.parameters()).device)
            with torch.no_grad():
                original_probe = block(probe_x)[0].float()
        clusters = nn.ModuleDict()
        layer_report = {}
        for cid, members in enumerate(layer_groups):
            if len(members) == 1:
                continue
            print(f"[QJL] layer {layer_idx} cluster {cid}: "
                  f"packing {len(members)} experts", flush=True)
            cluster, report = _make_cluster(
                block, family, members, saliency[layer_idx],
                samples.get(layer_idx, {}), heldout.get(layer_idx, {}), config, bank
            )
            clusters[str(cid)] = cluster
            layer_report[str(cid)] = report
            for expert_id in members:
                block.experts[expert_id] = nn.Identity()
        _install_block(block, family, layer_groups, clusters)
        if original_probe is not None:
            with torch.no_grad():
                approx_probe = block(probe_x)[0].float()
            layer_report["moe_output_relative_error"] = float(
                torch.linalg.vector_norm(approx_probe - original_probe)
                / torch.linalg.vector_norm(original_probe).clamp_min(1e-12)
            )
            if (config["residual_mode"] == "exact"
                    and layer_report["moe_output_relative_error"] > 0.01):
                raise AssertionError(
                    f"exact residual identity failed at layer {layer_idx}: "
                    f"relative error={layer_report['moe_output_relative_error']}"
                )
        reports[str(layer_idx)] = layer_report
        print(f"[QJL] layer {layer_idx}: {len(layer_groups)} groups, "
              f"{len(clusters)} compressed clusters", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return reports


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def save_qjl_checkpoint(model, family: str, groups: dict, config: dict,
                        path: str, diagnostics: dict = None):
    """Save only expert-side tensors; original HF backbone/router is source."""
    payload = {"format": "rescode-direct-qjl-v1", "family": family,
               "groups": groups, "config": config, "layers": {},
               "diagnostics": diagnostics or {}}
    for layer_idx, layer_groups in groups.items():
        block = _block(model, family, layer_idx)
        singletons = {i: {k: v.detach().cpu() for k, v in block.experts[i].state_dict().items()}
                      for group in layer_groups if len(group) == 1 for i in group}
        payload["layers"][layer_idx] = {
            "clusters": {cid: {k: v.detach().cpu() for k, v in cluster.state_dict().items()}
                         for cid, cluster in block._qjl_clusters.items()},
            "singletons": singletons,
        }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return {"checkpoint_bytes": os.path.getsize(path),
            "tensor_bytes": _tensor_bytes(payload),
            "metadata_and_zip_bytes": os.path.getsize(path) - _tensor_bytes(payload),
            "projection_gpu_resident_bytes": model._qjl_projection_bank.resident_bytes("cuda"),
            "projection_cpu_resident_bytes": model._qjl_projection_bank.resident_bytes("cpu")}


def load_qjl_checkpoint(model, path: str, backend: str = None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "rescode-direct-qjl-v1":
        raise ValueError("not a direct-QJL checkpoint")
    config = dict(payload["config"])
    if backend is not None:
        config["backend"] = backend
    family = payload["family"]
    bank = ProjectionBank(config["qjl_d"], config["projection_seed"])
    model._qjl_projection_bank = bank
    groups = {int(k): v for k, v in payload["groups"].items()}
    for layer_idx, layer_groups in groups.items():
        block = _block(model, family, layer_idx)
        device = next(block.parameters()).device
        clusters = nn.ModuleDict()
        for cid, state in payload["layers"][layer_idx]["clusters"].items():
            bases = {}
            residuals = {}
            for proj in ("gate", "up", "down"):
                bases[proj] = state[f"bases.{proj}.weight"].to(device)
            for i in layer_groups[int(cid)]:
                residuals[i] = {}
                for proj in ("gate", "up", "down"):
                    prefix = f"residuals.{i}.{proj}."
                    if prefix + "packed" in state:
                        bits = state[prefix + "packed"].to(device)
                        norms = state[prefix + "norms"].to(device)
                        residuals[i][proj] = PackedResidual(
                            bits, norms, bases[proj].shape[1], config["qjl_d"], bank,
                            config["backend"]
                        )
                    else:
                        residuals[i][proj] = ExactResidual(state[prefix + "weight"].to(device))
            clusters[str(cid)] = SharedBaseCluster(bases, residuals)
        for i, singleton_state in payload["layers"][layer_idx]["singletons"].items():
            block.experts[int(i)].load_state_dict(singleton_state)
        _install_block(block, family, layer_groups, clusters)
    return payload
