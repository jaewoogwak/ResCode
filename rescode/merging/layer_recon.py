import copy
from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .calib_stats import route, routing_config
from .forward_common import add_original_cluster_output


def _moe_block(model, layer_idx: int, backbone: str):
    layer = model.model.layers[layer_idx]
    return layer.block_sparse_moe if backbone == "mixtral" else layer.mlp


def snapshot_original_experts(model, cluster_maps, backbone: str) -> Dict[int, Dict[int, nn.Module]]:
    out: Dict[int, Dict[int, nn.Module]] = {}
    n_experts = n_copied = 0
    for layer_idx, cmap in cluster_maps.items():
        block = _moe_block(model, layer_idx, backbone)
        layer_out = {}
        for members in cmap.cluster_to_experts.values():
            if len(members) < 2:
                continue
            first = min(int(e) for e in members)
            for e in members:
                expert = block.experts[e]
                if int(e) == first:
                    expert = copy.deepcopy(expert)
                    n_copied += 1
                for p in expert.parameters():
                    p.requires_grad_(False)
                layer_out[int(e)] = expert
        out[layer_idx] = layer_out
        n_experts += len(layer_out)
    print(f"[RESCODE][L_rec] kept the originals of {n_experts} merged experts ({n_copied} copied) "
          f"for the original-layer target")
    return out


def attach_original_experts(model, original_experts, backbone: str):
    for layer_idx, experts in original_experts.items():
        star = _moe_block(model, layer_idx, backbone)._rescode
        star["orig_experts"] = experts
        star["teacher_active"] = False


def detach_original_experts(model):
    for block in _rescode_blocks(model):
        block._rescode.pop("orig_experts", None)
        block._rescode["teacher_active"] = False


def _rescode_blocks(model) -> List[nn.Module]:
    blocks = []
    for layer in model.model.layers:
        block = getattr(layer, "block_sparse_moe", None) or getattr(layer, "mlp", None)
        if block is not None and hasattr(block, "_rescode"):
            blocks.append(block)
    return blocks


@contextmanager
def original_experts(model):
    blocks = _rescode_blocks(model)
    for b in blocks:
        b._rescode["teacher_active"] = True
    try:
        yield
    finally:
        for b in blocks:
            b._rescode["teacher_active"] = False


def _block_output(output):
    return output[0] if isinstance(output, tuple) else output


class LayerReconstruction:
    def __init__(self, model, mode: str = "trajectory", backbone: str = "qwen"):
        if mode not in ("trajectory", "local"):
            raise ValueError(f"[RESCODE] unknown rec_mode={mode}")
        self.model = model
        self.mode = mode
        self.backbone = backbone
        self.top_k, self.norm_topk_prob = routing_config(model, backbone)
        self.local_terms: Dict[int, torch.Tensor] = {}
        self.blocks: List[Tuple[int, nn.Module]] = [
            (i, b) for i, b in enumerate(_rescode_blocks(model))
        ]
        self.student: Dict[int, torch.Tensor] = {}
        self.teacher: Dict[int, torch.Tensor] = {}
        self.backward_scale = None
        self._capture_teacher = False
        self._handles = [
            block.register_forward_hook(self._make_hook(i)) for i, block in self.blocks
        ]

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            out = _block_output(output)
            if self._capture_teacher:
                self.teacher[idx] = out.detach()
                return
            if not torch.is_grad_enabled():
                return
            if self.mode == "local":
                term = self._local_gap(module, inputs[0].detach())
                if self.backward_scale is not None and term.requires_grad:
                    (term * self.backward_scale).backward()
                    term = term.detach()
                self.local_terms[idx] = term
            else:
                self.student[idx] = out
        return hook

    def _local_gap(self, module, hidden: torch.Tensor) -> torch.Tensor:
        star = module._rescode
        flat = hidden.reshape(-1, hidden.shape[-1])
        with torch.no_grad():
            selected, weights = route(module, flat, self.backbone, self.top_k, self.norm_topk_prob)
            weights = weights.to(flat.dtype)
            original = torch.zeros_like(flat)
        compressed = None
        cmap = star["cluster_map"]
        for cluster_id, member_ids in cmap.cluster_to_experts.items():
            if cmap.is_singleton[cluster_id]:
                continue
            out = star["meta_experts"][cluster_id](
                flat_hidden_states=flat, selected_experts=selected, routing_weights=weights,
            ).float()
            compressed = out if compressed is None else compressed + out
            with torch.no_grad():
                add_original_cluster_output(original, flat, member_ids, selected, weights, star["orig_experts"])
        if compressed is None:
            return flat.new_zeros((), dtype=torch.float32)
        return (compressed - original.float()).pow(2).sum(dim=-1).mean()

    def begin_batch(self, batch: Dict[str, torch.Tensor]):
        self.student.clear()
        self.teacher.clear()
        self.local_terms.clear()
        if self.mode != "trajectory":
            return
        inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        self._capture_teacher = True
        try:
            with torch.no_grad(), original_experts(self.model):
                self.model.model(**inputs)
        finally:
            self._capture_teacher = False

    def loss(self) -> torch.Tensor:
        if self.mode == "local":
            terms = list(self.local_terms.values())
            self.local_terms.clear()
            if not terms:
                return torch.zeros((), device="cuda")
            return torch.stack([t.to(terms[0].device) for t in terms]).mean()
        terms = []
        for idx, s in self.student.items():
            t = self.teacher.get(idx)
            if t is None:
                continue
            diff = s.float() - t.float()
            terms.append(diff.pow(2).sum(dim=-1).mean())
        self.student.clear()
        self.teacher.clear()
        if not terms:
            return torch.zeros((), device="cuda")
        return torch.stack([t.to(terms[0].device) for t in terms]).mean()

    def num_layers(self) -> int:
        return len(self.blocks)

    def cleanup(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.student.clear()
        self.teacher.clear()
        self.local_terms.clear()
