"""Calibration teacher collection and code fitting for RESCODE.

Teacher = original model's per-cluster contribution
    y_C^teacher(x) = sum_{j in C cap T(x)} g_j(x) E_j(x)

Student = RESCODE merged-expert output with codes
    y_C^star(x) = W_d_bar (a_C(x) * h_C(x))
    a_C(x) = sum_{j in C cap T(x)} g_j(x) c_j

Loss = MSE(y_teacher, y_star) + lambda * ||c_j - 1||^2

Only the code parameters are trained. Merged expert weights, singleton
experts, and the router are kept frozen.

Memory note: storing per-cluster teacher tensors for all calibration tokens
is memory-heavy on large models. We support per-layer, per-cluster token
caps and stream tokens through fitting in mini-batches.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .cluster_map import ExpertClusterMap
from .config import RescodeConfig, PreActStarConfig
from .diagnostics import compute_reconstruction_error
from .meta_expert import RescodeMetaExpert


class _NullLogger:
    """No-op metric logger so fit loops can call `log.*` unconditionally."""
    enabled = False

    def log(self, _data, **_kw): pass
    def define_metric(self, *_args, **_kw): pass
    def summary_update(self, _data): pass
    def finish(self): pass


# --------------------------------------------------- preact param-group helpers
def _preact_param_groups(
    meta: RescodeMetaExpert,
    config: RescodeConfig,
) -> List[dict]:
    """Build per-LR optimizer param groups for one meta expert.

    Layout:
        - post code params (existing diag/scalar/lowrank) at config.fit_lr
          if either no preact or preact_star.train_post_codes=True
        - preact bias at preact_star.lr_preact_bias    if train_preact_bias
        - preact lowrank at preact_star.lr_preact_lowrank if train_preact_lowrank
        - token-adaptive down residual at preact_star.lr_down_residual
          if train_down_residual

    A preact-disabled meta uses the legacy single-group optimizer with
    config.fit_lr.
    """
    groups: List[dict] = []
    preact_cfg = config.preact_star
    preact_on = preact_cfg is not None and preact_cfg.enabled and meta.preact is not None

    # 1) post codes
    code_params = meta.code_parameters()
    train_post = (not preact_on) or preact_cfg.train_post_codes
    if code_params and train_post and not config.freeze_code and config.fit_codes:
        groups.append({
            "params": code_params,
            "lr": (preact_cfg.lr_post_code if preact_on else config.fit_lr),
            "name": "post_code",
        })

    if not preact_on:
        return groups

    # 2) preact bias
    bias_params = []
    for n in ("gate_bias_codes", "up_bias_codes"):
        if hasattr(meta.preact, n):
            bias_params.append(getattr(meta.preact, n))
    if bias_params and preact_cfg.train_preact_bias:
        groups.append({
            "params": bias_params,
            "lr": preact_cfg.lr_preact_bias,
            "name": "preact_bias",
        })

    # 3) preact lowrank
    lr_params = []
    for n in ("gate_A", "gate_B", "gate_z", "up_A", "up_B", "up_z"):
        if hasattr(meta.preact, n):
            lr_params.append(getattr(meta.preact, n))
    if lr_params and preact_cfg.train_preact_lowrank:
        groups.append({
            "params": lr_params,
            "lr": preact_cfg.lr_preact_lowrank,
            "name": "preact_lowrank",
        })

    # 4) down residual
    down_params = []
    for n in ("down_L", "down_R", "down_z"):
        if hasattr(meta.preact, n):
            down_params.append(getattr(meta.preact, n))
    if down_params and preact_cfg.train_down_residual:
        groups.append({
            "params": down_params,
            "lr": preact_cfg.lr_down_residual,
            "name": "down_residual",
        })

    return groups


def _preact_l2_term(meta: RescodeMetaExpert) -> torch.Tensor:
    """Sum of (per-group) mean(p^2) regularization for the preact params,
    weighted by per-group lambdas in PreActStarConfig.
    """
    if meta.preact is None:
        return None
    cfg = meta.preact.config
    total = None

    def _add(t, lam):
        nonlocal total
        if lam is None or lam == 0.0:
            return
        term = (lam * t.pow(2).mean())
        total = term if total is None else total + term

    for n in ("gate_bias_codes", "up_bias_codes"):
        if hasattr(meta.preact, n):
            _add(getattr(meta.preact, n), cfg.bias_l2_lambda)
    for n in ("gate_A", "gate_B", "gate_z", "up_A", "up_B", "up_z"):
        if hasattr(meta.preact, n):
            _add(getattr(meta.preact, n), cfg.lowrank_l2_lambda)
    for n in ("down_L", "down_R", "down_z"):
        if hasattr(meta.preact, n):
            _add(getattr(meta.preact, n), cfg.down_l2_lambda)
    return total


def _trainable_param_names_for_log(meta: RescodeMetaExpert, config: RescodeConfig) -> List[str]:
    """Helper: human-readable list of which params will receive gradients."""
    out: List[str] = []
    preact_cfg = config.preact_star
    preact_on = preact_cfg is not None and preact_cfg.enabled and meta.preact is not None

    if meta.code_parameters() and (
        (not preact_on or preact_cfg.train_post_codes)
        and config.fit_codes and not config.freeze_code
    ):
        for n in ("code", "code_U", "code_Z"):
            if hasattr(meta, n):
                out.append(n)

    if preact_on:
        for grp_filter, names in [
            (preact_cfg.train_preact_bias, ["gate_bias_codes", "up_bias_codes"]),
            (preact_cfg.train_preact_lowrank, ["gate_A", "gate_B", "gate_z", "up_A", "up_B", "up_z"]),
            (preact_cfg.train_down_residual, ["down_L", "down_R", "down_z"]),
        ]:
            if not grp_filter:
                continue
            for n in names:
                if hasattr(meta.preact, n):
                    out.append(f"preact.{n}")
    return out


# ---------------------------------------------------------------------- types
@dataclass
class LayerCalibration:
    """Per-layer calibration buffers, all CPU."""
    hidden_states: List[torch.Tensor] = field(default_factory=list)         # [T_i, D] per batch
    selected_experts: List[torch.Tensor] = field(default_factory=list)       # [T_i, K]
    routing_weights: List[torch.Tensor] = field(default_factory=list)        # [T_i, K]
    cluster_teachers: Dict[int, List[torch.Tensor]] = field(default_factory=dict)  # cid -> list of [T_i, D]


# ------------------------------------------------------ teacher collection
def _select_token_subset(buffers: LayerCalibration, max_tokens: int):
    """Randomly subsample tokens across all batches to at most max_tokens.

    NOTE: this destroys the alignment between the per-batch lists and the
    calib_loader iteration order, because each batch tensor gets cut down to
    a per-batch slice of a global random permutation. Only call this when the
    downstream consumer concatenates everything (i.e. layerwise fit). For
    joint LM-loss fitting we re-iterate calib_loader and need shape-aligned
    teacher tensors per batch — call sites in joint mode must skip this.
    """
    total = sum(t.shape[0] for t in buffers.hidden_states)
    if total <= max_tokens:
        return buffers
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(total, generator=g)[:max_tokens]
    perm, _ = torch.sort(perm)

    def _take(big_list):
        out = []
        offset = 0
        ptr = 0
        for chunk in big_list:
            n = chunk.shape[0]
            mask = (perm >= offset) & (perm < offset + n)
            if mask.any():
                local = perm[mask] - offset
                out.append(chunk.index_select(0, local))
            offset += n
        return out

    buffers.hidden_states = _take(buffers.hidden_states)
    buffers.selected_experts = _take(buffers.selected_experts)
    buffers.routing_weights = _take(buffers.routing_weights)
    new_teachers = {}
    for cid, lst in buffers.cluster_teachers.items():
        new_teachers[cid] = _take(lst)
    buffers.cluster_teachers = new_teachers
    return buffers


def _qwen_per_cluster_teacher_hook(
    layer_idx: int,
    cluster_map: ExpertClusterMap,
    buffers: LayerCalibration,
    norm_topk_prob: bool,
    top_k: int,
):
    """Build a forward pre-hook + post-hook combination on an original Qwen MoE
    block. Captures (hidden_states, selected_experts, routing_weights) and
    per-cluster teacher = sum_{j in C cap T(x)} g_j E_j(x)."""

    def hook(module, inputs, output):
        # `output` is (final_hidden_states, router_logits) but we re-compute
        # the gating to get aligned selected_experts/routing_weights here so
        # we don't have to monkey-patch the original block.
        hidden_states = inputs[0]
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = module.gate(flat)
        rw = F.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, top_k, dim=-1)
        if norm_topk_prob:
            rw = rw / rw.sum(dim=-1, keepdim=True)
        rw = rw.to(flat.dtype)

        T = flat.shape[0]
        for cluster_id, member_ids in cluster_map.cluster_to_experts.items():
            if cluster_map.is_singleton[cluster_id]:
                continue
            teacher = torch.zeros_like(flat)
            for j in member_ids:
                mask = (sel == j)
                if not mask.any():
                    continue
                top_x, idx = torch.where(mask)
                state = flat.index_select(0, top_x)
                expert = module.experts[j]
                contribution = expert(state) * rw[top_x, idx, None]
                teacher.index_add_(0, top_x, contribution.to(teacher.dtype))
            buffers.cluster_teachers.setdefault(cluster_id, []).append(teacher.detach().float().cpu())

        buffers.hidden_states.append(flat.detach().float().cpu())
        buffers.selected_experts.append(sel.detach().cpu())
        buffers.routing_weights.append(rw.detach().float().cpu())

    return hook


def _deepseek_per_cluster_teacher_hook(
    layer_idx: int,
    cluster_map: ExpertClusterMap,
    buffers: LayerCalibration,
    norm_topk_prob: bool,
    top_k: int,
):
    """Forward-hook builder for the original DeepSeek-V1/V2 MoE block.

    Unlike Qwen where `module.gate(...)` returns router logits, DeepSeek's
    `MoEGate.forward` returns `(topk_idx, topk_weight, aux_loss)`. We
    therefore reconstruct logits ourselves with the raw gate weight.
    The shared_experts pass-through is NOT part of the per-cluster teacher
    (it cancels out in the student-teacher MSE diff at the routed path).
    """

    def hook(module, inputs, output):
        hidden_states = inputs[0]
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = F.linear(flat, module.gate.weight, None)
        rw = F.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, top_k, dim=-1)
        if norm_topk_prob:
            rw = rw / rw.sum(dim=-1, keepdim=True)
        rw = rw.to(flat.dtype)

        T = flat.shape[0]
        for cluster_id, member_ids in cluster_map.cluster_to_experts.items():
            if cluster_map.is_singleton[cluster_id]:
                continue
            teacher = torch.zeros_like(flat)
            for j in member_ids:
                mask = (sel == j)
                if not mask.any():
                    continue
                top_x, idx = torch.where(mask)
                state = flat.index_select(0, top_x)
                expert = module.experts[j]
                contribution = expert(state) * rw[top_x, idx, None]
                teacher.index_add_(0, top_x, contribution.to(teacher.dtype))
            buffers.cluster_teachers.setdefault(cluster_id, []).append(teacher.detach().float().cpu())

        buffers.hidden_states.append(flat.detach().float().cpu())
        buffers.selected_experts.append(sel.detach().cpu())
        buffers.routing_weights.append(rw.detach().float().cpu())

    return hook


def _mixtral_per_cluster_teacher_hook(
    layer_idx: int,
    cluster_map: ExpertClusterMap,
    buffers: LayerCalibration,
    top_k: int,
):
    def hook(module, inputs, output):
        hidden_states = inputs[0]
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = module.gate(flat)
        rw = F.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, top_k, dim=-1)
        rw = rw / rw.sum(dim=-1, keepdim=True)
        rw = rw.to(flat.dtype)

        for cluster_id, member_ids in cluster_map.cluster_to_experts.items():
            if cluster_map.is_singleton[cluster_id]:
                continue
            teacher = torch.zeros_like(flat)
            for j in member_ids:
                mask = (sel == j)
                if not mask.any():
                    continue
                top_x, idx = torch.where(mask)
                state = flat.index_select(0, top_x)
                expert = module.experts[j]
                contribution = expert(state) * rw[top_x, idx, None]
                teacher.index_add_(0, top_x, contribution.to(teacher.dtype))
            buffers.cluster_teachers.setdefault(cluster_id, []).append(teacher.detach().float().cpu())

        buffers.hidden_states.append(flat.detach().float().cpu())
        buffers.selected_experts.append(sel.detach().cpu())
        buffers.routing_weights.append(rw.detach().float().cpu())

    return hook


def collect_teacher_signals_qwen(
    model: nn.Module,
    cluster_maps: Dict[int, ExpertClusterMap],
    dataloader,
    config: RescodeConfig,
) -> Dict[int, LayerCalibration]:
    """Run an UNCOMPRESSED Qwen MoE model on the calibration loader and capture
    per-layer per-cluster teacher signals."""
    buffers: Dict[int, LayerCalibration] = {
        layer_idx: LayerCalibration() for layer_idx in cluster_maps
    }
    handles = []
    norm_topk_prob = bool(getattr(model.config, "norm_topk_prob", True))
    top_k = int(model.config.num_experts_per_tok)
    for layer_idx, cmap in cluster_maps.items():
        block = model.model.layers[layer_idx].mlp
        h = block.register_forward_hook(
            _qwen_per_cluster_teacher_hook(layer_idx, cmap, buffers[layer_idx], norm_topk_prob, top_k)
        )
        handles.append(h)
    try:
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[RESCODE] collecting teacher signals (Qwen)"):
                batch = {k: v.cuda() for k, v in batch.items()}
                _ = model(**batch)
    finally:
        for h in handles:
            h.remove()

    # Only subsample at collection time when the downstream fit will
    # concatenate everything (layerwise mode). Joint mode iterates the calib
    # loader again and needs per-batch shape alignment, so subsampling here
    # would silently zero out the cluster-MSE term.
    if getattr(config, "fit_mode", "layerwise") != "joint":
        for layer_idx, buf in buffers.items():
            _select_token_subset(buf, config.max_fit_tokens_per_layer)
    return buffers


def collect_teacher_signals_deepseek(
    model: nn.Module,
    cluster_maps: Dict[int, ExpertClusterMap],
    dataloader,
    config: RescodeConfig,
) -> Dict[int, LayerCalibration]:
    """Run an UNCOMPRESSED DeepSeek-V1/V2 MoE model on the calibration loader
    and capture per-layer per-cluster teacher signals."""
    buffers: Dict[int, LayerCalibration] = {
        layer_idx: LayerCalibration() for layer_idx in cluster_maps
    }
    handles = []
    # DeepSeek default: norm_topk_prob=False. Respect config.
    norm_topk_prob = bool(getattr(model.config, "norm_topk_prob", False))
    top_k = int(model.config.num_experts_per_tok)
    for layer_idx, cmap in cluster_maps.items():
        block = model.model.layers[layer_idx].mlp
        h = block.register_forward_hook(
            _deepseek_per_cluster_teacher_hook(layer_idx, cmap, buffers[layer_idx], norm_topk_prob, top_k)
        )
        handles.append(h)
    try:
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[RESCODE] collecting teacher signals (DeepSeek)"):
                batch = {k: v.cuda() for k, v in batch.items()}
                _ = model(**batch)
    finally:
        for h in handles:
            h.remove()

    if getattr(config, "fit_mode", "layerwise") != "joint":
        for layer_idx, buf in buffers.items():
            _select_token_subset(buf, config.max_fit_tokens_per_layer)
    return buffers


def collect_teacher_signals_mixtral(
    model: nn.Module,
    cluster_maps: Dict[int, ExpertClusterMap],
    dataloader,
    config: RescodeConfig,
) -> Dict[int, LayerCalibration]:
    buffers: Dict[int, LayerCalibration] = {
        layer_idx: LayerCalibration() for layer_idx in cluster_maps
    }
    handles = []
    top_k = int(model.config.num_experts_per_tok)
    for layer_idx, cmap in cluster_maps.items():
        block = model.model.layers[layer_idx].block_sparse_moe
        h = block.register_forward_hook(
            _mixtral_per_cluster_teacher_hook(layer_idx, cmap, buffers[layer_idx], top_k)
        )
        handles.append(h)
    try:
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[RESCODE] collecting teacher signals (Mixtral)"):
                batch = {k: v.cuda() for k, v in batch.items()}
                _ = model(**batch)
    finally:
        for h in handles:
            h.remove()
    if getattr(config, "fit_mode", "layerwise") != "joint":
        for layer_idx, buf in buffers.items():
            _select_token_subset(buf, config.max_fit_tokens_per_layer)
    return buffers


# ============================================================ Online teacher
class OnlineTeacher:
    """Per-batch teacher — computes per-cluster teacher outputs on-the-fly.

    Wraps an UNMERGED reference model and registers forward hooks that capture
    `sum_{j in C cap T(x)} g_j(x) * E_j(x)` per non-singleton cluster on every
    forward pass. Outputs stay on GPU (no offline buffer accumulation), so
    memory is O(layers × clusters_per_layer × T × D) per call — independent of
    total batch count.

    Use this when the offline `collect_teacher_signals_*` would blow up CPU
    memory with large training data.

    Parameters
    ----------
    model : the original (unmerged) MoE model. Must remain untouched after
        construction (do not merge in-place; deepcopy first if needed).
    cluster_maps : per-layer cluster maps.
    backbone : "qwen", "qwen3", "mixtral", or "deepseek" — selects the
        per-layer MoE block accessor and the router-logits computation.
    """

    def __init__(self, model, cluster_maps, backbone="qwen", norm_topk_prob=None, top_k=None):
        self.model = model
        self.cluster_maps = cluster_maps
        self.backbone = backbone
        if norm_topk_prob is None:
            # DeepSeek defaults to False, Qwen* defaults to True (and Mixtral
            # always normalizes regardless).
            default_norm = False if backbone == "deepseek" else True
            norm_topk_prob = bool(getattr(model.config, "norm_topk_prob", default_norm))
        self.norm_topk_prob = norm_topk_prob
        if top_k is None:
            top_k = int(model.config.num_experts_per_tok)
        self.top_k = top_k
        # Per-call captures: layer_idx -> {cluster_id: tensor [T, D] on GPU}
        self.last: Dict[int, Dict[int, torch.Tensor]] = {}
        self._handles = []
        self._register()

    def _moe_block(self, layer_idx: int):
        if self.backbone in ("qwen", "qwen3", "deepseek"):
            return self.model.model.layers[layer_idx].mlp
        if self.backbone == "mixtral":
            return self.model.model.layers[layer_idx].block_sparse_moe
        raise ValueError(f"OnlineTeacher: unknown backbone {self.backbone}")

    def _make_hook(self, layer_idx: int, cmap: ExpertClusterMap):
        norm_topk = self.norm_topk_prob
        top_k = self.top_k
        last = self.last
        is_qwen_family = self.backbone in ("qwen", "qwen3")
        is_deepseek = (self.backbone == "deepseek")

        def hook(module, inputs, output):
            hidden_states = inputs[0]
            B, S, D = hidden_states.shape
            flat = hidden_states.reshape(-1, D)
            if is_deepseek:
                # MoEGate.forward returns (topk_idx, topk_weight, aux_loss);
                # rebuild logits from the raw gate weight matrix.
                router_logits = F.linear(flat, module.gate.weight, None)
            else:
                router_logits = module.gate(flat)
            rw = F.softmax(router_logits, dim=1, dtype=torch.float)
            rw, sel = torch.topk(rw, top_k, dim=-1)
            # Mixtral hook always normalizes; Qwen/Qwen3/DeepSeek respect config.
            if not (is_qwen_family or is_deepseek) or norm_topk:
                rw = rw / rw.sum(dim=-1, keepdim=True)
            rw = rw.to(flat.dtype)

            per_cluster: Dict[int, torch.Tensor] = {}
            for cluster_id, member_ids in cmap.cluster_to_experts.items():
                if cmap.is_singleton[cluster_id]:
                    continue
                teacher = torch.zeros_like(flat)
                for j in member_ids:
                    mask = (sel == j)
                    if not mask.any():
                        continue
                    top_x, idx = torch.where(mask)
                    state = flat.index_select(0, top_x)
                    expert = module.experts[j]
                    contribution = expert(state) * rw[top_x, idx, None]
                    teacher.index_add_(0, top_x, contribution.to(teacher.dtype))
                per_cluster[int(cluster_id)] = teacher.detach()
            last[int(layer_idx)] = per_cluster

        return hook

    def _register(self):
        for layer_idx, cmap in self.cluster_maps.items():
            block = self._moe_block(layer_idx)
            self._handles.append(block.register_forward_hook(self._make_hook(layer_idx, cmap)))

    @torch.no_grad()
    def forward(self, batch):
        """Run teacher on this batch; populate self.last and return it."""
        self.last.clear()
        self.model.eval()
        _ = self.model(**batch)
        return self.last

    def get(self, layer_idx: int, cluster_id: int) -> Optional[torch.Tensor]:
        layer_d = self.last.get(int(layer_idx))
        if layer_d is None:
            return None
        return layer_d.get(int(cluster_id))

    def cleanup(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.last.clear()


# ----------------------------------------------------------------- fitting
def _resolve_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


@torch.no_grad()
def _student_cluster_output(
    meta: RescodeMetaExpert,
    flat_x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Run RESCODE forward on a chunk and return [T, D] cluster output."""
    return meta(flat_x, selected_experts, routing_weights)


def _make_optimizer(params, config: RescodeConfig):
    """Build an optimizer.

    Accepts either a flat list of parameters or a list of param-group dicts
    (e.g. [{params, lr, name}, ...]) so different LRs can be applied per
    PreAct-STAR group.
    """
    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=config.fit_lr,
                                weight_decay=config.fit_weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=config.fit_lr,
                               weight_decay=config.fit_weight_decay,
                               momentum=0.9)
    return torch.optim.AdamW(params, lr=config.fit_lr,
                             weight_decay=config.fit_weight_decay)


def _make_scheduler(optim, config: RescodeConfig, total_steps: int):
    name = (config.lr_scheduler or "constant").lower()
    warmup = max(0, int(config.warmup_steps))

    def _lr_lambda_constant(step):
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        return 1.0

    def _lr_lambda_cosine(step):
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _lr_lambda_linear(step):
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 1.0 - min(1.0, progress))

    if name == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda_cosine)
    if name == "linear":
        return torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda_linear)
    return torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda_constant)


def fit_codes_for_block(
    block: nn.Module,
    layer_buffers: LayerCalibration,
    config: RescodeConfig,
    device: Optional[torch.device] = None,
    global_step_base: int = 0,
) -> Dict[int, Dict[str, float]]:
    """Fit codes of every meta expert in `block` against captured teacher signals.

    Returns per-cluster dict: {"mse_before": ..., "mse_after": ..., ...}.
    """
    star = block._rescode
    cmap: ExpertClusterMap = star["cluster_map"]
    meta_experts: Dict[int, RescodeMetaExpert] = star["meta_experts"]

    if device is None:
        # Use the device of the first projection weight we can find.
        for m in meta_experts.values():
            device = m.gate_proj.weight.device
            break

    fit_dtype = _resolve_dtype(config.fit_dtype)

    flat_x_all = torch.cat(layer_buffers.hidden_states, dim=0) if layer_buffers.hidden_states else None
    sel_all = torch.cat(layer_buffers.selected_experts, dim=0) if layer_buffers.selected_experts else None
    rw_all = torch.cat(layer_buffers.routing_weights, dim=0) if layer_buffers.routing_weights else None

    if flat_x_all is None:
        return {}

    results: Dict[int, Dict[str, float]] = {}

    for cid, meta in meta_experts.items():
        if cid not in layer_buffers.cluster_teachers or not layer_buffers.cluster_teachers[cid]:
            results[cid] = {"skipped": True, "reason": "no_teacher"}
            continue
        teacher_full = torch.cat(layer_buffers.cluster_teachers[cid], dim=0)

        # Subset to tokens that actually have at least one selected expert in C.
        member_set = set(meta.member_expert_ids)
        member_tensor = torch.tensor(list(member_set), dtype=sel_all.dtype)
        mask = torch.zeros(sel_all.shape[0], dtype=torch.bool)
        for j in member_tensor:
            mask = mask | (sel_all == j).any(dim=-1)
        if mask.sum().item() < config.min_fit_tokens:
            print(
                f"[RESCODE][fit] layer={cmap.layer_idx} cluster={cid} "
                f"only {int(mask.sum().item())} active tokens (< min_fit_tokens="
                f"{config.min_fit_tokens}); keeping codes at init."
            )
            results[cid] = {"skipped": True, "reason": "too_few_tokens",
                             "num_tokens": int(mask.sum().item())}
            continue

        x = flat_x_all[mask]
        sel = sel_all[mask]
        rw = rw_all[mask]
        teacher = teacher_full[mask]

        # Optional per-cluster cap
        if x.shape[0] > config.max_fit_tokens_per_cluster:
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(x.shape[0], generator=g)[:config.max_fit_tokens_per_cluster]
            x = x.index_select(0, idx)
            sel = sel.index_select(0, idx)
            rw = rw.index_select(0, idx)
            teacher = teacher.index_select(0, idx)

        x_dev = x.to(device=device, dtype=fit_dtype)
        sel_dev = sel.to(device=device)
        rw_dev = rw.to(device=device, dtype=fit_dtype)
        teacher_dev = teacher.to(device=device, dtype=fit_dtype)

        # PreAct-STAR-aware parameter selection. When preact is OFF this
        # collapses to the legacy code-only group with config.fit_lr.
        param_groups = _preact_param_groups(meta, config)
        flat_params = [p for grp in param_groups for p in grp["params"]]

        # Reconstruction MSE before fitting.
        with torch.no_grad():
            student_before = meta(x_dev, sel_dev, rw_dev)
        rec_before = compute_reconstruction_error(student_before.float(), teacher_dev.float())

        if not flat_params:
            results[cid] = {
                **{f"before_{k}": v for k, v in rec_before.items()},
                **{f"after_{k}": v for k, v in rec_before.items()},
                "fit_steps": 0,
                "skipped": True,
                "reason": "no_trainable_params",
            }
            continue

        # Defensive freeze: only the chosen params carry grad during fitting.
        # Without this, autograd retains intermediate activations through the
        # frozen merged projections across iterations and we waste both
        # memory and time.
        train_ids = {id(p) for p in flat_params}
        for p in meta.parameters():
            p.requires_grad_(id(p) in train_ids)

        # Log trainable param names once per cluster (debugging aid for
        # PreAct-STAR experiments where users sweep which groups are on).
        trainable_names = _trainable_param_names_for_log(meta, config)
        print(
            f"[RESCODE][fit] layer={cmap.layer_idx} cluster={cid} "
            f"trainable={trainable_names}"
        )
        optim = _make_optimizer(param_groups, config)
        sched = _make_scheduler(optim, config, total_steps=config.fit_steps)

        # Identity reference for ridge term.
        if meta.code_type in {"scalar", "diag"}:
            ridge_ref = torch.ones_like(meta.code)
        elif meta.code_type == "lowrank":
            ridge_ref = None  # ||code - 1||^2 ~ ||Z @ U.T||^2; fall back below
        else:
            ridge_ref = None

        log_every = max(1, int(getattr(config, "log_every", 25)))
        log = _NullLogger()
        layer_idx_int = int(cmap.layer_idx)

        preact_cfg: Optional[PreActStarConfig] = (
            config.preact_star if (config.preact_star is not None and config.preact_star.enabled
                                    and meta.preact is not None) else None
        )

        for step in range(config.fit_steps):
            optim.zero_grad(set_to_none=True)
            student = meta(x_dev, sel_dev, rw_dev)
            mse = F.mse_loss(student.float(), teacher_dev.float())
            ridge_term = mse.new_zeros(())
            if config.ridge_lambda > 0.0 and (preact_cfg is None or preact_cfg.train_post_codes):
                if meta.code_type in {"scalar", "diag"}:
                    ridge_term = (meta.code - ridge_ref).pow(2).mean()
                elif meta.code_type == "lowrank":
                    code_full = meta.code_Z @ meta.code_U.t()
                    ridge_term = code_full.pow(2).mean()
            preact_l2 = mse.new_zeros(())
            if preact_cfg is not None:
                t = _preact_l2_term(meta)
                if t is not None:
                    preact_l2 = t
            loss = mse + config.ridge_lambda * ridge_term + preact_l2
            loss.backward()

            # Optional grad clip (preact mode often needs it for the lowrank /
            # down residual factors).
            if preact_cfg is not None and preact_cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(flat_params, preact_cfg.grad_clip_norm)

            optim.step()
            sched.step()
            # post-update clipping
            if meta.code_type in {"scalar", "diag"} and (preact_cfg is None or preact_cfg.train_post_codes):
                meta.clip_codes_()
            if preact_cfg is not None:
                meta.preact.clip_biases_()

            if log.enabled and (step % log_every == 0 or step == config.fit_steps - 1):
                metrics = {
                    f"layer{layer_idx_int}/c{cid}/mse": float(mse.detach().item()),
                    f"layer{layer_idx_int}/c{cid}/ridge": float(ridge_term.detach().item()),
                    f"layer{layer_idx_int}/c{cid}/loss": float(loss.detach().item()),
                    f"layer{layer_idx_int}/c{cid}/lr": float(optim.param_groups[0]["lr"]),
                }
                if meta.code_type in {"scalar", "diag"}:
                    metrics[f"layer{layer_idx_int}/c{cid}/code_mean"] = float(meta.code.mean().item())
                    metrics[f"layer{layer_idx_int}/c{cid}/code_std"] = float(meta.code.std().item())
                    metrics[f"layer{layer_idx_int}/c{cid}/code_min"] = float(meta.code.min().item())
                    metrics[f"layer{layer_idx_int}/c{cid}/code_max"] = float(meta.code.max().item())
                log.log(metrics, step=global_step_base + step)
        # End fit loop.

        with torch.no_grad():
            student_after = meta(x_dev, sel_dev, rw_dev)
        rec_after = compute_reconstruction_error(student_after.float(), teacher_dev.float())

        if log.enabled:
            log.log({
                f"summary/layer{layer_idx_int}/c{cid}/mse_before": float(rec_before["mse"]),
                f"summary/layer{layer_idx_int}/c{cid}/mse_after": float(rec_after["mse"]),
                f"summary/layer{layer_idx_int}/c{cid}/cosine_before": float(rec_before["cosine"]),
                f"summary/layer{layer_idx_int}/c{cid}/cosine_after": float(rec_after["cosine"]),
                f"summary/layer{layer_idx_int}/c{cid}/num_tokens": int(x.shape[0]),
            }, step=global_step_base + config.fit_steps)

        results[cid] = {
            **{f"before_{k}": v for k, v in rec_before.items()},
            **{f"after_{k}": v for k, v in rec_after.items()},
            "fit_steps": config.fit_steps,
            "num_tokens": int(x.shape[0]),
        }
        print(
            f"[RESCODE][fit] layer={cmap.layer_idx} cluster={cid} N={x.shape[0]} "
            f"mse {rec_before['mse']:.4e} -> {rec_after['mse']:.4e} "
            f"cos {rec_before['cosine']:.4f} -> {rec_after['cosine']:.4f}"
        )

        # Re-freeze grad to avoid lingering autograd state from this layer.
        for p in flat_params:
            p.requires_grad_(not config.freeze_code)

    return results


def fit_all_codes(
    model: nn.Module,
    teacher_buffers: Dict[int, LayerCalibration],
    config: RescodeConfig,
) -> Dict[int, Dict[int, Dict[str, float]]]:
    """Iterate over MoE blocks and fit codes layer by layer."""
    layer_results: Dict[int, Dict[int, Dict[str, float]]] = {}
    global_step = 0
    for layer_idx, layer in enumerate(model.model.layers):
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        if layer_idx not in teacher_buffers:
            continue
        layer_results[layer_idx] = fit_codes_for_block(
            block=block,
            layer_buffers=teacher_buffers[layer_idx],
            config=config,
            global_step_base=global_step,
        )
        # Each block consumes ~fit_steps iterations.
        global_step += int(config.fit_steps) + 1

    return layer_results


# --------------------------------------------------------------------- joint
def _collect_all_code_params(model: nn.Module):
    params = []
    for layer in model.model.layers:
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        for meta in block._rescode["meta_experts"].values():
            params.extend(meta.code_parameters())
    return params


def _collect_all_preact_params(model: nn.Module):
    """Return all PreAct-STAR parameters across the model (flat list)."""
    params = []
    for layer in model.model.layers:
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        for meta in block._rescode["meta_experts"].values():
            if meta.preact is not None:
                params.extend(meta.preact.preact_parameter_list())
    return params


def _build_joint_param_groups(model: nn.Module, config: RescodeConfig):
    """Per-LR optimizer param groups for the joint LM-loss fit.

    Mirrors `_preact_param_groups` but applies it model-wide. When PreAct is
    off this falls back to a single legacy group of code parameters at
    config.fit_lr.
    """
    preact_cfg = config.preact_star
    preact_on = preact_cfg is not None and preact_cfg.enabled

    groups = []
    code_params: List[torch.nn.Parameter] = []
    bias_params, lr_params, down_params = [], [], []
    for layer in model.model.layers:
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        for meta in block._rescode["meta_experts"].values():
            code_params.extend(meta.code_parameters())
            if meta.preact is None:
                continue
            for n in ("gate_bias_codes", "up_bias_codes"):
                if hasattr(meta.preact, n):
                    bias_params.append(getattr(meta.preact, n))
            for n in ("gate_A", "gate_B", "gate_z", "up_A", "up_B", "up_z"):
                if hasattr(meta.preact, n):
                    lr_params.append(getattr(meta.preact, n))
            for n in ("down_L", "down_R", "down_z"):
                if hasattr(meta.preact, n):
                    down_params.append(getattr(meta.preact, n))

    train_post = (not preact_on) or preact_cfg.train_post_codes
    if code_params and train_post and not config.freeze_code and config.fit_codes:
        lr = preact_cfg.lr_post_code if preact_on else config.fit_lr
        groups.append({"params": code_params, "lr": lr, "name": "post_code"})
    if preact_on:
        if bias_params and preact_cfg.train_preact_bias:
            groups.append({"params": bias_params,
                            "lr": preact_cfg.lr_preact_bias, "name": "preact_bias"})
        if lr_params and preact_cfg.train_preact_lowrank:
            groups.append({"params": lr_params,
                            "lr": preact_cfg.lr_preact_lowrank, "name": "preact_lowrank"})
        if down_params and preact_cfg.train_down_residual:
            groups.append({"params": down_params,
                            "lr": preact_cfg.lr_down_residual, "name": "down_residual"})
    return groups


def _clip_all_codes(model: nn.Module):
    for layer in model.model.layers:
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        for meta in block._rescode["meta_experts"].values():
            meta.clip_codes_()
            if meta.preact is not None:
                meta.preact.clip_biases_()


def _ridge_term_for_model(model: nn.Module) -> torch.Tensor:
    total = None
    n = 0
    for layer in model.model.layers:
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        for meta in block._rescode["meta_experts"].values():
            if meta.code_type in {"scalar", "diag"}:
                term = (meta.code - 1.0).pow(2).mean()
            elif meta.code_type == "lowrank":
                code_full = meta.code_Z @ meta.code_U.t()
                term = code_full.pow(2).mean()
            else:
                continue
            total = term if total is None else total + term
            n += 1
    if total is None:
        # Need a tensor that lives on a known device with grad info.
        for p in _collect_all_code_params(model):
            return p.new_zeros(())
        for p in _collect_all_preact_params(model):
            return p.new_zeros(())
        return torch.zeros(())
    return total / max(1, n)


def _preact_l2_for_model(model: nn.Module) -> torch.Tensor:
    """Sum of `_preact_l2_term` across all clusters (already weighted)."""
    total = None
    for layer in model.model.layers:
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        for meta in block._rescode["meta_experts"].values():
            if meta.preact is None:
                continue
            t = _preact_l2_term(meta)
            if t is None:
                continue
            total = t if total is None else total + t
    if total is None:
        for p in _collect_all_preact_params(model):
            return p.new_zeros(())
        for p in _collect_all_code_params(model):
            return p.new_zeros(())
        return torch.zeros(())
    return total


def _register_meta_capture_hooks(model: nn.Module):
    """Hook each RescodeMetaExpert.forward so we can grab its per-token output
    (the cluster contribution) during a model-level forward pass.

    Returns a (handles, store) pair where store is a dict
        (layer_idx, cluster_id) -> last cluster contribution tensor
    The hook stores the FORWARD output of meta() which has shape [T, hidden]
    where T is the flat token count of the input."""

    store: Dict[Tuple[int, int], torch.Tensor] = {}

    def _make_hook(layer_idx, cid):
        def hook(_module, _inputs, output):
            store[(layer_idx, cid)] = output
        return hook

    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        for cid, meta in block._rescode["meta_experts"].items():
            handles.append(meta.register_forward_hook(_make_hook(layer_idx, cid)))
    return handles, store


@torch.no_grad()
def _eval_val_lm_loss(model: nn.Module, val_loader) -> float:
    """Mean LM cross-entropy over a held-out validation loader.

    Run with `torch.no_grad()`, model in eval mode, no code clipping. Returns
    a single float (mean across batches, equally weighted).
    """
    if val_loader is None:
        return float("nan")
    losses = []
    for batch in val_loader:
        batch = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"]
        out = model(**batch)
        if out.loss is not None:
            losses.append(float(out.loss.detach().item()))
    return sum(losses) / len(losses) if losses else float("nan")


def _teacher_per_batch_from_buffers(
    teacher_buffers: Optional[Dict[int, LayerCalibration]],
) -> Optional[Dict[Tuple[int, int], List[torch.Tensor]]]:
    if teacher_buffers is None:
        return None
    out: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    for lid, buf in teacher_buffers.items():
        for cid, tlist in buf.cluster_teachers.items():
            out[(lid, cid)] = tlist
    return out


@torch.no_grad()
def _eval_val_joint_loss(
    model: nn.Module,
    val_loader,
    val_teacher_per_batch: Dict[Tuple[int, int], List[torch.Tensor]],
    config: RescodeConfig,
    fit_dtype: torch.dtype,
    student_store: Dict[Tuple[int, int], torch.Tensor],
    val_online_teacher: Optional["OnlineTeacher"] = None,
) -> Dict[str, float]:
    """Validation loss using the same joint objective as training.

    The active RescodeMetaExpert forward hooks fill `student_store`, and
    `val_teacher_per_batch` must be aligned with `val_loader` iteration order.
    Returns batch-mean LM loss, MSE, regularizers, and joint loss.
    """
    if val_loader is None:
        return {}

    loss_vals: List[float] = []
    lm_vals: List[float] = []
    mse_vals: List[float] = []
    ridge_vals: List[float] = []
    preact_l2_vals: List[float] = []
    mse_counts: List[int] = []

    for batch_idx, batch in enumerate(val_loader):
        batch = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"]

        student_store.clear()
        if val_online_teacher is not None and config.lambda_mse > 0:
            val_online_teacher.forward(batch)
        outputs = model(**batch)
        lm_loss = outputs.loss

        mse_total = lm_loss.new_zeros(())
        mse_count = 0
        for (lid, cid), student_out in student_store.items():
            if val_online_teacher is not None:
                teacher = val_online_teacher.get(lid, cid)
                if teacher is None:
                    continue
                teacher = teacher.to(student_out.device, fit_dtype)
            else:
                tlist = val_teacher_per_batch.get((lid, cid))
                if tlist is None or batch_idx >= len(tlist):
                    continue
                teacher = tlist[batch_idx].to(student_out.device, fit_dtype)
            if teacher.shape != student_out.shape:
                if teacher.numel() == student_out.numel():
                    teacher = teacher.view_as(student_out)
                else:
                    continue
            mse_total = mse_total + F.mse_loss(student_out.float(), teacher.float())
            mse_count += 1
        if mse_count > 0:
            mse_total = mse_total / mse_count

        ridge_term = _ridge_term_for_model(model)
        preact_l2 = _preact_l2_for_model(model)
        val_loss = (
            config.lambda_lm * lm_loss
            + config.lambda_mse * mse_total
            + config.ridge_lambda * ridge_term
            + preact_l2
        )

        loss_vals.append(float(val_loss.detach().item()))
        lm_vals.append(float(lm_loss.detach().item()))
        mse_vals.append(float(mse_total.detach().item()) if mse_count else 0.0)
        ridge_vals.append(float(ridge_term.detach().item()))
        preact_l2_vals.append(float(preact_l2.detach().item()))
        mse_counts.append(mse_count)

    if not loss_vals:
        return {}
    return {
        "loss": sum(loss_vals) / len(loss_vals),
        "lm_loss": sum(lm_vals) / len(lm_vals),
        "mse": sum(mse_vals) / len(mse_vals),
        "ridge": sum(ridge_vals) / len(ridge_vals),
        "preact_l2": sum(preact_l2_vals) / len(preact_l2_vals),
        "num_clusters": sum(mse_counts) / len(mse_counts),
    }


def fit_codes_joint_lm(
    model: nn.Module,
    teacher_buffers: Dict[int, LayerCalibration],
    calib_loader,
    config: RescodeConfig,
    val_loader=None,
    val_teacher_buffers: Optional[Dict[int, LayerCalibration]] = None,
    val_every: int = 10,
    online_teacher: Optional["OnlineTeacher"] = None,
    val_online_teacher: Optional["OnlineTeacher"] = None,
) -> Dict[str, float]:
    """Joint fitting: per-cluster MSE + LM cross-entropy on the same calibration
    data. Only code parameters are updated; everything else is frozen.

    Iterates calib_loader for `config.joint_epochs` passes. The MSE term is
    re-derived each step against the cached teacher cluster contributions
    (per-batch ordered to match calib_loader's iteration).

    For simplicity we re-run the calibration loader from scratch each epoch and
    rely on the fact that the dataloader is deterministic (shuffle=False).
    """
    # Collect ALL trainable RESCODE / PreAct-STAR params into per-LR groups.
    param_groups = _build_joint_param_groups(model, config)
    flat_params = [p for grp in param_groups for p in grp["params"]]
    if not flat_params:
        print("[RESCODE][joint] no trainable parameters; skipping.")
        return {}

    # Freeze everything else.
    train_ids = {id(p) for p in flat_params}
    for p in model.parameters():
        p.requires_grad_(id(p) in train_ids)

    optim = _make_optimizer(param_groups, config)
    accum_steps = max(1, int(getattr(config, "joint_grad_accum", 1)))
    total_batches = max(1, int(config.joint_epochs) * len(calib_loader))
    total_steps = max(1, math.ceil(total_batches / accum_steps))
    sched = _make_scheduler(optim, config, total_steps=total_steps)
    preact_cfg = config.preact_star if (config.preact_star is not None and config.preact_star.enabled) else None
    grad_clip = preact_cfg.grad_clip_norm if preact_cfg is not None else None
    print(
        "[RESCODE][joint] groups=" +
        ", ".join([f"{g['name']}(n={len(g['params'])},lr={g['lr']})" for g in param_groups])
    )
    print(
        f"[RESCODE][joint] batches={total_batches} "
        f"grad_accum={accum_steps} optimizer_steps={total_steps}"
    )

    handles, student_store = _register_meta_capture_hooks(model)

    # Build teacher per-batch lists keyed (layer_idx, cluster_id). Each entry
    # is a list[Tensor] aligned with the order of calib_loader iteration.
    # The collect_teacher_signals_* function appended one tensor per batch in
    # iteration order, so we assume the same order is reproduced here.
    teacher_per_batch = _teacher_per_batch_from_buffers(teacher_buffers) or {}
    val_teacher_per_batch = _teacher_per_batch_from_buffers(val_teacher_buffers)

    log = _NullLogger()
    fit_dtype = _resolve_dtype(config.fit_dtype)

    model.eval()  # freeze dropout / norm momentum
    # We *do* need autograd; the codes carry grad, everything else is frozen.

    global_step = 0
    last_metrics = {}
    # Exponential-moving-average of train lm_loss across batches (renders a
    # clean trend in logs instead of batch-content zig-zag).
    ema_alpha = 0.1
    lm_loss_ema = None
    loss_ema = None
    mse_ema = None
    # Best-val checkpointing: if validation teacher buffers are available,
    # use the same joint objective as training; otherwise fall back to held-out
    # LM loss for backward-compatible runs.
    best_val_metric = float("inf")
    best_val_metric_name = "joint/val_loss" if val_teacher_per_batch is not None else "joint/val_lm_loss"
    best_val_loss = float("inf")
    best_val_lm = float("inf")
    best_val_mse = float("inf")
    best_val_step = -1
    best_state = None  # dict[id(p)->cpu fp32 tensor]
    def _snapshot_params():
        return {id(p): p.detach().to("cpu", dtype=torch.float32, copy=True)
                for p in flat_params}
    def _restore_params(snap):
        with torch.no_grad():
            for p in flat_params:
                t = snap.get(id(p))
                if t is None:
                    continue
                p.copy_(t.to(p.device, dtype=p.dtype))

    def _run_validation(step: int):
        nonlocal best_val_metric, best_val_metric_name
        nonlocal best_val_loss, best_val_lm, best_val_mse, best_val_step, best_state
        if val_loader is None or val_every <= 0:
            return
        if val_teacher_per_batch is not None or val_online_teacher is not None:
            val_metrics = _eval_val_joint_loss(
                model=model,
                val_loader=val_loader,
                val_teacher_per_batch=val_teacher_per_batch or {},
                config=config,
                fit_dtype=fit_dtype,
                student_store=student_store,
                val_online_teacher=val_online_teacher,
            )
        else:
            val_metrics = {}

        if val_metrics:
            val_lm = val_metrics.get("lm_loss", float("nan"))
            val_loss = val_metrics.get("loss", float("nan"))
            val_mse = val_metrics.get("mse", float("nan"))
            last_metrics["joint/val_loss"] = val_loss
            last_metrics["joint/val_lm_loss"] = val_lm
            last_metrics["joint/val_mse"] = val_mse
            last_metrics["joint/val_ridge"] = val_metrics.get("ridge", 0.0)
            last_metrics["joint/val_preact_l2"] = val_metrics.get("preact_l2", 0.0)
            last_metrics["joint/val_num_clusters"] = val_metrics.get("num_clusters", 0.0)
            val_criterion = val_loss
        else:
            val_lm = _eval_val_lm_loss(model, val_loader)
            val_loss = float("nan")
            val_mse = float("nan")
            last_metrics["joint/val_lm_loss"] = val_lm
            val_criterion = val_lm

        is_best = val_criterion < best_val_metric
        if is_best:
            best_val_metric = val_criterion
            best_val_metric_name = "joint/val_loss" if val_metrics else "joint/val_lm_loss"
            best_val_lm = val_lm
            best_val_loss = val_loss
            best_val_mse = val_mse
            best_val_step = step
            best_state = _snapshot_params()
        if val_metrics:
            print(
                f"[RESCODE][joint][val] step={step} "
                f"val_loss={val_loss:.4f} val_lm={val_lm:.4f} "
                f"val_mse={val_mse:.4e}"
                + (" *best*" if is_best else "")
            )
        else:
            print(
                f"[RESCODE][joint][val] step={step} "
                f"val_lm={val_lm:.4f}"
                + (" *best*" if is_best else "")
            )

    try:
        optim.zero_grad(set_to_none=True)
        _run_validation(step=0)
        if log.enabled and any(k.startswith("joint/val_") for k in last_metrics):
            log.log(last_metrics, step=0)

        for epoch in range(int(config.joint_epochs)):
            epoch_lm_sum, epoch_lm_count = 0.0, 0
            for batch_idx, batch in enumerate(calib_loader):
                batch = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
                if "labels" not in batch:
                    batch["labels"] = batch["input_ids"]

                student_store.clear()

                # Online teacher: forward the unmerged reference model on this
                # same batch BEFORE student forward, so per-cluster outputs are
                # captured into online_teacher.last. Only spend the cost when
                # MSE is actually used.
                if online_teacher is not None and config.lambda_mse > 0:
                    online_teacher.forward(batch)

                outputs = model(**batch)
                lm_loss = outputs.loss

                # MSE against teacher: align by (layer, cluster).
                # Source: online forward (this batch) OR offline buffer lookup.
                mse_total = lm_loss.new_zeros(())
                mse_count = 0
                shape_skip_warned = False
                for (lid, cid), student_out in student_store.items():
                    if online_teacher is not None:
                        teacher = online_teacher.get(lid, cid)
                        if teacher is None:
                            continue
                        teacher = teacher.to(student_out.device, fit_dtype)
                    else:
                        tlist = teacher_per_batch.get((lid, cid))
                        if tlist is None or batch_idx >= len(tlist):
                            continue
                        teacher = tlist[batch_idx].to(student_out.device, fit_dtype)
                    if teacher.shape != student_out.shape:
                        if teacher.numel() == student_out.numel():
                            teacher = teacher.view_as(student_out)
                        else:
                            # Surface this once: it usually means the teacher
                            # buffers were subsampled at collection time, which
                            # silently zeros out the cluster-MSE term. The
                            # collection path now skips subsampling for joint
                            # mode, so seeing this in practice is a regression.
                            if epoch == 0 and batch_idx == 0 and not shape_skip_warned:
                                print(
                                    f"[RESCODE][joint][warn] shape skip "
                                    f"layer={lid} cluster={cid} "
                                    f"teacher={tuple(teacher.shape)} "
                                    f"student={tuple(student_out.shape)}"
                                )
                                shape_skip_warned = True
                            continue
                    mse_total = mse_total + F.mse_loss(student_out.float(),
                                                       teacher.float())
                    mse_count += 1
                if mse_count > 0:
                    mse_total = mse_total / mse_count

                ridge_term = _ridge_term_for_model(model)
                preact_l2 = _preact_l2_for_model(model)

                loss = (
                    config.lambda_lm * lm_loss
                    + config.lambda_mse * mse_total
                    + config.ridge_lambda * ridge_term
                    + preact_l2
                )
                (loss / accum_steps).backward()

                _lm = float(lm_loss.detach().item())
                _loss = float(loss.detach().item())
                _mse = float(mse_total.detach().item()) if mse_count else 0.0
                lm_loss_ema = _lm if lm_loss_ema is None else (1 - ema_alpha) * lm_loss_ema + ema_alpha * _lm
                loss_ema = _loss if loss_ema is None else (1 - ema_alpha) * loss_ema + ema_alpha * _loss
                mse_ema = _mse if mse_ema is None else (1 - ema_alpha) * mse_ema + ema_alpha * _mse
                epoch_lm_sum += _lm
                epoch_lm_count += 1
                last_metrics = {
                    "joint/loss": _loss,
                    "joint/lm_loss": _lm,
                    "joint/lm_loss_ema": lm_loss_ema,
                    "joint/loss_ema": loss_ema,
                    "joint/mse": _mse,
                    "joint/mse_ema": mse_ema,
                    "joint/ridge": float(ridge_term.detach().item()),
                    "joint/lr": float(optim.param_groups[0]["lr"]),
                    "joint/epoch": epoch,
                    "joint/batch": batch_idx,
                    "joint/grad_accum": accum_steps,
                    "joint/num_clusters_in_batch": mse_count,
                }

                is_accum_boundary = ((batch_idx + 1) % accum_steps == 0)
                is_last_batch = (batch_idx + 1) == len(calib_loader)
                if is_accum_boundary or is_last_batch:
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(flat_params, grad_clip)
                    optim.step()
                    sched.step()
                    _clip_all_codes(model)
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                    if (
                        val_loader is not None
                        and val_every > 0
                        and (global_step % val_every == 0)
                    ):
                        _run_validation(step=global_step)

                    if log.enabled and (global_step % max(1, config.log_every) == 0):
                        log.log(last_metrics, step=global_step)

            epoch_lm_mean = (epoch_lm_sum / epoch_lm_count) if epoch_lm_count else float("nan")
            if log.enabled:
                # Push the epoch-mean as part of the next available step so
                # the chart shows a clean per-epoch curve alongside per-step.
                log.log(
                    {
                        "joint/lm_loss_epoch_mean": epoch_lm_mean,
                        "joint/epoch_done": epoch + 1,
                    },
                    step=max(0, global_step),
                )
            print(
                f"[RESCODE][joint] epoch {epoch+1}/{config.joint_epochs} "
                f"last_loss={last_metrics.get('joint/loss', float('nan')):.4f} "
                f"lm={last_metrics.get('joint/lm_loss', float('nan')):.4f} "
                f"epoch_mean_lm={epoch_lm_mean:.4f} "
                f"mse={last_metrics.get('joint/mse', float('nan')):.4e} "
                f"n_clusters_with_mse={last_metrics.get('joint/num_clusters_in_batch', 0)}"
            )
    finally:
        for h in handles:
            h.remove()

    # Restore best-val params if we saw at least one val improvement.
    if best_state is not None and val_loader is not None:
        _restore_params(best_state)
        if best_val_metric_name == "joint/val_loss":
            print(
                f"[RESCODE][joint] restored best-val state from step={best_val_step} "
                f"val_loss={best_val_loss:.4f} val_lm={best_val_lm:.4f} "
                f"val_mse={best_val_mse:.4e}"
            )
        else:
            print(
                f"[RESCODE][joint] restored best-val state from step={best_val_step} "
                f"val_lm={best_val_lm:.4f}"
            )
    if log.enabled:
        log.summary_update({
            "joint/final_loss": last_metrics.get("joint/loss"),
            "joint/final_lm_loss": last_metrics.get("joint/lm_loss"),
            "joint/final_mse": last_metrics.get("joint/mse"),
            "joint/total_steps": global_step,
            "joint/planned_optimizer_steps": total_steps,
            "joint/grad_accum": accum_steps,
            "joint/best_val_metric": best_val_metric if best_val_metric != float("inf") else None,
            "joint/best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
            "joint/best_val_lm": best_val_lm if best_val_lm != float("inf") else None,
            "joint/best_val_mse": best_val_mse if best_val_mse != float("inf") else None,
            "joint/best_val_step": best_val_step,
        })
    return last_metrics


def freeze_non_code_parameters(model: nn.Module, include_preact: bool = True):
    """Set requires_grad=False on every parameter except RESCODE codes.

    When `include_preact=True` (default), PreAct-STAR parameters
    (`gate_bias_codes`, `up_bias_codes`, lowrank A/B/z, down_L/R/z) are also
    kept trainable. Per-cluster fitting in `fit_codes_for_block` further
    refines this by config-driven group filters.
    """
    keep = set()
    for layer in model.model.layers:
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        for meta in block._rescode["meta_experts"].values():
            for p in meta.code_parameters():
                keep.add(id(p))
            if include_preact and meta.preact is not None:
                for p in meta.preact.preact_parameter_list():
                    keep.add(id(p))
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)
