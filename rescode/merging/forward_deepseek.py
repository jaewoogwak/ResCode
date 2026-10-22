"""RESCODE forward for DeepSeek-V1/V2 MoE sparse blocks.

The function `rescode_deepseek_moe_forward` is monkey-patched onto a
`DeepseekMoE` / `DeepseekV2MoE` (via `MethodType`) after RESCODE conversion.

Key differences vs. Qwen2/Qwen3:
    - `self.gate` is a `MoEGate` module (NOT nn.Linear). Its `forward` returns
      `(topk_idx, topk_weight, aux_loss)` and not router logits. So we compute
      router logits ourselves via `F.linear(hidden, self.gate.weight, None)`.
    - `top_k` and `norm_topk_prob` live on `self.gate`, not `self`.
    - DeepSeek has `self.shared_experts` (single MLP, despite the plural name)
      which is added on top of the routed contribution.
    - The stock DeepSeek decoder layer calls `y = self.mlp(hidden_states)` -
      it expects a single tensor return, not a (output, router_logits) tuple.
      So this forward returns ONLY `final_hidden_states`.
"""

import torch
import torch.nn.functional as F

from .forward_common import add_original_cluster_output


def rescode_deepseek_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for `DeepseekMoE.forward` / `DeepseekV2MoE.forward`.

    Requires the block to have a `_rescode` attribute holding:
        - cluster_map: ExpertClusterMap for this layer
        - meta_experts: dict[cluster_id -> RescodeMetaExpert] for non-singletons
        - singleton_experts: dict[cluster_id -> nn.Module] for singletons
    """
    star = self._rescode
    cmap = star["cluster_map"]
    meta_experts = star["meta_experts"]
    singleton_experts = star["singleton_experts"]

    identity = hidden_states
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # MoEGate is not nn.Linear; reconstruct router logits ourselves.
    router_logits = F.linear(hidden_states, self.gate.weight, None)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    top_k = int(self.gate.top_k)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if bool(getattr(self.gate, "norm_topk_prob", False)):
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    T = hidden_states.shape[0]
    final_hidden_states = torch.zeros(
        (T, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    for cluster_id, member_ids in cmap.cluster_to_experts.items():
        if cmap.is_singleton[cluster_id]:
            j = member_ids[0]
            mask = (selected_experts == j)
            if not mask.any():
                continue
            top_x, idx = torch.where(mask)
            current_state = hidden_states.index_select(0, top_x)
            expert = singleton_experts[cluster_id]
            current_hidden = expert(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden.to(final_hidden_states.dtype))
        elif star.get("teacher_active"):
            add_original_cluster_output(final_hidden_states, hidden_states, member_ids,
                                        selected_experts, routing_weights, star["orig_experts"])
        else:
            meta = meta_experts[cluster_id]
            cluster_out = meta(
                flat_hidden_states=hidden_states,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
            )
            final_hidden_states = final_hidden_states + cluster_out.to(final_hidden_states.dtype)

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    # DeepSeek shared experts (single MLP; the attribute name is plural but
    # the module is one MLP). Stock modeling calls it as
    # `y = y + self.shared_experts(identity)`.
    if hasattr(self, "shared_experts") and self.shared_experts is not None:
        shared_out = self.shared_experts(identity)
        final_hidden_states = final_hidden_states + shared_out

    return final_hidden_states
