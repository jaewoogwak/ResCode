"""RESCODE forward for Mixtral sparse MoE blocks."""

from typing import Tuple

import torch
import torch.nn.functional as F

from .forward_common import add_original_cluster_output


def rescode_mixtral_moe_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for `MixtralSparseMoeBlock.forward`.

    Requires `self._rescode` like Qwen's variant. Preserves the original
    router; cluster aggregation is purely an expert-side rewrite.
    """
    star = self._rescode
    cmap = star["cluster_map"]
    meta_experts = star["meta_experts"]
    singleton_experts = star["singleton_experts"]

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
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
    return final_hidden_states, router_logits
