"""Helpers shared by the RESCODE MoE forwards."""

import torch


def add_original_cluster_output(final_hidden_states, hidden_states, member_ids,
                                selected_experts, routing_weights, original_experts):
    """Add the ORIGINAL contribution of a merged cluster, sum_{j in G cap T_k(x)} pi_j(x) E_j(x).

    Used for the original-layer target of L_rec (see layer_recon.py); the
    experts in `original_experts` are frozen copies taken before merging.
    """
    for j in member_ids:
        mask = selected_experts == j
        if not mask.any():
            continue
        top_x, idx = torch.where(mask)
        out = original_experts[j](hidden_states.index_select(0, top_x)) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, out.to(final_hidden_states.dtype))
