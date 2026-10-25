import torch


def add_original_cluster_output(final_hidden_states, hidden_states, member_ids,
                                selected_experts, routing_weights, original_experts):
    for j in member_ids:
        mask = selected_experts == j
        if not mask.any():
            continue
        top_x, idx = torch.where(mask)
        out = original_experts[j](hidden_states.index_select(0, top_x)) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, out.to(final_hidden_states.dtype))
