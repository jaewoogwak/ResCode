import os
from typing import Dict, Optional

import torch

from .calib_stats import iter_moe_blocks, load_or_collect_calib_stats
from .config import parse_lowrank_target
from .lowrank_init import _get_proj_weights
from .expert_clusters import form_clusters, groups_to_labels, residual_params_fn_for


def ffn_name(layer_idx: int, backbone: str) -> str:
    return f"model.layers.{layer_idx}.{'block_sparse_moe' if backbone == 'mixtral' else 'mlp'}"


def descriptor_distances(model, grouper, dataloader, backbone: str) -> Dict[int, torch.Tensor]:
    grouper.compute_saliency_and_descriptors(model, dataloader)
    descriptors = grouper.descriptor_state_dict()
    out = {}
    for layer_idx, _ in iter_moe_blocks(model, backbone):
        o = descriptors[ffn_name(layer_idx, backbone)].double().cpu()
        out[layer_idx] = torch.cdist(o, o)
    return out


def cluster_from_stats(
    model,
    grouper,
    dataloader,
    backbone: str,
    num_groups: int,
    singleton_ratio: float,
    rank: int,
    lowrank_target: Optional[str],
    linkage: str = "average",
    budget_aware: bool = True,
    cache_path: Optional[str] = None,
    distances: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, float]:
    stats = load_or_collect_calib_stats(model, dataloader, backbone, cache_path=cache_path)
    if distances is not None:
        stats = {layer_idx: dict(s, dist=distances[layer_idx]) for layer_idx, s in stats.items()}
    _, block = next(iter_moe_blocks(model, backbone))
    d_ff, d_model = _get_proj_weights(block.experts[0], backbone)[0].shape
    n_targets = len(parse_lowrank_target(lowrank_target)) if lowrank_target else 0
    groups, summary = form_clusters(
        stats,
        num_groups=num_groups,
        singleton_ratio=singleton_ratio,
        expert_params=3 * d_model * d_ff,
        residual_params_fn=residual_params_fn_for(d_model, d_ff, rank, n_targets),
        linkage=linkage,
        budget_aware=budget_aware,
    )
    for layer_idx, block in iter_moe_blocks(model, backbone):
        name = ffn_name(layer_idx, backbone)
        grouper._group_state_dict[name] = groups_to_labels(groups[layer_idx], len(block.experts))
        grouper._saliency_state_dict[name] = stats[layer_idx]["saliency"]
        grouper._usage_frequency_state_dict[name] = stats[layer_idx]["freq"]
    return summary


def residual_init_weights(grouper, hc_merge: str, delta: float):
    base = None
    if hc_merge == "saliency":
        base = {k: v + float(delta) for k, v in grouper.saliency_state_dict().items()}
    return grouper.usage_frequency_state_dict(), base


def write_budget_line(result_path: Optional[str], summary: Optional[Dict[str, float]], minutes: float):
    if summary is None or not result_path:
        return
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    with open(result_path, "a") as f:
        f.write(
            f"[budget] expert-memory compression = {100 * summary['compression_ratio']:.3f}% "
            f"(compressed {summary['compressed_params']:,} / original {summary['original_params']:,}; "
            f"residual {summary['residual_params']:,}; extra merges {summary['extra_merges']}); "
            f"compression time {minutes:.1f} min\n"
        )


def start_layer_recon(model, original_experts, backbone: str, mode: str):
    from .layer_recon import LayerReconstruction, attach_original_experts
    attach_original_experts(model, original_experts, backbone)
    rec = LayerReconstruction(model, mode=mode, backbone=backbone)
    print(f"[RESCODE] L_rec enabled (target={mode}, layers={rec.num_layers()})")
    return rec


def stop_layer_recon(model, rec):
    from .layer_recon import detach_original_experts
    rec.cleanup()
    detach_original_experts(model)
    torch.cuda.empty_cache()
