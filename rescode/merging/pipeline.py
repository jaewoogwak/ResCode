"""Idea 1 / Idea 3 helpers shared by the rescode_* entry points."""

import os
from typing import Dict, Optional

import torch

from .calib_stats import iter_moe_blocks, load_or_collect_calib_stats
from .config import parse_lowrank_target
from .lowrank_init import _get_proj_weights
from .expert_clusters import form_clusters, groups_to_labels, residual_params_fn_for


def ffn_name(layer_idx: int, backbone: str) -> str:
    return f"model.layers.{layer_idx}.{'block_sparse_moe' if backbone == 'mixtral' else 'mlp'}"


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
) -> Dict[str, float]:
    """Idea 1 from one pass over the calibration set (paper Sec. 4.2).

    Fills the grouper's group labels, saliency a_i and routing frequency omega_i,
    which the saliency-weighted merge and the residual init read, and returns
    the expert-memory budget summary.
    """
    stats = load_or_collect_calib_stats(model, dataloader, backbone, cache_path=cache_path)
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


def residual_init_weights(grouper, clustering: str, hc_merge: str, delta: float):
    """(freq_dict, base_weight_dict) for compute_lowrank_preact_init_data.

    Covariances are weighted by the routing frequencies omega_i (App. E) and
    residuals are taken w.r.t. the saliency-weighted base S_G of Eq. (3).
    """
    if clustering == "dist":
        freq = grouper.usage_frequency_state_dict()
        base = None
        if hc_merge == "saliency":
            base = {k: v + float(delta) for k, v in grouper.saliency_state_dict().items()}
        return freq, base
    return (grouper.usage_frequency_state_dict() if hc_merge == "freq" else None), None


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
