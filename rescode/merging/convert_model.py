"""Convert an RESCODE merged model into a RESCODE model.

The conversion is purely a forward-pass rewrite: the original router is kept,
non-singleton clusters are replaced with a single `RescodeMetaExpert`, and
singleton clusters keep their original expert. The block's `forward` method
is monkey-patched in place so generation paths remain unchanged in signature.
"""

from copy import deepcopy
from types import MethodType
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .cluster_map import ExpertClusterMap
from .config import RescodeConfig
from .meta_expert import RescodeMetaExpert
from .forward_qwen import rescode_qwen_moe_forward
from .forward_qwen3 import rescode_qwen3_moe_forward
from .forward_mixtral import rescode_mixtral_moe_forward
from .forward_deepseek import rescode_deepseek_moe_forward


# --------------------------------------------------------------------- helpers
def _qwen_get_projections(expert: nn.Module):
    return expert.gate_proj, expert.up_proj, expert.down_proj


def _mixtral_get_projections(expert: nn.Module):
    # Mixtral: w1 = gate_proj, w3 = up_proj, w2 = down_proj
    return expert.w1, expert.w3, expert.w2


def _deepseek_get_projections(expert: nn.Module):
    # DeepSeek MLP uses the same names as Qwen.
    return expert.gate_proj, expert.up_proj, expert.down_proj


def _build_meta_for_cluster(
    moe_block: nn.Module,
    member_ids: List[int],
    get_projs,
    config: RescodeConfig,
    preact_precomputed_init: Optional[dict] = None,
) -> RescodeMetaExpert:
    """Pick the merged-expert weight (RESCODE bound it to member_ids[0]) and
    wrap it in a RescodeMetaExpert with codes initialized to identity."""
    rep_expert = moe_block.experts[member_ids[0]]
    gate_proj, up_proj, down_proj = get_projs(rep_expert)
    # With `config.preact_star` enabled the meta expert also holds a
    # `PreActModule` with the residual bases and codes.
    preact_cfg = config.preact_star if config.preact_enabled else None
    return RescodeMetaExpert(
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
        member_expert_ids=member_ids,
        code_type=config.code_type,
        code_rank=config.code_rank if config.code_type == "lowrank" else None,
        init=config.code_init,
        code_clip_min=config.code_clip_min,
        code_clip_max=config.code_clip_max,
        freeze_code=config.freeze_code,
        preact_config=preact_cfg,
        preact_precomputed_init=preact_precomputed_init,
    )


def _attach_to_block(
    moe_block: nn.Module,
    cluster_map: ExpertClusterMap,
    meta_experts: Dict[int, RescodeMetaExpert],
    singleton_experts: Dict[int, nn.Module],
    forward_fn,
    physical_module_list: nn.ModuleList,
):
    """Attach RESCODE state and rewire forward of one MoE block."""
    moe_block._rescode = {
        "cluster_map": cluster_map,
        "meta_experts": meta_experts,
        "singleton_experts": singleton_experts,
    }
    # Register meta + singleton modules so they move with `.to(device)` /
    # `.state_dict()` and are tracked as parameters of the model.
    moe_block.add_module("rescode_physical_experts", physical_module_list)
    moe_block.forward = MethodType(forward_fn, moe_block)


def _convert_one_block(
    moe_block: nn.Module,
    cluster_map: ExpertClusterMap,
    config: RescodeConfig,
    get_projs,
    forward_fn,
    layer_init_data: Optional[Dict[int, dict]] = None,
):
    meta_experts: Dict[int, RescodeMetaExpert] = {}
    singleton_experts: Dict[int, nn.Module] = {}
    physical_modules: List[nn.Module] = []

    for cluster_id, member_ids in cluster_map.cluster_to_experts.items():
        if len(member_ids) == 1:
            singleton_experts[cluster_id] = moe_block.experts[member_ids[0]]
            physical_modules.append(singleton_experts[cluster_id])
        else:
            cluster_init = (layer_init_data or {}).get(cluster_id) if layer_init_data is not None else None
            meta = _build_meta_for_cluster(
                moe_block, member_ids, get_projs, config,
                preact_precomputed_init=cluster_init,
            )
            meta_experts[cluster_id] = meta
            physical_modules.append(meta)

    _attach_to_block(
        moe_block=moe_block,
        cluster_map=cluster_map,
        meta_experts=meta_experts,
        singleton_experts=singleton_experts,
        forward_fn=forward_fn,
        physical_module_list=nn.ModuleList(physical_modules),
    )


# ------------------------------------------------------------- public API
def convert_qwen_to_rescode(
    model: nn.Module,
    cluster_maps: Dict[int, ExpertClusterMap],
    config: RescodeConfig,
    preact_init_data: Optional[Dict[int, Dict[int, dict]]] = None,
) -> nn.Module:
    """Convert a Qwen2-MoE model in place to RESCODE.

    Assumes RESCODE merging has already produced shared expert weights bound
    to member_ids[0] of each cluster. Cluster member ordering is taken from
    `cluster_maps[layer_idx].cluster_to_experts`.

    `preact_init_data[layer_idx][cluster_id] = {"gate": (B, A, Z), "up": ...}`
    is consumed by PreActModule when `lowrank_init != "lora_zero_B"`.
    """
    for layer_idx, cmap in cluster_maps.items():
        moe_block = model.model.layers[layer_idx].mlp
        layer_init = (preact_init_data or {}).get(layer_idx) if preact_init_data is not None else None
        _convert_one_block(
            moe_block=moe_block,
            cluster_map=cmap,
            config=config,
            get_projs=_qwen_get_projections,
            forward_fn=rescode_qwen_moe_forward,
            layer_init_data=layer_init,
        )
    return model


def convert_mixtral_to_rescode(
    model: nn.Module,
    cluster_maps: Dict[int, ExpertClusterMap],
    config: RescodeConfig,
    preact_init_data: Optional[Dict[int, Dict[int, dict]]] = None,
) -> nn.Module:
    """Convert a Mixtral model in place to RESCODE."""
    for layer_idx, cmap in cluster_maps.items():
        moe_block = model.model.layers[layer_idx].block_sparse_moe
        layer_init = (preact_init_data or {}).get(layer_idx) if preact_init_data is not None else None
        _convert_one_block(
            moe_block=moe_block,
            cluster_map=cmap,
            config=config,
            get_projs=_mixtral_get_projections,
            forward_fn=rescode_mixtral_moe_forward,
            layer_init_data=layer_init,
        )
    return model


def convert_qwen3_to_rescode(
    model: nn.Module,
    cluster_maps: Dict[int, ExpertClusterMap],
    config: RescodeConfig,
    preact_init_data: Optional[Dict[int, Dict[int, dict]]] = None,
) -> nn.Module:
    """Convert a Qwen3-MoE model in place to RESCODE.

    Same as the Qwen2 path (expert MLPs use `gate_proj/up_proj/down_proj`)
    but swaps in the Qwen3 forward (no shared-expert path).
    """
    for layer_idx, cmap in cluster_maps.items():
        moe_block = model.model.layers[layer_idx].mlp
        layer_init = (preact_init_data or {}).get(layer_idx) if preact_init_data is not None else None
        _convert_one_block(
            moe_block=moe_block,
            cluster_map=cmap,
            config=config,
            get_projs=_qwen_get_projections,
            forward_fn=rescode_qwen3_moe_forward,
            layer_init_data=layer_init,
        )
    return model


def convert_deepseek_to_rescode(
    model: nn.Module,
    cluster_maps: Dict[int, ExpertClusterMap],
    config: RescodeConfig,
    preact_init_data: Optional[Dict[int, Dict[int, dict]]] = None,
) -> nn.Module:
    """Convert a DeepSeek-V1/V2 MoE model in place to RESCODE.

    Layer 0 is dense in DeepSeek; callers should not put it in `cluster_maps`.
    """
    for layer_idx, cmap in cluster_maps.items():
        moe_block = model.model.layers[layer_idx].mlp
        layer_init = (preact_init_data or {}).get(layer_idx) if preact_init_data is not None else None
        _convert_one_block(
            moe_block=moe_block,
            cluster_map=cmap,
            config=config,
            get_projs=_deepseek_get_projections,
            forward_fn=rescode_deepseek_moe_forward,
            layer_init_data=layer_init,
        )
    return model
