"""Save and load RESCODE state.

We save:
    - per-layer cluster maps
    - per-layer per-cluster code tensors (and lowrank factors)
    - the RescodeConfig used to build them
    - layer-level summary stats (cluster sizes, num_singletons)

The merged expert weights themselves are part of the underlying model state
dict; saving/loading the model with HuggingFace's `save_pretrained` /
`torch.save(model.state_dict())` is left to the caller.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .cluster_map import ExpertClusterMap
from .config import RescodeConfig
from .meta_expert import RescodeMetaExpert


def _iter_moe_blocks(model: nn.Module):
    """Yield (layer_idx, moe_block) for each block carrying _rescode state."""
    for layer_idx, layer in enumerate(model.model.layers):
        for attr in ("mlp", "block_sparse_moe"):
            block = getattr(layer, attr, None)
            if block is not None and hasattr(block, "_rescode"):
                yield layer_idx, block
                break


# Low-rank residual matrices - bulk of checkpoint size; safe to bf16 for storage.
# bias_codes / preact_std are tiny + accuracy-critical -> kept fp32.
_LOWRANK_RESIDUAL_KEYS = (
    "gate_A", "gate_B", "gate_z",
    "up_A", "up_B", "up_z",
    "down_L", "down_R", "down_z",
)
_FP32_PRESERVE_KEYS = (
    "gate_bias_codes", "up_bias_codes",
    "gate_preact_std", "up_preact_std",
)


def _serialize_meta(
    meta: RescodeMetaExpert,
    lowrank_save_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {
        "code_type": meta.code_type,
        "member_expert_ids": list(meta.member_expert_ids),
        "intermediate_size": int(meta.intermediate_size),
    }
    if meta.code_type == "scalar":
        out["code"] = meta.code.detach().cpu()
    elif meta.code_type == "diag":
        out["code"] = meta.code.detach().cpu()
    elif meta.code_type == "lowrank":
        out["code_U"] = meta.code_U.detach().cpu()
        out["code_Z"] = meta.code_Z.detach().cpu()
        out["code_rank"] = int(meta.code_rank)
    elif meta.code_type == "none":
        pass

    # Residual module (optional). Only the params that exist (config-controlled)
    # get serialized.
    if meta.preact is not None:
        preact_dump: Dict[str, torch.Tensor] = {
            "config": meta.preact.config.to_dict(),
        }
        for n in _LOWRANK_RESIDUAL_KEYS:
            if hasattr(meta.preact, n):
                t = getattr(meta.preact, n).detach().cpu()
                if lowrank_save_dtype is not None and t.is_floating_point():
                    t = t.to(lowrank_save_dtype)
                preact_dump[n] = t
        for n in _FP32_PRESERVE_KEYS:
            if hasattr(meta.preact, n):
                preact_dump[n] = getattr(meta.preact, n).detach().cpu()
        out["preact"] = preact_dump
    return out


def save_rescode_state(
    model: nn.Module,
    config: RescodeConfig,
    path: str,
    lowrank_save_dtype: Optional[torch.dtype] = torch.bfloat16,
):
    """Pickle a dict with cluster maps, codes, and config to `path`.

    `lowrank_save_dtype` controls the storage dtype of the bulk low-rank
    residual matrices (gate/up/down A,B,z,L,R). Default `bfloat16` halves
    the checkpoint size with ~3-bit mantissa precision loss; bias_codes and
    preact_std buffers stay fp32. Pass `None` (or `torch.float32`) to keep
    everything in the original fp32 dtype.
    """
    payload: Dict = {
        "compression_method": "rescode",
        "base_method": "rescode",
        "router_preserved": True,
        "config": config.to_dict(),
        "lowrank_save_dtype": (
            str(lowrank_save_dtype).replace("torch.", "")
            if lowrank_save_dtype is not None else "float32"
        ),
        "layers": {},
    }

    for layer_idx, block in _iter_moe_blocks(model):
        star = block._rescode
        cmap: ExpertClusterMap = star["cluster_map"]
        meta_experts = star["meta_experts"]

        layer_dump = {
            "cluster_map": cmap.to_dict(),
            "meta": {
                int(cid): _serialize_meta(meta, lowrank_save_dtype=lowrank_save_dtype)
                for cid, meta in meta_experts.items()
            },
        }
        payload["layers"][int(layer_idx)] = layer_dump

    torch.save(payload, path)


def _load_meta_into(meta: RescodeMetaExpert, dump: Dict):
    if dump.get("code_type", meta.code_type) != meta.code_type:
        raise ValueError(
            f"[RESCODE] code_type mismatch on load: file={dump.get('code_type')} "
            f"vs model={meta.code_type}"
        )
    if meta.code_type == "scalar" or meta.code_type == "diag":
        with torch.no_grad():
            meta.code.copy_(dump["code"].to(meta.code.device).to(meta.code.dtype))
    elif meta.code_type == "lowrank":
        with torch.no_grad():
            meta.code_U.copy_(dump["code_U"].to(meta.code_U.device).to(meta.code_U.dtype))
            meta.code_Z.copy_(dump["code_Z"].to(meta.code_Z.device).to(meta.code_Z.dtype))

    # Residual module (optional)
    preact_dump = dump.get("preact")
    if preact_dump is not None and meta.preact is not None:
        with torch.no_grad():
            for n in (
                "gate_bias_codes", "up_bias_codes",
                "gate_A", "gate_B", "gate_z",
                "up_A", "up_B", "up_z",
                "down_L", "down_R", "down_z",
                "gate_preact_std", "up_preact_std",
            ):
                if hasattr(meta.preact, n) and n in preact_dump:
                    dst = getattr(meta.preact, n)
                    src = preact_dump[n].to(dst.device).to(dst.dtype)
                    dst.copy_(src)
            if (
                hasattr(meta.preact, "down_z")
                and "down_z" not in preact_dump
                and ("down_L" in preact_dump or "down_R" in preact_dump)
            ):
                # Backward compatibility for old down_residual checkpoints:
                # q_d=sum rho*1 = 1, so L(q_d * R h) equals the old L(R h).
                meta.preact.down_z.fill_(1.0)
    elif preact_dump is not None and meta.preact is None:
        print(
            "[RESCODE][load] file has preact tensors but the model has no "
            "preact module attached; preact params skipped."
        )


def load_rescode_state(model: nn.Module, path: str) -> RescodeConfig:
    """Load codes from `path` into an already-converted RESCODE model.

    The model must have been built via `convert_*_to_rescode` already, and
    its cluster structure must match the file (same layers, same cluster
    membership, same code_type).
    """
    payload = torch.load(path, map_location="cpu")
    cfg = RescodeConfig.from_dict(payload["config"])

    file_layers = payload["layers"]

    for layer_idx, block in _iter_moe_blocks(model):
        if int(layer_idx) not in file_layers:
            raise KeyError(f"[RESCODE] layer {layer_idx} missing from save file.")
        layer_dump = file_layers[int(layer_idx)]
        meta_experts = block._rescode["meta_experts"]
        for cid_str, dump in layer_dump["meta"].items():
            cid = int(cid_str)
            if cid not in meta_experts:
                raise KeyError(
                    f"[RESCODE] layer {layer_idx} cluster {cid} present in file but "
                    f"not in model (singleton/non-singleton mismatch)."
                )
            _load_meta_into(meta_experts[cid], dump)
    return cfg
