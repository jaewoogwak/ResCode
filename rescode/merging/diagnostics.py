"""Diagnostics helpers for RESCODE: cluster stats, reconstruction error, and
code parameter statistics."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .cluster_map import ExpertClusterMap
from .meta_expert import RescodeMetaExpert


def log_cluster_stats(cluster_maps: Dict[int, ExpertClusterMap]) -> List[Dict]:
    """Print + return per-layer cluster summary stats."""
    summaries: List[Dict] = []
    for layer_idx in sorted(cluster_maps.keys()):
        cmap = cluster_maps[layer_idx]
        sizes = sorted(cmap.cluster_sizes(), reverse=True)
        summary = {
            "layer_idx": layer_idx,
            "num_original_experts": cmap.num_original_experts,
            "num_clusters": cmap.num_clusters,
            "num_singletons": cmap.num_singletons,
            "num_meta_clusters": cmap.num_meta_clusters,
            "largest_cluster_size": cmap.largest_cluster_size,
            "cluster_sizes": sizes,
            "compression_ratio": (
                cmap.num_clusters / cmap.num_original_experts
                if cmap.num_original_experts else 1.0
            ),
        }
        print(
            f"[RESCODE][cluster] layer={layer_idx} "
            f"E={cmap.num_original_experts} -> P={cmap.num_clusters} "
            f"(largest={cmap.largest_cluster_size}, "
            f"singletons={cmap.num_singletons}/{cmap.num_clusters}, "
            f"meta={cmap.num_meta_clusters}, sizes={sizes[:8]}...)"
        )
        summaries.append(summary)
    return summaries


@torch.no_grad()
def compute_reconstruction_error(
    student_outputs: torch.Tensor,
    teacher_outputs: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Mean-squared error and cosine similarity between student/teacher cluster
    contributions. Both tensors must already be at the same shape."""
    s = student_outputs.float()
    t = teacher_outputs.float()
    diff = s - t
    mse = (diff * diff).mean().item()
    teacher_norm_sq = (t * t).sum().clamp_min(eps)
    rel_l2 = (diff * diff).sum().item() / max(teacher_norm_sq.item(), eps)

    s_flat = s.reshape(s.shape[0], -1)
    t_flat = t.reshape(t.shape[0], -1)
    cos = torch.nn.functional.cosine_similarity(s_flat, t_flat, dim=-1, eps=eps).mean().item()

    return {"mse": mse, "rel_l2": rel_l2, "cosine": cos}


@torch.no_grad()
def log_preact_statistics(model: nn.Module) -> List[Dict]:
    """Print + return per-(layer, cluster) residual parameter statistics.

    Reports norms of bias codes and lowrank factors so users can see whether
    the optimizer is actually moving them away from their zero-init state.
    """
    out: List[Dict] = []
    for layer_idx, layer in enumerate(model.model.layers):
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        for cid, meta in block._rescode["meta_experts"].items():
            if meta.preact is None:
                continue
            stat: Dict = {"layer_idx": layer_idx, "cluster_id": cid}
            for n in (
                "gate_bias_codes", "up_bias_codes",
                "gate_A", "gate_B", "gate_z",
                "up_A", "up_B", "up_z",
                "down_L", "down_R", "down_z",
            ):
                if hasattr(meta.preact, n):
                    t = getattr(meta.preact, n).detach().float()
                    stat[f"{n}_norm"] = float(t.norm().item())
                    stat[f"{n}_mean"] = float(t.mean().item())
                    stat[f"{n}_std"] = float(t.std().item())
                    if n in ("gate_bias_codes", "up_bias_codes"):
                        stat[f"{n}_min"] = float(t.min().item())
                        stat[f"{n}_max"] = float(t.max().item())
            out.append(stat)
            print(
                f"[RESCODE][preact] layer={layer_idx} cluster={cid} "
                + " ".join(
                    f"{k}={v:.3e}" for k, v in stat.items()
                    if k.endswith("_norm")
                )
            )
    return out


def log_giant_cluster_summary(
    cluster_maps,
    layer_results: Optional[Dict] = None,
) -> List[Dict]:
    """Pick the largest cluster per layer and print its before/after metrics.

    Qwen 60->30 typically has one giant cluster (~31 members) per layer with
    most calibration tokens routed to it; that's where reconstruction is
    hardest, so we surface it explicitly. `layer_results` is the dict
    returned by `fit_all_codes` (or compatible).
    """
    summaries: List[Dict] = []
    for layer_idx in sorted(cluster_maps.keys()):
        cmap = cluster_maps[layer_idx]
        if not cmap.clusters:
            continue
        # Find the largest non-singleton cluster.
        candidates = [
            (cid, members) for cid, members in cmap.cluster_to_experts.items()
            if not cmap.is_singleton[cid]
        ]
        if not candidates:
            continue
        cid, members = max(candidates, key=lambda kv: len(kv[1]))
        s = {
            "layer_idx": layer_idx,
            "cluster_id": cid,
            "cluster_size": len(members),
        }
        if layer_results and layer_idx in layer_results and cid in layer_results[layer_idx]:
            r = layer_results[layer_idx][cid]
            for k in ("before_mse", "after_mse", "before_cosine", "after_cosine", "num_tokens"):
                if k in r:
                    s[k] = r[k]
            if "before_mse" in s and "after_mse" in s and s["before_mse"] > 0:
                s["mse_relative_improvement"] = (
                    (s["before_mse"] - s["after_mse"]) / s["before_mse"]
                )
        summaries.append(s)
        msg = (
            f"[RESCODE][giant] layer={layer_idx} cluster={cid} size={len(members)}"
        )
        if "before_mse" in s and "after_mse" in s:
            msg += (
                f" mse {s['before_mse']:.4e} -> {s['after_mse']:.4e}"
                f" ({100.0 * s.get('mse_relative_improvement', 0.0):+.2f}%)"
            )
        if "before_cosine" in s and "after_cosine" in s:
            msg += f" cos {s['before_cosine']:.4f} -> {s['after_cosine']:.4f}"
        print(msg)
    return summaries


@torch.no_grad()
def log_code_statistics(model: nn.Module) -> List[Dict]:
    """Print + return per-(layer, cluster) code statistics."""
    out: List[Dict] = []
    for layer_idx, layer in enumerate(model.model.layers):
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        meta_experts = block._rescode["meta_experts"]
        for cid, meta in meta_experts.items():
            stat: Dict = {"layer_idx": layer_idx, "cluster_id": cid, "code_type": meta.code_type}
            if meta.code_type == "none":
                out.append(stat)
                continue
            if meta.code_type in {"scalar", "diag"}:
                code = meta.code.detach().float()
            else:
                code = (1.0 + meta.code_Z.detach().float() @ meta.code_U.detach().float().t())
            stat.update({
                "mean": code.mean().item(),
                "std": code.std().item(),
                "min": code.min().item(),
                "max": code.max().item(),
            })
            if meta.code_clip_min is not None:
                stat["fraction_clipped_min"] = (code <= meta.code_clip_min + 1e-9).float().mean().item()
            if meta.code_clip_max is not None:
                stat["fraction_clipped_max"] = (code >= meta.code_clip_max - 1e-9).float().mean().item()
            print(
                f"[RESCODE][codes] layer={layer_idx} cluster={cid} "
                f"mean={stat['mean']:.4f} std={stat['std']:.4f} "
                f"min={stat['min']:.4f} max={stat['max']:.4f}"
            )
            out.append(stat)
    return out
