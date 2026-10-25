import os
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def iter_moe_blocks(model, backbone: str) -> Iterator[Tuple[int, torch.nn.Module]]:
    for layer_idx, layer in enumerate(model.model.layers):
        block = layer.block_sparse_moe if backbone == "mixtral" else layer.mlp
        if hasattr(block, "experts"):
            yield layer_idx, block


def route(block, flat: torch.Tensor, backbone: str, top_k: int, norm_topk_prob: bool):
    if backbone == "deepseek":
        logits = F.linear(flat, block.gate.weight, None)
    else:
        logits = block.gate(flat)
    probs = F.softmax(logits, dim=-1, dtype=torch.float)
    weights, selected = torch.topk(probs, top_k, dim=-1)
    if backbone == "mixtral" or norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return selected, weights


def routing_config(model, backbone: str) -> Tuple[int, bool]:
    cfg = model.config
    top_k = int(cfg.num_experts_per_tok)
    norm_topk_prob = True if backbone == "mixtral" else bool(getattr(cfg, "norm_topk_prob", False))
    return top_k, norm_topk_prob


@torch.no_grad()
def collect_calib_stats(
    model,
    dataloader,
    backbone: str,
    token_chunk: int = 2048,
) -> Dict[int, Dict[str, torch.Tensor]]:
    top_k, norm_topk_prob = routing_config(model, backbone)
    acc: Dict[int, Dict[str, torch.Tensor]] = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, _output):
            hidden = inputs[0]
            flat = hidden.reshape(-1, hidden.shape[-1])
            selected, weights = route(module, flat, backbone, top_k, norm_topk_prob)
            n = len(module.experts)
            st = acc.get(layer_idx)
            if st is None:
                st = {
                    "sal": torch.zeros(n, dtype=torch.float64, device=flat.device),
                    "freq": torch.zeros(n, dtype=torch.float64, device=flat.device),
                    "gram": torch.zeros(n, n, dtype=torch.float64, device=flat.device),
                    "tokens": 0,
                }
                acc[layer_idx] = st
            gate = torch.zeros(flat.shape[0], n, dtype=torch.float32, device=flat.device)
            gate.scatter_(1, selected, weights.float())
            routed = torch.zeros_like(gate).scatter_(1, selected, 1.0)
            st["freq"] += routed.sum(0).double()
            for s in range(0, flat.shape[0], token_chunk):
                xc = flat[s:s + token_chunk]
                outs = torch.stack([expert(xc) for expert in module.experts]).float()
                st["sal"] += (outs.norm(dim=-1) * gate[s:s + token_chunk].T).sum(1).double()
                o = outs.reshape(n, -1).double()
                st["gram"] += o @ o.T
                del outs, o
            st["tokens"] += flat.shape[0]
        return hook

    for layer_idx, block in iter_moe_blocks(model, backbone):
        handles.append(block.register_forward_hook(make_hook(layer_idx)))
    model.eval()
    try:
        for batch in tqdm(dataloader, desc="[RESCODE] calibration statistics (saliency, freq, dist_out)"):
            batch = {k: v.cuda() for k, v in batch.items() if k != "labels"}
            model(**batch)
    finally:
        for h in handles:
            h.remove()

    stats = {}
    for layer_idx, st in acc.items():
        t = float(st["tokens"])
        g = st["gram"]
        sq = torch.diagonal(g)
        dist = (sq[:, None] + sq[None, :] - 2.0 * g).clamp_min(0.0) / t
        dist.fill_diagonal_(0.0)
        stats[layer_idx] = {
            "saliency": (st["sal"] / t).float().cpu(),
            "freq": (st["freq"] / t).float().cpu(),
            "dist": dist.float().cpu(),
            "energy": (sq / t).float().cpu(),
            "tokens": int(st["tokens"]),
        }
    torch.cuda.empty_cache()
    return stats


def load_or_collect_calib_stats(model, dataloader, backbone: str, cache_path: Optional[str] = None):
    if cache_path and os.path.isfile(cache_path):
        print(f"[RESCODE] loading calibration statistics from {cache_path}")
        return torch.load(cache_path)
    stats = collect_calib_stats(model, dataloader, backbone)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp = f"{cache_path}.tmp{os.getpid()}"
        torch.save(stats, tmp)
        os.replace(tmp, cache_path)
        print(f"[RESCODE] saved calibration statistics to {cache_path}")
    return stats
