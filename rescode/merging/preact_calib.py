"""Calibration pass for preact std normalization.

Computes per-channel std of the merged gate/up preactivations on the merged
model, and writes them into each `PreActModule.gate_preact_std` /
`up_preact_std` buffer. This calibration is OPTIONAL: when
`preact_star.normalize_by_preact_std=False` (or no preact_star is configured)
this whole pass is a no-op.
"""

from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def collect_preact_stats(
    model: nn.Module,
    dataloader,
    desc: str = "[RESCODE] preact std calibration",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Run the (already converted) RESCODE model forward and accumulate
    g0 = W_g_bar x and u0 = W_u_bar x across all calibration tokens, computing
    per-channel std. Writes the result into each meta expert's PreActModule
    buffers and also returns it for inspection."""

    stats: Dict[int, Dict[str, torch.Tensor]] = {}
    handles = []

    def _make_hook(layer_idx, meta):
        # `meta.gate_proj` / `meta.up_proj` are Linear modules whose output we
        # want the std of. PyTorch hooks fire on the modules' forward outputs.
        gate_acc = {"sumsq": None, "sum": None, "n": 0}
        up_acc = {"sumsq": None, "sum": None, "n": 0}

        def _hook_gate(_m, _inp, out):
            o = out.detach().to(torch.float32).reshape(-1, out.shape[-1])
            sq = o.pow(2).sum(dim=0).cpu()
            sm = o.sum(dim=0).cpu()
            n = o.shape[0]
            if gate_acc["sumsq"] is None:
                gate_acc["sumsq"] = sq
                gate_acc["sum"] = sm
            else:
                gate_acc["sumsq"] += sq
                gate_acc["sum"] += sm
            gate_acc["n"] += n

        def _hook_up(_m, _inp, out):
            o = out.detach().to(torch.float32).reshape(-1, out.shape[-1])
            sq = o.pow(2).sum(dim=0).cpu()
            sm = o.sum(dim=0).cpu()
            n = o.shape[0]
            if up_acc["sumsq"] is None:
                up_acc["sumsq"] = sq
                up_acc["sum"] = sm
            else:
                up_acc["sumsq"] += sq
                up_acc["sum"] += sm
            up_acc["n"] += n

        return _hook_gate, _hook_up, gate_acc, up_acc

    accs: Dict[int, Dict[int, dict]] = {}
    # Iterate RESCODE blocks and attach hooks only to non-singleton meta
    # experts that have a preact module.
    for layer_idx, layer in enumerate(model.model.layers):
        block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if block is None or not hasattr(block, "_rescode"):
            continue
        meta_experts = block._rescode["meta_experts"]
        for cid, meta in meta_experts.items():
            if meta.preact is None:
                continue
            cfg = meta.preact.config
            if not cfg.normalize_by_preact_std:
                continue
            hg, hu, ga, ua = _make_hook(layer_idx, meta)
            handles.append(meta.gate_proj.register_forward_hook(hg))
            handles.append(meta.up_proj.register_forward_hook(hu))
            accs.setdefault(layer_idx, {})[cid] = {"gate": ga, "up": ua, "meta": meta}

    if not accs:
        return stats

    try:
        model.eval()
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
            _ = model(**batch)
    finally:
        for h in handles:
            h.remove()

    for layer_idx, by_cid in accs.items():
        stats[layer_idx] = {}
        for cid, slot in by_cid.items():
            ga = slot["gate"]
            ua = slot["up"]
            meta = slot["meta"]
            cfg = meta.preact.config
            if ga["n"] == 0 or ua["n"] == 0:
                continue
            mean_g = ga["sum"] / ga["n"]
            var_g = ga["sumsq"] / ga["n"] - mean_g.pow(2)
            std_g = var_g.clamp_min(0.0).sqrt() + cfg.preact_std_eps

            mean_u = ua["sum"] / ua["n"]
            var_u = ua["sumsq"] / ua["n"] - mean_u.pow(2)
            std_u = var_u.clamp_min(0.0).sqrt() + cfg.preact_std_eps

            with torch.no_grad():
                meta.preact.gate_preact_std.copy_(
                    std_g.to(meta.preact.gate_preact_std.device,
                             meta.preact.gate_preact_std.dtype)
                )
                meta.preact.up_preact_std.copy_(
                    std_u.to(meta.preact.up_preact_std.device,
                             meta.preact.up_preact_std.dtype)
                )
            stats[layer_idx][cid] = {
                "gate_preact_std_mean": float(std_g.mean().item()),
                "up_preact_std_mean": float(std_u.mean().item()),
                "gate_preact_mean_mean": float(mean_g.mean().item()),
                "up_preact_mean_mean": float(mean_u.mean().item()),
                "num_tokens": int(ga["n"]),
            }
            print(
                f"[RESCODE][preact_std] layer={layer_idx} cluster={cid} "
                f"N={int(ga['n'])} gate_std_mean={float(std_g.mean()):.4f} "
                f"up_std_mean={float(std_u.mean()):.4f}"
            )
    return stats
