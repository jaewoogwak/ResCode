"""Collect original routed inputs for direct-QJL base fitting.

The existing ResCode statistics/grouping path is used by the launcher. This
module only captures the additional projection inputs and router-probability
squares that the QJL variance objective requires.
"""

from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rescode.merging.calib_stats import iter_moe_blocks, route, routing_config


@torch.no_grad()
def collect_projection_samples(model, dataloader, family: str, layers,
                               max_train: int = 256, max_heldout: int = 64):
    layers = set(layers)
    top_k, norm = routing_config(model, family)
    # Mutable accumulators: layer -> expert -> split -> (hidden, middle, q)
    store = defaultdict(lambda: defaultdict(lambda: {
        "train": {"hidden": [], "middle": [], "q": []},
        "heldout": {"hidden": [], "middle": [], "q": []},
        "seen": 0, "train_n": 0, "heldout_n": 0,
    }))
    probes = {}
    handles = []

    # More efficient hook implementation: take only the first still-needed
    # routed token indices for each split and evaluate original SwiGLU once.
    def efficient_hook(layer_idx):
        def hook(block, inputs, _output):
            flat = inputs[0].reshape(-1, inputs[0].shape[-1])
            if layer_idx not in probes:
                probes[layer_idx] = (inputs[0][:, :min(16, inputs[0].shape[1])].detach().cpu(),
                                     None)
            selected, weights = route(block, flat, family, top_k, norm)
            for eid, expert in enumerate(block.experts):
                token, slot = torch.where(selected == eid)
                if not token.numel():
                    continue
                st = store[layer_idx][eid]
                picks = {"train": [], "heldout": []}
                for j in range(token.numel()):
                    split = "heldout" if st["seen"] % 5 == 0 else "train"
                    st["seen"] += 1
                    cap = max_heldout if split == "heldout" else max_train
                    if st[split + "_n"] < cap:
                        picks[split].append(j)
                        st[split + "_n"] += 1
                for split, positions in picks.items():
                    if not positions:
                        continue
                    p = torch.tensor(positions, device=token.device)
                    x = flat.index_select(0, token.index_select(0, p))
                    q = weights[token.index_select(0, p), slot.index_select(0, p)].float().square()
                    if family == "mixtral":
                        middle = F.silu(expert.w1(x)) * expert.w3(x)
                    else:
                        middle = F.silu(expert.gate_proj(x)) * expert.up_proj(x)
                    st[split]["hidden"].append(x.detach().cpu())
                    st[split]["middle"].append(middle.detach().cpu())
                    st[split]["q"].append(q.detach().cpu())
        return hook

    for layer_idx, block in iter_moe_blocks(model, family):
        if layer_idx in layers:
            handles.append(block.register_forward_hook(efficient_hook(layer_idx)))
    model.eval()
    try:
        for batch in tqdm(dataloader, desc="[QJL] routed projection inputs"):
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    output = {"train": {}, "heldout": {}}
    for layer_idx, experts in store.items():
        for split in output:
            output[split][layer_idx] = {"gate": {}, "up": {}, "down": {}}
        for eid, st in experts.items():
            for split in output:
                if not st[split]["hidden"]:
                    continue
                x = torch.cat(st[split]["hidden"])
                middle = torch.cat(st[split]["middle"])
                q = torch.cat(st[split]["q"])
                output[split][layer_idx]["gate"][eid] = (x, q)
                output[split][layer_idx]["up"][eid] = (x, q)
                output[split][layer_idx]["down"][eid] = (middle, q)
    return output["train"], output["heldout"], probes
