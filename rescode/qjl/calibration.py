"""Collect routed inputs and router-probability squares for QJL base fitting."""

from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rescode.merging.calib_stats import iter_moe_blocks, route, routing_config


class ProjectionSampleCollector:
    """Retain the first routed samples for each expert, in dataloader order."""

    def __init__(self, family: str, layers=None, max_train: int = 256,
                 max_heldout: int = 64):
        self.family = family
        self.layers = None if layers is None else set(layers)
        self.max_train = max_train
        self.max_heldout = max_heldout
        self.store = defaultdict(lambda: defaultdict(lambda: {
            "train": {"hidden": [], "middle": [], "q": []},
            "heldout": {"hidden": [], "middle": [], "q": []},
            "seen": 0, "train_n": 0, "heldout_n": 0,
        }))
        self.probes = {}
        self.complete_layers = set()
        self.captured = False

    def complete_for(self, layers) -> bool:
        return all(layer in self.complete_layers for layer in layers)

    def capture(self, layer_idx, block, hidden, selected, weights):
        if self.layers is not None and layer_idx not in self.layers:
            return
        self.captured = True
        if layer_idx in self.complete_layers:
            return
        flat = hidden.reshape(-1, hidden.shape[-1])
        if layer_idx not in self.probes:
            self.probes[layer_idx] = (
                hidden[:, :min(16, hidden.shape[1])].detach().cpu(), None
            )
        layer_store = self.store[layer_idx]
        for eid, expert in enumerate(block.experts):
            st = layer_store[eid]
            if st["train_n"] >= self.max_train and st["heldout_n"] >= self.max_heldout:
                continue
            token, slot = torch.where(selected == eid)
            count = token.numel()
            if not count:
                continue
            positions = torch.arange(count, device=token.device)
            heldout_mask = (positions + st["seen"]) % 5 == 0
            st["seen"] += count
            for split, mask, cap in (("train", ~heldout_mask, self.max_train),
                                     ("heldout", heldout_mask, self.max_heldout)):
                remaining = cap - st[split + "_n"]
                if remaining <= 0:
                    continue
                chosen = torch.where(mask)[0][:remaining]
                if not chosen.numel():
                    continue
                st[split + "_n"] += chosen.numel()
                selected_token = token.index_select(0, chosen)
                x = flat.index_select(0, selected_token)
                q = weights[selected_token, slot.index_select(0, chosen)].float().square()
                if self.family == "mixtral":
                    middle = F.silu(expert.w1(x)) * expert.w3(x)
                else:
                    middle = F.silu(expert.gate_proj(x)) * expert.up_proj(x)
                st[split]["hidden"].append(x.detach().cpu())
                st[split]["middle"].append(middle.detach().cpu())
                st[split]["q"].append(q.detach().cpu())
        if all(st["train_n"] >= self.max_train and st["heldout_n"] >= self.max_heldout
               for st in layer_store.values()):
            self.complete_layers.add(layer_idx)

    def result(self):
        output = {"train": {}, "heldout": {}}
        for layer_idx, experts in self.store.items():
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
        return output["train"], output["heldout"], self.probes


@torch.no_grad()
def collect_projection_samples(model, dataloader, family: str, layers,
                               max_train: int = 256, max_heldout: int = 64):
    layers = set(layers)
    collector = ProjectionSampleCollector(family, layers, max_train, max_heldout)
    top_k, norm = routing_config(model, family)
    handles = []

    def make_hook(layer_idx):
        def hook(block, inputs, _output):
            if layer_idx in collector.complete_layers:
                return
            hidden = inputs[0]
            selected, weights = route(block, hidden.reshape(-1, hidden.shape[-1]),
                                      family, top_k, norm)
            collector.capture(layer_idx, block, hidden, selected, weights)
        return hook

    for layer_idx, block in iter_moe_blocks(model, family):
        if layer_idx in layers:
            handles.append(block.register_forward_hook(make_hook(layer_idx)))
    model.eval()
    try:
        for batch in tqdm(dataloader, desc="[QJL] routed projection inputs"):
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            model(**batch)
            if collector.complete_for(layers):
                print("[QJL] projection sample caps reached; ending pass early", flush=True)
                break
    finally:
        for handle in handles:
            handle.remove()
    return collector.result()
