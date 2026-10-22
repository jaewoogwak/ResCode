"""Idea 1 cluster formation from calibration statistics (paper Sec. 4.2, Alg. 1).

Step 1  the top-ceil(rho * N) experts of each layer by saliency a_i are kept as
        singleton clusters.
Step 2  the remaining experts are grouped by agglomerative clustering on the
        output distance dist_out(i, j), repeatedly merging the two closest
        clusters until the target number of clusters is reached.

Budget  Problem 1 requires |Theta_hat_E| <= tau * |Theta_E|, and the residual
        bases A_G, B_G and codes z_i of every non-singleton cluster count
        toward Theta_hat_E. When the nominal per-layer cluster count leaves the
        model over budget once those residual parameters are counted, extra
        merges are taken one at a time, spread over layers (one extra merge per
        layer per round), each in the layer whose next merge has the smallest
        output distance relative to the layer's median expert distance. This
        gives layer-specific cluster counts C_l (Property 2) and keeps the
        expert memory within the reported compression ratio.
"""

import math
from typing import Dict, List, Sequence, Tuple

import torch


def agglomerative_merges(dist: torch.Tensor, linkage: str = "average") -> List[Tuple[int, int, float]]:
    """Full merge sequence of agglomerative clustering on a dissimilarity matrix.

    Returns (keep, absorbed, height) triples, where `keep` and `absorbed` are
    cluster representatives (row indices of `dist`) and `height` is the linkage
    dissimilarity at which they were merged.
    """
    if linkage not in ("average", "complete", "single", "ward"):
        raise ValueError(f"[RESCODE] unsupported linkage={linkage}")
    d = dist.double().clone()
    n = d.shape[0]
    d.fill_diagonal_(float("inf"))
    size = [1] * n
    alive = list(range(n))
    merges = []
    for _ in range(n - 1):
        idx = torch.tensor(alive)
        sub = d[idx][:, idx]
        flat = int(torch.argmin(sub).item())
        a, b = alive[flat // len(alive)], alive[flat % len(alive)]
        a, b = min(a, b), max(a, b)
        h = float(d[a, b].item())
        for k in alive:
            if k in (a, b):
                continue
            if linkage == "average":
                v = (size[a] * d[a, k] + size[b] * d[b, k]) / (size[a] + size[b])
            elif linkage == "complete":
                v = torch.maximum(d[a, k], d[b, k])
            elif linkage == "single":
                v = torch.minimum(d[a, k], d[b, k])
            else:  # ward (Lance-Williams update on squared distances)
                nk = size[k]
                v = ((nk + size[a]) * d[a, k] + (nk + size[b]) * d[b, k] - nk * h) / (nk + size[a] + size[b])
            d[a, k] = v
            d[k, a] = v
        size[a] += size[b]
        alive.remove(b)
        merges.append((a, b, h))
    return merges


def _groups_after(n: int, merges: Sequence[Tuple[int, int, float]], num_merges: int) -> List[List[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in merges[:num_merges]:
        parent[find(b)] = find(a)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


class _LayerPlan:
    def __init__(self, layer_idx, pinned, rest, merges, num_merges, scale):
        self.layer_idx = layer_idx
        self.pinned = pinned
        self.rest = rest
        self.merges = merges
        self.num_merges = num_merges
        self.scale = scale
        self.extra = 0

    def groups(self) -> List[List[int]]:
        local = _groups_after(len(self.rest), self.merges, self.num_merges)
        return [[p] for p in self.pinned] + [[self.rest[i] for i in g] for g in local]

    def can_merge(self) -> bool:
        return self.num_merges < len(self.merges)

    def next_cost(self) -> float:
        return self.merges[self.num_merges][2] / max(self.scale, 1e-12)


def expert_memory(groups: List[List[int]], expert_params: int, residual_params_fn) -> int:
    """Parameters of one compressed layer: one expert per cluster + residuals of non-singletons."""
    return sum(expert_params + (residual_params_fn(len(g)) if len(g) > 1 else 0) for g in groups)


def form_clusters(
    stats: Dict[int, Dict[str, torch.Tensor]],
    num_groups: int,
    singleton_ratio: float,
    expert_params: int,
    residual_params_fn,
    linkage: str = "average",
    budget_aware: bool = True,
) -> Tuple[Dict[int, List[List[int]]], Dict[str, float]]:
    """Cluster every layer from `stats` (see calib_stats.collect_calib_stats).

    Args:
      num_groups:        nominal clusters per layer C (e.g. 30 of 60 at 50%).
      singleton_ratio:   rho in [0, 1); ceil(rho * N) experts per layer are pinned.
      expert_params:     parameters of one original expert.
      residual_params_fn: |G| -> residual parameters of a non-singleton cluster.
      budget_aware:      enforce sum_l memory_l <= L * C * expert_params.

    Returns ({layer_idx: list of member lists}, summary dict).
    """
    plans = []
    for layer_idx in sorted(stats):
        st = stats[layer_idx]
        sal = st["saliency"].float()
        n = sal.numel()
        k_pin = min(n, int(math.ceil(singleton_ratio * n - 1e-9))) if singleton_ratio > 0 else 0
        order = sorted(range(n), key=lambda i: (-float(sal[i]), i))
        pinned = sorted(order[:k_pin])
        rest = [i for i in range(n) if i not in set(pinned)]
        target_rest = num_groups - len(pinned)
        if target_rest < 1:
            raise ValueError(f"[RESCODE] layer {layer_idx}: {len(pinned)} singletons >= {num_groups} clusters")
        sub = st["dist"][rest][:, rest]
        merges = agglomerative_merges(sub, linkage=linkage) if len(rest) > 1 else []
        num_merges = max(0, len(rest) - target_rest)
        # typical expert dissimilarity of the layer, so merge costs compare across layers
        off_diag = sub[~torch.eye(len(rest), dtype=torch.bool)] if len(rest) > 1 else sub.flatten()
        scale = float(off_diag.median()) if off_diag.numel() else 1.0
        plans.append(_LayerPlan(layer_idx, pinned, rest, merges, num_merges, scale))

    def layer_mem(plan):
        return expert_memory(plan.groups(), expert_params, residual_params_fn)

    budget = len(plans) * num_groups * expert_params
    mem = {p.layer_idx: layer_mem(p) for p in plans}
    extra_merges = 0
    if budget_aware:
        while sum(mem.values()) > budget:
            candidates = [p for p in plans if p.can_merge()]
            if not candidates:
                raise RuntimeError("[RESCODE] cannot meet the expert-memory budget")
            # spread extra merges over layers (fewest extra merges first), cheapest merge within a round
            best = min(candidates, key=lambda p: (p.extra, p.next_cost(), p.layer_idx))
            best.extra += 1
            best.num_merges += 1
            mem[best.layer_idx] = layer_mem(best)
            extra_merges += 1

    groups = {p.layer_idx: p.groups() for p in plans}
    n_experts = sum(len(p.pinned) + len(p.rest) for p in plans)
    original = n_experts * expert_params
    total = sum(mem.values())
    residual = total - sum(len(g) for g in groups.values()) * expert_params
    summary = {
        "budget_params": budget,
        "compressed_params": total,
        "residual_params": residual,
        "original_params": original,
        "kept_ratio": total / original,
        "compression_ratio": 1.0 - total / original,
        "extra_merges": extra_merges,
    }
    for p in plans:
        g = groups[p.layer_idx]
        sizes = sorted((len(x) for x in g), reverse=True)
        print(f"[RESCODE][cluster] layer={p.layer_idx} C={len(g)} pinned={len(p.pinned)} "
              f"meta={sum(1 for s in sizes if s > 1)} sizes={[s for s in sizes if s > 1]}")
    print(f"[RESCODE][budget] expert params {total:,} / original {original:,} "
          f"-> compression {100 * summary['compression_ratio']:.3f}% "
          f"(budget {100 * (1 - budget / original):.3f}%, residual {residual:,}, extra merges {extra_merges})")
    return groups, summary


def residual_params_fn_for(hidden_size: int, intermediate_size: int, rank: int, num_targets: int):
    """Residual parameters A_G, B_G (r x (d_in + d_out)) and codes z_i (|G| x r) per restored branch."""
    return lambda g: num_targets * (rank * (hidden_size + intermediate_size) + g * rank)


def groups_to_labels(groups: List[List[int]], num_experts: int) -> torch.Tensor:
    labels = torch.empty(num_experts, dtype=torch.long)
    for cid, members in enumerate(groups):
        for e in members:
            labels[e] = cid
    return labels
