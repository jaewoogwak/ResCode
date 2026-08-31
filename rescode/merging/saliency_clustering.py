"""
Saliency-weighted hierarchical clustering for MoE expert merging.

Objective:
    cost(C) = Σ_{i ∈ C} s_i · ‖ f_i − base(C) ‖₂
    total   = Σ_C cost(C)

    where f_i is a per-expert feature (e.g. routed-output mean) and s_i is a
    pruning-style saliency (REAP: mean over routed tokens of g̃·‖f(x)‖).

Low-saliency experts are "cheap to merge" (their reconstruction error is
down-weighted), while high-saliency experts tend to stay as singletons or
form tight clusters — directly aligned with end-to-end layer-output error.

The base(C) used in the cost matches the base that the merger will later use
(e.g., freq-weighted average), so the clustering objective is an upper bound
on the merge-induced layer error.
"""

from __future__ import annotations

from typing import Optional

import torch

from rescode.utils.constants import FP32_EPS


@torch.no_grad()
def _weighted_centroid(
    features: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """weights-weighted mean along dim 0. features: (N, D), weights: (N,)."""
    w = weights.clamp_min(FP32_EPS)
    w = w / w.sum()
    return (features * w.unsqueeze(-1)).sum(dim=0)


@torch.no_grad()
def _cluster_cost(
    members: list,
    features: torch.Tensor,
    saliencies: torch.Tensor,
    base: torch.Tensor,
) -> float:
    sub = features[members] - base.unsqueeze(0)
    dist = sub.norm(dim=-1)
    return (saliencies[members] * dist).sum().item()


@torch.no_grad()
def saliency_weighted_hierarchical_clustering(
    features: torch.Tensor,
    saliencies: torch.Tensor,
    num_groups: int,
    freq: Optional[torch.Tensor] = None,
    base_weighting: str = "freq",
    verbose: bool = False,
):
    """Greedy hierarchical clustering minimizing total saliency-weighted reconstruction cost.

    Args:
        features: (N, D) — per-expert feature vectors.
        saliencies: (N,) — per-expert saliency (≥ 0).
        num_groups: target number of clusters K.
        freq: (N,) — per-expert usage frequency (for base_weighting='freq').
        base_weighting: how base(C) is computed.
            'freq'     → Σ freq_i · f_i / Σ freq_i   (matches freq merge)
            'uniform'  → mean of f_i
            'saliency' → Σ s_i · f_i / Σ s_i

    Returns:
        labels: (N,) LongTensor — cluster id per expert, contiguous starting at 0.
        dom_experts: list[int] — representative (medoid) expert per cluster.
    """
    device = features.device
    N = features.shape[0]
    features = features.float()
    saliencies = saliencies.float().to(device)
    if freq is not None:
        freq = freq.float().to(device)
    else:
        freq = torch.ones(N, device=device)

    def _base(members):
        if base_weighting == "freq":
            w = freq[members]
        elif base_weighting == "saliency":
            w = saliencies[members]
        else:
            w = torch.ones(len(members), device=device)
        return _weighted_centroid(features[members], w)

    # Each expert starts as its own singleton cluster (cost = 0).
    cluster_members: dict[int, list[int]] = {i: [i] for i in range(N)}
    cluster_bases: dict[int, torch.Tensor] = {i: features[i].clone() for i in range(N)}
    cluster_costs: dict[int, float] = {i: 0.0 for i in range(N)}

    step = 0
    while len(cluster_members) > num_groups:
        keys = list(cluster_members.keys())
        best_delta = float("inf")
        best_pick = None

        # O(K^2) pair search; K is tiny (≤60) so fine.
        for ii in range(len(keys)):
            for jj in range(ii + 1, len(keys)):
                ca, cb = keys[ii], keys[jj]
                members = cluster_members[ca] + cluster_members[cb]
                new_base = _base(members)
                new_cost = _cluster_cost(members, features, saliencies, new_base)
                delta = new_cost - cluster_costs[ca] - cluster_costs[cb]
                if delta < best_delta:
                    best_delta = delta
                    best_pick = (ca, cb, members, new_base, new_cost)

        ca, cb, members, new_base, new_cost = best_pick
        cluster_members[ca] = members
        cluster_bases[ca] = new_base
        cluster_costs[ca] = new_cost
        del cluster_members[cb], cluster_bases[cb], cluster_costs[cb]

        if verbose:
            total = sum(cluster_costs.values())
            print(
                f"  [sal-hier] step {step}: merge ({ca},{cb}) |C|={len(members)} "
                f"Δ={best_delta:.4e} total={total:.4e}"
            )
        step += 1

    # Build label vector + medoid of each cluster
    labels = torch.zeros(N, dtype=torch.long)
    dom_experts: list[int] = []
    for new_id, (cid, members) in enumerate(cluster_members.items()):
        for m in members:
            labels[m] = new_id
        base = cluster_bases[cid]
        dists = (features[members] - base.unsqueeze(0)).norm(dim=-1)
        dom_experts.append(members[int(dists.argmin().item())])

    total_cost = sum(cluster_costs.values())
    print(
        f"[sal-hier] final: K={len(cluster_members)}, total cost={total_cost:.4e}"
    )
    return labels, dom_experts


@torch.no_grad()
def saliency_weighted_kmedoids(
    features: torch.Tensor,
    saliencies: torch.Tensor,
    num_groups: int,
    max_iter: int = 100,
    seed: int = 42,
):
    """Saliency-weighted K-medoids (PAM).

    Objective:
        min  Σ_C  Σ_{i ∈ C} s_i · ‖f_i − f_{m(C)}‖₂    s.t. m(C) ∈ C ⊂ {1..N}

    - Assignment step:   expert i → cluster of nearest medoid (unweighted distance).
    - Medoid update:     for cluster C, pick m* = argmin_{j ∈ C} Σ_{i ∈ C} s_i · d(f_i, f_j).
                         High-saliency members thus pull the medoid toward themselves.
    - Init:              saliency-biased k-means++ — next seed ∝ s_i · min_m d(f_i, f_m)².

    Args:
        features: (N, D).
        saliencies: (N,) ≥ 0.
        num_groups: K.
        max_iter: cap on PAM swap iterations.
        seed: RNG seed for init.

    Returns:
        labels: (N,) LongTensor, contiguous 0..K-1.
        dom_experts: list[int] — the K medoid indices.
    """
    device = features.device
    N = features.shape[0]
    features = features.float()
    saliencies = saliencies.float().to(device).clamp_min(FP32_EPS)

    dist = torch.cdist(features, features, p=2)  # (N, N)

    g = torch.Generator(device="cpu").manual_seed(int(seed))

    # Saliency-biased k-means++ init
    first = int(torch.multinomial(saliencies.cpu() / saliencies.cpu().sum(), 1, generator=g).item())
    medoids: list[int] = [first]
    for _ in range(1, num_groups):
        d_near = dist[:, medoids].min(dim=1).values  # (N,)
        probs = saliencies * d_near.pow(2)
        probs[medoids] = 0.0
        if probs.sum().item() <= 0:
            # degenerate: pick any remaining expert
            remaining = [i for i in range(N) if i not in medoids]
            medoids.append(remaining[0])
            continue
        nxt = int(torch.multinomial(probs.cpu() / probs.cpu().sum(), 1, generator=g).item())
        medoids.append(nxt)

    medoids_t = torch.tensor(medoids, dtype=torch.long, device=device)

    for it in range(max_iter):
        # Assignment
        d_to_med = dist[:, medoids_t]  # (N, K)
        assignments = d_to_med.argmin(dim=1)  # (N,)

        # Medoid update
        new_medoids = medoids_t.clone()
        for k in range(num_groups):
            members = torch.where(assignments == k)[0]
            if members.numel() == 0:
                continue
            # cost(j) = Σ_{i ∈ C} s_i · d(f_i, f_j)  for j ∈ C
            sub_dist = dist[members][:, members]               # (|C|, |C|)
            weighted = (saliencies[members].unsqueeze(1) * sub_dist).sum(dim=0)  # (|C|,)
            best_local = int(weighted.argmin().item())
            new_medoids[k] = members[best_local]

        if torch.equal(new_medoids, medoids_t):
            break
        medoids_t = new_medoids

    # Final assignment (contiguous labels in [0, K))
    d_to_med = dist[:, medoids_t]
    assignments = d_to_med.argmin(dim=1).to("cpu")

    # Compute total cost for logging
    total = 0.0
    for k in range(num_groups):
        members = torch.where(assignments == k)[0]
        if members.numel() == 0:
            continue
        m = medoids_t[k].item()
        total += (saliencies[members].cpu() * dist[members, m].cpu()).sum().item()
    print(
        f"[sal-kmed] final: K={num_groups}, iters={it + 1}, total cost={total:.4e}"
    )

    dom_experts = [int(medoids_t[k].item()) for k in range(num_groups)]
    return assignments.long(), dom_experts


@torch.no_grad()
def layer_recon_hierarchical_clustering(
    selected: torch.Tensor,
    gates: torch.Tensor,
    outs: torch.Tensor,
    num_experts: int,
    num_groups: int,
    verbose: bool = False,
):
    """Greedy hierarchical clustering minimizing layer-output reconstruction error.

    Objective:
        cost(C) = E_t || Σ_{i ∈ C} g_i(t) · (f_i(t) − μ_C) ||²
                = (1/T) Σ_t || M_C(t) − G_C(t)·μ_C ||²
        μ_C     = Σ_t M_C(t) / Σ_t G_C(t)           (gate-weighted mean activation)
        M_C(t)  = Σ_{i ∈ C ∩ topk(t)} g_i(t) · f_i(t)
        G_C(t)  = Σ_{i ∈ C ∩ topk(t)} g_i(t)

    The objective captures 3 signals simultaneously:
      - Diagonal (i=j):  saliency-style ‖g_i f_i‖² — smaller experts get merged first
      - Off-diag (i≠j): g_i(t)g_j(t)·⟨f_i, f_j⟩ — gate co-activation × output correlation
                        → experts firing at different timesteps are cheap to merge

    Args:
        selected: (T, K) long — expert index per top-k slot
        gates:    (T, K) float — renormalized top-k gate values
        outs:     (T, K, D) float — expert output at each (token, slot)
        num_experts: E
        num_groups: target K clusters

    Returns:
        labels: (E,) long contiguous 0..num_groups-1
        dom_experts: list[int] of length num_groups
    """
    T, K_topk, D = outs.shape
    E = num_experts
    device = outs.device

    def _build_mask(members_list):
        """(T, K_topk) boolean — True where expert slot is in members."""
        members_t = torch.tensor(members_list, device=device, dtype=torch.long)
        return torch.isin(selected, members_t)

    def _cluster_cost(members_list):
        mask = _build_mask(members_list)
        w = gates * mask.float()               # (T, K_topk)
        M = (w.unsqueeze(-1) * outs).sum(dim=1)  # (T, D)
        G = w.sum(dim=1)                        # (T,)
        G_total = G.sum()
        if G_total.item() <= 0.0:
            return 0.0
        mu = M.sum(dim=0) / G_total              # (D,)
        err = M - G.unsqueeze(-1) * mu.unsqueeze(0)  # (T, D)
        return float(err.pow(2).sum(dim=-1).mean().item())

    # Singleton initialization
    cluster_members: dict[int, list[int]] = {i: [i] for i in range(E)}
    cluster_costs: dict[int, float] = {i: _cluster_cost([i]) for i in range(E)}

    step = 0
    while len(cluster_members) > num_groups:
        keys = list(cluster_members.keys())
        best_delta = float("inf")
        best_pick = None
        for ii in range(len(keys)):
            for jj in range(ii + 1, len(keys)):
                ca, cb = keys[ii], keys[jj]
                members = cluster_members[ca] + cluster_members[cb]
                new_cost = _cluster_cost(members)
                delta = new_cost - cluster_costs[ca] - cluster_costs[cb]
                if delta < best_delta:
                    best_delta = delta
                    best_pick = (ca, cb, members, new_cost)

        ca, cb, members, new_cost = best_pick
        cluster_members[ca] = members
        cluster_costs[ca] = new_cost
        del cluster_members[cb], cluster_costs[cb]

        if verbose:
            total = sum(cluster_costs.values())
            print(
                f"  [layer-recon] step {step}: merge ({ca},{cb}) |C|={len(members)} "
                f"Δ={best_delta:.4e} total={total:.4e}"
            )
        step += 1

    # Labels + medoids
    labels = torch.zeros(E, dtype=torch.long)
    dom_experts: list[int] = []
    for new_id, (cid, members) in enumerate(cluster_members.items()):
        for m in members:
            labels[m] = new_id
        # Medoid: member minimizing single-expert cost (most "typical" of its token distribution)
        best_dom = min(members, key=lambda i: cluster_costs.get(i, float("inf"))
                        if i == cid else _cluster_cost([i]))
        dom_experts.append(best_dom)

    total_final = sum(cluster_costs.values())
    print(f"[layer-recon] final: K={len(cluster_members)}, total cost={total_final:.4e}")
    return labels, dom_experts
