"""Sub-cluster split methods for dual residual.

Each method takes a cluster's member expert ids (and possibly auxiliary data)
and returns a binary assignment list of length |member_ids|, with values
in {0, 1} indicating which sub-cluster each member belongs to.

Methods:
  A. balanced  — most balanced cut from the hierarchical merge tree.
  B. kmeans    — 2-way k-means on member expert weight vectors.
  C. coactivation — 2-way spectral clustering on routing co-occurrence.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------- helpers
def _build_binary_trees(merge_seq: Sequence[Tuple[int, int]]):
    """Replay a hierarchical merge sequence into binary trees.

    Each leaf carries `{'leaf': sample_id, 'members': [sample_id]}`.
    Each internal node carries `{'left': subtree, 'right': subtree, 'members': [...]}`.
    Returns a dict {root_rep -> tree_node} after all merges.
    """
    rep: Dict[int, int] = {}
    tree: Dict[int, dict] = {}
    for (i, j) in merge_seq:
        for s in (i, j):
            if s not in rep:
                rep[s] = s
                tree[s] = {"leaf": s, "members": [s]}
        ri = rep[i]
        rj = rep[j]
        if ri == rj:
            continue
        merged = {
            "left": tree[ri],
            "right": tree[rj],
            "members": tree[ri]["members"] + tree[rj]["members"],
        }
        for m in tree[rj]["members"]:
            rep[m] = ri
        tree[ri] = merged
        del tree[rj]
    return rep, tree


# ---------------------------------------------------------- A: balanced cut
def compute_split_balanced(
    member_ids: Sequence[int],
    merge_seq: Sequence[Tuple[int, int]],
) -> List[int]:
    """Most balanced binary partition from the merge tree.

    For every internal node of the cluster's binary tree, compute |left| vs
    |right| restricted to cluster members; pick the cut with minimum |diff|.
    Falls back to alternating assignment if the merge sequence does not
    contain the requested members.
    """
    n = len(member_ids)
    if n <= 1:
        return [0] * n
    if n == 2:
        return [0, 1]

    rep, tree = _build_binary_trees(merge_seq)
    member_set = set(int(m) for m in member_ids)
    # Identify root of cluster: rep of any member.
    if member_ids[0] not in rep:
        # fallback — list of members not in merge tree at all (shouldn't happen)
        half = n // 2
        return [0] * half + [1] * (n - half)
    root_rep = rep[member_ids[0]]
    # Sanity — all members should share the same rep
    if not all(rep.get(int(m)) == root_rep for m in member_ids):
        half = n // 2
        return [0] * half + [1] * (n - half)
    root = tree[root_rep]

    best = {"diff": float("inf"), "left": None, "right": None}

    def visit(node: dict):
        if "leaf" in node:
            return
        left_in = [m for m in node["left"]["members"] if m in member_set]
        right_in = [m for m in node["right"]["members"] if m in member_set]
        if left_in and right_in:
            diff = abs(len(left_in) - len(right_in))
            if diff < best["diff"]:
                best["diff"] = diff
                best["left"] = set(left_in)
                best["right"] = set(right_in)
        visit(node["left"])
        visit(node["right"])

    visit(root)

    if best["left"] is None:
        half = n // 2
        return [0] * half + [1] * (n - half)
    return [0 if int(m) in best["left"] else 1 for m in member_ids]


# ---------------------------------------------------------- B: 2-way k-means
@torch.no_grad()
def _kmeans2(X: torch.Tensor, max_iter: int = 50, seed: int = 0) -> List[int]:
    """Naive 2-way k-means on rows of X. Returns list of {0,1} labels."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = X.shape[0]
    # Init: pick the two farthest rows as initial centers
    pdist = torch.cdist(X, X, p=2)
    flat = pdist.view(-1)
    far_idx = int(torch.argmax(flat).item())
    a, b = far_idx // n, far_idx % n
    centers = X[[a, b]].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(max_iter):
        d = torch.cdist(X, centers, p=2)            # [n, 2]
        new_labels = torch.argmin(d, dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(2):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(dim=0)
    # Guard against degenerate single-cluster output (force at least 1 member each)
    if (labels == 0).all() or (labels == 1).all():
        labels[:] = 0
        labels[n // 2 :] = 1
    return labels.tolist()


def compute_split_kmeans(
    member_ids: Sequence[int],
    member_weights: torch.Tensor,
) -> List[int]:
    """2-way k-means on flattened member weight vectors.

    member_weights: tensor of shape [K, D] where K = |member_ids|, D = flattened
    feature dim (typically gate_W.flatten() or [gate_W; up_W].flatten()).
    """
    n = len(member_ids)
    if n <= 1:
        return [0] * n
    if n == 2:
        return [0, 1]
    return _kmeans2(member_weights.float())


# ---------------------------------------------------------- C: co-activation
def compute_split_coactivation(
    member_ids: Sequence[int],
    coact_matrix: torch.Tensor,
) -> List[int]:
    """2-way split based on routing co-occurrence among members.

    coact_matrix: [K, K] symmetric, M[i,j] = how often member i and member j
    fire together at top-k (or normalized variant). Higher = more similar.

    We convert similarity → distance via D = max(M) - M, then run 2-way
    k-means on the rows of D treated as feature vectors. (Equivalent to a
    rough spectral split without needing full eigendecomposition.)
    """
    n = len(member_ids)
    if n <= 1:
        return [0] * n
    if n == 2:
        return [0, 1]
    M = coact_matrix.float()
    D = M.max() - M
    return _kmeans2(D)
