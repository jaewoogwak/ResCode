"""Spectral / SVD-aligned initialization for `lowrank_preact`.

Two new init methods are provided in addition to the existing LoRA-style
zero-B init:

    1. "covariance":
       Use freq-weighted residual covariances as basis.
         B = top-r eigvecs of   S_out = sum_i omega_i D_i D_i^T
         A = top-r eigvecs of   S_in  = sum_i omega_i D_i^T D_i  (transposed)
         z_{i,l} = (1/alpha) b_l^T D_i a_l
       The model's forward computes effective residual alpha B diag(z_i) A.
       With this z, alpha (1/alpha) = 1 cancels exactly, so the effective residual
       at init equals the rank-r projection of D_i - INDEPENDENT of (alpha, r).
       This is the actual rank-r approximation of the per-expert residual.

    2. "expert_svd_align":
       Truncated SVD per expert, align components across experts via Hungarian
       (or greedy fallback) matching to spectral prototype slots, then build
       shared B,A by sigma-weighted averaging of aligned components. z is then
       re-projected onto the finalized (B,A) using the same (1/alpha) scaling.
       Per-expert SVD components are temporary and discarded.

Per-expert residuals are D_i = W_i - W_bar where W_bar is the cluster base
(`base_weight_dict`) or the freq-weighted cluster mean.

Works on gate/up/down token-adaptive lowrank residuals. Gate/up use the
`method` selected by the caller ("covariance" | "expert_svd_align"); the down
residual, when requested via `init_down=True`, always uses the covariance
recipe (its triple is stored under the "down" key as (L, R, Z)). Callers that
skip `init_down` fall back to the zero-output initialization in `PreActModule`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .config import parse_lowrank_target

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ----------------------------------------------------- low-level math helpers
def _topr_eigvecs_psd(M: torch.Tensor, r: int) -> torch.Tensor:
    """Top-r eigenvectors (descending eigenvalue) of a symmetric PSD matrix."""
    eigvals, eigvecs = torch.linalg.eigh(M)
    top = eigvecs[:, -r:].flip(-1)  # eigh returns ascending; take last r and flip
    return top.contiguous()


@torch.no_grad()
def compute_covariance_init(
    D_list: List[torch.Tensor],
    freqs: torch.Tensor,
    rank: int,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Init Method 1: freq-weighted covariance basis.

    Args:
      D_list: list of K residual matrices, each (m, d).
      freqs : tensor of K non-negative weights omega_i.
      rank  : r, the residual rank.
      alpha : residual scale alpha.

    Returns:
      B : (m, r)
      A : (r, d)
      Z : (K, r)  - z_{i,l} = (1/alpha) b_l^T D_i a_l
                   so the model's forward `alpha B diag(z_i) A` equals the
                   rank-r projection of D_i exactly (alpha,r-independent).
    """
    assert len(D_list) > 0
    K = len(D_list)
    m, d = D_list[0].shape
    device = D_list[0].device
    dtype = torch.float32

    omega = freqs.to(device=device, dtype=dtype)
    omega = omega / (omega.sum() + 1e-12)

    S_out = torch.zeros(m, m, device=device, dtype=dtype)
    S_in = torch.zeros(d, d, device=device, dtype=dtype)
    for i in range(K):
        D = D_list[i].to(device=device, dtype=dtype)
        S_out += omega[i] * (D @ D.T)
        S_in += omega[i] * (D.T @ D)

    B = _topr_eigvecs_psd(S_out, rank)        # (m, r)
    A_T = _topr_eigvecs_psd(S_in, rank)       # (d, r)
    A = A_T.T.contiguous()                    # (r, d)

    Z = torch.zeros(K, rank, device=device, dtype=dtype)
    scale = 1.0 / max(float(alpha), 1e-12)
    for i in range(K):
        D = D_list[i].to(device=device, dtype=dtype)
        # z_{i,l} = (1/alpha) b_l^T D_i a_l = (1/alpha) diag(B^T D A^T)
        # -> model effective residual alpha B diag(z_i) A = proj_r(D_i) (alpha,r-independent)
        Z[i] = scale * torch.diagonal(B.T @ D @ A_T)
    return B, A, Z


def _hungarian_or_greedy_assign(sim: torch.Tensor, alignment: str = "hungarian") -> List[Tuple[int, int]]:
    """Return list of (row, col) assignments maximizing total similarity.

    sim shape: (rank, rank) where rows = expert components, cols = slots.
    """
    r = sim.shape[0]
    if alignment == "hungarian" and _HAVE_SCIPY:
        row_ind, col_ind = linear_sum_assignment(-sim.detach().cpu().numpy())
        return list(zip(row_ind.tolist(), col_ind.tolist()))
    # Greedy fallback: walk pairs in descending similarity, skip used rows/cols.
    flat = sim.flatten()
    order = torch.argsort(flat, descending=True).cpu().tolist()
    used_rows, used_cols = set(), set()
    assigns: List[Tuple[int, int]] = []
    for idx in order:
        a, l = divmod(idx, r)
        if a in used_rows or l in used_cols:
            continue
        used_rows.add(a)
        used_cols.add(l)
        assigns.append((a, l))
        if len(assigns) == r:
            break
    return assigns


@torch.no_grad()
def compute_expert_svd_align_init(
    D_list: List[torch.Tensor],
    freqs: torch.Tensor,
    rank: int,
    alpha: float = 1.0,
    alignment: str = "hungarian",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Init Method 2: per-expert truncated SVD -> align -> weighted average.

    Args mirror `compute_covariance_init`. `alignment` is "hungarian" (needs
    scipy; falls back to greedy if unavailable) or "greedy".
    """
    assert len(D_list) > 0
    K = len(D_list)
    m, d = D_list[0].shape
    device = D_list[0].device
    dtype = torch.float32

    omega = freqs.to(device=device, dtype=dtype)
    omega = omega / (omega.sum() + 1e-12)

    # 1) Per-expert truncated SVD (kept temporarily)
    Us, Ss, Vs = [], [], []
    for D in D_list:
        D_f = D.to(device=device, dtype=dtype)
        U, S, Vh = torch.linalg.svd(D_f, full_matrices=False)
        Us.append(U[:, :rank])      # (m, r)
        Ss.append(S[:rank])         # (r,)
        Vs.append(Vh[:rank].T)      # (d, r)

    # 2) Prototype basis from spectral / covariance init
    B_proto, A_proto, _ = compute_covariance_init(D_list, freqs, rank, alpha)
    A_proto_T = A_proto.T.contiguous()   # (d, r)

    aligned_U = torch.zeros(K, m, rank, device=device, dtype=dtype)
    aligned_V = torch.zeros(K, d, rank, device=device, dtype=dtype)
    aligned_S = torch.zeros(K, rank, device=device, dtype=dtype)

    # 3) Align each expert's components to prototype slots
    for i in range(K):
        U_i, S_i, V_i = Us[i], Ss[i], Vs[i]
        # similarity[a, l] = |u_a^T b_l| * |v_a^T a_l|
        sim_u = (U_i.T @ B_proto).abs()       # (r, r)
        sim_v = (V_i.T @ A_proto_T).abs()     # (r, r)
        sim = sim_u * sim_v
        for (a, l) in _hungarian_or_greedy_assign(sim, alignment=alignment):
            sign_u = torch.sign(U_i[:, a] @ B_proto[:, l] + 1e-12)
            sign_v = torch.sign(V_i[:, a] @ A_proto_T[:, l] + 1e-12)
            aligned_U[i, :, l] = U_i[:, a] * sign_u
            aligned_V[i, :, l] = V_i[:, a] * sign_v
            aligned_S[i, l] = S_i[a] * sign_u * sign_v

    # 4) Shared B, A by weighted average + per-column normalize
    weights = omega.view(K, 1) * aligned_S    # (K, r)
    B_acc = (aligned_U * weights.view(K, 1, rank)).sum(0)   # (m, r)
    Vacc = (aligned_V * weights.view(K, 1, rank)).sum(0)    # (d, r)
    B = B_acc / (B_acc.norm(dim=0, keepdim=True) + 1e-12)
    A_T = Vacc / (Vacc.norm(dim=0, keepdim=True) + 1e-12)
    A = A_T.T.contiguous()  # (r, d)

    # 5) Re-project Z onto finalized (B, A) using same (1/alpha) scaling as
    # `compute_covariance_init` - keeps init residual = projection.
    Z = torch.zeros(K, rank, device=device, dtype=dtype)
    scale = 1.0 / max(float(alpha), 1e-12)
    for i in range(K):
        D = D_list[i].to(device=device, dtype=dtype)
        Z[i] = scale * torch.diagonal(B.T @ D @ A_T)
    return B, A, Z


# ------------------------------------------------------ top-level driver
def _get_moe_block(model, layer_idx: int, backbone: str):
    if backbone in ("qwen", "qwen3", "deepseek"):
        # All Qwen-family + DeepSeek expose the MoE block at `.mlp` (DeepSeek
        # dense layers also live there but should be filtered out by the
        # caller via cluster_maps / sparse_layer_indices).
        return model.model.layers[layer_idx].mlp
    if backbone == "mixtral":
        return model.model.layers[layer_idx].block_sparse_moe
    raise ValueError(f"Unknown backbone: {backbone}")


def _get_proj_weights(expert, backbone: str):
    """Return (gate_W, up_W, down_W) for a single original expert."""
    if backbone in ("qwen", "qwen3", "deepseek"):
        return expert.gate_proj.weight, expert.up_proj.weight, expert.down_proj.weight
    if backbone == "mixtral":
        # Mixtral: w1 = gate, w3 = up, w2 = down
        return expert.w1.weight, expert.w3.weight, expert.w2.weight
    raise ValueError(f"Unknown backbone: {backbone}")


def _ffn_name(layer_idx: int, backbone: str) -> str:
    if backbone in ("qwen", "qwen3", "deepseek"):
        return f"model.layers.{layer_idx}.mlp"
    return f"model.layers.{layer_idx}.block_sparse_moe"


def _compute_BAZ_for_subset(
    member_ids_subset,
    member_idx_in_full,            # list[int]: positions in full member list
    full_gate_Ws,                  # list of full member gate W tensors
    full_up_Ws,
    freqs_full_local,              # freqs over the FULL cluster, in local order
    rank, alpha, method_fn, target_gate, target_up,
    full_down_Ws=None,
    target_down=False,
    down_rank=None,
    down_alpha=1.0,
):
    """Compute (B, A, Z) on a subset of cluster members. Returns dict with
    'gate'/'up'/'down' triples (each B, A, Z on cpu). Down always uses the
    covariance recipe."""
    if len(member_ids_subset) < 1:
        return None
    w = freqs_full_local[member_idx_in_full].clone()
    if float(w.sum()) <= 0.0:
        w = torch.ones_like(w)
    sub_gate = [full_gate_Ws[i] for i in member_idx_in_full]
    sub_up = [full_up_Ws[i] for i in member_idx_in_full]
    w_dev = w.to(sub_gate[0].device).to(torch.float32)
    w_norm = w_dev / (w_dev.sum() + 1e-12)
    out = {}
    if target_gate:
        W_bar = sum(w_norm[i] * sub_gate[i].to(torch.float32)
                    for i in range(len(sub_gate)))
        D = [sub_gate[i].to(torch.float32) - W_bar for i in range(len(sub_gate))]
        B, A, Z = method_fn(D, w_dev, rank, alpha)
        out["gate"] = (B.cpu(), A.cpu(), Z.cpu())
    if target_up:
        W_bar = sum(w_norm[i] * sub_up[i].to(torch.float32)
                    for i in range(len(sub_up)))
        D = [sub_up[i].to(torch.float32) - W_bar for i in range(len(sub_up))]
        B, A, Z = method_fn(D, w_dev, rank, alpha)
        out["up"] = (B.cpu(), A.cpu(), Z.cpu())
    if target_down and full_down_Ws is not None:
        sub_down = [full_down_Ws[i] for i in member_idx_in_full]
        W_bar = sum(w_norm[i] * sub_down[i].to(torch.float32)
                    for i in range(len(sub_down)))
        D = [sub_down[i].to(torch.float32) - W_bar for i in range(len(sub_down))]
        B, A, Z = compute_covariance_init(D, w_dev, int(down_rank), float(down_alpha))
        out["down"] = (B.cpu(), A.cpu(), Z.cpu())
    return out


@torch.no_grad()
def compute_lowrank_preact_init_data(
    model,
    cluster_maps,                       # Dict[layer_idx, ExpertClusterMap]
    freq_dict: Dict[str, torch.Tensor],
    rank: int,
    alpha: float,
    target: str,                        # gate/up/down combinations
    method: str,                        # "covariance" | "expert_svd_align"
    alignment: str,                     # "hungarian" | "greedy"
    backbone: str,                      # "qwen" | "mixtral"
    verbose: bool = True,
    split_assignments: Optional[Dict[int, Dict[int, list]]] = None,
    init_gate_up: bool = True,
    init_down: bool = False,
    down_rank: Optional[int] = None,
    down_alpha: float = 1.0,
    base_weight_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[int, Dict[int, dict]]:
    """Walk the model layer-by-layer, capture per-expert weight residuals, and
    compute (B, A, Z) for each non-singleton cluster.

    MUST be called BEFORE RESCODE merging overwrites expert weights.

    `init_gate_up` computes gate/up triples with `method`; `init_down` computes
    the down triple with the covariance recipe (rank `down_rank`, scale
    `down_alpha`), stored under the "down" key as (L, R, Z).

    If `split_assignments[layer_idx][cluster_id]` is provided (a list of
    {0,1} of length |members|), the cluster is treated as DUAL - per-sub-cluster
    (B,A,Z) triples are returned and the per-cluster entry shape becomes:
        {"gate": {"a": (B_a,A_a,Z_a), "b": (B_b,A_b,Z_b)},
         "up":   {"a": ...,           "b": ...},
         "down": {"a": ...,           "b": ...},
         "sub_assignment": [0/1, 0/1, ...]}
    Singleton-only-side splits (e.g., (1, k-1)) fall back to single-residual
    init for that cluster (returns the original (B,A,Z) shape).

    Returns:
        init_data[layer_idx][cluster_id] = single or dual structure as above
        Singleton clusters are absent from the inner dict.
    """
    if init_gate_up and method not in {"covariance", "expert_svd_align"}:
        raise ValueError(f"Unknown lowrank_init method: {method}")

    targets = set(parse_lowrank_target(target))
    target_gate = "gate" in targets and init_gate_up
    target_up = "up" in targets and init_gate_up
    target_down = bool(init_down)
    if target_down and down_rank is None:
        down_rank = rank

    init_data: Dict[int, Dict[int, dict]] = {}

    method_fn = (
        compute_covariance_init
        if method == "covariance"
        else (lambda D, w, r, a: compute_expert_svd_align_init(D, w, r, a, alignment))
    )

    for layer_idx, cmap in cluster_maps.items():
        moe_block = _get_moe_block(model, layer_idx, backbone)
        ffn = _ffn_name(layer_idx, backbone)
        freqs_full = freq_dict.get(ffn) if freq_dict is not None else None
        base_full = base_weight_dict.get(ffn) if base_weight_dict is not None else None

        layer_out: Dict[int, dict] = {}
        for cluster_id, member_ids in cmap.cluster_to_experts.items():
            if len(member_ids) < 2:
                continue  # singleton - skip

            # Per-member freq weights omega_i (in member order)
            if freqs_full is not None:
                w_full = freqs_full[torch.as_tensor(member_ids, dtype=torch.long)].clone()
            else:
                w_full = torch.ones(len(member_ids))

            # Per-member weight matrices
            gate_Ws, up_Ws, down_Ws = [], [], []
            for mid in member_ids:
                expert = moe_block.experts[mid]
                g, u, d = _get_proj_weights(expert, backbone)
                gate_Ws.append(g.detach())
                up_Ws.append(u.detach())
                down_Ws.append(d.detach())

            # Decide single vs dual for this cluster
            sub_asg = None
            if split_assignments is not None:
                sub_asg = split_assignments.get(layer_idx, {}).get(cluster_id)

            do_dual = (
                sub_asg is not None
                and len(set(int(x) for x in sub_asg)) == 2
                and min(sum(1 for x in sub_asg if x == 0),
                        sum(1 for x in sub_asg if x == 1)) >= 1
            )

            if do_dual:
                idx_a = [i for i, s in enumerate(sub_asg) if int(s) == 0]
                idx_b = [i for i, s in enumerate(sub_asg) if int(s) == 1]
                # Need at least 1 member per side; if a side is singleton, the
                # covariance init still returns a degenerate (zero residual)
                # rank-r triple - that's fine, it just means that side starts
                # near zero and the optimizer can adapt.
                triples_a = _compute_BAZ_for_subset(
                    member_ids, idx_a, gate_Ws, up_Ws, w_full,
                    rank, alpha, method_fn, target_gate, target_up,
                    full_down_Ws=down_Ws, target_down=target_down,
                    down_rank=down_rank, down_alpha=down_alpha,
                )
                triples_b = _compute_BAZ_for_subset(
                    member_ids, idx_b, gate_Ws, up_Ws, w_full,
                    rank, alpha, method_fn, target_gate, target_up,
                    full_down_Ws=down_Ws, target_down=target_down,
                    down_rank=down_rank, down_alpha=down_alpha,
                )
                # The returned Z is sized for the SUBSET; we need to expand it
                # back to a full [num_members, rank] tensor with zeros for the
                # other side.
                cluster_entry: dict = {"sub_assignment": list(int(x) for x in sub_asg)}
                for kind, idx_sub, triples_sub in (
                    ("a", idx_a, triples_a),
                    ("b", idx_b, triples_b),
                ):
                    if triples_sub is None:
                        continue
                    for proj in ("gate", "up", "down"):
                        if proj not in triples_sub:
                            continue
                        B_s, A_s, Z_sub = triples_sub[proj]
                        Z_full = torch.zeros(len(member_ids), Z_sub.shape[1], dtype=Z_sub.dtype)
                        for k, i in enumerate(idx_sub):
                            Z_full[i] = Z_sub[k]
                        if proj not in cluster_entry:
                            cluster_entry[proj] = {}
                        cluster_entry[proj][kind] = (B_s, A_s, Z_full)
                layer_out[cluster_id] = cluster_entry
            else:
                # Single residual
                w_dev = w_full.to(gate_Ws[0].device).to(torch.float32)
                if float(w_dev.sum()) <= 0.0:
                    w_dev = torch.ones_like(w_dev)
                w_norm = w_dev / (w_dev.sum() + 1e-12)
                if base_full is not None:
                    b = base_full[torch.as_tensor(member_ids, dtype=torch.long)].float()
                    w_norm = (b / (b.sum() + 1e-7)).to(w_dev.device)
                cluster_entry = {}
                if target_gate:
                    W_bar_g = sum(w_norm[i] * gate_Ws[i].to(torch.float32)
                                  for i in range(len(member_ids)))
                    if base_full is not None:
                        W_bar_g = W_bar_g.to(gate_Ws[0].dtype).to(torch.float32)
                    D_g = [gate_Ws[i].to(torch.float32) - W_bar_g for i in range(len(member_ids))]
                    B, A, Z = method_fn(D_g, w_dev, rank, alpha)
                    cluster_entry["gate"] = (B.cpu(), A.cpu(), Z.cpu())
                    del D_g, W_bar_g
                if target_up:
                    W_bar_u = sum(w_norm[i] * up_Ws[i].to(torch.float32)
                                  for i in range(len(member_ids)))
                    if base_full is not None:
                        W_bar_u = W_bar_u.to(up_Ws[0].dtype).to(torch.float32)
                    D_u = [up_Ws[i].to(torch.float32) - W_bar_u for i in range(len(member_ids))]
                    B, A, Z = method_fn(D_u, w_dev, rank, alpha)
                    cluster_entry["up"] = (B.cpu(), A.cpu(), Z.cpu())
                    del D_u, W_bar_u
                if target_down:
                    W_bar_d = sum(w_norm[i] * down_Ws[i].to(torch.float32)
                                  for i in range(len(member_ids)))
                    if base_full is not None:
                        W_bar_d = W_bar_d.to(down_Ws[0].dtype).to(torch.float32)
                    D_d = [down_Ws[i].to(torch.float32) - W_bar_d for i in range(len(member_ids))]
                    B, A, Z = compute_covariance_init(D_d, w_dev, int(down_rank), float(down_alpha))
                    cluster_entry["down"] = (B.cpu(), A.cpu(), Z.cpu())
                    del D_d, W_bar_d
                layer_out[cluster_id] = cluster_entry

            del gate_Ws, up_Ws, down_Ws

        init_data[layer_idx] = layer_out
        if verbose:
            n_meta = sum(1 for v in cmap.cluster_to_experts.values() if len(v) >= 2)
            n_dual = sum(1 for ce in layer_out.values() if "sub_assignment" in ce)
            print(f"[RESCODE][lowrank-init] layer={layer_idx}  "
                  f"meta_clusters={n_meta}  dual={n_dual}  "
                  f"method={method}  rank={rank}  alpha={alpha}  "
                  f"down={'covariance' if target_down else 'skip'}")

    return init_data
