"""RescodeMetaExpert: a single physical SwiGLU MLP shared by a cluster of
original logical experts, conditioned by per-member code vectors.

Three output equations are supported, selected by `preact` presence and
`PreActStarConfig.paper_postact_aggregation`:

(1) Vanilla RESCODE (preact module disabled):

    h_C(x) = act(W_g_bar x) * (W_u_bar x)
    y_C(x) = W_d_bar ( a_C(x) * h_C(x) )

    where a_C(x) = sum_{j in C cap T(x)} g_j(x) c_j is a gate-weighted
    aggregation of per-member codes c_j. With c_j = 1 this collapses to
    RESCODE's gate_sum * E_bar(x).

(2) PostAct (paper Eq.(4), DEFAULT when preact enabled):

    For each occurrence a = (token t, member j ∈ C ∩ T(x)):
        g_a = (W_bar_g x_t) + alpha * B_g (z_g_j * (A_g x_t)) + beta_g_j (opt)
        u_a = (W_bar_u x_t) + alpha * B_u (z_u_j * (A_u x_t)) + beta_u_j (opt)
        h_a = SiLU(g_a) * u_a
        y_a = W_bar_d h_a + alpha_d * L_d (z_d_j * (R_d h_a))
    Cluster contribution:
        y_C(t) = sum_{a: token(a)=t} pi_a * y_a   (+ G_C(t) * bias if any)

    Equivalent to running a per-member SMoE expert with
    W_hat_j = W_bar + B diag(z_j) A in each branch (paper-faithful).
    Cost: K SwiGLU evaluations per cluster per token.

(3) PreAct-STAR (router-conditioned pre-activation modulation):

    rho_j(x)  = g_j(x) / G_C(x), G_C = sum_j g_j
    beta_g(x) = sum_j rho_j(x) b_g_j
    beta_u(x) = sum_j rho_j(x) b_u_j
    delta_g   = alpha * B_g (q_g(x) ⊙ A_g x), q_g = sum_j rho_j z_g_j
    delta_u   = alpha * B_u (q_u(x) ⊙ A_u x), q_u = sum_j rho_j z_u_j
    delta_d   = alpha_d * L_d (q_d(x) ⊙ R_d h), q_d = sum_j rho_j z_d_j
    g'(x)     = g0 + s_g ⊙ beta_g + delta_g
    u'(x)     = u0 + s_u ⊙ beta_u + delta_u
    h_C(x)    = SiLU(g'(x)) ⊙ u'(x)
    y_C(x)    = G_C(x) * (W_d_bar (a'_C(x) * h_C(x)) + delta_d)

    First-order approximation of (2) under nonlinear SwiGLU; cheaper
    (1 SwiGLU evaluation per cluster per token).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PreActStarConfig


def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


# ---------------------------------------------------------------------- helper
def compute_cluster_gate_and_rho(
    selected_experts: torch.Tensor,    # [T, top_k] long
    routing_weights: torch.Tensor,     # [T, top_k] float
    local_table: torch.Tensor,         # buffer mapping orig_eid -> local_idx (or -1)
    eps: float = 1.0e-6,
):
    """Build the per-occurrence ρ_j(x) and per-token cluster gate G_C(x) for one cluster.

    A "occurrence" is a (token, top_k_slot) pair where the selected expert id
    falls in this cluster. A single token can have multiple occurrences if it
    routes to two cluster members in its top-k.

    Returns
    -------
    unique_tokens     : [T_C] long       — token ids that are active in this cluster
    inverse           : [A]   long       — for each occurrence, its index in unique_tokens
    local_member_ids  : [A]   long       — local id (0..K-1) of the chosen member
    gates             : [A]   float      — raw routing weight r_j(x) at the occurrence
    cluster_gate      : [T_C] float      — G_C(x) = sum of gates for that token
    rho               : [A]   float      — gates / cluster_gate[inverse]
    """
    device = selected_experts.device
    flat_se = selected_experts.reshape(-1).to(device)
    flat_rw = routing_weights.reshape(-1).to(device)
    T = selected_experts.shape[0]
    token_id = (
        torch.arange(T, device=device).unsqueeze(-1)
        .expand_as(selected_experts).reshape(-1)
    )

    max_eid = local_table.shape[0] - 1
    valid_mask = (flat_se >= 0) & (flat_se <= max_eid)
    if valid_mask.any():
        local_lookup = torch.where(
            valid_mask,
            local_table[flat_se.clamp(min=0, max=max_eid)],
            torch.full_like(flat_se, -1),
        )
    else:
        local_lookup = torch.full_like(flat_se, -1)
    keep = local_lookup >= 0
    if not keep.any():
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        empty_float = torch.zeros(0, dtype=flat_rw.dtype, device=device)
        return empty_long, empty_long, empty_long, empty_float, empty_float, empty_float

    token_kept = token_id[keep]
    local_kept = local_lookup[keep]
    gate_kept = flat_rw[keep]

    unique_tokens, inverse = torch.unique(token_kept, return_inverse=True)
    N = unique_tokens.shape[0]

    cluster_gate = torch.zeros(
        N, device=device, dtype=gate_kept.dtype if gate_kept.is_floating_point() else torch.float32,
    )
    cluster_gate.index_add_(0, inverse, gate_kept.to(cluster_gate.dtype))

    rho = gate_kept.to(cluster_gate.dtype) / (cluster_gate[inverse] + eps)

    return unique_tokens, inverse, local_kept, gate_kept.to(cluster_gate.dtype), cluster_gate, rho


# ============================================================ PreActModule
class PreActModule(nn.Module):
    """Optional preactivation modulation parameters for one cluster.

    Built only when PreActStarConfig.enabled is True. Carries:
        - per-member preact biases b_g_j, b_u_j        [K, d_ff]
        - router-conditioned low-rank preact factors:
            gate_A: [r,    d_model]   (random, std=init_std)
            gate_B: [d_ff, r]         (zero — keeps initial output equal to RESCODE)
            gate_z: [K,    r]         (random, std=init_std)
        - down-proj low-rank residual:
            down_L: [d_model, r_d]
            down_R: [r_d,    d_ff]
            down_z: [K,      r_d]
            Initialized by `down_residual_init`: "covariance" (default) copies
            the precomputed rank-r projection of the per-expert down residual,
            "zero_L"/"zero_R" start at Δ=0.
        - optional preact std buffers for normalization:
            gate_preact_std: [d_ff]
            up_preact_std:   [d_ff]

    With the zero-style inits (lora_zero_B / zero_L / zero_R) every learnable
    parameter starts so that the cluster output is *exactly* equal to vanilla
    RESCODE with the same `code_type`. The covariance-style inits (the default
    for both gate/up and down) instead start each residual at the rank-r
    projection of the corresponding per-expert weight residual.
    """

    def __init__(
        self,
        num_members: int,
        intermediate_size: int,
        hidden_size: int,
        config: PreActStarConfig,
        device: torch.device,
        precomputed_init: Optional[dict] = None,
    ):
        super().__init__()
        self.config = config
        self.num_members = num_members
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

        if config.gate_bias:
            self.gate_bias_codes = nn.Parameter(
                torch.zeros(num_members, intermediate_size, device=device, dtype=torch.float32)
            )
        if config.up_bias:
            self.up_bias_codes = nn.Parameter(
                torch.zeros(num_members, intermediate_size, device=device, dtype=torch.float32)
            )

        # ---- dual residual sub-cluster assignment --------------------------
        # When dual is enabled and the cluster is large enough to split, the
        # caller passes a [num_members] long tensor of {0,1} via
        # `precomputed_init["sub_assignment"]`. Singleton clusters or clusters
        # below `lowrank_split_min_size` keep `dual_active=False`.
        sub_assignment = None
        if precomputed_init is not None:
            sub_assignment = precomputed_init.get("sub_assignment")
        self.dual_active = bool(
            getattr(config, "lowrank_dual", False)
            and sub_assignment is not None
            and num_members >= int(getattr(config, "lowrank_split_min_size", 3))
            and len(set(int(x) for x in sub_assignment)) == 2
        )
        if self.dual_active:
            sa = torch.as_tensor(sub_assignment, dtype=torch.long, device=device)
            self.register_buffer("sub_assignment", sa, persistent=True)

        # ---- gate lowrank --------------------------------------------------
        if config.has_lowrank_gate:
            r = int(config.lowrank_rank)
            self.gate_A = nn.Parameter(torch.empty(r, hidden_size, device=device, dtype=torch.float32))
            self.gate_B = nn.Parameter(torch.empty(intermediate_size, r, device=device, dtype=torch.float32))
            self.gate_z = nn.Parameter(torch.empty(num_members, r, device=device, dtype=torch.float32))
            pre_g = (precomputed_init or {}).get("gate")
            if self.dual_active and isinstance(pre_g, dict) and "a" in pre_g:
                pre_g_a = pre_g.get("a")
            else:
                pre_g_a = pre_g
            self._init_lowrank_triple(
                A=self.gate_A, B=self.gate_B, Z=self.gate_z,
                config=config, device=device,
                precomputed=pre_g_a,
                kind="gate",
            )
            if self.dual_active:
                self.gate_A_b = nn.Parameter(torch.empty(r, hidden_size, device=device, dtype=torch.float32))
                self.gate_B_b = nn.Parameter(torch.empty(intermediate_size, r, device=device, dtype=torch.float32))
                self.gate_z_b = nn.Parameter(torch.empty(num_members, r, device=device, dtype=torch.float32))
                pre_g_b = pre_g.get("b") if isinstance(pre_g, dict) else None
                self._init_lowrank_triple(
                    A=self.gate_A_b, B=self.gate_B_b, Z=self.gate_z_b,
                    config=config, device=device,
                    precomputed=pre_g_b,
                    kind="gate_b",
                )

        # ---- up lowrank ----------------------------------------------------
        if config.has_lowrank_up:
            r = int(config.lowrank_rank)
            self.up_A = nn.Parameter(torch.empty(r, hidden_size, device=device, dtype=torch.float32))
            self.up_B = nn.Parameter(torch.empty(intermediate_size, r, device=device, dtype=torch.float32))
            self.up_z = nn.Parameter(torch.empty(num_members, r, device=device, dtype=torch.float32))
            pre_u = (precomputed_init or {}).get("up")
            if self.dual_active and isinstance(pre_u, dict) and "a" in pre_u:
                pre_u_a = pre_u.get("a")
            else:
                pre_u_a = pre_u
            self._init_lowrank_triple(
                A=self.up_A, B=self.up_B, Z=self.up_z,
                config=config, device=device,
                precomputed=pre_u_a,
                kind="up",
            )
            if self.dual_active:
                self.up_A_b = nn.Parameter(torch.empty(r, hidden_size, device=device, dtype=torch.float32))
                self.up_B_b = nn.Parameter(torch.empty(intermediate_size, r, device=device, dtype=torch.float32))
                self.up_z_b = nn.Parameter(torch.empty(num_members, r, device=device, dtype=torch.float32))
                pre_u_b = pre_u.get("b") if isinstance(pre_u, dict) else None
                self._init_lowrank_triple(
                    A=self.up_A_b, B=self.up_B_b, Z=self.up_z_b,
                    config=config, device=device,
                    precomputed=pre_u_b,
                    kind="up_b",
                )

        if config.down_residual:
            r_d = int(config.down_residual_rank)
            pre_d = (precomputed_init or {}).get("down")
            if self.dual_active and isinstance(pre_d, dict) and "a" in pre_d:
                pre_d_a = pre_d.get("a")
                pre_d_b = pre_d.get("b")
            else:
                pre_d_a = pre_d if not isinstance(pre_d, dict) else None
                pre_d_b = None
            self.down_L = nn.Parameter(torch.empty(hidden_size, r_d, device=device, dtype=torch.float32))
            self.down_R = nn.Parameter(torch.empty(r_d, intermediate_size, device=device, dtype=torch.float32))
            self.down_z = nn.Parameter(torch.empty(num_members, r_d, device=device, dtype=torch.float32))
            self._init_down_triple(
                L=self.down_L, R=self.down_R, Z=self.down_z,
                config=config, precomputed=pre_d_a, kind="down",
            )
            if self.dual_active:
                # Dual down residual — same init scheme, separate params.
                self.down_L_b = nn.Parameter(torch.empty(hidden_size, r_d, device=device, dtype=torch.float32))
                self.down_R_b = nn.Parameter(torch.empty(r_d, intermediate_size, device=device, dtype=torch.float32))
                self.down_z_b = nn.Parameter(torch.empty(num_members, r_d, device=device, dtype=torch.float32))
                self._init_down_triple(
                    L=self.down_L_b, R=self.down_R_b, Z=self.down_z_b,
                    config=config, precomputed=pre_d_b, kind="down_b",
                )

        # Std buffers — start at 1.0 so normalization is a no-op until calibration runs.
        if config.normalize_by_preact_std:
            self.register_buffer(
                "gate_preact_std",
                torch.ones(intermediate_size, device=device, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "up_preact_std",
                torch.ones(intermediate_size, device=device, dtype=torch.float32),
                persistent=True,
            )

    # ----------------------------------------------- lowrank init dispatch
    @torch.no_grad()
    def _init_lowrank_triple(
        self,
        A: nn.Parameter,
        B: nn.Parameter,
        Z: nn.Parameter,
        config: PreActStarConfig,
        device: torch.device,
        precomputed: Optional[tuple],
        kind: str,
    ) -> None:
        """Initialize one (A, B, z) triple according to `config.lowrank_init`.

        - "lora_zero_B" (default): A,z ~ N(0, init_std²), B = 0.  ΔW = 0 at init.
        - "covariance" / "expert_svd_align": use precomputed (B, A, Z) triple
          if provided. Falls back to LoRA-style init with a warning if not.
        """
        init_mode = config.lowrank_init
        if init_mode == "lora_zero_B" or precomputed is None:
            if init_mode != "lora_zero_B" and precomputed is None:
                print(f"[RESCODE][PreActModule][warn] lowrank_init={init_mode} requested "
                      f"for {kind} but no precomputed_init provided — falling back to lora_zero_B.")
            A.data.normal_(mean=0.0, std=config.lowrank_init_std)
            B.data.zero_()
            Z.data.normal_(mean=0.0, std=config.lowrank_init_std)
            return

        # Covariance / SVD-align: copy precomputed (B, A, Z)
        B_pre, A_pre, Z_pre = precomputed
        if A_pre.shape != A.shape:
            raise ValueError(
                f"[RESCODE] precomputed A shape {tuple(A_pre.shape)} != "
                f"expected {tuple(A.shape)} for {kind}."
            )
        if B_pre.shape != B.shape:
            raise ValueError(
                f"[RESCODE] precomputed B shape {tuple(B_pre.shape)} != "
                f"expected {tuple(B.shape)} for {kind}."
            )
        if Z_pre.shape != Z.shape:
            raise ValueError(
                f"[RESCODE] precomputed Z shape {tuple(Z_pre.shape)} != "
                f"expected {tuple(Z.shape)} for {kind}."
            )
        A.data.copy_(A_pre.to(device=device, dtype=A.dtype))
        B.data.copy_(B_pre.to(device=device, dtype=B.dtype))
        Z.data.copy_(Z_pre.to(device=device, dtype=Z.dtype))

    @torch.no_grad()
    def _init_down_triple(
        self,
        L: nn.Parameter,
        R: nn.Parameter,
        Z: nn.Parameter,
        config: PreActStarConfig,
        precomputed: Optional[tuple],
        kind: str,
    ) -> None:
        """Initialize one down-residual (L, R, z) triple per `config.down_residual_init`.

        - "covariance" (default): copy the precomputed (L, R, Z) triple so the
          effective residual α_d·L·diag(z_j)·R starts at the rank-r projection
          of the per-expert down residual — same recipe as gate/up. Falls back
          to zero_L with a warning when no precomputed triple is available.
        - "zero_L": L = 0, R,z ~ N(0, init_std²).  Δ = 0 at init.
        - "zero_R": R = 0, L,z ~ N(0, init_std²).  Δ = 0 at init.
        """
        init_mode = config.down_residual_init
        if init_mode == "covariance" and precomputed is None:
            print(f"[RESCODE][PreActModule][warn] down_residual_init=covariance requested "
                  f"for {kind} but no precomputed_init provided — falling back to zero_L.")
            init_mode = "zero_L"
        if init_mode == "covariance":
            # Stored by `compute_lowrank_preact_init_data` as (B, A, Z) = (L, R, Z).
            L_pre, R_pre, Z_pre = precomputed
            for name, pre, param in (("L", L_pre, L), ("R", R_pre, R), ("z", Z_pre, Z)):
                if pre.shape != param.shape:
                    raise ValueError(
                        f"[RESCODE] precomputed down {name} shape {tuple(pre.shape)} != "
                        f"expected {tuple(param.shape)} for {kind}."
                    )
            L.data.copy_(L_pre.to(device=L.device, dtype=L.dtype))
            R.data.copy_(R_pre.to(device=R.device, dtype=R.dtype))
            Z.data.copy_(Z_pre.to(device=Z.device, dtype=Z.dtype))
        elif init_mode == "zero_R":
            L.data.normal_(mean=0.0, std=config.down_residual_init_std)
            R.data.zero_()
            Z.data.normal_(mean=0.0, std=config.down_residual_init_std)
        else:  # zero_L
            L.data.zero_()
            R.data.normal_(mean=0.0, std=config.down_residual_init_std)
            Z.data.normal_(mean=0.0, std=config.down_residual_init_std)

    # -------------------------------- introspection / param-group helpers
    def named_preact_params(self):
        """Yield (group_name, name, parameter) tuples for optimizer parameter groups.

        group_name is one of {bias, lowrank, down}. The 'name' is local
        (`gate_bias_codes`, `gate_A`, ...).
        """
        for n in ("gate_bias_codes", "up_bias_codes"):
            if hasattr(self, n):
                yield "bias", n, getattr(self, n)
        for n in ("gate_A", "gate_B", "gate_z", "up_A", "up_B", "up_z",
                  "gate_A_b", "gate_B_b", "gate_z_b",
                  "up_A_b", "up_B_b", "up_z_b"):
            if hasattr(self, n):
                yield "lowrank", n, getattr(self, n)
        for n in ("down_L", "down_R", "down_z",
                  "down_L_b", "down_R_b", "down_z_b"):
            if hasattr(self, n):
                yield "down", n, getattr(self, n)

    def preact_parameter_list(self):
        return [p for _, _, p in self.named_preact_params()]

    def has_any_param(self) -> bool:
        return any(True for _ in self.named_preact_params())

    @torch.no_grad()
    def clip_biases_(self):
        cfg = self.config
        if cfg.bias_clip_min is None and cfg.bias_clip_max is None:
            return
        lo = cfg.bias_clip_min if cfg.bias_clip_min is not None else -float("inf")
        hi = cfg.bias_clip_max if cfg.bias_clip_max is not None else float("inf")
        for n in ("gate_bias_codes", "up_bias_codes"):
            if hasattr(self, n):
                getattr(self, n).clamp_(lo, hi)


# ============================================================ RescodeMetaExpert
class RescodeMetaExpert(nn.Module):
    """Single physical meta expert for a cluster.

    Holds references (or copies) of merged gate/up/down projections plus a
    per-member code that modulates the FFN intermediate activation.

    Code shapes (num_members = |C|, d_ff = intermediate_size):
        none      -> no code parameter, treated as ones
        scalar    -> [num_members]
        diag      -> [num_members, d_ff]
        lowrank   -> Z: [num_members, rank], U: [d_ff, rank];
                     code_j = 1 + Z_j @ U.T

    PreAct-STAR (optional, off by default): if `preact_config` is provided and
    enabled, the module also carries router-conditioned preactivation
    parameters. See module docstring at top of file for the equations.
    """

    def __init__(
        self,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        member_expert_ids: List[int],
        code_type: str = "diag",
        code_rank: Optional[int] = None,
        init: str = "ones",
        code_clip_min: Optional[float] = 0.0,
        code_clip_max: Optional[float] = 2.0,
        freeze_code: bool = False,
        act_fn=_silu,
        preact_config: Optional[PreActStarConfig] = None,
        preact_precomputed_init: Optional[dict] = None,
    ):
        super().__init__()
        if code_type not in {"none", "scalar", "diag", "lowrank"}:
            raise ValueError(f"[RESCODE] unknown code_type={code_type}")
        self.code_type = code_type
        self.code_rank = code_rank
        self.code_clip_min = code_clip_min
        self.code_clip_max = code_clip_max
        self.freeze_code = freeze_code

        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn

        self.intermediate_size = int(gate_proj.out_features)
        self.hidden_size = int(gate_proj.in_features)
        self.member_expert_ids = list(member_expert_ids)
        self.num_members = len(self.member_expert_ids)

        device = gate_proj.weight.device

        # Local-id table: original expert id -> position in member_expert_ids.
        local_table = torch.full(
            (max(self.member_expert_ids) + 1,),
            -1, dtype=torch.long, device=device,
        )
        for local_idx, eid in enumerate(self.member_expert_ids):
            local_table[eid] = local_idx
        self.register_buffer("local_table", local_table, persistent=False)

        self._init_codes(init=init, device=device)

        if self.freeze_code:
            for p in self._code_parameters():
                p.requires_grad_(False)

        # Optional PreAct-STAR module.
        self.preact: Optional[PreActModule] = None
        if preact_config is not None and preact_config.enabled:
            self.preact = PreActModule(
                num_members=self.num_members,
                intermediate_size=self.intermediate_size,
                hidden_size=self.hidden_size,
                config=preact_config,
                device=device,
                precomputed_init=preact_precomputed_init,
            )

    # ------------------------------------------------------------------ codes
    def _init_codes(self, init: str, device: torch.device):
        if self.code_type == "none":
            return
        if self.code_type == "scalar":
            data = torch.ones(self.num_members, dtype=torch.float32, device=device)
            if init != "ones":
                raise ValueError("[RESCODE] only 'ones' init is supported for scalar.")
            self.code = nn.Parameter(data)
            return
        if self.code_type == "diag":
            data = torch.ones(
                self.num_members, self.intermediate_size,
                dtype=torch.float32, device=device,
            )
            if init != "ones":
                raise ValueError("[RESCODE] only 'ones' init is supported for diag.")
            self.code = nn.Parameter(data)
            return
        if self.code_type == "lowrank":
            if self.code_rank is None or self.code_rank <= 0:
                raise ValueError("[RESCODE] code_rank must be a positive integer for lowrank.")
            U = torch.randn(self.intermediate_size, self.code_rank, device=device) * 1e-3
            Z = torch.zeros(self.num_members, self.code_rank, device=device)
            self.code_U = nn.Parameter(U.float())
            self.code_Z = nn.Parameter(Z.float())
            return

    def _code_parameters(self):
        if self.code_type == "none":
            return []
        if self.code_type in {"scalar", "diag"}:
            return [self.code]
        if self.code_type == "lowrank":
            return [self.code_U, self.code_Z]
        return []

    def code_parameters(self):
        return list(self._code_parameters())

    @torch.no_grad()
    def clip_codes_(self):
        if self.code_type == "none":
            return
        if self.code_clip_min is None and self.code_clip_max is None:
            return
        if self.code_type in {"scalar", "diag"}:
            self.code.clamp_(
                self.code_clip_min if self.code_clip_min is not None else -float("inf"),
                self.code_clip_max if self.code_clip_max is not None else float("inf"),
            )

    def materialize_code(self, local_idx: torch.Tensor) -> torch.Tensor:
        if self.code_type == "none":
            raise RuntimeError("[RESCODE] materialize_code called with code_type=none.")
        if self.code_type == "scalar":
            return self.code.index_select(0, local_idx)
        if self.code_type == "diag":
            return self.code.index_select(0, local_idx)
        if self.code_type == "lowrank":
            Z_sel = self.code_Z.index_select(0, local_idx)
            code = 1.0 + Z_sel @ self.code_U.t()
            if self.code_clip_min is not None or self.code_clip_max is not None:
                lo = self.code_clip_min if self.code_clip_min is not None else -float("inf")
                hi = self.code_clip_max if self.code_clip_max is not None else float("inf")
                code = code.clamp(lo, hi)
            return code
        raise RuntimeError("unreachable")

    # --------------------------------------------------------- aggregation
    def aggregate(
        self,
        local_member_indices: torch.Tensor,    # [A]    long, in [0, num_members)
        gates: torch.Tensor,                   # [A]    float
        inverse: torch.Tensor,                 # [A]    long, in [0, N)
        num_unique: int,
        accum_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate gate-weighted codes into per-token vectors (vanilla RESCODE)."""
        device = gates.device
        gates_f = gates.to(accum_dtype)

        gate_sum = torch.zeros(num_unique, device=device, dtype=accum_dtype)
        gate_sum.index_add_(0, inverse, gates_f)

        if self.code_type == "none":
            return gate_sum, gate_sum

        if self.code_type == "scalar":
            codes = self.code.index_select(0, local_member_indices).to(accum_dtype)
            weighted = gates_f * codes
            code_agg = torch.zeros(num_unique, device=device, dtype=accum_dtype)
            code_agg.index_add_(0, inverse, weighted)
            return code_agg, gate_sum

        if self.code_type == "diag":
            codes = self.code.index_select(0, local_member_indices).to(accum_dtype)
            weighted = gates_f.unsqueeze(-1) * codes
            code_agg = torch.zeros(
                num_unique, self.intermediate_size, device=device, dtype=accum_dtype,
            )
            code_agg.index_add_(0, inverse, weighted)
            return code_agg, gate_sum

        if self.code_type == "lowrank":
            codes = self.materialize_code(local_member_indices).to(accum_dtype)
            weighted = gates_f.unsqueeze(-1) * codes
            code_agg = torch.zeros(
                num_unique, self.intermediate_size, device=device, dtype=accum_dtype,
            )
            code_agg.index_add_(0, inverse, weighted)
            return code_agg, gate_sum

        raise RuntimeError("unreachable")

    # ----------------------------------------------------- forward (low level)
    def forward_aggregated(
        self,
        unique_x: torch.Tensor,         # [N, hidden]
        code_agg: torch.Tensor,         # [N] or [N, d_ff]
        gate_sum: torch.Tensor,         # [N]
    ) -> torch.Tensor:
        """Vanilla RESCODE forward (no preact). Kept for backward compat."""
        proj_dtype = self.gate_proj.weight.dtype
        x = unique_x.to(proj_dtype) if unique_x.dtype != proj_dtype else unique_x
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)  # [N, d_ff]

        if self.code_type == "none":
            h_coded = h * gate_sum.unsqueeze(-1).to(h.dtype)
        elif self.code_type == "scalar":
            h_coded = h * code_agg.unsqueeze(-1).to(h.dtype)
        else:  # diag / lowrank
            h_coded = h * code_agg.to(h.dtype)

        out = F.linear(h_coded, self.down_proj.weight, bias=None)
        if self.down_proj.bias is not None:
            out = out + gate_sum.to(out.dtype).unsqueeze(-1) * self.down_proj.bias
        return out

    # --------------------------------------------- preact-aware aggregation
    def _preact_aggregate(
        self,
        local_member_indices: torch.Tensor,
        rho: torch.Tensor,
        inverse: torch.Tensor,
        num_unique: int,
        accum_dtype: torch.dtype = torch.float32,
    ):
        """Aggregate ρ-weighted preact biases / lowrank codes for the cluster.

        Returns a dict keyed by what the preact path needs:
            beta_g [N, d_ff] or None
            beta_u [N, d_ff] or None
            q_g    [N, r]    or None
            q_u    [N, r]    or None
            q_d    [N, r_d]  or None
            post_code_agg [N] or [N, d_ff] or None  (rho-weighted post code,
                                                     i.e. sum_j rho_j c_j)
        """
        assert self.preact is not None
        device = rho.device
        cfg = self.preact.config
        rho_f = rho.to(accum_dtype)
        out = {}

        # ---- gate / up bias ---------------------------------------------
        if cfg.gate_bias and hasattr(self.preact, "gate_bias_codes"):
            b_sel = self.preact.gate_bias_codes.index_select(0, local_member_indices).to(accum_dtype)
            beta = torch.zeros(num_unique, self.intermediate_size, device=device, dtype=accum_dtype)
            beta.index_add_(0, inverse, rho_f.unsqueeze(-1) * b_sel)
            out["beta_g"] = beta
        else:
            out["beta_g"] = None

        if cfg.up_bias and hasattr(self.preact, "up_bias_codes"):
            b_sel = self.preact.up_bias_codes.index_select(0, local_member_indices).to(accum_dtype)
            beta = torch.zeros(num_unique, self.intermediate_size, device=device, dtype=accum_dtype)
            beta.index_add_(0, inverse, rho_f.unsqueeze(-1) * b_sel)
            out["beta_u"] = beta
        else:
            out["beta_u"] = None

        # ---- dual residual sub-cluster mask (only when dual_active) -------
        dual_active = bool(getattr(self.preact, "dual_active", False))
        if dual_active:
            sa_sel = self.preact.sub_assignment.index_select(0, local_member_indices).to(torch.long)
            mask_a = (sa_sel == 0).to(accum_dtype)
            mask_b = (sa_sel == 1).to(accum_dtype)
        else:
            mask_a = None
            mask_b = None

        # ---- lowrank z aggregation --------------------------------------
        if cfg.has_lowrank_gate:
            r = int(cfg.lowrank_rank)
            z_sel = self.preact.gate_z.index_select(0, local_member_indices).to(accum_dtype)
            qg = torch.zeros(num_unique, r, device=device, dtype=accum_dtype)
            rho_a = (rho_f * mask_a) if dual_active else rho_f
            qg.index_add_(0, inverse, rho_a.unsqueeze(-1) * z_sel)
            out["q_g"] = qg
            if dual_active:
                z_sel_b = self.preact.gate_z_b.index_select(0, local_member_indices).to(accum_dtype)
                qg_b = torch.zeros(num_unique, r, device=device, dtype=accum_dtype)
                qg_b.index_add_(0, inverse, (rho_f * mask_b).unsqueeze(-1) * z_sel_b)
                out["q_g_b"] = qg_b
        else:
            out["q_g"] = None

        if cfg.has_lowrank_up:
            r = int(cfg.lowrank_rank)
            z_sel = self.preact.up_z.index_select(0, local_member_indices).to(accum_dtype)
            qu = torch.zeros(num_unique, r, device=device, dtype=accum_dtype)
            rho_a = (rho_f * mask_a) if dual_active else rho_f
            qu.index_add_(0, inverse, rho_a.unsqueeze(-1) * z_sel)
            out["q_u"] = qu
            if dual_active:
                z_sel_b = self.preact.up_z_b.index_select(0, local_member_indices).to(accum_dtype)
                qu_b = torch.zeros(num_unique, r, device=device, dtype=accum_dtype)
                qu_b.index_add_(0, inverse, (rho_f * mask_b).unsqueeze(-1) * z_sel_b)
                out["q_u_b"] = qu_b
        else:
            out["q_u"] = None

        if cfg.has_lowrank_down and hasattr(self.preact, "down_z"):
            r_d = int(cfg.down_residual_rank)
            z_sel = self.preact.down_z.index_select(0, local_member_indices).to(accum_dtype)
            qd = torch.zeros(num_unique, r_d, device=device, dtype=accum_dtype)
            rho_a = (rho_f * mask_a) if dual_active else rho_f
            qd.index_add_(0, inverse, rho_a.unsqueeze(-1) * z_sel)
            out["q_d"] = qd
            if dual_active:
                z_sel_b = self.preact.down_z_b.index_select(0, local_member_indices).to(accum_dtype)
                qd_b = torch.zeros(num_unique, r_d, device=device, dtype=accum_dtype)
                qd_b.index_add_(0, inverse, (rho_f * mask_b).unsqueeze(-1) * z_sel_b)
                out["q_d_b"] = qd_b
        else:
            out["q_d"] = None

        # ---- rho-weighted post-code (a'_C = sum_j rho_j c_j) -------------
        if self.code_type == "none":
            out["post_code_agg"] = None  # treated as 1.0
        elif self.code_type == "scalar":
            c = self.code.index_select(0, local_member_indices).to(accum_dtype)
            agg = torch.zeros(num_unique, device=device, dtype=accum_dtype)
            agg.index_add_(0, inverse, rho_f * c)
            out["post_code_agg"] = agg
        elif self.code_type == "diag":
            c = self.code.index_select(0, local_member_indices).to(accum_dtype)
            agg = torch.zeros(num_unique, self.intermediate_size, device=device, dtype=accum_dtype)
            agg.index_add_(0, inverse, rho_f.unsqueeze(-1) * c)
            out["post_code_agg"] = agg
        elif self.code_type == "lowrank":
            c = self.materialize_code(local_member_indices).to(accum_dtype)
            agg = torch.zeros(num_unique, self.intermediate_size, device=device, dtype=accum_dtype)
            agg.index_add_(0, inverse, rho_f.unsqueeze(-1) * c)
            out["post_code_agg"] = agg
        return out

    # --------------------------------------------- paper Eq.(4) postact forward
    def forward_postact(
        self,
        unique_x: torch.Tensor,                 # [N, hidden]
        local_member_indices: torch.Tensor,     # [A] long, in [0, K)
        gates: torch.Tensor,                    # [A] float, original pi_j
        inverse: torch.Tensor,                  # [A] long, in [0, N)
        num_unique: int,
        accum_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Paper-faithful per-expert SwiGLU forward (Eq. 4) then pi-weighted sum.

        For each occurrence a = (token t, member j):
            g_a = (W_bar_g x_t) + alpha * B_g (z_g_j * (A_g x_t)) + beta_g_j (opt)
            u_a = (W_bar_u x_t) + alpha * B_u (z_u_j * (A_u x_t)) + beta_u_j (opt)
            h_a = SiLU(g_a) * u_a
            (optional diag post-code:  h_a <- c_j * h_a)
            y_a = W_bar_d h_a + alpha_d * L_d (z_d_j * (R_d h_a))   (down residual)
        Cluster contribution per token t:
            y_t = sum_{a: token(a)=t} pi_a * y_a    (+ G_C(t) * bias if any)

        Equivalent to running a per-member SMoE expert with
        W_hat_j = W_bar + B diag(z_j) A in each branch (matches paper Eq. 4),
        at the cost of K SwiGLU evaluations per cluster per token.
        """
        assert self.preact is not None
        cfg = self.preact.config
        proj_dtype = self.gate_proj.weight.dtype
        device = unique_x.device

        # Per-occurrence inputs.
        x_occ_f = unique_x.index_select(0, inverse).to(accum_dtype)       # [A, hidden]
        x_occ_p = x_occ_f.to(proj_dtype) if proj_dtype != accum_dtype else x_occ_f

        g0_occ = self.gate_proj(x_occ_p).to(accum_dtype)                  # [A, d_ff]
        u0_occ = self.up_proj(x_occ_p).to(accum_dtype)                    # [A, d_ff]

        gates_f = gates.to(accum_dtype)                                    # [A]

        # ---- per-member preact bias (optional) ---------------------------
        if cfg.gate_bias and hasattr(self.preact, "gate_bias_codes"):
            b_g = self.preact.gate_bias_codes.index_select(0, local_member_indices).to(accum_dtype)
            if cfg.normalize_by_preact_std:
                b_g = b_g * self.preact.gate_preact_std.to(accum_dtype).unsqueeze(0)
            g0_occ = g0_occ + b_g
        if cfg.up_bias and hasattr(self.preact, "up_bias_codes"):
            b_u = self.preact.up_bias_codes.index_select(0, local_member_indices).to(accum_dtype)
            if cfg.normalize_by_preact_std:
                b_u = b_u * self.preact.up_preact_std.to(accum_dtype).unsqueeze(0)
            u0_occ = u0_occ + b_u

        # ---- per-member token-adaptive lowrank residual ------------------
        if cfg.has_lowrank_gate:
            A_g_x = F.linear(x_occ_f, self.preact.gate_A.to(accum_dtype))        # [A, r]
            z_g = self.preact.gate_z.index_select(0, local_member_indices).to(accum_dtype)  # [A, r]
            delta_g = F.linear(z_g * A_g_x, self.preact.gate_B.to(accum_dtype)) * cfg.lowrank_alpha
            g0_occ = g0_occ + delta_g
        if cfg.has_lowrank_up:
            A_u_x = F.linear(x_occ_f, self.preact.up_A.to(accum_dtype))
            z_u = self.preact.up_z.index_select(0, local_member_indices).to(accum_dtype)
            delta_u = F.linear(z_u * A_u_x, self.preact.up_B.to(accum_dtype)) * cfg.lowrank_alpha
            u0_occ = u0_occ + delta_u

        h_occ = self.act_fn(g0_occ) * u0_occ                                     # [A, d_ff]

        # ---- optional diag/scalar post-code per member -------------------
        if self.code_type == "scalar":
            c = self.code.index_select(0, local_member_indices).to(accum_dtype)  # [A]
            h_occ = h_occ * c.unsqueeze(-1)
        elif self.code_type == "diag":
            c = self.code.index_select(0, local_member_indices).to(accum_dtype)  # [A, d_ff]
            h_occ = h_occ * c
        elif self.code_type == "lowrank":
            c = self.materialize_code(local_member_indices).to(accum_dtype)
            h_occ = h_occ * c

        # ---- per-member down projection (with optional residual) ---------
        y_occ = F.linear(h_occ.to(proj_dtype), self.down_proj.weight, bias=None).to(accum_dtype)
        if cfg.has_lowrank_down:
            R_h = F.linear(h_occ, self.preact.down_R.to(accum_dtype))            # [A, r_d]
            z_d = self.preact.down_z.index_select(0, local_member_indices).to(accum_dtype)
            delta_y = F.linear(z_d * R_h, self.preact.down_L.to(accum_dtype))    # [A, d_model]
            y_occ = y_occ + cfg.down_residual_alpha * delta_y

        # pi-weighted contribution per occurrence, scatter to tokens.
        y_occ = y_occ * gates_f.unsqueeze(-1)                                    # [A, d_model]

        out = torch.zeros(num_unique, self.hidden_size, device=device, dtype=accum_dtype)
        out.index_add_(0, inverse, y_occ)

        # Optional down_proj bias: each occurrence contributes pi_a * bias, summing
        # to G_C(t) * bias per token (matches preact path's gate_sum * bias).
        if self.down_proj.bias is not None:
            gate_sum = torch.zeros(num_unique, device=device, dtype=accum_dtype)
            gate_sum.index_add_(0, inverse, gates_f)
            out = out + gate_sum.unsqueeze(-1) * self.down_proj.bias.to(accum_dtype).unsqueeze(0)

        return out

    def forward_aggregated_preact(
        self,
        unique_x: torch.Tensor,
        gate_sum: torch.Tensor,
        preact_aggs: dict,
    ) -> torch.Tensor:
        """PreAct-STAR forward path on already-aggregated per-token quantities.

        Computes y_C = G_C * (Wbar_d (a' * h) + alpha_d L(q_d * R(a' * h))), where
            h = SiLU(g0 + s_g*beta_g + delta_g) * (u0 + s_u*beta_u + delta_u)
        and a' is the rho-weighted post-code (or 1 if code_type=none).
        """
        assert self.preact is not None
        cfg = self.preact.config
        proj_dtype = self.gate_proj.weight.dtype
        # Always run preact arithmetic in fp32 for stability. The merged
        # gate/up/down projections still execute in their own dtype (typically
        # bf16) and we only cast to/from fp32 at the boundaries.
        accum_dtype = torch.float32

        x_proj = unique_x.to(proj_dtype) if unique_x.dtype != proj_dtype else unique_x
        x_f = unique_x.to(accum_dtype)
        g0 = self.gate_proj(x_proj).to(accum_dtype)
        u0 = self.up_proj(x_proj).to(accum_dtype)

        # bias paths ----------------------------------------------------------
        beta_g = preact_aggs.get("beta_g")
        beta_u = preact_aggs.get("beta_u")
        if beta_g is not None:
            beta_g = beta_g.to(accum_dtype)
            if cfg.normalize_by_preact_std:
                beta_g = beta_g * self.preact.gate_preact_std.to(accum_dtype).unsqueeze(0)
        if beta_u is not None:
            beta_u = beta_u.to(accum_dtype)
            if cfg.normalize_by_preact_std:
                beta_u = beta_u * self.preact.up_preact_std.to(accum_dtype).unsqueeze(0)

        # lowrank paths -------------------------------------------------------
        dual_active = bool(getattr(self.preact, "dual_active", False))
        delta_g = None
        delta_u = None
        if cfg.has_lowrank_gate:
            qg = preact_aggs["q_g"].to(accum_dtype)                              # [N, r]
            sg = F.linear(x_f, self.preact.gate_A.to(accum_dtype))               # [N, r]
            delta_g = F.linear(qg * sg, self.preact.gate_B.to(accum_dtype)) * cfg.lowrank_alpha
            if dual_active and "q_g_b" in preact_aggs:
                qg_b = preact_aggs["q_g_b"].to(accum_dtype)
                sg_b = F.linear(x_f, self.preact.gate_A_b.to(accum_dtype))
                delta_g = delta_g + F.linear(
                    qg_b * sg_b, self.preact.gate_B_b.to(accum_dtype)
                ) * cfg.lowrank_alpha
        if cfg.has_lowrank_up:
            qu = preact_aggs["q_u"].to(accum_dtype)
            su = F.linear(x_f, self.preact.up_A.to(accum_dtype))
            delta_u = F.linear(qu * su, self.preact.up_B.to(accum_dtype)) * cfg.lowrank_alpha
            if dual_active and "q_u_b" in preact_aggs:
                qu_b = preact_aggs["q_u_b"].to(accum_dtype)
                su_b = F.linear(x_f, self.preact.up_A_b.to(accum_dtype))
                delta_u = delta_u + F.linear(
                    qu_b * su_b, self.preact.up_B_b.to(accum_dtype)
                ) * cfg.lowrank_alpha

        g_eff = g0
        u_eff = u0
        if beta_g is not None:
            g_eff = g_eff + beta_g
        if delta_g is not None:
            g_eff = g_eff + delta_g
        if beta_u is not None:
            u_eff = u_eff + beta_u
        if delta_u is not None:
            u_eff = u_eff + delta_u

        h = self.act_fn(g_eff) * u_eff                                       # [N, d_ff]

        # post-code: a'_C(x) * h
        post = preact_aggs.get("post_code_agg")
        if post is None:  # code_type == 'none'
            h_coded = h
        elif self.code_type == "scalar":
            h_coded = h * post.to(accum_dtype).unsqueeze(-1)
        else:  # diag / lowrank
            h_coded = h * post.to(accum_dtype)

        # base down-proj output (in fp32 for residual stability)
        base = F.linear(h_coded.to(proj_dtype), self.down_proj.weight, bias=None).to(accum_dtype)

        # token-adaptive down residual Δy = L(q_d ⊙ R h)
        if cfg.has_lowrank_down:
            tmp = F.linear(h_coded, self.preact.down_R.to(accum_dtype))           # [N, r_d]
            qd = preact_aggs.get("q_d")
            if qd is not None:
                tmp = tmp * qd.to(accum_dtype)
            delta_out = F.linear(tmp, self.preact.down_L.to(accum_dtype))         # [N, d_model]
            base = base + cfg.down_residual_alpha * delta_out
            if dual_active and "q_d_b" in preact_aggs:
                tmp_b = F.linear(h_coded, self.preact.down_R_b.to(accum_dtype))
                qd_b = preact_aggs.get("q_d_b")
                if qd_b is not None:
                    tmp_b = tmp_b * qd_b.to(accum_dtype)
                delta_out_b = F.linear(tmp_b, self.preact.down_L_b.to(accum_dtype))
                base = base + cfg.down_residual_alpha * delta_out_b

        if self.down_proj.bias is not None:
            # apply the per-cluster gate-sum gain to the bias too.
            base = base + self.down_proj.bias.to(accum_dtype).unsqueeze(0)
            # NOTE: in vanilla RESCODE this contributes gate_sum * bias; we
            # also multiply by gate_sum below, so the final contribution is
            # gate_sum * (W_d h + bias) — same shape, semantics preserved.

        out = gate_sum.unsqueeze(-1).to(accum_dtype) * base
        return out

    # -------------------------------------- one-shot convenience for tests
    def forward(
        self,
        flat_hidden_states: torch.Tensor,    # [T, hidden]
        selected_experts: torch.Tensor,      # [T, top_k]
        routing_weights: torch.Tensor,       # [T, top_k]
        accum_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute the cluster contribution y_C(x) for all tokens.

        Tokens that have no member expert in this cluster receive zero output.
        Output shape: [T, hidden].
        """
        T, hidden = flat_hidden_states.shape
        device = flat_hidden_states.device
        member_set = self.local_table.to(device)
        max_eid = member_set.shape[0] - 1

        flat_se = selected_experts.reshape(-1).to(device)
        flat_rw = routing_weights.reshape(-1).to(device)
        token_id = torch.arange(T, device=device).unsqueeze(-1).expand_as(selected_experts).reshape(-1)

        valid_mask = (flat_se >= 0) & (flat_se <= max_eid)
        if valid_mask.any():
            local_lookup = torch.where(
                valid_mask,
                member_set[flat_se.clamp(min=0, max=max_eid)],
                torch.full_like(flat_se, -1),
            )
        else:
            local_lookup = torch.full_like(flat_se, -1)
        keep = local_lookup >= 0

        if not keep.any():
            return flat_hidden_states.new_zeros(T, hidden)

        token_kept = token_id[keep]
        local_kept = local_lookup[keep]
        gate_kept = flat_rw[keep]

        unique_tokens, inverse = torch.unique(token_kept, return_inverse=True)
        N = unique_tokens.shape[0]

        unique_x = flat_hidden_states.index_select(0, unique_tokens)

        if self.preact is None:
            code_agg, gate_sum = self.aggregate(
                local_member_indices=local_kept,
                gates=gate_kept,
                inverse=inverse,
                num_unique=N,
                accum_dtype=accum_dtype,
            )
            out = self.forward_aggregated(unique_x, code_agg, gate_sum)
        elif self.preact.config.paper_postact_aggregation:
            # Paper Eq. (4): per-expert SwiGLU then pi-weighted sum.
            out = self.forward_postact(
                unique_x=unique_x,
                local_member_indices=local_kept,
                gates=gate_kept,
                inverse=inverse,
                num_unique=N,
                accum_dtype=accum_dtype,
            )
        else:
            # preact-aware path (router-conditioned pre-activation)
            gates_f = gate_kept.to(accum_dtype)
            gate_sum = torch.zeros(N, device=device, dtype=accum_dtype)
            gate_sum.index_add_(0, inverse, gates_f)
            rho = gates_f / (gate_sum[inverse] + self.preact.config.eps)

            preact_aggs = self._preact_aggregate(
                local_member_indices=local_kept,
                rho=rho,
                inverse=inverse,
                num_unique=N,
                accum_dtype=accum_dtype,
            )
            out = self.forward_aggregated_preact(
                unique_x=unique_x,
                gate_sum=gate_sum,
                preact_aggs=preact_aggs,
            )

        full_out = flat_hidden_states.new_zeros(T, hidden)
        full_out.index_add_(0, unique_tokens, out.to(full_out.dtype))
        return full_out
