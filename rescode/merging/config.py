from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple


LOWRANK_TARGET_ORDER = ("gate", "up", "down")


def parse_lowrank_target(target: str) -> Tuple[str, ...]:
    """Parse a token-adaptive lowrank residual target string.

    Accepted forms include ``gate``, ``up``, ``down``, ``gate_up``,
    ``gate_up_down``, ``gate,up,down``, ``gate+up+down``, and ``all``.
    The return value is canonicalized in gate/up/down order.
    """
    raw = str(target).strip().lower()
    normalized = (
        raw.replace("+", "_")
        .replace(",", "_")
        .replace("-", "_")
        .replace(" ", "")
    )
    if normalized == "all":
        normalized = "gate_up_down"
    parts = [p for p in normalized.split("_") if p]
    unknown = sorted(set(parts) - set(LOWRANK_TARGET_ORDER))
    if unknown or not parts:
        legal = {
            "gate", "up", "down",
            "gate_up", "gate_down", "up_down", "gate_up_down",
            "gate,up,down", "gate+up+down", "all",
        }
        raise ValueError(
            f"[RESCODE] preact lowrank_target must be a combination of "
            f"gate/up/down (examples: {sorted(legal)}), got {target}."
        )
    return tuple(t for t in LOWRANK_TARGET_ORDER if t in parts)


def canonical_lowrank_target(target: str) -> str:
    return "_".join(parse_lowrank_target(target))


@dataclass
class PreActStarConfig:
    """Optional router-conditioned preactivation modulation on top of RESCODE.

    The cluster MLP becomes:
        rho_j(x) = r_j(x) / G_C(x)               (cluster-internal ratio)
        beta_g(x) = sum_j rho_j(x) b_g_j         (preact gate bias)
        beta_u(x) = sum_j rho_j(x) b_u_j         (preact up bias)
        delta_g(x) = B_g (q_g(x) ⊙ A_g x)        (router-conditioned lowrank)
        delta_u(x) = B_u (q_u(x) ⊙ A_u x)
        delta_d(x) = L_d (q_d(x) ⊙ R_d h_C(x))
        g'(x) = g0(x) + s_g ⊙ beta_g(x) + alpha * delta_g(x)
        u'(x) = u0(x) + s_u ⊙ beta_u(x) + alpha * delta_u(x)
        h_C(x) = SiLU(g'(x)) ⊙ u'(x)
        y_C(x) = G_C(x) * (Wbar_d h_C(x) + alpha_d * delta_d(x))

    All new parameters are zero-init (or LoRA-style with B=0) so the cluster
    output is exactly equal to vanilla RESCODE at construction time. This
    means `enabled=True` with no fitting must reproduce the existing model
    bit-for-bit (modulo a single backward-compat output multiply).

    `enabled=False` (default) makes this whole struct a no-op; `RescodeConfig.preact_star`
    is `None` by default, so the existing pipeline keeps the original code
    paths.
    """

    enabled: bool = False

    # Primary / secondary knobs: preact bias
    gate_bias: bool = False
    up_bias: bool = False

    # Token-adaptive low-rank residual targets. gate/up act on preactivations;
    # down acts on the post-SwiGLU hidden state before the down projection.
    lowrank_preact: bool = False
    lowrank_target: str = "gate_up"     # gate | up | down | gate_up | ... | all
    lowrank_rank: int = 16
    lowrank_alpha: float = 1.0
    lowrank_init: str = "lora_zero_B"   # lora_zero_B | covariance | expert_svd_align
    lowrank_init_std: float = 1.0e-3
    lowrank_init_alignment: str = "hungarian"  # hungarian | greedy (only for expert_svd_align)

    # ---------------- Dual residual (split a cluster into 2 sub-residuals) ----
    # When enabled, each cluster of size >= 2 holds two parallel rank-r residuals
    # (gate/up/down). Members are partitioned into sub-cluster A and B; the
    # forward computes
    #   delta_g = α [ B^A (q^A_g ⊙ A^A x) + B^B (q^B_g ⊙ A^B x) ]
    # where q^A = Σ_{j∈A} ρ_j z^A_j and q^B is analogous (ρ normalized over the
    # whole original cluster).
    # The actual per-cluster sub-cluster assignment is computed externally and
    # passed in via `precomputed_init["sub_assignment"]` (a [num_members] long
    # tensor in {0,1}). `lowrank_split_method` is informational only — the
    # resolution to an assignment happens in the caller.
    lowrank_dual: bool = False
    lowrank_split_method: str = "balanced"  # balanced | kmeans | coactivation
    lowrank_split_min_size: int = 3         # only split clusters of size >= this

    # ---------------- Cluster forward aggregation choice ---------------------
    # When True (default), the cluster runs paper Eq. (4): for each cluster
    # member j ∈ C ∩ T(x), evaluate SwiGLU with W_hat_j = W̄ + B·diag(z_j)·A and
    # take a π_j-weighted sum. Cost is K SwiGLU evaluations per cluster per
    # token. When False, the cluster runs the cheaper PreAct path: rho-weighted
    # aggregation of per-member residual params and a single SwiGLU evaluation
    # per cluster per token (first-order approximation under nonlinear SwiGLU).
    paper_postact_aggregation: bool = True

    # Down-projection token-adaptive low-rank residual.
    # Kept as an explicit bool for backward compatibility; target strings that
    # include "down" enable it automatically.
    down_residual: bool = False
    down_residual_rank: int = 16
    down_residual_alpha: float = 1.0
    # covariance (default): same freq/uniform-weighted covariance recipe as
    #   gate/up — the residual starts at the rank-r projection of the
    #   per-expert down residual.
    # zero_L (L = 0, R ~ small) | zero_R (R = 0, L ~ small): Δ = 0 at init.
    down_residual_init: str = "covariance"
    down_residual_init_std: float = 1.0e-3
    down_residual_target_clusters: str = "non_singleton"  # non_singleton | all | largest_only | top_error

    # router-conditioning behavior
    use_cluster_ratio: bool = True       # always use rho_j = r_j / G_C; raw r_j is forbidden
    eps: float = 1.0e-6

    # preact std normalization
    normalize_by_preact_std: bool = True
    preact_std_eps: float = 1.0e-6

    # bias clipping / regularization
    bias_clip_min: Optional[float] = -4.0
    bias_clip_max: Optional[float] = 4.0
    bias_l2_lambda: float = 1.0e-4
    lowrank_l2_lambda: float = 1.0e-4
    down_l2_lambda: float = 1.0e-4
    grad_clip_norm: Optional[float] = 1.0

    # learning-rate per parameter group (used when fitting preact)
    lr_post_code: float = 1.0e-2
    lr_preact_bias: float = 1.0e-2
    lr_preact_lowrank: float = 3.0e-3
    lr_down_residual: float = 3.0e-3

    # which preact params are trainable in the fit pass
    train_post_codes: bool = False       # explicit 'old code' fit toggle
    train_preact_bias: bool = True
    train_preact_lowrank: bool = True
    train_down_residual: bool = True

    # diagnostics
    report_preact_stats: bool = True
    report_giant_cluster_summary: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreActStarConfig":
        known = {f for f in cls.__dataclass_fields__}
        d2 = {k: v for k, v in d.items() if k in known}
        return cls(**d2)

    def __post_init__(self):
        targets = parse_lowrank_target(self.lowrank_target)
        self.lowrank_target = "_".join(targets)
        if self.lowrank_preact and "down" in targets:
            self.down_residual = True
        legal_inits = {"lora_zero_B", "covariance", "expert_svd_align"}
        if self.lowrank_init not in legal_inits:
            raise ValueError(
                f"[RESCODE] preact lowrank_init must be one of {legal_inits}, "
                f"got {self.lowrank_init}."
            )
        legal_aligns = {"hungarian", "greedy"}
        if self.lowrank_init_alignment not in legal_aligns:
            raise ValueError(
                f"[RESCODE] preact lowrank_init_alignment must be one of {legal_aligns}, "
                f"got {self.lowrank_init_alignment}."
            )
        legal_down_inits = {"covariance", "zero_L", "zero_R"}
        if self.down_residual_init not in legal_down_inits:
            raise ValueError(
                f"[RESCODE] preact down_residual_init must be one of {legal_down_inits}, "
                f"got {self.down_residual_init}."
            )
        if not self.use_cluster_ratio:
            raise ValueError(
                "[RESCODE] preact_star.use_cluster_ratio=False is not supported. "
                "Cluster gate must be applied at output scale and rho = r/G_C "
                "must be used inside the preactivation modulation, otherwise the "
                "router gate is double-counted."
            )

    @property
    def has_lowrank_gate(self) -> bool:
        return self.lowrank_preact and "gate" in parse_lowrank_target(self.lowrank_target)

    @property
    def has_lowrank_up(self) -> bool:
        return self.lowrank_preact and "up" in parse_lowrank_target(self.lowrank_target)

    @property
    def has_lowrank_down(self) -> bool:
        return self.down_residual

    def has_any_trainable_preact_param(self) -> bool:
        return (
            self.gate_bias or self.up_bias
            or self.lowrank_preact or self.down_residual
        )


@dataclass
class RescodeConfig:
    """Configuration for RESCODE: Router-Preserving Coded Meta-Expert Merging.

    `code_type=none` makes the model mathematically equivalent to RESCODE merged
    output and is used for sanity testing. `code_type=diag` is the main method.

    `preact_star` is an OPTIONAL router-conditioned preactivation modulation
    that lives in front of SiLU. It defaults to None (disabled) so the existing
    pipeline is byte-identical.
    """

    enabled: bool = True

    # -- cluster source --------------------------------------------------------
    cluster_source: str = "rescode"
    cluster_path: Optional[str] = None

    # -- code type -------------------------------------------------------------
    code_type: str = "diag"            # none | scalar | diag | lowrank
    code_init: str = "ones"
    freeze_code: bool = False
    code_rank: int = 16

    # -- fitting ---------------------------------------------------------------
    fit_codes: bool = True
    fit_steps: int = 300
    fit_lr: float = 1.0e-2
    fit_weight_decay: float = 0.0
    ridge_lambda: float = 1.0e-4
    max_fit_tokens_per_layer: int = 32768
    max_fit_tokens_per_cluster: int = 32768
    min_fit_tokens: int = 128

    # -- clipping / stability --------------------------------------------------
    code_clip_min: Optional[float] = 0.0
    code_clip_max: Optional[float] = 2.0
    fit_dtype: str = "fp32"

    # -- calibration -----------------------------------------------------------
    calib_dataset: str = "c4"
    calib_nseq: int = 32
    calib_seqlen: int = 2048
    calib_batch_size: int = 1

    # -- save / load -----------------------------------------------------------
    save_rescode_path: Optional[str] = None
    load_rescode_path: Optional[str] = None

    # -- diagnostics -----------------------------------------------------------
    report_reconstruction_error: bool = True
    report_cluster_stats: bool = True

    # -- optimizer / scheduler -------------------------------------------------
    optimizer: str = "adamw"            # adamw | adam | sgd
    lr_scheduler: str = "constant"       # constant | cosine | linear
    warmup_steps: int = 0
    log_every: int = 25                  # metric log cadence (in optimizer steps)

    # -- joint LM-loss fitting --------------------------------------------------
    fit_mode: str = "layerwise"          # layerwise (default) | joint
    lambda_lm: float = 1.0               # weight on LM cross-entropy term
    lambda_mse: float = 1.0              # weight on per-cluster MSE term
    joint_epochs: int = 5                # full passes over calib loader
    joint_grad_accum: int = 1            # gradient accumulation

    # -- preact-star (NEW, OPTIONAL) -------------------------------------------
    # `None` means: do nothing; legacy RESCODE stays bit-identical.
    preact_star: Optional[PreActStarConfig] = None

    # -- meta ------------------------------------------------------------------
    seed: int = 0

    def __post_init__(self):
        legal_code_types = {"none", "scalar", "diag", "lowrank"}
        if self.code_type not in legal_code_types:
            raise ValueError(
                f"[RESCODE] code_type must be one of {legal_code_types}, "
                f"got {self.code_type}."
            )
        if self.fit_dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError(
                f"[RESCODE] fit_dtype must be one of fp32/fp16/bf16, got {self.fit_dtype}."
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict already handles nested dataclasses; keep explicit None.
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RescodeConfig":
        known = {f for f in cls.__dataclass_fields__}
        d2 = {k: v for k, v in d.items() if k in known}
        # Re-hydrate the nested preact_star block if present.
        ps = d2.get("preact_star", None)
        if isinstance(ps, dict):
            d2["preact_star"] = PreActStarConfig.from_dict(ps)
        return cls(**d2)

    @property
    def preact_enabled(self) -> bool:
        return self.preact_star is not None and self.preact_star.enabled
