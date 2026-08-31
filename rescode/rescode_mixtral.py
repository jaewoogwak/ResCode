# -*- coding: utf-8 -*-
"""End-to-end RESCODE pipeline for Mixtral, mirroring rescode_qwen.py.

Supports the same PreAct-STAR + joint-LM fitting paths as the Qwen entry so we
can transfer the recipes that beat RESCODE on Qwen 30e to Mixtral 6e/4e.
"""

import gc
import os
import time
from typing import Optional

import torch
from fire import Fire
from transformers import MixtralForCausalLM, AutoTokenizer

from rescode.evaluation import evaluate_fewshot, get_calib_dataloder
from rescode.merging.grouping_mixtral import (
    ExpertsGrouperForMixtral,
    merge_by_groups_with_usage_weighted,
    merge_by_groups_with_saliency_weighted,
    merge_by_groups_within_and_across_models,
)

from rescode.merging import (
    RescodeConfig,
    PreActStarConfig,
    parse_lowrank_target,
    build_cluster_maps_from_group_state,
    convert_mixtral_to_rescode,
    save_rescode_state,
    load_rescode_state,
    log_cluster_stats,
    log_code_statistics,
    log_preact_statistics,
    log_giant_cluster_summary,
    collect_teacher_signals_mixtral,
    collect_preact_stats,
    fit_all_codes,
    fit_codes_joint_lm,
    freeze_non_code_parameters,
    report_eval_summary,
)


@torch.no_grad()
def _collect_coactivation_matrices_mixtral(
    model,
    dataloader,
    num_layers: int,
    num_experts: int,
    top_k: int,
    max_batches: int = 16,
):
    """Brief calibration pass on the original Mixtral MoE to count co-activations.
    Returns dict[layer_idx -> [num_experts, num_experts] float tensor]."""
    import torch.nn.functional as F
    coact = {
        layer_idx: torch.zeros(num_experts, num_experts, dtype=torch.float32)
        for layer_idx in range(num_layers)
    }
    handles = []
    captured = {}

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            captured[layer_idx] = output.detach()
        return hook

    for layer_idx in range(num_layers):
        moe = model.model.layers[layer_idx].block_sparse_moe
        if not hasattr(moe, "gate"):
            continue
        h = moe.gate.register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    model.eval()
    n_done = 0
    try:
        for batch in dataloader:
            if n_done >= max_batches:
                break
            batch = {k: v.cuda() for k, v in batch.items() if hasattr(v, 'cuda')}
            if "labels" in batch:
                batch.pop("labels")
            captured.clear()
            try:
                model(**batch)
            except Exception as e:
                print(f"[RESCODE-Mixtral][coact] forward batch {n_done} failed: {e}")
                break
            for layer_idx, logits in captured.items():
                rw = F.softmax(logits.float(), dim=-1)
                _, sel = torch.topk(rw, top_k, dim=-1)
                onehot = F.one_hot(sel, num_classes=num_experts).any(dim=1).float()
                coact[layer_idx] += (onehot.T @ onehot).cpu()
            n_done += 1
    finally:
        for h in handles:
            h.remove()
    print(f"[RESCODE-Mixtral][coact] done: {n_done} batches processed.")
    return coact


def _build_dataloader(
    tokenizer,
    n_sentences,
    batch_size,
    seqlen=2048,
    seed=42,
    data_path: Optional[str] = None,
    shuffle_docs: bool = True,
):
    return get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=seqlen,
        n_blocks_for_stat=n_sentences,
        batch_size=batch_size,
        num_workers=4,
        seed=seed,
        data_path=data_path,
        shuffle_docs=shuffle_docs,
    )


def run_rescode_mixtral(
    task: str = "winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte",
    num_average_groups: int = 4,
    model_name: str = "mistralai/Mixtral-8x7B-v0.1",
    similarity_base: str = "expert-output",
    # excl-topk specific (only when similarity_base="expert-output-excl-topk")
    exclude_top_percent: float = 15.0,
    excl_mode: str = "per-layer",
    # RESCODE specific
    rescode_delta: float = 1.0e-3,
    cluster: str = "hierarchical",
    linkage: str = "average",
    descriptor_tokens: str = "all",  # "all" = same tokens for every expert (paper) | "routed"
    hc_merge: str = "freq",
    hc_mode: str = "normal",
    n_sentences: int = 32,
    cluster_n_sentences: int = 32,   # clustering / preact-std calib (always small)
    train_data_path: str = "rescode/data/c4-train.00000-of-01024.json",
    val_data_path: str = "PATH/TO/YOUR/val-calib.json",
    train_batch_size: int = 2,
    eval_batch_size: int = 16,
    start_layer: int = 0,
    group_limit: int = 4,
    data_limit: int = 50000,
    code_type: str = "diag",
    code_init: str = "ones",
    freeze_code: bool = False,
    code_rank: int = 16,
    fit_codes: bool = True,
    fit_steps: int = 300,
    fit_lr: float = 1.0e-2,
    fit_weight_decay: float = 0.0,
    ridge_lambda: float = 1.0e-4,
    code_clip_min: float = 0.0,
    code_clip_max: float = 2.0,
    fit_dtype: str = "fp32",
    max_fit_tokens_per_layer: int = 32768,
    max_fit_tokens_per_cluster: int = 32768,
    min_fit_tokens: int = 128,
    save_rescode_path: Optional[str] = None,
    load_rescode_path: Optional[str] = None,
    # optimizer / scheduler
    optimizer: str = "adamw",
    lr_scheduler: str = "constant",
    warmup_steps: int = 0,
    log_every: int = 25,
    # joint LM-loss fitting
    fit_mode: str = "layerwise",
    lambda_lm: float = 1.0,
    lambda_mse: float = 1.0,
    joint_epochs: int = 5,
    joint_grad_accum: int = 1,
    joint_val_n_sentences: int = 8,
    joint_val_every: int = 10,
    joint_val_seed: int = 4321,
    # PreAct-STAR (off by default; backward compatible)
    preact_star: bool = False,
    preact_mode: str = "gate_bias",     # gate_bias | gate_up_bias | token_adaptive_lowrank | lowrank_preact | lowrank_preact_plus_down | custom
    preact_gate_bias: Optional[bool] = None,
    preact_up_bias: Optional[bool] = None,
    preact_lowrank: Optional[bool] = None,
    preact_lowrank_target: str = "gate_up",
    preact_lowrank_rank: int = 16,
    preact_lowrank_alpha: float = 1.0,
    preact_lowrank_init_std: float = 1.0e-3,
    preact_lowrank_init: str = "lora_zero_B",       # lora_zero_B | covariance | expert_svd_align
    preact_lowrank_init_alignment: str = "hungarian",  # hungarian | greedy (only for expert_svd_align)
    # Dual residual: split each cluster into 2 sub-clusters with parallel residuals.
    preact_lowrank_dual: bool = False,
    preact_lowrank_split_method: str = "balanced",  # balanced | kmeans | coactivation
    preact_lowrank_split_min_size: int = 3,
    paper_postact_aggregation: bool = True,         # paper Eq.(4) per-expert SwiGLU (default)
    preact_coact_n_batches: int = 16,
    preact_down_residual: Optional[bool] = None,
    preact_down_residual_rank: int = 16,
    preact_down_residual_alpha: float = 1.0,
    preact_down_residual_init: str = "covariance",   # covariance | zero_L | zero_R
    preact_normalize_by_std: bool = True,
    preact_bias_clip_min: float = -4.0,
    preact_bias_clip_max: float = 4.0,
    preact_bias_l2_lambda: float = 1.0e-4,
    preact_lowrank_l2_lambda: float = 1.0e-4,
    preact_down_l2_lambda: float = 1.0e-4,
    preact_grad_clip_norm: Optional[float] = 1.0,
    preact_lr_post_code: float = 1.0e-2,
    preact_lr_bias: float = 1.0e-2,
    preact_lr_lowrank: float = 3.0e-3,
    preact_lr_down: float = 3.0e-3,
    fit_preact: bool = False,
    train_post_codes: bool = False,
    # teacher mode: see rescode_qwen.py for description.
    teacher_mode: str = "online",
    # output
    output_path: Optional[str] = None,
    result_path: Optional[str] = None,
    do_eval: bool = True,
    num_fewshot: int = 0,
    seed: int = 0,
    calib_seed: int = 42,   # calibration-sampling seed (varied jointly with `seed` for multi-seed runs)
):
    print(
        "[RESCODE-Mixtral] model=", model_name,
        " num_average_groups=", num_average_groups,
        " code_type=", code_type,
        " fit_codes=", fit_codes,
        " preact_star=", preact_star,
        " preact_mode=", preact_mode if preact_star else "(disabled)",
    )
    torch.manual_seed(seed)

    # ---- Build PreActStarConfig (None when disabled) ----
    preact_cfg: Optional[PreActStarConfig] = None
    if preact_star:
        gate_b = preact_gate_bias
        up_b = preact_up_bias
        lr_p = preact_lowrank
        d_res = preact_down_residual
        targets = set(parse_lowrank_target(preact_lowrank_target))
        if preact_mode == "gate_bias":
            gate_b = True if gate_b is None else gate_b
            up_b = False if up_b is None else up_b
            lr_p = False if lr_p is None else lr_p
            d_res = False if d_res is None else d_res
        elif preact_mode == "gate_up_bias":
            gate_b = True if gate_b is None else gate_b
            up_b = True if up_b is None else up_b
            lr_p = False if lr_p is None else lr_p
            d_res = False if d_res is None else d_res
        elif preact_mode == "lowrank_preact":
            gate_b = False if gate_b is None else gate_b
            up_b = False if up_b is None else up_b
            lr_p = True if lr_p is None else lr_p
            d_res = ("down" in targets) if d_res is None else d_res
        elif preact_mode == "lowrank_preact_plus_down":
            gate_b = False if gate_b is None else gate_b
            up_b = False if up_b is None else up_b
            lr_p = True if lr_p is None else lr_p
            d_res = True if d_res is None else d_res
        elif preact_mode in {
            "token_adaptive_lowrank",
            "token_adaptive_lowrank_residual",
            "talr",
        }:
            gate_b = False if gate_b is None else gate_b
            up_b = False if up_b is None else up_b
            lr_p = True if lr_p is None else lr_p
            d_res = ("down" in targets) if d_res is None else d_res
        elif preact_mode == "custom":
            gate_b = bool(gate_b) if gate_b is not None else False
            up_b = bool(up_b) if up_b is not None else False
            lr_p = bool(lr_p) if lr_p is not None else False
            d_res = bool(d_res) if d_res is not None else False
        else:
            raise ValueError(f"[RESCODE-Mixtral] unknown preact_mode={preact_mode}")

        preact_cfg = PreActStarConfig(
            enabled=True,
            gate_bias=bool(gate_b),
            up_bias=bool(up_b),
            lowrank_preact=bool(lr_p),
            lowrank_target=preact_lowrank_target,
            lowrank_rank=preact_lowrank_rank,
            lowrank_alpha=preact_lowrank_alpha,
            lowrank_init_std=preact_lowrank_init_std,
            lowrank_init=preact_lowrank_init,
            lowrank_init_alignment=preact_lowrank_init_alignment,
            down_residual=bool(d_res),
            down_residual_rank=preact_down_residual_rank,
            down_residual_alpha=preact_down_residual_alpha,
            down_residual_init=preact_down_residual_init,
            normalize_by_preact_std=preact_normalize_by_std,
            bias_clip_min=preact_bias_clip_min,
            bias_clip_max=preact_bias_clip_max,
            bias_l2_lambda=preact_bias_l2_lambda,
            lowrank_l2_lambda=preact_lowrank_l2_lambda,
            down_l2_lambda=preact_down_l2_lambda,
            grad_clip_norm=preact_grad_clip_norm,
            lr_post_code=preact_lr_post_code,
            lr_preact_bias=preact_lr_bias,
            lr_preact_lowrank=preact_lr_lowrank,
            lr_down_residual=preact_lr_down,
            train_post_codes=bool(train_post_codes),
            train_preact_bias=bool(fit_preact),
            train_preact_lowrank=bool(fit_preact),
            train_down_residual=bool(fit_preact),
            lowrank_dual=bool(preact_lowrank_dual),
            lowrank_split_method=str(preact_lowrank_split_method),
            lowrank_split_min_size=int(preact_lowrank_split_min_size),
            paper_postact_aggregation=bool(paper_postact_aggregation),
        )

    star_cfg = RescodeConfig(
        cluster_source="rescode",
        code_type=code_type,
        code_init=code_init,
        freeze_code=freeze_code,
        code_rank=code_rank,
        fit_codes=fit_codes,
        fit_steps=fit_steps,
        fit_lr=fit_lr,
        fit_weight_decay=fit_weight_decay,
        ridge_lambda=ridge_lambda,
        code_clip_min=code_clip_min,
        code_clip_max=code_clip_max,
        fit_dtype=fit_dtype,
        max_fit_tokens_per_layer=max_fit_tokens_per_layer,
        max_fit_tokens_per_cluster=max_fit_tokens_per_cluster,
        min_fit_tokens=min_fit_tokens,
        save_rescode_path=save_rescode_path,
        load_rescode_path=load_rescode_path,
        seed=seed,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        warmup_steps=warmup_steps,
        log_every=log_every,
        fit_mode=fit_mode,
        lambda_lm=lambda_lm,
        lambda_mse=lambda_mse,
        joint_epochs=joint_epochs,
        joint_grad_accum=joint_grad_accum,
        preact_star=preact_cfg,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = MixtralForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    # Two dataloaders:
    #   - cluster_dataloader: small calib for clustering, merge, preact std calib.
    #   - dataloader: full train data for teacher signals + joint LM training.
    dataloader = _build_dataloader(
        tokenizer,
        n_sentences,
        train_batch_size,
        seed=calib_seed,
        data_path=train_data_path,
    )
    cluster_dataloader = _build_dataloader(
        tokenizer,
        cluster_n_sentences,
        train_batch_size,
        seed=calib_seed,
        data_path=train_data_path,
    )

    grouper = ExpertsGrouperForMixtral(
        config=model.config,
        similarity_base=similarity_base,
        start_layer=start_layer,
        group_limit=group_limit,
        data_limit=data_limit,
        cluster=cluster,
        linkage=linkage,
        descriptor_tokens=descriptor_tokens,
    )

    print(f"[RESCODE-Mixtral] computing RESCODE clusters (calib n={cluster_n_sentences})...")
    t0 = time.time()
    needs_freq = (hc_merge == "freq") or (similarity_base == "expert-output-excl-topk")
    if needs_freq:
        grouper.compute_all_usages(model, cluster_dataloader)
    dom_experts = grouper.cluster_experts(
        model=model, dataloader=cluster_dataloader, num_groups=num_average_groups,
        exclude_top_percent=float(exclude_top_percent),
        excl_mode=str(excl_mode),
    )
    print(f"[RESCODE-Mixtral] clustering took {time.time() - t0:.1f}s")

    cluster_maps = build_cluster_maps_from_group_state(
        group_state_dict=grouper.group_state_dict(),
        layer_indices=grouper.sparse_layer_indices,
        ffn_name_template="model.layers.{layer_idx}.block_sparse_moe",
    )
    if star_cfg.report_cluster_stats:
        log_cluster_stats(cluster_maps)

    # 2) Collect teacher signals BEFORE merging.
    # Skip when lambda_mse=0 — the buffers go unused and at large N_SENTENCES
    # they easily exceed container memory.
    teacher_buffers = None
    will_fit_preact = (
        preact_cfg is not None and (
            preact_cfg.train_preact_bias
            or preact_cfg.train_preact_lowrank
            or preact_cfg.train_down_residual
        )
    )
    needs_teacher = star_cfg.fit_codes or (will_fit_preact and float(lambda_mse) > 0.0)
    use_online_teacher = (
        needs_teacher
        and not star_cfg.fit_codes
        and str(teacher_mode).lower() == "online"
    )
    if needs_teacher and not use_online_teacher:
        print("[RESCODE-Mixtral] collecting teacher signals from original model (offline)...")
        teacher_buffers = collect_teacher_signals_mixtral(
            model=model,
            cluster_maps=cluster_maps,
            dataloader=dataloader,
            config=star_cfg,
        )
    elif use_online_teacher:
        print("[RESCODE-Mixtral] online teacher mode — skipping offline buffer collection.")
    elif will_fit_preact:
        print("[RESCODE-Mixtral] skipping teacher (lambda_mse=0).")
    val_loader = None
    val_teacher_buffers = None
    # Build val_loader whenever joint training will run — LM-only val loss is
    # useful even without teacher (lambda_mse=0).
    # An empty or missing val_data_path disables joint validation (and the
    # best-val checkpoint restore) instead of crashing at load time.
    has_val_data = bool(val_data_path) and os.path.isfile(str(val_data_path))
    wants_joint_val = (
        fit_mode == "joint"
        and (star_cfg.fit_codes or will_fit_preact)
        and joint_val_n_sentences
        and joint_val_n_sentences > 0
    )
    if wants_joint_val and not has_val_data:
        print(f"[RESCODE-Mixtral] val_data_path missing ({val_data_path!r}) — joint validation disabled.")
    if wants_joint_val and has_val_data:
        val_loader = _build_dataloader(
            tokenizer,
            joint_val_n_sentences,
            train_batch_size,
            seed=joint_val_seed,
            data_path=val_data_path,
            shuffle_docs=False,
        )
        if not needs_teacher:
            print("[RESCODE-Mixtral] val_loader built; teacher val collection skipped (lambda_mse=0).")
        elif use_online_teacher:
            print("[RESCODE-Mixtral] online val teacher mode — skipping offline val buffer collection.")
        else:
            print("[RESCODE-Mixtral] collecting validation teacher signals from original model (offline)...")
            val_teacher_buffers = collect_teacher_signals_mixtral(
                model=model,
                cluster_maps=cluster_maps,
                dataloader=val_loader,
                config=star_cfg,
            )

    # 2.6) Online teacher: deepcopy unmerged model BEFORE merging.
    online_teacher = None
    val_online_teacher = None
    if use_online_teacher:
        import copy
        from rescode.merging import OnlineTeacher
        print("[RESCODE-Mixtral] cloning original model for online teacher...")
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        online_teacher = OnlineTeacher(teacher_model, cluster_maps, backbone="mixtral")
        if val_loader is not None:
            val_online_teacher = online_teacher

    # 2.6) Optional: dual-residual sub-cluster assignment per cluster.
    split_assignments = None
    if preact_cfg is not None and preact_cfg.lowrank_dual:
        from rescode.merging.split_methods import (
            compute_split_balanced, compute_split_kmeans, compute_split_coactivation,
        )
        method = preact_cfg.lowrank_split_method
        min_size = int(preact_cfg.lowrank_split_min_size)
        print(f"[RESCODE-Mixtral] computing dual-residual split assignments "
              f"(method={method}, min_size={min_size})...")
        split_assignments = {}

        coact_per_layer = None
        if method == "coactivation":
            print(f"[RESCODE-Mixtral] collecting routing co-activation "
                  f"(n_batches={preact_coact_n_batches})...")
            coact_per_layer = _collect_coactivation_matrices_mixtral(
                model=model,
                dataloader=cluster_dataloader,
                num_layers=model.config.num_hidden_layers,
                num_experts=model.config.num_local_experts,
                top_k=model.config.num_experts_per_tok,
                max_batches=int(preact_coact_n_batches),
            )

        for layer_idx, cmap in cluster_maps.items():
            sa_layer = {}
            for cluster_id, member_ids in cmap.cluster_to_experts.items():
                if len(member_ids) < min_size:
                    continue
                if method == "balanced":
                    # Balanced split needs hierarchical merge tree per layer.
                    # Mixtral's small expert pool (8) makes this rarely useful; skip.
                    continue
                elif method == "kmeans":
                    moe = model.model.layers[layer_idx].block_sparse_moe
                    rows = []
                    for mid in member_ids:
                        e = moe.experts[mid]
                        rows.append(torch.cat([e.w1.weight.flatten(),
                                               e.w3.weight.flatten()], dim=0))
                    X = torch.stack(rows).float().cpu()
                    asg = compute_split_kmeans(member_ids, X)
                elif method == "coactivation":
                    if coact_per_layer is None or layer_idx not in coact_per_layer:
                        continue
                    M = coact_per_layer[layer_idx]
                    idx = torch.as_tensor(member_ids, dtype=torch.long)
                    sub_M = M[idx][:, idx]
                    asg = compute_split_coactivation(member_ids, sub_M)
                else:
                    raise ValueError(f"unknown split_method={method}")
                if len(set(int(x) for x in asg)) < 2:
                    continue
                sa_layer[cluster_id] = list(int(x) for x in asg)
            if sa_layer:
                split_assignments[layer_idx] = sa_layer
        n_dual_total = sum(len(v) for v in split_assignments.values())
        print(f"[RESCODE-Mixtral] dual splits computed: {n_dual_total} clusters across "
              f"{len(split_assignments)} layers (method={method})")

    # 2.7) Optional: precompute spectral / SVD-aligned init for lowrank_preact
    # using the ORIGINAL per-expert weights (before merging overwrites them).
    preact_init_data = None
    wants_gate_up_init = (
        preact_cfg is not None
        and preact_cfg.lowrank_preact
        and bool(set(parse_lowrank_target(preact_cfg.lowrank_target)) & {"gate", "up"})
        and preact_cfg.lowrank_init in {"covariance", "expert_svd_align"}
    )
    wants_down_init = (
        preact_cfg is not None
        and preact_cfg.down_residual
        and preact_cfg.down_residual_init == "covariance"
    )
    if wants_gate_up_init or wants_down_init:
        from rescode.merging import compute_lowrank_preact_init_data
        print(f"[RESCODE-Mixtral] computing lowrank_preact init "
              f"(method={preact_cfg.lowrank_init}, target={preact_cfg.lowrank_target}, "
              f"rank={preact_cfg.lowrank_rank}, α={preact_cfg.lowrank_alpha}, "
              f"down_init={preact_cfg.down_residual_init})...")
        preact_init_data = compute_lowrank_preact_init_data(
            model=model,
            cluster_maps=cluster_maps,
            freq_dict=grouper.usage_frequency_state_dict() if hc_merge == "freq" else None,
            rank=int(preact_cfg.lowrank_rank),
            alpha=float(preact_cfg.lowrank_alpha),
            target=preact_cfg.lowrank_target,
            method=preact_cfg.lowrank_init,
            alignment=preact_cfg.lowrank_init_alignment,
            backbone="mixtral",
            split_assignments=split_assignments,
            init_gate_up=wants_gate_up_init,
            init_down=wants_down_init,
            down_rank=int(preact_cfg.down_residual_rank),
            down_alpha=float(preact_cfg.down_residual_alpha),
        )

    # Inject sub_assignment even if init=lora_zero_B
    if preact_cfg is not None and preact_cfg.lowrank_dual and split_assignments is not None:
        if preact_init_data is None:
            preact_init_data = {}
        for layer_idx, sa_layer in split_assignments.items():
            preact_init_data.setdefault(layer_idx, {})
            for cluster_id, asg in sa_layer.items():
                ce = preact_init_data[layer_idx].setdefault(cluster_id, {})
                ce["sub_assignment"] = asg

    # 3) RESCODE merging.
    if hc_merge == "freq":
        print("[RESCODE-Mixtral] merging via usage-frequency-weighted averaging...")
        model = merge_by_groups_with_usage_weighted(
            model, grouper=grouper,
            merging_layers=list(range(start_layer, model.config.num_hidden_layers)),
        )
    elif hc_merge == "saliency":
        print(f"[RESCODE-Mixtral] merging via RESCODE saliency-weighted averaging (δ={rescode_delta})...")
        model = merge_by_groups_with_saliency_weighted(
            model, grouper=grouper,
            merging_layers=list(range(start_layer, model.config.num_hidden_layers)),
            delta=float(rescode_delta),
        )
    else:
        print(f"[RESCODE-Mixtral] merging via merge='{hc_merge}'...")
        model = merge_by_groups_within_and_across_models(
            mixtral_model=model,
            grouper=grouper,
            dataloader=cluster_dataloader,
            merge=hc_merge,
            mode=hc_mode,
            partition=1,
            dominant_alone=False,
            core_experts=dom_experts,
            ingredient="act",
        )

    # 4) Convert to RESCODE.
    print("[RESCODE-Mixtral] converting MoE blocks to RESCODE...")
    model = convert_mixtral_to_rescode(model, cluster_maps, star_cfg, preact_init_data=preact_init_data)

    # 4.5) Optional preact std calibration on the merged model.
    if star_cfg.preact_enabled and star_cfg.preact_star.normalize_by_preact_std:
        print("[RESCODE-Mixtral] calibrating preact std on merged model...")
        collect_preact_stats(model, cluster_dataloader)

    # 5) Fit codes / preact params (or load).
    do_fit_post_code = star_cfg.fit_codes and code_type != "none"
    do_fit_preact = star_cfg.preact_enabled and (
        star_cfg.preact_star.train_preact_bias
        or star_cfg.preact_star.train_preact_lowrank
        or star_cfg.preact_star.train_down_residual
    )
    layer_results = None
    if star_cfg.load_rescode_path is not None:
        print(f"[RESCODE-Mixtral] loading RESCODE state from {star_cfg.load_rescode_path}")
        load_rescode_state(model, star_cfg.load_rescode_path)
    elif do_fit_post_code or do_fit_preact:
        freeze_non_code_parameters(model, include_preact=star_cfg.preact_enabled)
        if fit_mode == "joint":
            print(f"[RESCODE-Mixtral] joint LM-loss fitting "
                  f"(lambda_lm={lambda_lm}, lambda_mse={lambda_mse}, "
                  f"epochs={joint_epochs}, "
                  f"val_n={joint_val_n_sentences}, "
                  f"val_every={joint_val_every})")
            fit_codes_joint_lm(
                model=model,
                teacher_buffers=teacher_buffers,
                calib_loader=dataloader,
                config=star_cfg,
                val_loader=val_loader,
                val_teacher_buffers=val_teacher_buffers,
                val_every=int(joint_val_every),
                online_teacher=online_teacher,
                val_online_teacher=val_online_teacher,
            )
        else:
            print("[RESCODE-Mixtral] layerwise fitting (post codes + preact, as configured)...")
            layer_results = fit_all_codes(model, teacher_buffers, star_cfg)
        del teacher_buffers
        if val_teacher_buffers is not None:
            del val_teacher_buffers
        if online_teacher is not None:
            online_teacher.cleanup()
            del teacher_model  # noqa: F821
            online_teacher = None
            val_online_teacher = None
        gc.collect()
        torch.cuda.empty_cache()

    if star_cfg.report_reconstruction_error:
        log_code_statistics(model)
    if star_cfg.preact_enabled and star_cfg.preact_star.report_preact_stats:
        log_preact_statistics(model)
    if star_cfg.preact_enabled and star_cfg.preact_star.report_giant_cluster_summary:
        log_giant_cluster_summary(cluster_maps, layer_results=layer_results)

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_path, "model.pth"))
        print(f"[RESCODE-Mixtral] saved model state_dict to {output_path}/model.pth")
    if star_cfg.save_rescode_path is not None:
        os.makedirs(os.path.dirname(star_cfg.save_rescode_path) or ".", exist_ok=True)
        save_rescode_state(model, star_cfg, star_cfg.save_rescode_path)
        print(f"[RESCODE-Mixtral] saved RESCODE state to {star_cfg.save_rescode_path}")

    if do_eval:
        if result_path:
            os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        if isinstance(task, tuple):
            task = list(task)
        results_payload = evaluate_fewshot(
            model, tokenizer=tokenizer, task=task, num_fewshot=num_fewshot,
            eval_batch_size=eval_batch_size, output_path=result_path, log=True,
        )
        report_eval_summary(
            results_payload=results_payload,
            task_list=task,
            result_path=result_path,
            header=(
                f"RESCODE-Mixtral | {model_name} | num_groups={num_average_groups} "
                f"| code_type={code_type}"
            ),
        )


if __name__ == "__main__":
    Fire(run_rescode_mixtral)
