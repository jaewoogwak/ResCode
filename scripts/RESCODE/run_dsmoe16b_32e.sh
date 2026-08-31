#!/usr/bin/env bash
# ============================================================================
# DeepSeek-MoE-16B-Base (V1)  →  64 / 32 experts  (50% compression)
#
# Defaults follow the training protocol in the paper's Appendix E.
# Paper reference at this setting: Table 9, avg = 0.5018.
#
# DeepSeek-MoE-16B: 28 layers, layer 0 dense, 27 MoE layers each with
#   - n_routed_experts=64, top_k=6, hidden=2048, moe_intermediate=1408
#   - n_shared_experts=2 (kept as-is; not merged)
#
# Stock modeling does NOT accept `output_router_logits=True`. Prefer
# CLUSTER_BASE=rescode (the default) or expert-output.
#
# Override via env vars.
# ============================================================================

set -e

# --- 1. Identity ----------------------------------------------------------
export MODEL="${MODEL:-deepseek-ai/deepseek-moe-16b-base}"
export NUM_EXPERTS="${NUM_EXPERTS:-32}"             # 50% of 64
export GPU="${GPU:-0}"
export TAG="${TAG:-default}"

# --- 3. Idea 1 Step 1 (saliency-aware cluster formation) ------------------
export CLUSTER_BASE="${CLUSTER_BASE:-rescode}"
export SALIENCY_TOP_TAU="${SALIENCY_TOP_TAU:-15.0}"

# --- 4. Idea 1 Step 2 (saliency-weighted base construction) ---------------
export BASE_WEIGHT="${BASE_WEIGHT:-saliency}"
export DELTA="${DELTA:-1e-3}"

# --- 5. Idea 2 (diagonal residual-code) -----------------------------------
export RANK="${RANK:-128}"
export ALPHA="${ALPHA:-2.0}"
export INIT="${INIT:-covariance}"
export TARGET="${TARGET:-gate_up_down}"

# --- 6. Idea 3 (output-guided fitting) ------------------------------------
export JOINT_EPOCHS="${JOINT_EPOCHS:-3}"
export JOINT_GRAD_ACCUM="${JOINT_GRAD_ACCUM:-8}"
export LAMBDA_LM="${LAMBDA_LM:-1.0}"
export LAMBDA_MSE="${LAMBDA_MSE:-0.01}"

# --- 2. Data --------------------------------------------------------------
export N_SENTENCES="${N_SENTENCES:-2400}"

cd "$(dirname "$0")/../.."
exec bash scripts/RESCODE/run.sh
