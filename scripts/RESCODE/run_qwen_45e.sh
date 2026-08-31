#!/usr/bin/env bash
# ============================================================================
# Qwen1.5-MoE-A2.7B-Chat  →  60 / 45 experts  (25% compression)
#
# Defaults follow the training protocol in the paper's Appendix E, with the
# singleton ratio ρ=15% used under moderate compression (Section 5.7).
# Paper reference at this setting: Table 3, avg = 0.5878.
#
# Idea 1 (saliency-aware restoration): ρ=15% singleton pin, saliency-weighted base.
# Idea 2 (diagonal residual-code):     r=128, α=2.0, covariance init, target=gate_up_down.
# Idea 3 (output-guided fitting):      joint LM + small L_rec (λ_mse=0.01, λ_lm=1), 3 epochs.
#                                      Set LAMBDA_MSE=0 for the LM-only variant.
#
# Override anything via env vars, e.g.:
#   ALPHA=4.0 INIT=expert_svd_align bash scripts/RESCODE/run_qwen_45e.sh
# ============================================================================

set -e

# --- 1. Identity ----------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
export NUM_EXPERTS="${NUM_EXPERTS:-45}"
export GPU="${GPU:-0}"
export TAG="${TAG:-default}"

# --- 3. Idea 1 Step 1 (clustering + saliency pin) -------------------------
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
