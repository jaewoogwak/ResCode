#!/usr/bin/env bash
# ============================================================================
# Qwen1.5-MoE-A2.7B-Chat  →  60 / 30 experts  (50% compression)
#
# Defaults follow the training protocol in the paper's Appendix E. Under
# aggressive compression the singleton ratio ρ (SALIENCY_TOP_TAU) is set to 0
# (Idea-1 Step-1 is reduced to descriptor clustering), while saliency-weighted
# base construction (Idea-1 Step-2) is kept on.
# Paper reference at this setting: Table 3, avg = 0.5466.
#
# Idea 1: ρ=0, saliency-weighted base (Step-2 still applies, Step-1 pin off).
# Idea 2: r=128, α=2.0, covariance init, target=gate_up_down.
# Idea 3: joint LM + small L_rec (λ_mse=0.01, λ_lm=1), 3 epochs.
#         Set LAMBDA_MSE=0 for the LM-only variant if memory is tight.
#
# Override via env vars.
# ============================================================================

set -e

# --- 1. Identity ----------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
export NUM_EXPERTS="${NUM_EXPERTS:-30}"
export GPU="${GPU:-0}"
export TAG="${TAG:-default}"

# --- 3. Idea 1 Step 1 (clustering — pin disabled at aggressive ratio) -----
export CLUSTER_BASE="${CLUSTER_BASE:-rescode}"     # still rescode so saliency is computed
export SALIENCY_TOP_TAU="${SALIENCY_TOP_TAU:-0}"  # τ=0 — no singleton pin

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
