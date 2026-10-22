#!/usr/bin/env bash
# ============================================================================
# Qwen1.5-MoE-A2.7B-Chat  ->  60 / 30 experts  (50% compression)
#
# Defaults follow the training protocol in the paper's Appendix E.
# Paper reference at this setting: Table 3, avg = 0.5466.
# This configuration measured avg = 0.5487 (seed 0 / calib seed 42) at
# 50.02% expert-memory compression (residual bases and codes included).
#
# Idea 1: rho = 0.15 singleton pin, saliency a_i / routing frequency omega_i /
#         dist_out over the full calibration set, saliency-weighted base.
# Idea 2: r = 128, alpha = 4.0, covariance init (residuals w.r.t. S_G), gate/up/down.
# Idea 3: L_rec + lambda L_LM with lambda = 1 (REC_MODE=local), 3 epochs,
#         8 sequences per optimizer step.
#
# Override via env vars.
# ============================================================================

set -e

# --- 1. Identity ----------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
export NUM_EXPERTS="${NUM_EXPERTS:-30}"
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
export ALPHA="${ALPHA:-4.0}"
export INIT="${INIT:-covariance}"
export TARGET="${TARGET:-gate_up_down}"

# --- 6. Idea 3 (output-guided fitting) ------------------------------------
export JOINT_EPOCHS="${JOINT_EPOCHS:-3}"
export JOINT_GRAD_ACCUM="${JOINT_GRAD_ACCUM:-8}"
export LAMBDA_LM="${LAMBDA_LM:-1.0}"
export LAMBDA_MSE="${LAMBDA_MSE:-1.0}"
export REC_MODE="${REC_MODE:-local}"

# --- 2. Data --------------------------------------------------------------
export N_SENTENCES="${N_SENTENCES:-2400}"

cd "$(dirname "$0")/../.."
exec bash scripts/RESCODE/run.sh
