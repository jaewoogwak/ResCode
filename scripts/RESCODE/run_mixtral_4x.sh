#!/usr/bin/env bash
# ============================================================================
# Mixtral-8x7B-v0.1  →  8 / 4 experts  (50% compression)
#
# Defaults follow the training protocol in the paper's Appendix E. Under
# aggressive compression the singleton ratio ρ (SALIENCY_TOP_TAU) is set to 0
# (Section 5.7); saliency-weighted base construction (Idea-1 Step-2) stays on.
# Paper reference at this setting: Table 3, avg = 0.6096.
#
# Idea 1: ρ=0, saliency-weighted base.
# Idea 2: r=128, α=1.0, covariance init, target=gate_up_down.
# Idea 3: joint LM + small L_rec (λ_mse=0.01, λ_lm=1), 3 epochs.
#         λ_mse>0 keeps a frozen teacher copy on GPU (~2× model memory);
#         set LAMBDA_MSE=0 for the LM-only variant if memory is tight.
# ============================================================================

set -e

# --- 1. Identity ----------------------------------------------------------
export MODEL="${MODEL:-mistralai/Mixtral-8x7B-v0.1}"
export NUM_EXPERTS="${NUM_EXPERTS:-4}"
export GPU="${GPU:-0}"
# CALIB_DATASET (c4|wikipedia|slimpajama|dclm|starcoder) selects the
# calibration corpus and, by default, also names the output dir.
export CALIB_DATASET="${CALIB_DATASET:-dclm}"
if [ -z "${TAG:-}" ]; then
  export TAG="calib-${CALIB_DATASET}"
fi

# --- 3. Idea 1 Step 1 (clustering — pin disabled at aggressive ratio) -----
export CLUSTER_BASE="${CLUSTER_BASE:-rescode}"
export SALIENCY_TOP_TAU="${SALIENCY_TOP_TAU:-0}"

# --- 4. Idea 1 Step 2 (saliency-weighted base construction) ---------------
export BASE_WEIGHT="${BASE_WEIGHT:-saliency}"
export DELTA="${DELTA:-1e-3}"

# --- 5. Idea 2 (diagonal residual-code) -----------------------------------
export RANK="${RANK:-128}"
export ALPHA="${ALPHA:-1.0}"
export INIT="${INIT:-covariance}"
export TARGET="${TARGET:-gate_up_down}"

# --- 6. Idea 3 (output-guided fitting) ------------------------------------
export JOINT_EPOCHS="${JOINT_EPOCHS:-3}"
export JOINT_GRAD_ACCUM="${JOINT_GRAD_ACCUM:-8}"
export LAMBDA_LM="${LAMBDA_LM:-1.0}"
export LAMBDA_MSE="${LAMBDA_MSE:-0.01}"

# --- 2. Data --------------------------------------------------------------
export N_SENTENCES="${N_SENTENCES:-2080}"

cd "$(dirname "$0")/../.."
exec bash scripts/RESCODE/run.sh
