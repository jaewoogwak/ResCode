#!/usr/bin/env bash
# ============================================================================
# RESCODE — generic launcher.
# Reproduces the paper-aligned RESCODE pipeline (I1 + I2 + I3) on any SMoE
# model and compression ratio.
#
# All knobs follow the paper's notation. Internal Python flag names are kept
# for backward compatibility, but the user-facing shell variables match the
# paper.
#
#   I1  Saliency-aware restoration
#       Step 1: cluster formation with output similarity + singleton pin ρ
#       Step 2: saliency-weighted base construction
#               α_i^G = (a_i + δ) / Σ_j (a_j + δ)
#   I2  Diagonal Residual-code:    D_i ≈ B_G · diag(z_i) · A_G
#   I3  Output-guided fitting:     L = λ_mse · Σ_G L_rec(G) + λ_lm · L_LM
#
# Override anything via env vars. Examples:
#   MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat NUM_EXPERTS=30 GPU=0 bash scripts/RESCODE/run.sh
#   MODEL=mistralai/Mixtral-8x7B-v0.1 NUM_EXPERTS=4 SALIENCY_TOP_TAU=0 bash ...
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."   # to repo root

# ---------------------------------------------------------------------------
# 1. Experiment identity
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
NUM_EXPERTS="${NUM_EXPERTS:-30}"          # compressed expert count per layer
GPU="${GPU:-0}"
TAG="${TAG:-}"
SEED="${SEED:-0}"                         # optimization seed
CALIB_SEED="${CALIB_SEED:-42}"            # calibration-sampling seed; the paper's
                                          # multi-seed runs (App. F.5) vary SEED
                                          # and CALIB_SEED jointly

# ---------------------------------------------------------------------------
# 2. Data / calibration
# ---------------------------------------------------------------------------
TASK="${TASK:-winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte}"

# CALIB_DATASET picks a registered calibration corpus by short name (paper
# default: dclm). Prepare a corpus once with
#   python scripts/RESCODE/prepare_calib_datasets.py --dataset <name>
# (`c4` instead expects the pre-downloaded shard below). TRAIN_DATA overrides
# the resolved path directly.
CALIB_DATASET="${CALIB_DATASET:-dclm}"
case "$CALIB_DATASET" in
  c4)           _DEFAULT_TRAIN="rescode/data/c4-train.00000-of-01024.json" ;;
  wikipedia)    _DEFAULT_TRAIN="rescode/data/wikipedia-calib.jsonl" ;;
  slimpajama)   _DEFAULT_TRAIN="rescode/data/slimpajama-calib.jsonl" ;;
  dclm)         _DEFAULT_TRAIN="rescode/data/dclm-calib.jsonl" ;;
  starcoder)    _DEFAULT_TRAIN="rescode/data/starcoder-calib.jsonl" ;;
  *) echo "[RESCODE] unknown CALIB_DATASET=$CALIB_DATASET" >&2; exit 1 ;;
esac
TRAIN_DATA="${TRAIN_DATA:-$_DEFAULT_TRAIN}"
if [ ! -f "$TRAIN_DATA" ]; then
  echo "[RESCODE] calibration file missing: $TRAIN_DATA" >&2
  echo "[RESCODE]   run: python scripts/RESCODE/prepare_calib_datasets.py --dataset $CALIB_DATASET" >&2
  exit 1
fi

# VAL_DATA: held-out validation calibration corpus for joint LM-loss val.
# Point it to a separately prepared jsonl file. When empty (the default),
# joint validation (and best-val checkpoint restore) is disabled and
# training proceeds normally.
VAL_DATA="${VAL_DATA:-}"
if [ -n "$VAL_DATA" ] && [ ! -f "$VAL_DATA" ]; then
  echo "[RESCODE] val calibration file missing: $VAL_DATA — joint validation disabled." >&2
  VAL_DATA=""
fi
N_SENTENCES="${N_SENTENCES:-2400}"        # train calibration blocks
CLUSTER_N_SENTENCES="${CLUSTER_N_SENTENCES:-32}"
TRAIN_BATCH="${TRAIN_BATCH:-2}"
EVAL_BATCH="${EVAL_BATCH:-16}"
MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-32768}"
MAX_TOKENS_PER_CLUSTER="${MAX_TOKENS_PER_CLUSTER:-32768}"

# ---------------------------------------------------------------------------
# 3. Clustering — Idea 1 Step 1 (saliency-aware cluster formation)
# ---------------------------------------------------------------------------
CLUSTER_BASE="${CLUSTER_BASE:-rescode}"           # rescode | expert-output
                                                  #   rescode = output-similarity + singleton pin (Step 1)
                                                  #   expert-output = output-similarity only (no saliency pin)
SALIENCY_TOP_TAU="${SALIENCY_TOP_TAU:-15.0}"     # singleton ratio ρ, in %: top-saliency experts pinned
                                                  # as singletons. ρ=15% at moderate compression,
                                                  # ρ=0 at aggressive compression (paper Sec. 5.7)
EXCL_MODE="${EXCL_MODE:-per-layer}"
DESCRIPTOR_TOKENS="${DESCRIPTOR_TOKENS:-all}"    # descriptor token set for output-similarity clustering
                                                  #   all    = every expert evaluated on the SAME
                                                  #            calibration tokens (paper, default)
                                                  #   routed = mean over tokens routed to each expert

# ---------------------------------------------------------------------------
# 4. Base construction — Idea 1 Step 2 (saliency-weighted base)
# ---------------------------------------------------------------------------
BASE_WEIGHT="${BASE_WEIGHT:-saliency}"           # saliency | freq
                                                  #   saliency = α_i = (a_i+δ)/Σ(a_j+δ)  (paper)
                                                  #   freq     = usage-frequency weighting (HC-SMoE-style)
DELTA="${DELTA:-1e-3}"                           # δ floor in α_i (Step 2)

# ---------------------------------------------------------------------------
# 5. Residual structure — Idea 2 (diagonal residual-code)
#    D_i ≈ B_G · diag(z_i) · A_G
# ---------------------------------------------------------------------------
PREACT_STAR="${PREACT_STAR:-True}"               # Idea 2 on/off (False = no residual code)
RANK="${RANK:-128}"                              # r — A_G ∈ R^{r×d_in}, B_G ∈ R^{d_out×r}, z_i ∈ R^r
ALPHA="${ALPHA:-2.0}"                            # residual scale
INIT="${INIT:-covariance}"                       # covariance | lora_zero_B | expert_svd_align
INIT_ALIGN="${INIT_ALIGN:-hungarian}"            # only used by expert_svd_align init
INIT_STD="${INIT_STD:-1e-3}"                     # only used by lora_zero_B / zero_L / zero_R inits
DOWN_INIT="${DOWN_INIT:-covariance}"             # down-proj residual init: covariance | zero_L | zero_R
TARGET="${TARGET:-gate_up_down}"                 # which projection branches get a residual
                                                  # gate | up | down | gate_up | gate_up_down

# Aggregation choice for the cluster forward (Idea 2 path)
#   postact (default) : POST-activation aggregation matching paper Eq.(4)
#                       exactly: for each cluster member j∈C∩T(x), evaluate
#                       SwiGLU with W_hat_j = W̄+B·diag(z_j)·A and take a
#                       π_j-weighted sum. Costs K× SwiGLU evaluations per
#                       cluster per token (memory/throughput scales with
#                       cluster size).
#   preact            : router-conditioned PRE-activation modulation.
#                       SwiGLU is evaluated once per cluster on the
#                       rho-aggregated pre-activations. Fast (1× SwiGLU
#                       per cluster per token), first-order approximation
#                       of paper Eq.(4) under nonlinear SwiGLU.
AGGREGATION="${AGGREGATION:-postact}"            # postact | preact
case "$AGGREGATION" in
  preact)  POSTACT_FLAG="--paper_postact_aggregation=False" ;;
  postact) POSTACT_FLAG="--paper_postact_aggregation=True"  ;;
  *) echo "[RESCODE] unknown AGGREGATION=$AGGREGATION (expected preact|postact)" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# 6. Residual training — Idea 3 (output-guided fitting)
#    L = λ_mse · Σ_G L_rec(G) + λ_lm · L_LM
# ---------------------------------------------------------------------------
FIT_PREACT="${FIT_PREACT:-True}"                 # Idea 3 on/off (False = no LM-guided fitting)
JOINT_EPOCHS="${JOINT_EPOCHS:-3}"                # n_ep = 3 (paper App. E)
JOINT_GRAD_ACCUM="${JOINT_GRAD_ACCUM:-8}"
LAMBDA_LM="${LAMBDA_LM:-1.0}"                    # λ_lm — LM cross-entropy weight
LAMBDA_MSE="${LAMBDA_MSE:-0.01}"                 # λ_mse — coefficient on L_rec(G). Small by design so
                                                  # results stay close to LM-only while the paper's
                                                  # output-reconstruction term is exercised. λ_mse>0
                                                  # keeps a frozen teacher copy on GPU (~2x model
                                                  # memory); set to 0.0 (LM-only) if memory is tight
TEACHER_MODE="${TEACHER_MODE:-online}"           # online | offline (online keeps unmerged model on GPU)
JOINT_VAL_N="${JOINT_VAL_N:-200}"
JOINT_VAL_EVERY="${JOINT_VAL_EVERY:-10}"
LOG_EVERY="${LOG_EVERY:-1}"

# ---------------------------------------------------------------------------
# 7. Optimizer / scheduler / regularization
# ---------------------------------------------------------------------------
LR="${LR:-1.5e-4}"                               # learning rate for A,B,z
LR_DOWN="${LR_DOWN:-$LR}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"           # constant | cosine | linear
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
L2="${L2:-1e-4}"                                 # L2 on A,B,z params
L2_DOWN="${L2_DOWN:-$L2}"

# ---------------------------------------------------------------------------
# 8. Output / load-mode
# ---------------------------------------------------------------------------
RESULT_NAME="${RESULT_NAME:-result.txt}"

# Pick model family: detect from MODEL string
if [[ "$MODEL" == *Mixtral* || "$MODEL" == *mixtral* ]]; then
  MODEL_FAMILY="mixtral"
  PY_ENTRY="rescode/rescode_mixtral.py"
elif [[ "$MODEL" == *Qwen3* || "$MODEL" == *qwen3* ]]; then
  MODEL_FAMILY="qwen3"
  PY_ENTRY="rescode/rescode_qwen3.py"
elif [[ "$MODEL" == *eepSeek* || "$MODEL" == *deepseek* ]]; then
  MODEL_FAMILY="deepseek"
  PY_ENTRY="rescode/rescode_deepseek.py"
else
  MODEL_FAMILY="qwen"
  PY_ENTRY="rescode/rescode_qwen.py"
fi

OUT_SUFFIX="rescode-${MODEL_FAMILY}-${NUM_EXPERTS}e"
[ -n "$TAG" ] && OUT_SUFFIX="${OUT_SUFFIX}-${TAG}"
OUT="results/${MODEL_FAMILY}/${OUT_SUFFIX}"
mkdir -p "$OUT"

# Optional reload mode: LOAD=auto -> reuse $OUT/rescode_state.pt and skip fitting
LOAD="${LOAD:-}"
LOAD_ARGS=""
if [ -n "$LOAD" ]; then
  if [ "$LOAD" = "auto" ] || [ "$LOAD" = "1" ]; then
    LOAD_PATH="$OUT/rescode_state.pt"
  else
    LOAD_PATH="$LOAD"
  fi
  LOAD_ARGS="--load_rescode_path=$LOAD_PATH"
fi

# ---------------------------------------------------------------------------
# 0. Environment (HF cache, accelerate)
#    User-specific paths — override via env vars before running.
#    ACCELERATE: path to the accelerate executable inside your virtualenv
#                (e.g. $HOME/venvs/rescode/bin/accelerate)
#    HF_HOME   : HuggingFace cache (where models are downloaded)
# ---------------------------------------------------------------------------
ACC="${ACCELERATE:-accelerate}"          # 'accelerate' on PATH, or a virtualenv path
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PORT="${PORT:-$((30000 + 10#$GPU * 100 + NUM_EXPERTS))}"

# warmup_steps = 5 % of optimizer steps
BATCHES_PER_EPOCH=$(( (N_SENTENCES + TRAIN_BATCH - 1) / TRAIN_BATCH ))
EST_OPT_STEPS=$(( (BATCHES_PER_EPOCH * JOINT_EPOCHS + JOINT_GRAD_ACCUM - 1) / JOINT_GRAD_ACCUM ))
WARMUP_STEPS="${WARMUP_STEPS:-$((EST_OPT_STEPS / 20))}"
[ "$WARMUP_STEPS" -lt 1 ] && WARMUP_STEPS=1

# ---------------------------------------------------------------------------
# Summary echo
# ---------------------------------------------------------------------------
echo "[RESCODE] ============================================================"
echo "[RESCODE]   model=$MODEL  family=$MODEL_FAMILY"
echo "[RESCODE]   num_experts=$NUM_EXPERTS  gpu=$GPU  out=$OUT"
echo "[RESCODE]   calib_dataset=$CALIB_DATASET  train_data=$TRAIN_DATA"
echo "[RESCODE]   ----- Idea 1 (saliency-aware restoration) -----"
echo "[RESCODE]   cluster_base=$CLUSTER_BASE  saliency_top_tau=$SALIENCY_TOP_TAU  base_weight=$BASE_WEIGHT  delta=$DELTA  descriptor_tokens=$DESCRIPTOR_TOKENS"
echo "[RESCODE]   ----- Idea 2 (diagonal residual-code) ---------"
echo "[RESCODE]   preact_star=$PREACT_STAR  rank=$RANK  alpha=$ALPHA  init=$INIT  down_init=$DOWN_INIT  target=$TARGET  aggregation=$AGGREGATION"
echo "[RESCODE]   ----- Idea 3 (output-guided fitting) ----------"
echo "[RESCODE]   fit_preact=$FIT_PREACT  epochs=$JOINT_EPOCHS  grad_accum=$JOINT_GRAD_ACCUM  est_opt_steps=$EST_OPT_STEPS"
echo "[RESCODE]   lambda_lm=$LAMBDA_LM  lambda_mse=$LAMBDA_MSE  teacher=$TEACHER_MODE"
echo "[RESCODE]   lr=$LR  sched=$LR_SCHEDULER  warmup=$WARMUP_STEPS  wd=$WEIGHT_DECAY  l2=$L2"
echo "[RESCODE] ============================================================"

# ---------------------------------------------------------------------------
# Launch
# Internal Python flags follow the legacy name scheme; user-facing knobs are
# remapped above.
# ---------------------------------------------------------------------------
"$ACC" launch --config_file static/single_gpu.yaml \
  --gpu_ids "$GPU" \
  --main_process_port "$PORT" "$PY_ENTRY" \
  --model_name="$MODEL" \
  --task="$TASK" \
  --seed="$SEED" \
  --calib_seed="$CALIB_SEED" \
  --train_data_path="$TRAIN_DATA" \
  --val_data_path="$VAL_DATA" \
  --n_sentences="$N_SENTENCES" \
  --cluster_n_sentences="$CLUSTER_N_SENTENCES" \
  --train_batch_size="$TRAIN_BATCH" \
  --eval_batch_size="$EVAL_BATCH" \
  --max_fit_tokens_per_layer="$MAX_TOKENS_PER_LAYER" \
  --max_fit_tokens_per_cluster="$MAX_TOKENS_PER_CLUSTER" \
  --num_average_groups="$NUM_EXPERTS" \
  --similarity_base="$CLUSTER_BASE" \
  --exclude_top_percent="$SALIENCY_TOP_TAU" \
  --excl_mode="$EXCL_MODE" \
  --cluster=hierarchical \
  --linkage=average \
  --descriptor_tokens="$DESCRIPTOR_TOKENS" \
  --hc_merge="$BASE_WEIGHT" \
  --rescode_delta="$DELTA" \
  --preact_star=$PREACT_STAR \
  --preact_mode=token_adaptive_lowrank \
  --preact_lowrank_target="$TARGET" \
  --preact_lowrank_rank="$RANK" \
  --preact_lowrank_alpha="$ALPHA" \
  --preact_lowrank_init="$INIT" \
  --preact_lowrank_init_alignment="$INIT_ALIGN" \
  --preact_lowrank_init_std="$INIT_STD" \
  --preact_lowrank_dual=False \
  $POSTACT_FLAG \
  --preact_down_residual_rank="$RANK" \
  --preact_down_residual_alpha="$ALPHA" \
  --preact_down_residual_init="$DOWN_INIT" \
  --preact_lowrank_l2_lambda="$L2" \
  --preact_down_l2_lambda="$L2_DOWN" \
  --preact_normalize_by_std=True \
  --preact_grad_clip_norm="$GRAD_CLIP" \
  --preact_lr_lowrank="$LR" \
  --preact_lr_down="$LR_DOWN" \
  --code_type=none \
  --fit_codes=False \
  --fit_preact=$FIT_PREACT \
  --fit_mode=joint \
  --teacher_mode="$TEACHER_MODE" \
  --joint_epochs="$JOINT_EPOCHS" \
  --joint_grad_accum="$JOINT_GRAD_ACCUM" \
  --lambda_lm="$LAMBDA_LM" \
  --lambda_mse="$LAMBDA_MSE" \
  --joint_val_n_sentences="$JOINT_VAL_N" \
  --joint_val_every="$JOINT_VAL_EVERY" \
  --lr_scheduler="$LR_SCHEDULER" \
  --warmup_steps="$WARMUP_STEPS" \
  --fit_weight_decay="$WEIGHT_DECAY" \
  --log_every="$LOG_EVERY" \
  --do_eval=True \
  --output_path="$OUT" \
  --save_rescode_path="$OUT/rescode_state.pt" \
  --result_path="$OUT/$RESULT_NAME" \
  $LOAD_ARGS \
  |& tee "$OUT/log.txt"
